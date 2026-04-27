"""
LLM 独立进程

通过 ZMQ 接收来自主运行时的传感器 token 快照，
维护滚动上下文窗口，定期触发 LLM 推理，
把动作指令/生成文字发回主进程。

为什么是独立进程：
  - LLM 推理（~300ms）不能阻塞反射引擎（<20ms）
  - LLM 的 GPU 显存和主进程的编码器 GPU 显存分开管理
  - 独立进程崩溃不影响反射层继续运行

运行：
  python -m sensorium.llm.process --model Qwen/Qwen2.5-7B-Instruct
  python -m sensorium.llm.process --model path/to/model.gguf --backend llama_cpp
"""

import argparse
import asyncio
import json
import logging
import time
from typing import Any

import zmq
import zmq.asyncio

from .context import ContextWindow
from .vocab import SensoriumTokenizer

logger = logging.getLogger(__name__)


# ——— 动作词汇表（LLM 从这里选一个输出）———
# 比让 LLM 自由生成文字快 10 倍，适合实时控制

ACTION_VOCAB = [
    "none",           # 不做任何事
    "wag",            # 轻摇/表示开心
    "calm",           # 进入平静模式
    "alert",          # 警戒/注意
    "speak",          # 触发语音输出（配合 text 字段）
    "turn_head",      # 转头朝向声音
    "flinch",         # 缩避
    "sleep",          # 进入休眠
    "wake",           # 唤醒
    "purr",           # 发出满足声
]


# ——— 模型后端 ———

class TransformersBackend:
    """用 HuggingFace Transformers 跑 LLM（开发调试用）。"""

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"加载模型：{model_path}")
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, input_ids: list[int], max_new_tokens: int = 64) -> str:
        import torch

        ids = torch.tensor([input_ids], dtype=torch.long).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.hf_tokenizer.eos_token_id,
            )
        new_ids = out[0][len(input_ids):]
        return self.hf_tokenizer.decode(new_ids, skip_special_tokens=True)


class LlamaCppBackend:
    """用 llama-cpp-python 跑量化模型（Jetson Orin 部署用）。"""

    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096):
        from llama_cpp import Llama

        logger.info(f"加载 GGUF 模型：{model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,   # -1 = 全部层放 GPU
            n_ctx=n_ctx,
            verbose=False,
        )

    def generate(self, input_ids: list[int], max_new_tokens: int = 64) -> str:
        out = self.llm(
            input_ids,
            max_tokens=max_new_tokens,
            temperature=0.0,             # 贪心解码，稳定性更好
            echo=False,
        )
        return out["choices"][0]["text"].strip()


# ——— 系统提示词 ———

SYSTEM_PROMPT = """你是一个具身 AI 机器人的感知推理核心。

你会持续收到多模态传感器 token 流：
  [VIS_xxx] = 视觉原子（摄像头看到的场景模式）
  [AUD_xxx] = 听觉原子（麦克风捕捉的声音模式）
  [IMU_xxx] = 运动原子（当前身体运动状态）
  [TAC_xxx] = 触觉原子（身体被触碰的模式）

根据当前传感器状态和历史上下文，从以下动作中选择最合适的一个输出：
  none / wag / calm / alert / speak / turn_head / flinch / sleep / wake / purr

如果选择 speak，在动作后附上要说的话，格式：speak: <内容>
否则只输出动作名称，不要其他内容。"""


# ——— LLM 进程主类 ———

class LLMProcess:
    """
    Args:
        model_path:   模型路径（HuggingFace model ID 或 .gguf 文件路径）
        backend:      "transformers" 或 "llama_cpp"
        sub_addr:     ZMQ 订阅地址（接收传感器 token）
        pub_addr:     ZMQ 发布地址（发送动作指令）
        inference_hz: LLM 推理频率（Hz），默认 3Hz（每 333ms 推理一次）
        context_secs: 上下文保留时长（秒）
        vocab_path:   SensoriumTokenizer 的基础模型路径
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "transformers",
        sub_addr: str = "tcp://127.0.0.1:5555",
        pub_addr: str = "tcp://127.0.0.1:5556",
        inference_hz: float = 3.0,
        context_secs: float = 30.0,
        vocab_path: str | None = None,
    ):
        self.model_path = model_path
        self.inference_hz = inference_hz

        # 模型后端
        if backend == "llama_cpp":
            self._backend = LlamaCppBackend(model_path)
        else:
            self._backend = TransformersBackend(model_path)

        # 分词器（用基础模型路径，或和 LLM 同路径）
        tok_path = vocab_path or model_path
        self._tokenizer = SensoriumTokenizer.from_pretrained(tok_path)

        # 上下文窗口
        self._context = ContextWindow(max_seconds=context_secs)

        # ZMQ
        self._zmq_ctx = zmq.asyncio.Context()
        self._sub = self._zmq_ctx.socket(zmq.SUB)
        self._sub.connect(sub_addr)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self._pub = self._zmq_ctx.socket(zmq.PUB)
        self._pub.bind(pub_addr)

        self._latest_snapshot: dict[str, list[int]] = {}
        self._running = False

        # 系统提示 token（只编码一次）
        self._system_ids = self._tokenizer.tokenizer.encode(
            SYSTEM_PROMPT, add_special_tokens=True
        )

    async def _receive_loop(self) -> None:
        """持续接收主进程发来的 token 快照，更新上下文。"""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._sub.recv(), timeout=1.0)
                snapshot: dict[str, list[int]] = json.loads(msg.decode())
                self._latest_snapshot = snapshot

                # 把新到的 token 推入上下文窗口
                for modality, tokens in snapshot.items():
                    if tokens:
                        self._context.push_sensor(modality, tokens)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning(f"接收 token 快照出错：{e}")

    async def _inference_loop(self) -> None:
        """以固定频率触发 LLM 推理，把动作指令发回主进程。"""
        interval = 1.0 / self.inference_hz

        while self._running:
            t0 = time.monotonic()

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._run_inference
                )
                if result:
                    await self._pub.send(json.dumps(result).encode())
                    logger.debug(f"LLM 输出：{result}")

            except Exception as e:
                logger.error(f"LLM 推理出错：{e}")

            elapsed = time.monotonic() - t0
            sleep = max(0.0, interval - elapsed)
            await asyncio.sleep(sleep)

    def _run_inference(self) -> dict[str, Any] | None:
        """同步推理，在线程池里执行（不阻塞事件循环）。"""
        if not self._latest_snapshot:
            return None

        # 构建输入序列：系统提示 + 上下文历史 + 当前快照
        input_ids = list(self._system_ids)
        context_ids = self._context.build_sequence(
            current_snapshot=self._latest_snapshot,
            tokenizer=self._tokenizer,
        )
        input_ids.extend(context_ids)

        # 截断：保留最后 N 个 token（LLM 上下文窗口限制）
        max_ctx = 3500
        if len(input_ids) > max_ctx:
            input_ids = input_ids[-max_ctx:]

        raw = self._backend.generate(input_ids, max_new_tokens=32)
        return self._parse_output(raw)

    def _parse_output(self, raw: str) -> dict[str, Any]:
        """把 LLM 的原始文字输出解析成结构化动作指令。"""
        raw = raw.strip().lower()

        # 检查是否是合法动作
        for action in ACTION_VOCAB:
            if raw.startswith(action):
                result: dict[str, Any] = {"action": action}
                # speak: <内容> 格式
                if action == "speak" and ":" in raw:
                    result["text"] = raw.split(":", 1)[1].strip()
                return result

        # 兜底：不做任何事
        logger.debug(f"LLM 输出无法解析为动作：{raw!r}，回退到 none")
        return {"action": "none"}

    async def run(self) -> None:
        self._running = True
        logger.info("LLM 进程启动，等待 token 流...")

        try:
            await asyncio.gather(
                self._receive_loop(),
                self._inference_loop(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("LLM 进程正在关闭...")
        finally:
            self._running = False
            self._zmq_ctx.destroy()


# ——— 入口 ———

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensorium LLM 推理进程")
    p.add_argument("--model",         type=str, required=True,
                   help="模型路径（HuggingFace ID 或 .gguf 文件）")
    p.add_argument("--backend",       type=str, default="transformers",
                   choices=["transformers", "llama_cpp"])
    p.add_argument("--vocab-path",    type=str, default=None,
                   help="SensoriumTokenizer 基础模型路径，默认与 --model 相同")
    p.add_argument("--sub-addr",      type=str, default="tcp://127.0.0.1:5555")
    p.add_argument("--pub-addr",      type=str, default="tcp://127.0.0.1:5556")
    p.add_argument("--inference-hz",  type=float, default=3.0)
    p.add_argument("--context-secs",  type=float, default=30.0)
    p.add_argument("--device",        type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    proc = LLMProcess(
        model_path=args.model,
        backend=args.backend,
        sub_addr=args.sub_addr,
        pub_addr=args.pub_addr,
        inference_hz=args.inference_hz,
        context_secs=args.context_secs,
        vocab_path=args.vocab_path,
    )
    asyncio.run(proc.run())
