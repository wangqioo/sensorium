"""
LLM 进程入口（与 run.py 分开运行）

用法：
  # 开发调试（HuggingFace 模型）
  python run_llm.py --model Qwen/Qwen2.5-3B-Instruct

  # Jetson 生产部署（GGUF 量化）
  python run_llm.py --model models/qwen2.5-7b-q4_k_m.gguf --backend llama_cpp

  # 两个进程在不同终端里同时跑：
  terminal 1: python run.py --imu-checkpoint checkpoints/imu/best.ckpt ...
  terminal 2: python run_llm.py --model Qwen/Qwen2.5-3B-Instruct
"""

from sensorium.llm.process import parse_args, LLMProcess
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
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
