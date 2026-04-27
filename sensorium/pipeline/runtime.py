"""
Sensorium 运行时主管道

替代 ROS2 的轻量级 asyncio 事件循环。
所有传感器驱动并行运行，原子编码器异步消费数据，
反射引擎直接在热路径上调用，LLM 通过 ZMQ 跨进程通信。

延迟预算：
  传感器采集 → 原子编码：  <10ms（GPU 推理）
  原子编码   → 反射引擎：  <1ms（字典查表）
  反射引擎   → 动作执行：  <5ms（线程调度）
  总反射延迟：              <20ms（远低于动物反射弧 ~50ms）

  LLM 推理：~300ms（异步，不阻塞上面的路径）
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import zmq
import zmq.asyncio

from ..drivers import CameraDriver, MicrophoneDriver, IMUDriver, TactileDriver
from ..encoders import IMUEncoder, TactileEncoder, VisualEncoder, AudioEncoder
from ..reflex import ReflexEngine

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    # 传感器
    camera_device: int = 0
    camera_fps: int = 30
    camera_width: int = 320
    camera_height: int = 240

    mic_device: int | None = None
    mic_sample_rate: int = 24000
    mic_chunk_ms: int = 100

    imu_bus: int = 1
    imu_hz: int = 100

    tactile_port: str = "/dev/ttyUSB0"
    tactile_baudrate: int = 921600

    # 编码器
    imu_checkpoint: str = ""
    tactile_checkpoint: str = ""
    device: str = "cuda"          # "cuda" / "cpu" / "mps"

    # LLM 进程通信（ZMQ）
    zmq_pub_addr: str = "tcp://127.0.0.1:5555"  # 我们 publish 原子 token
    zmq_sub_addr: str = "tcp://127.0.0.1:5556"  # 接收 LLM 的回复

    # 时序
    imu_window_size: int = 50      # IMU 编码窗口采样数
    tactile_window_size: int = 50  # 触觉编码窗口帧数
    llm_publish_hz: float = 5.0   # 每秒向 LLM 推送几次 token 快照


class SensoriumRuntime:
    """
    主运行时。负责：
      1. 启动所有传感器驱动（各自独立线程）
      2. asyncio 事件循环消费传感器数据
      3. 调用原子编码器（GPU 推理）
      4. 触发反射引擎（同步调用，热路径）
      5. 定期向 LLM 进程发布 token 快照（ZMQ）
      6. 接收 LLM 的输出并执行动作

    使用：
        cfg = RuntimeConfig(imu_checkpoint="checkpoints/imu/best.ckpt", ...)
        runtime = SensoriumRuntime(cfg, robot=MyRobot())
        asyncio.run(runtime.run())
    """

    def __init__(self, cfg: RuntimeConfig, robot=None):
        self.cfg = cfg
        self.robot = robot
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # 传感器驱动
        self.cam_driver = CameraDriver(
            device_id=cfg.camera_device,
            fps=cfg.camera_fps,
            width=cfg.camera_width,
            height=cfg.camera_height,
        )
        self.mic_driver = MicrophoneDriver(
            device=cfg.mic_device,
            sample_rate=cfg.mic_sample_rate,
            chunk_ms=cfg.mic_chunk_ms,
        )
        self.imu_driver = IMUDriver(bus_id=cfg.imu_bus, sample_hz=cfg.imu_hz)
        self.tac_driver = TactileDriver(port=cfg.tactile_port, baudrate=cfg.tactile_baudrate)

        # 原子编码器（延迟加载，避免启动时占用显存）
        self._imu_enc: IMUEncoder | None = None
        self._tac_enc: TactileEncoder | None = None
        self._vis_enc: VisualEncoder | None = None

        # 反射引擎
        self.reflex = ReflexEngine()

        # 滑动缓冲区（用于积累窗口数据）
        self._imu_buf: deque = deque(maxlen=cfg.imu_window_size)
        self._tac_buf: deque = deque(maxlen=cfg.tactile_window_size)

        # 最新 token 快照（供 LLM 消费）
        self._latest_tokens: dict[str, list[int]] = {
            "VIS": [], "AUD": [], "IMU": [], "TAC": []
        }

        # ZMQ（异步）
        self._zmq_ctx: zmq.asyncio.Context | None = None
        self._zmq_pub: zmq.asyncio.Socket | None = None
        self._zmq_sub: zmq.asyncio.Socket | None = None

        self._running = False

    # ——— 编码器懒加载 ———

    def _load_encoders(self) -> None:
        if self.cfg.imu_checkpoint:
            self._imu_enc = IMUEncoder().to(self.device)
            ckpt = torch.load(self.cfg.imu_checkpoint, map_location=self.device)
            self._imu_enc.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
            self._imu_enc.eval()

        if self.cfg.tactile_checkpoint:
            self._tac_enc = TactileEncoder().to(self.device)
            ckpt = torch.load(self.cfg.tactile_checkpoint, map_location=self.device)
            self._tac_enc.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
            self._tac_enc.eval()

        self._vis_enc = VisualEncoder().to(self.device)
        self._vis_enc.eval()

    # ——— 传感器消费协程 ———

    async def _consume_imu(self) -> None:
        """持续消费 IMU 队列，积累窗口，到窗口满时编码。"""
        while self._running:
            try:
                sample = await asyncio.wait_for(self.imu_driver.queue.get(), timeout=0.5)
                self._imu_buf.append(
                    np.concatenate([sample.accel, sample.gyro])  # (6,)
                )

                if len(self._imu_buf) == self.cfg.imu_window_size and self._imu_enc:
                    window = torch.tensor(
                        np.stack(self._imu_buf), dtype=torch.float32
                    ).unsqueeze(0).to(self.device)  # (1, T, 6)

                    with torch.no_grad():
                        token_id = self._imu_enc.tokenize(window).item()

                    self._latest_tokens["IMU"] = [token_id]
                    self.reflex.push("IMU", token_id)

            except asyncio.TimeoutError:
                continue

    async def _consume_tactile(self) -> None:
        """持续消费触觉队列，积累窗口，到窗口满时编码 + 检查反射。"""
        while self._running:
            try:
                sample = await asyncio.wait_for(self.tac_driver.queue.get(), timeout=0.5)
                self._tac_buf.append(sample.body_map)  # (H, W)

                # 单帧快速反射检测（无需等满整个窗口）
                # 头部平均压力超过阈值 → 立即触发反射
                head_pressure = sample.body_map[0:2, :].mean()
                if head_pressure > 0.15:
                    # 用一个特殊"高频触发 token"直接推给反射引擎
                    # 此 token ID 在 Stage 1 后填入，这里用 -1 表示占位
                    self.reflex.push("TAC", -1)  # TODO: 填入实际 token ID

                if len(self._tac_buf) == self.cfg.tactile_window_size and self._tac_enc:
                    window = torch.tensor(
                        np.stack(self._tac_buf), dtype=torch.float32
                    ).unsqueeze(0).to(self.device)  # (1, T, H, W)

                    with torch.no_grad():
                        token_id = self._tac_enc.tokenize(window).item()

                    self._latest_tokens["TAC"] = [token_id]
                    self.reflex.push("TAC", token_id)

            except asyncio.TimeoutError:
                continue

    async def _consume_camera(self) -> None:
        """每 3 帧取一帧做视觉编码（10 VIS token/秒）。"""
        frame_count = 0
        while self._running:
            try:
                cam_frame = await asyncio.wait_for(self.cam_driver.queue.get(), timeout=0.5)
                frame_count += 1

                if frame_count % 3 != 0 or self._vis_enc is None:
                    continue

                import cv2
                # BGR → RGB，resize → tensor
                rgb = cv2.cvtColor(cam_frame.frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224, 224))
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

                with torch.no_grad():
                    token_id = self._vis_enc.tokenize(tensor).item()

                self._latest_tokens["VIS"].append(token_id)
                if len(self._latest_tokens["VIS"]) > 10:
                    self._latest_tokens["VIS"].pop(0)
                self.reflex.push("VIS", token_id)

            except asyncio.TimeoutError:
                continue

    # ——— ZMQ：向 LLM 进程发布 token ———

    async def _publish_to_llm(self) -> None:
        """定期把最新 token 快照发给 LLM 进程（5Hz）。"""
        interval = 1.0 / self.cfg.llm_publish_hz
        while self._running:
            await asyncio.sleep(interval)
            if self._zmq_pub:
                import json
                msg = json.dumps(self._latest_tokens).encode()
                await self._zmq_pub.send(msg)

    async def _receive_from_llm(self) -> None:
        """接收 LLM 的响应并执行动作。"""
        while self._running:
            if self._zmq_sub:
                try:
                    msg = await asyncio.wait_for(self._zmq_sub.recv(), timeout=1.0)
                    import json
                    response = json.loads(msg.decode())
                    action = response.get("action")
                    if action and self.robot:
                        getattr(self.robot, action, lambda: None)()
                except asyncio.TimeoutError:
                    continue

    # ——— 主入口 ———

    async def run(self) -> None:
        """启动整个系统。Ctrl+C 优雅退出。"""
        loop = asyncio.get_running_loop()

        logger.info("加载编码器...")
        self._load_encoders()

        logger.info("启动传感器驱动...")
        self.cam_driver.start(loop)
        self.mic_driver.start(loop)
        self.imu_driver.start(loop)
        self.tac_driver.start(loop)

        logger.info("初始化 ZMQ...")
        self._zmq_ctx = zmq.asyncio.Context()
        self._zmq_pub = self._zmq_ctx.socket(zmq.PUB)
        self._zmq_pub.bind(self.cfg.zmq_pub_addr)
        self._zmq_sub = self._zmq_ctx.socket(zmq.SUB)
        self._zmq_sub.connect(self.cfg.zmq_sub_addr)
        self._zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self._running = True
        logger.info("Sensorium 运行中...")

        try:
            await asyncio.gather(
                self._consume_imu(),
                self._consume_tactile(),
                self._consume_camera(),
                self._publish_to_llm(),
                self._receive_from_llm(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("收到退出信号，正在关闭...")
        finally:
            self._running = False
            self.cam_driver.stop()
            self.mic_driver.stop()
            self.imu_driver.stop()
            self.tac_driver.stop()
            if self._zmq_ctx:
                self._zmq_ctx.destroy()
            logger.info("Sensorium 已关闭。")
