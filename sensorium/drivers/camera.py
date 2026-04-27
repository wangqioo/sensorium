"""
摄像头驱动

直接用 OpenCV 读帧，在独立线程里跑，通过 asyncio.Queue 传给处理层。
不经过任何中间件。
"""

import time
import threading
import asyncio
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraFrame:
    timestamp: float      # monotonic 时间戳（秒）
    frame: np.ndarray     # (H, W, 3) BGR uint8


class CameraDriver:
    """
    Args:
        device_id:  摄像头设备号，默认 0
        width:      采集分辨率宽
        height:     采集分辨率高
        fps:        目标帧率
        queue:      接收帧的 asyncio.Queue，None 则内部创建
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 320,
        height: int = 240,
        fps: int = 30,
        queue: asyncio.Queue | None = None,
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.queue: asyncio.Queue = queue or asyncio.Queue(maxsize=5)

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """启动采集线程，loop 是主事件循环的引用（用于线程安全入队）。"""
        self._loop = loop
        self._cap = cv2.VideoCapture(self.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小 buffer，降低延迟

        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.device_id}")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def _capture_loop(self) -> None:
        interval = 1.0 / self.fps
        next_time = time.monotonic()

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            ts = time.monotonic()

            if ret and self._loop:
                item = CameraFrame(timestamp=ts, frame=frame)
                # 从线程安全地往 asyncio 队列放数据
                # 队列满时丢弃最老帧（保持实时性）
                if self.queue.qsize() >= self.queue.maxsize:
                    try:
                        self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                asyncio.run_coroutine_threadsafe(self.queue.put(item), self._loop)

            # 主动节流到目标帧率
            next_time += interval
            sleep = next_time - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
