"""
麦克风驱动

用 sounddevice 做流式音频采集，callback 模式，
最小延迟，直接把音频块扔进 asyncio.Queue。
"""

import time
import asyncio
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioChunk:
    timestamp: float      # monotonic 时间戳（块的起始时间）
    samples: np.ndarray   # (N,) float32，单声道，[-1, 1]
    sample_rate: int


class MicrophoneDriver:
    """
    Args:
        device:      sounddevice 设备 ID 或名称，None 使用系统默认
        sample_rate: 采样率，默认 24000Hz（WavTokenizer 要求）
        chunk_ms:    每块音频时长（毫秒），默认 100ms
        queue:       接收数据的 asyncio.Queue
    """

    def __init__(
        self,
        device=None,
        sample_rate: int = 24000,
        chunk_ms: int = 100,
        queue: asyncio.Queue | None = None,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_ms / 1000)
        self.queue: asyncio.Queue = queue or asyncio.Queue(maxsize=20)
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        import sounddevice as sd

        self._loop = loop

        def callback(indata: np.ndarray, frames: int, time_info, status):
            ts = time.monotonic()
            chunk = AudioChunk(
                timestamp=ts,
                samples=indata[:, 0].copy(),  # 取第一声道
                sample_rate=self.sample_rate,
            )
            if self._loop:
                asyncio.run_coroutine_threadsafe(self.queue.put(chunk), self._loop)

        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=callback,
            latency="low",
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
