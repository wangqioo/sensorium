"""
触觉驱动（FSR 阵列，Arduino → UART 串口）

Arduino 负责 ADC 采样，通过串口以固定格式发送给 Jetson。
串口协议（Arduino 端固件约定）：
  每帧：[0xAA][len=N_sensors*2][sensor_0_hi][sensor_0_lo]...[checksum]
  帧率：100Hz（Arduino 端定时）
  数据：10bit ADC 值（0-1023），两字节 big-endian 传输
"""

import time
import threading
import asyncio
import struct
from dataclasses import dataclass

import numpy as np

N_SENSORS = 80          # FSR 传感器总数
FRAME_MAGIC = 0xAA      # 帧头魔数
BODY_MAP_H = 10
BODY_MAP_W = 8


@dataclass
class TactileSample:
    timestamp: float          # monotonic 时间戳
    pressure: np.ndarray      # (N_SENSORS,) float32，值域 [0, 1]
    body_map: np.ndarray      # (H, W) float32，映射到身体地图


# 传感器物理位置到身体地图的映射表
# 格式：sensor_index → (row, col)
# 根据实际硬件布线填写
SENSOR_TO_BODY_MAP: dict[int, tuple[int, int]] = {
    # 头部：传感器 0-7 → rows 0-1, cols 0-7
    **{i: (i // 4, i % 8) for i in range(8)},
    # 背部：传感器 8-39 → rows 2-5, cols 0-7
    **{8 + i: (2 + i // 8, i % 8) for i in range(32)},
    # 腹部：传感器 40-55 → rows 6-9, cols 2-5
    **{40 + i: (6 + i // 4, 2 + i % 4) for i in range(16)},
    # 左侧：传感器 56-63 → rows 6-9, cols 0-1
    **{56 + i: (6 + i // 2, i % 2) for i in range(8)},
    # 右侧：传感器 64-71 → rows 6-9, cols 6-7
    **{64 + i: (6 + i // 2, 6 + i % 2) for i in range(8)},
}


class TactileDriver:
    """
    Args:
        port:      串口设备路径，如 "/dev/ttyUSB0" 或 "/dev/ttyACM0"
        baudrate:  波特率，与 Arduino 固件设置一致，默认 921600
        n_sensors: FSR 传感器数量
        queue:     接收数据的 asyncio.Queue
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 921600,
        n_sensors: int = N_SENSORS,
        queue: asyncio.Queue | None = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.n_sensors = n_sensors
        self.queue: asyncio.Queue = queue or asyncio.Queue(maxsize=200)

        self._serial = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        # 预计算身体地图映射矩阵（避免每帧计算）
        self._map_rows = np.array(
            [SENSOR_TO_BODY_MAP.get(i, (0, 0))[0] for i in range(n_sensors)]
        )
        self._map_cols = np.array(
            [SENSOR_TO_BODY_MAP.get(i, (0, 0))[1] for i in range(n_sensors)]
        )

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        import serial

        self._loop = loop
        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=0.05,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial:
            self._serial.close()

    def _read_loop(self) -> None:
        frame_size = 2 + self.n_sensors * 2 + 1  # magic + len + data + checksum

        while not self._stop_event.is_set():
            try:
                # 对齐帧头
                byte = self._serial.read(1)
                if not byte or byte[0] != FRAME_MAGIC:
                    continue

                # 读剩余帧
                rest = self._serial.read(frame_size - 1)
                if len(rest) < frame_size - 1:
                    continue

                ts = time.monotonic()

                # 校验和（所有数据字节异或）
                checksum = 0
                for b in rest[:-1]:
                    checksum ^= b
                if checksum != rest[-1]:
                    continue

                # 解析 ADC 值
                n = rest[0]
                if n != self.n_sensors * 2:
                    continue

                raw = struct.unpack(f">{self.n_sensors}H", rest[1:1 + self.n_sensors * 2])
                pressure = np.array(raw, dtype=np.float32) / 1023.0  # 归一化到 [0, 1]

                # 映射到身体地图
                body_map = np.zeros((BODY_MAP_H, BODY_MAP_W), dtype=np.float32)
                body_map[self._map_rows, self._map_cols] = pressure

                sample = TactileSample(
                    timestamp=ts,
                    pressure=pressure,
                    body_map=body_map,
                )
                asyncio.run_coroutine_threadsafe(self.queue.put(sample), self._loop)

            except Exception:
                pass
