"""
IMU 驱动（MPU-6050 / ICM-42688 via I2C）

直接用 smbus2 读 I2C 寄存器，100Hz 采样，
在独立线程里跑，通过 asyncio.Queue 传给处理层。
"""

import time
import threading
import asyncio
import struct
from dataclasses import dataclass

import numpy as np

# MPU-6050 寄存器地址
MPU6050_ADDR      = 0x68
PWR_MGMT_1        = 0x6B
ACCEL_XOUT_H      = 0x3B
GYRO_XOUT_H       = 0x43
ACCEL_CONFIG      = 0x1C
GYRO_CONFIG       = 0x1B

# 量程设置（±2g 加速度，±250°/s 角速度）
ACCEL_SCALE = 16384.0  # LSB/g
GYRO_SCALE  = 131.0    # LSB/(°/s)
G_MS2       = 9.80665  # 1g = 9.80665 m/s²


@dataclass
class IMUSample:
    timestamp: float      # monotonic 时间戳
    accel: np.ndarray     # (3,) m/s²，[ax, ay, az]
    gyro: np.ndarray      # (3,) rad/s，[gx, gy, gz]


class IMUDriver:
    """
    Args:
        bus_id:    I2C 总线号，Jetson Orin 上通常是 1 或 7
        address:   IMU I2C 地址，默认 0x68（MPU-6050/ICM-42688）
        sample_hz: 采样频率，默认 100Hz
        queue:     接收数据的 asyncio.Queue
    """

    def __init__(
        self,
        bus_id: int = 1,
        address: int = MPU6050_ADDR,
        sample_hz: int = 100,
        queue: asyncio.Queue | None = None,
    ):
        self.bus_id = bus_id
        self.address = address
        self.sample_hz = sample_hz
        self.queue: asyncio.Queue = queue or asyncio.Queue(maxsize=200)

        self._bus = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            import smbus2
            self._bus = smbus2.SMBus(self.bus_id)
            # 唤醒 MPU-6050
            self._bus.write_byte_data(self.address, PWR_MGMT_1, 0x00)
            time.sleep(0.1)
        except Exception as e:
            raise RuntimeError(f"IMU I2C 初始化失败（bus={self.bus_id}）：{e}")

        self._loop = loop
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._bus:
            self._bus.close()

    def _read_loop(self) -> None:
        interval = 1.0 / self.sample_hz
        next_time = time.monotonic()

        while not self._stop_event.is_set():
            try:
                raw = self._bus.read_i2c_block_data(self.address, ACCEL_XOUT_H, 14)
                ts = time.monotonic()

                # 解析加速度（字节 0-5）
                ax, ay, az = struct.unpack(">hhh", bytes(raw[0:6]))
                # 解析角速度（字节 8-13，跳过温度）
                gx, gy, gz = struct.unpack(">hhh", bytes(raw[8:14]))

                sample = IMUSample(
                    timestamp=ts,
                    accel=np.array([ax, ay, az], dtype=np.float32) / ACCEL_SCALE * G_MS2,
                    gyro=np.array([gx, gy, gz], dtype=np.float32) / GYRO_SCALE * (np.pi / 180),
                )

                asyncio.run_coroutine_threadsafe(self.queue.put(sample), self._loop)

            except Exception:
                pass  # I2C 偶发失败不中断采集

            next_time += interval
            sleep = next_time - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
