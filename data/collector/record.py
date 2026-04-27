"""
多模态同步录制工具

同时采集：摄像头 / 麦克风 / IMU / FSR 触觉阵列
所有传感器流打统一时间戳，保存到一个 session 目录。

session 目录结构：
  data/sessions/session_20240427_143022/
    frames/           ← jpg 帧，文件名 = 时间戳（微秒）
    audio.wav         ← 同步音频
    imu.npy           ← (N, 6) [ax,ay,az,gx,gy,gz]
    imu_ts.npy        ← (N,) 时间戳
    tactile.npy       ← (N, H, W) 压力地图
    tactile_ts.npy    ← (N,) 时间戳
    meta.json         ← 录制元信息

用法：
  python data/collector/record.py                        # 默认配置
  python data/collector/record.py --duration 300         # 录 5 分钟
  python data/collector/record.py --no-tactile           # 没有触觉传感器时
  python data/collector/record.py --session-name test01  # 自定义 session 名
"""

import argparse
import asyncio
import json
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sensorium.drivers import CameraDriver, MicrophoneDriver, IMUDriver, TactileDriver
from sensorium.drivers.tactile import BODY_MAP_H, BODY_MAP_W


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensorium 多模态录制工具")
    p.add_argument("--output-dir",    default="data/sessions")
    p.add_argument("--session-name",  default=None)
    p.add_argument("--duration",      type=float, default=None, help="录制时长（秒），None=手动停止")
    p.add_argument("--camera-device", type=int,   default=0)
    p.add_argument("--camera-fps",    type=int,   default=30)
    p.add_argument("--imu-bus",       type=int,   default=1)
    p.add_argument("--tactile-port",  type=str,   default="/dev/ttyUSB0")
    p.add_argument("--no-imu",        action="store_true")
    p.add_argument("--no-tactile",    action="store_true")
    return p.parse_args()


class Recorder:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        name = args.session_name or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = Path(args.output_dir) / name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "frames").mkdir(exist_ok=True)

        # 数据缓冲
        self.imu_data:     list[np.ndarray] = []
        self.imu_ts:       list[float]      = []
        self.tactile_data: list[np.ndarray] = []
        self.tactile_ts:   list[float]      = []
        self.audio_frames: list[np.ndarray] = []
        self.frame_count   = 0

        # 驱动
        self.cam = CameraDriver(device_id=args.camera_device, fps=args.camera_fps)
        self.mic = MicrophoneDriver(sample_rate=24000, chunk_ms=100)
        self.imu = IMUDriver(bus_id=args.imu_bus) if not args.no_imu else None
        self.tac = TactileDriver(port=args.tactile_port) if not args.no_tactile else None

        self._running = False
        self._start_ts: float = 0.0

    # ——— 消费协程 ———

    async def _consume_camera(self) -> None:
        frames_dir = self.session_dir / "frames"
        while self._running:
            try:
                item = await asyncio.wait_for(self.cam.queue.get(), timeout=0.5)
                fname = frames_dir / f"{int(item.timestamp * 1e6):020d}.jpg"
                cv2.imwrite(str(fname), item.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.frame_count += 1
            except asyncio.TimeoutError:
                continue

    async def _consume_mic(self) -> None:
        while self._running:
            try:
                item = await asyncio.wait_for(self.mic.queue.get(), timeout=0.5)
                self.audio_frames.append(item.samples)
            except asyncio.TimeoutError:
                continue

    async def _consume_imu(self) -> None:
        if not self.imu:
            return
        while self._running:
            try:
                item = await asyncio.wait_for(self.imu.queue.get(), timeout=0.5)
                self.imu_data.append(np.concatenate([item.accel, item.gyro]))
                self.imu_ts.append(item.timestamp)
            except asyncio.TimeoutError:
                continue

    async def _consume_tactile(self) -> None:
        if not self.tac:
            return
        while self._running:
            try:
                item = await asyncio.wait_for(self.tac.queue.get(), timeout=0.5)
                self.tactile_data.append(item.body_map)
                self.tactile_ts.append(item.timestamp)
            except asyncio.TimeoutError:
                continue

    async def _duration_timer(self) -> None:
        if self.args.duration:
            await asyncio.sleep(self.args.duration)
            self._running = False

    async def _progress_reporter(self) -> None:
        while self._running:
            elapsed = time.monotonic() - self._start_ts
            print(
                f"\r录制中 {elapsed:6.1f}s | "
                f"帧:{self.frame_count:5d} | "
                f"IMU:{len(self.imu_ts):6d} | "
                f"触觉:{len(self.tactile_ts):6d}",
                end="", flush=True,
            )
            await asyncio.sleep(1.0)

    # ——— 保存 ———

    def _save(self) -> None:
        print("\n\n正在保存数据...")

        # IMU
        if self.imu_data:
            np.save(self.session_dir / "imu.npy",    np.stack(self.imu_data))
            np.save(self.session_dir / "imu_ts.npy", np.array(self.imu_ts))

        # 触觉
        if self.tactile_data:
            np.save(self.session_dir / "tactile.npy",    np.stack(self.tactile_data))
            np.save(self.session_dir / "tactile_ts.npy", np.array(self.tactile_ts))

        # 音频（WAV）
        if self.audio_frames:
            audio = np.concatenate(self.audio_frames)
            audio_i16 = (audio * 32767).astype(np.int16)
            wav_path = self.session_dir / "audio.wav"
            with wave.open(str(wav_path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_i16.tobytes())

        # 元信息
        meta = {
            "session":        self.session_dir.name,
            "start_time":     datetime.now().isoformat(),
            "duration_s":     time.monotonic() - self._start_ts,
            "n_frames":       self.frame_count,
            "n_imu":          len(self.imu_ts),
            "n_tactile":      len(self.tactile_ts),
            "camera_fps":     self.args.camera_fps,
            "has_imu":        self.imu is not None,
            "has_tactile":    self.tac is not None,
        }
        with open(self.session_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"✓ 保存完成：{self.session_dir}")
        print(f"  视频帧：{self.frame_count}")
        print(f"  IMU：   {len(self.imu_ts)} 采样点")
        print(f"  触觉：  {len(self.tactile_ts)} 采样点")

    # ——— 主运行 ———

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        self.cam.start(loop)
        self.mic.start(loop)
        if self.imu:  self.imu.start(loop)
        if self.tac:  self.tac.start(loop)

        self._running = True
        self._start_ts = time.monotonic()

        print(f"开始录制 → {self.session_dir}")
        if not self.args.duration:
            print("按 Ctrl+C 停止录制\n")

        try:
            await asyncio.gather(
                self._consume_camera(),
                self._consume_mic(),
                self._consume_imu(),
                self._consume_tactile(),
                self._duration_timer(),
                self._progress_reporter(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            self._running = False
        finally:
            self.cam.stop()
            self.mic.stop()
            if self.imu:  self.imu.stop()
            if self.tac:  self.tac.stop()
            self._save()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(Recorder(args).run())
