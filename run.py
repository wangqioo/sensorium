"""
Sensorium 主入口

用法：
  python run.py                                    # 默认配置
  python run.py --imu-checkpoint checkpoints/imu/best.ckpt
  python run.py --device cpu                       # 无 GPU 时
"""

import asyncio
import argparse
import logging

from sensorium.pipeline import SensoriumRuntime
from sensorium.pipeline.runtime import RuntimeConfig
from sensorium.reflex import ReflexEngine, ReflexRule


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> RuntimeConfig:
    p = argparse.ArgumentParser(description="Sensorium 多模态感知系统")
    p.add_argument("--camera-device",       type=int,   default=0)
    p.add_argument("--imu-bus",             type=int,   default=1)
    p.add_argument("--tactile-port",        type=str,   default="/dev/ttyUSB0")
    p.add_argument("--imu-checkpoint",      type=str,   default="")
    p.add_argument("--tactile-checkpoint",  type=str,   default="")
    p.add_argument("--device",              type=str,   default="cuda")
    p.add_argument("--zmq-pub",             type=str,   default="tcp://127.0.0.1:5555")
    p.add_argument("--zmq-sub",             type=str,   default="tcp://127.0.0.1:5556")
    args = p.parse_args()

    return RuntimeConfig(
        camera_device=args.camera_device,
        imu_bus=args.imu_bus,
        tactile_port=args.tactile_port,
        imu_checkpoint=args.imu_checkpoint,
        tactile_checkpoint=args.tactile_checkpoint,
        device=args.device,
        zmq_pub_addr=args.zmq_pub,
        zmq_sub_addr=args.zmq_sub,
    )


def main() -> None:
    cfg = parse_args()

    # TODO: 替换成实际机器人对象
    robot = None

    runtime = SensoriumRuntime(cfg, robot=robot)

    # 注册反射规则（Stage 1 训练完后填入真实 atom_id）
    # runtime.reflex.add_rule(ReflexRule(
    #     name="头部被抚摸",
    #     modality="TAC",
    #     atom_id=87,       # 训练后填入
    #     action=robot.wag,
    #     priority=5,
    # ))

    asyncio.run(runtime.run())


if __name__ == "__main__":
    main()
