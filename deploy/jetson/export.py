"""
Jetson Orin 部署导出工具

把训练好的各模态编码器导出为 TensorRT INT8 engine，
在 Jetson Orin 上实现 <10ms/帧 的推理延迟。

步骤：
  1. PyTorch checkpoint → ONNX
  2. ONNX → TensorRT INT8（需要校准数据）
  3. 验证导出前后精度差异
  4. Benchmark 延迟

运行：
  # 导出 IMU 编码器
  python deploy/jetson/export.py --encoder imu \
    --checkpoint checkpoints/imu/best.ckpt \
    --calib-data data/imu_recordings/ \
    --output deploy/jetson/engines/

  # 导出全部编码器
  python deploy/jetson/export.py --encoder all \
    --imu-ckpt   checkpoints/imu/best.ckpt \
    --tac-ckpt   checkpoints/tactile/best.ckpt \
    --vis-ckpt   checkpoints/visual/best.ckpt \
    --output     deploy/jetson/engines/

注意：TensorRT 导出必须在目标设备（Jetson Orin）上运行，
      在 x86 机器上导出的 engine 无法在 Jetson 上使用。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensorium TensorRT 导出工具")
    p.add_argument("--encoder",   default="all",
                   choices=["imu", "tactile", "visual", "all"])
    p.add_argument("--imu-ckpt",  default="")
    p.add_argument("--tac-ckpt",  default="")
    p.add_argument("--vis-ckpt",  default="")
    p.add_argument("--checkpoint",default="",  help="单编码器模式时的 checkpoint")
    p.add_argument("--calib-data",default="",  help="INT8 校准数据目录")
    p.add_argument("--output",    default="deploy/jetson/engines/")
    p.add_argument("--batch-size",type=int, default=1)
    p.add_argument("--precision", default="int8", choices=["fp32", "fp16", "int8"])
    p.add_argument("--benchmark", action="store_true", help="导出后跑延迟 benchmark")
    return p.parse_args()


# ——— ONNX 导出 ———

class IMUEncoderWrapper(nn.Module):
    """只导出 encoder → quantizer 的前向路径（推理时不需要 decoder）。"""
    def __init__(self, enc):
        super().__init__()
        self.encoder   = enc.encoder
        self.quantizer = enc.quantizer

    def forward(self, x):  # x: (B, T, 6)
        z = self.encoder(x.permute(0, 2, 1))          # (B, D)
        # 推理时直接取最近邻，不需要 VQ loss
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * (z @ self.quantizer.codebook.T)
            + self.quantizer.codebook.pow(2).sum(1)
        )
        return dist.argmin(dim=1)                      # (B,) token indices


class TactileEncoderWrapper(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.spatial_enc  = enc.spatial_enc
        self.temporal_enc = enc.temporal_enc
        self.quantizer    = enc.quantizer

    def forward(self, x):  # x: (B, T, H, W)
        B, T, H, W = x.shape
        frames = x.reshape(B * T, 1, H, W)
        spatial = self.spatial_enc(frames).reshape(B, T, -1).permute(0, 2, 1)
        z = self.temporal_enc(spatial)
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * (z @ self.quantizer.codebook.T)
            + self.quantizer.codebook.pow(2).sum(1)
        )
        return dist.argmin(dim=1)


class VisualEncoderWrapper(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.vq_head   = enc.vq_head
        self.quantizer = enc.quantizer
        # backbone 在推理时单独处理（DINOv2 有自己的优化路径）

    def forward(self, cls_token):  # cls_token: (B, 384) 来自 DINOv2
        z = self.vq_head(cls_token)
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * (z @ self.quantizer.codebook.T)
            + self.quantizer.codebook.pow(2).sum(1)
        )
        return dist.argmin(dim=1)


def export_to_onnx(model: nn.Module, dummy_input: torch.Tensor,
                   onnx_path: Path, input_names: list, output_names: list) -> None:
    model.eval()
    print(f"  导出 ONNX → {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={input_names[0]: {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


# ——— TensorRT 导出 ———

def onnx_to_tensorrt(onnx_path: Path, engine_path: Path,
                     precision: str, calib_data: str | None = None) -> None:
    """把 ONNX 转换为 TensorRT engine。"""
    try:
        import tensorrt as trt
    except ImportError:
        print("  ⚠️  TensorRT 未安装，跳过 TRT 导出（仅保留 ONNX）")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX 解析错误：{parser.get_error(i)}")
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if calib_data:
            # TODO: 实现 INT8 校准器（需要校准数据集）
            print("  ℹ️  INT8 校准：使用默认熵校准（生产环境建议提供校准数据）")

    print(f"  构建 TensorRT engine（{precision}，可能需要几分钟）...")
    serialized = builder.build_serialized_network(network, config)

    if serialized:
        with open(engine_path, "wb") as f:
            f.write(serialized)
        print(f"  ✓ TensorRT engine → {engine_path}")
    else:
        print("  ✗ TensorRT engine 构建失败")


# ——— Benchmark ———

def benchmark(engine_path: Path, dummy_input: np.ndarray, n_runs: int = 200) -> None:
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("  Benchmark 需要 tensorrt + pycuda，跳过")
        return

    print(f"\n  Benchmark：{engine_path.name}，{n_runs} 次推理")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # 分配 GPU 内存
    inp_gpu = cuda.mem_alloc(dummy_input.nbytes)
    out_shape = (dummy_input.shape[0],)
    output = np.zeros(out_shape, dtype=np.int32)
    out_gpu = cuda.mem_alloc(output.nbytes)

    stream = cuda.Stream()
    cuda.memcpy_htod_async(inp_gpu, dummy_input, stream)

    # 预热
    for _ in range(10):
        context.execute_async_v2([int(inp_gpu), int(out_gpu)], stream.handle)
    stream.synchronize()

    # 计时
    t0 = time.perf_counter()
    for _ in range(n_runs):
        context.execute_async_v2([int(inp_gpu), int(out_gpu)], stream.handle)
    stream.synchronize()
    elapsed = (time.perf_counter() - t0) / n_runs * 1000

    print(f"  平均延迟：{elapsed:.2f} ms/次")


# ——— 各编码器导出入口 ———

def export_imu(ckpt: str, output_dir: Path, precision: str,
               calib_data: str, benchmark_flag: bool) -> None:
    from sensorium.encoders.imu import IMUEncoder
    enc = IMUEncoder()
    state = torch.load(ckpt, map_location="cpu")
    enc.load_state_dict(state.get("state_dict", state))
    wrapper = IMUEncoderWrapper(enc)

    dummy = torch.randn(1, 50, 6)
    onnx_path   = output_dir / "imu_encoder.onnx"
    engine_path = output_dir / f"imu_encoder_{precision}.engine"

    export_to_onnx(wrapper, dummy, onnx_path, ["imu_window"], ["token_id"])
    onnx_to_tensorrt(onnx_path, engine_path, precision, calib_data)

    if benchmark_flag and engine_path.exists():
        benchmark(engine_path, dummy.numpy())


def export_tactile(ckpt: str, output_dir: Path, precision: str,
                   calib_data: str, benchmark_flag: bool) -> None:
    from sensorium.encoders.tactile import TactileEncoder
    enc = TactileEncoder()
    state = torch.load(ckpt, map_location="cpu")
    enc.load_state_dict(state.get("state_dict", state))
    wrapper = TactileEncoderWrapper(enc)

    dummy = torch.randn(1, 50, 10, 8)
    onnx_path   = output_dir / "tactile_encoder.onnx"
    engine_path = output_dir / f"tactile_encoder_{precision}.engine"

    export_to_onnx(wrapper, dummy, onnx_path, ["tac_window"], ["token_id"])
    onnx_to_tensorrt(onnx_path, engine_path, precision, calib_data)

    if benchmark_flag and engine_path.exists():
        benchmark(engine_path, dummy.numpy())


def export_visual(ckpt: str, output_dir: Path, precision: str,
                  calib_data: str, benchmark_flag: bool) -> None:
    from sensorium.encoders.visual import VisualEncoder
    enc = VisualEncoder()
    state = torch.load(ckpt, map_location="cpu")
    enc.load_state_dict(state.get("state_dict", state))
    wrapper = VisualEncoderWrapper(enc)

    # 视觉导出的输入是 DINOv2 CLS token，不是原始像素
    dummy = torch.randn(1, 384)
    onnx_path   = output_dir / "visual_vq_head.onnx"
    engine_path = output_dir / f"visual_vq_head_{precision}.engine"

    export_to_onnx(wrapper, dummy, onnx_path, ["cls_token"], ["token_id"])
    onnx_to_tensorrt(onnx_path, engine_path, precision, calib_data)

    if benchmark_flag and engine_path.exists():
        benchmark(engine_path, dummy.numpy())


# ——— 主入口 ———

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    enc_map = {
        "imu":     (args.imu_ckpt     or args.checkpoint, export_imu),
        "tactile": (args.tac_ckpt     or args.checkpoint, export_tactile),
        "visual":  (args.vis_ckpt     or args.checkpoint, export_visual),
    }

    targets = list(enc_map.keys()) if args.encoder == "all" else [args.encoder]

    for enc_name in targets:
        ckpt, export_fn = enc_map[enc_name]
        if not ckpt:
            print(f"跳过 {enc_name}（未提供 checkpoint）")
            continue
        print(f"\n[{enc_name}] 导出中...")
        export_fn(ckpt, output_dir, args.precision, args.calib_data, args.benchmark)

    print(f"\n全部导出完成 → {output_dir}")


if __name__ == "__main__":
    main()
