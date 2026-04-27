"""
码本检查工具

Stage 1 训练完成后，用此工具检查码本中每个原子的含义，
找到对应"头部轻抚"、"突然撞击"等模式的 atom_id，
然后填入 sensorium/reflex/engine.py 的反射规则中。

用法：
  # 检查 IMU 码本（可视化每个原子对应的运动波形）
  python tools/inspect_codebook.py --encoder imu \
    --checkpoint checkpoints/imu/best.ckpt \
    --data-dir data/imu_recordings/

  # 检查触觉码本（可视化每个原子对应的压力地图）
  python tools/inspect_codebook.py --encoder tactile \
    --checkpoint checkpoints/tactile/best.ckpt \
    --data-dir data/tactile_recordings/

  # 打印使用频率最高的 Top-20 原子（快速定位重要原子）
  python tools/inspect_codebook.py --encoder imu \
    --checkpoint checkpoints/imu/best.ckpt \
    --mode top20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensorium 码本检查工具")
    p.add_argument("--encoder",    required=True, choices=["imu", "tactile", "visual"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir",   default=None)
    p.add_argument("--mode",       default="top20",
                   choices=["top20", "interactive", "export"])
    p.add_argument("--output",     default="codebook_summary.json")
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


def load_encoder(enc_type: str, ckpt: str, device: torch.device):
    if enc_type == "imu":
        from sensorium.encoders.imu import IMUEncoder
        enc = IMUEncoder()
    elif enc_type == "tactile":
        from sensorium.encoders.tactile import TactileEncoder
        enc = TactileEncoder()
    elif enc_type == "visual":
        from sensorium.encoders.visual import VisualEncoder
        enc = VisualEncoder()

    state = torch.load(ckpt, map_location=device)
    enc.load_state_dict(state.get("state_dict", state))
    enc.eval().to(device)
    return enc


def run_over_data(enc, enc_type: str, data_dir: Path, device: torch.device) -> np.ndarray:
    """在数据集上跑编码器，收集每个样本的 atom_id，返回频率统计。"""
    all_indices = []

    if enc_type == "imu":
        window = 50
        for f in sorted(data_dir.glob("*.npy")):
            data = np.load(f).astype(np.float32)
            data[:, :3] /= 20.0; data[:, 3:] /= 10.0
            for i in range(0, len(data) - window, window):
                w = torch.from_numpy(data[i:i+window]).unsqueeze(0).to(device)
                idx = enc.tokenize(w).item()
                all_indices.append(idx)

    elif enc_type == "tactile":
        window = 50
        for f in sorted(data_dir.glob("*.npy")):
            data = np.load(f).astype(np.float32) / 1023.0
            for i in range(0, len(data) - window, window):
                w = torch.from_numpy(data[i:i+window]).unsqueeze(0).to(device)
                idx = enc.tokenize(w).item()
                all_indices.append(idx)

    return np.array(all_indices)


def print_top20(enc, indices: np.ndarray | None = None) -> dict:
    """打印使用频率最高的 Top-20 原子。"""
    codebook_size = enc.quantizer.codebook_size
    ema_usage = enc.quantizer.ema_cluster_size.cpu().numpy()
    total = ema_usage.sum()

    if total == 0 and indices is not None:
        # EMA 统计为空（新加载模型），用实际推理统计
        counts = np.bincount(indices, minlength=codebook_size).astype(float)
        total = counts.sum()
    else:
        counts = ema_usage

    top_idx = np.argsort(counts)[::-1][:20]

    print(f"\n{'='*60}")
    print(f"{'Rank':>4}  {'atom_id':>8}  {'频率':>8}  {'用途标注（训练后手填）'}")
    print(f"{'='*60}")
    summary = {}
    for rank, idx in enumerate(top_idx):
        pct = counts[idx] / total * 100 if total > 0 else 0
        label = _guess_label(enc, int(idx))
        print(f"{rank+1:>4}  {idx:>8}  {pct:>7.1f}%  {label}")
        summary[int(idx)] = {"rank": rank + 1, "freq_pct": round(pct, 2), "label": label}

    dead = int((counts < 1).sum())
    print(f"\n码本利用率：{(codebook_size - dead) / codebook_size:.1%}  "
          f"（{dead}/{codebook_size} 个死码）")
    return summary


def _guess_label(enc, atom_id: int) -> str:
    """
    根据码本向量的特征粗略猜测原子的语义（仅供参考）。
    IMU：分析向量的能量分布来区分静止/运动/振动。
    """
    vec = enc.quantizer.codebook[atom_id].cpu().detach().numpy()
    energy = float(np.linalg.norm(vec))

    # 非常粗略的启发式，仅作参考
    if energy < 0.3:
        return "← 低能量（可能是静止）"
    elif energy > 0.9:
        return "← 高能量（可能是剧烈运动）"
    return ""


def export_summary(summary: dict, output: str) -> None:
    import json
    with open(output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n码本摘要已保存：{output}")
    print("请根据上表，在 sensorium/reflex/engine.py 中填入对应的 atom_id。")


def interactive_mode(enc, enc_type: str) -> None:
    """交互式查询：输入 atom_id，显示该原子的详细信息。"""
    print("\n交互模式（输入 atom_id 查询，q 退出）")
    while True:
        raw = input("atom_id> ").strip()
        if raw.lower() == "q":
            break
        try:
            idx = int(raw)
        except ValueError:
            continue
        if idx < 0 or idx >= enc.quantizer.codebook_size:
            print(f"  超出范围（0 ~ {enc.quantizer.codebook_size - 1}）")
            continue

        vec = enc.quantizer.codebook[idx].cpu().detach().numpy()
        freq = enc.quantizer.ema_cluster_size[idx].item()
        print(f"\natom_id = {idx}")
        print(f"  使用频率（EMA）：{freq:.1f}")
        print(f"  向量能量：       {np.linalg.norm(vec):.4f}")
        print(f"  向量均值/标准差：{vec.mean():.4f} / {vec.std():.4f}")

        # 尝试可视化（需要 matplotlib）
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 2))
            plt.bar(range(len(vec)), vec, width=1.0)
            plt.title(f"atom_id={idx} 码本向量")
            plt.tight_layout()
            fname = f"/tmp/atom_{idx}.png"
            plt.savefig(fname, dpi=80)
            plt.close()
            print(f"  向量图：{fname}")
        except ImportError:
            pass


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"加载 {args.encoder} 编码器：{args.checkpoint}")
    enc = load_encoder(args.encoder, args.checkpoint, device)

    indices = None
    if args.data_dir:
        print(f"在数据集上推理收集统计：{args.data_dir}")
        indices = run_over_data(enc, args.encoder, Path(args.data_dir), device)
        print(f"共 {len(indices)} 个样本")

    if args.mode in ("top20", "export"):
        summary = print_top20(enc, indices)
        if args.mode == "export":
            export_summary(summary, args.output)

    elif args.mode == "interactive":
        if indices is not None:
            print_top20(enc, indices)
        interactive_mode(enc, args.encoder)


if __name__ == "__main__":
    main()
