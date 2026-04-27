"""
多模态训练语料生成工具

把录制的 session 目录转换成三种训练数据：

1. IMU/触觉/视觉的 .npy 文件 → 各编码器 Stage 1 训练数据
2. 同步对齐的多模态快照     → Stage 2 对齐训练数据
3. 传感器 token + ASR 文字  → Stage 3 LLM 训练的 JSONL

运行：
  # 处理所有 session，生成全部格式
  python data/preprocessor/build_dataset.py --sessions-dir data/sessions/

  # 只生成 LLM 训练语料（需要先跑 Stage 1 编码器）
  python data/preprocessor/build_dataset.py \
    --sessions-dir data/sessions/ \
    --mode llm \
    --imu-ckpt checkpoints/imu/best.ckpt \
    --tactile-ckpt checkpoints/tactile/best.ckpt \
    --visual-ckpt checkpoints/visual/best.ckpt
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions-dir", required=True)
    p.add_argument("--output-dir",   default="data/processed")
    p.add_argument("--mode",         default="all",
                   choices=["all", "stage1", "stage2", "llm"])
    p.add_argument("--imu-ckpt",     default="")
    p.add_argument("--tactile-ckpt", default="")
    p.add_argument("--visual-ckpt",  default="")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--whisper-model",default="base",
                   help="Whisper 模型大小（tiny/base/small/medium）")
    return p.parse_args()


# ——— Stage 1：拆分各模态数据到训练目录 ———

def build_stage1(sessions_dir: Path, output_dir: Path) -> None:
    """把各 session 的 IMU / 触觉数据分别合并到对应子目录。"""
    imu_dir = output_dir / "imu_recordings"
    tac_dir = output_dir / "tactile_recordings"
    vid_dir = output_dir / "video_recordings"
    for d in [imu_dir, tac_dir, vid_dir]:
        d.mkdir(parents=True, exist_ok=True)

    sessions = [d for d in sessions_dir.iterdir() if d.is_dir()]
    print(f"处理 {len(sessions)} 个 session...")

    for session in sessions:
        name = session.name

        # IMU
        imu_path = session / "imu.npy"
        if imu_path.exists():
            dst = imu_dir / f"{name}.npy"
            if not dst.exists():
                import shutil; shutil.copy(imu_path, dst)

        # 触觉
        tac_path = session / "tactile.npy"
        if tac_path.exists():
            dst = tac_dir / f"{name}.npy"
            if not dst.exists():
                import shutil; shutil.copy(tac_path, dst)

        # 视频（把帧打包成 mp4）
        frames_dir = session / "frames"
        if frames_dir.exists():
            mp4_path = vid_dir / f"{name}.mp4"
            if not mp4_path.exists():
                _frames_to_video(frames_dir, mp4_path)

    print(f"✓ Stage 1 数据 → {output_dir}")


def _frames_to_video(frames_dir: Path, out_path: Path, fps: int = 10) -> None:
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return
    sample = cv2.imread(str(frames[0]))
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.imread(str(f)))
    writer.release()


# ——— Stage 2：构建同步对齐数据集 ———

def build_stage2(sessions_dir: Path, output_dir: Path) -> None:
    """
    为每个 session 生成对齐数据子目录，包含同步的帧、IMU、触觉和时间戳。
    """
    aligned_dir = output_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    for session in sessions_dir.iterdir():
        if not session.is_dir():
            continue

        imu_path = session / "imu.npy"
        tac_path = session / "tactile.npy"
        imu_ts_path = session / "imu_ts.npy"
        tac_ts_path = session / "tactile_ts.npy"
        frames_dir  = session / "frames"

        if not all(p.exists() for p in [imu_path, tac_path, imu_ts_path, frames_dir]):
            continue

        dst = aligned_dir / session.name
        if dst.exists():
            continue
        dst.mkdir()

        imu_data = np.load(imu_path)
        tac_data = np.load(tac_path)
        imu_ts   = np.load(imu_ts_path)

        # 用 IMU 时间戳作为主时间轴
        np.save(dst / "imu.npy",        imu_data)
        np.save(dst / "tactile.npy",    _align_to_imu(tac_data, np.load(tac_ts_path), imu_ts))
        np.save(dst / "timestamps.npy", imu_ts)

        # 帧目录软链接（节省空间）
        frame_link = dst / "frames"
        if not frame_link.exists():
            frame_link.symlink_to(frames_dir.resolve())

    print(f"✓ Stage 2 对齐数据 → {aligned_dir}")


def _align_to_imu(
    data: np.ndarray,
    data_ts: np.ndarray,
    imu_ts: np.ndarray,
) -> np.ndarray:
    """把任意传感器数据插值/对齐到 IMU 时间轴。"""
    # 最近邻插值
    indices = np.searchsorted(data_ts, imu_ts).clip(0, len(data) - 1)
    return data[indices]


# ——— Stage 3：生成 LLM 训练 JSONL ———

def build_llm_dataset(
    sessions_dir: Path,
    output_dir: Path,
    imu_ckpt: str,
    tactile_ckpt: str,
    visual_ckpt: str,
    device: str,
    whisper_model: str,
) -> None:
    """
    对每个 session：
      1. 用训练好的编码器把传感器数据转成 token 序列
      2. 用 Whisper 对 audio.wav 做 ASR 转写
      3. 把 token + 文字按时间戳对齐，输出 JSONL
    """
    import whisper
    from sensorium.encoders import IMUEncoder, TactileEncoder, VisualEncoder

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # 加载编码器
    imu_enc = _load_enc(IMUEncoder, imu_ckpt, dev)
    tac_enc = _load_enc(TactileEncoder, tactile_ckpt, dev)
    vis_enc = _load_enc(VisualEncoder, visual_ckpt, dev)

    # 加载 Whisper
    print(f"加载 Whisper-{whisper_model}...")
    asr = whisper.load_model(whisper_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "multimodal_sequences.jsonl"

    total = 0
    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for session in sorted(sessions_dir.iterdir()):
            if not session.is_dir():
                continue
            rows = _process_session_for_llm(session, imu_enc, tac_enc, vis_enc, asr, dev)
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(rows)
            print(f"  {session.name}: {len(rows)} 条")

    print(f"✓ LLM 训练语料 → {jsonl_path}（共 {total} 条）")


def _load_enc(cls, ckpt: str, device: torch.device):
    enc = cls()
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        enc.load_state_dict(state.get("state_dict", state))
    enc.eval().to(device)
    return enc


def _process_session_for_llm(
    session: Path,
    imu_enc, tac_enc, vis_enc,
    asr,
    device: torch.device,
) -> list[dict]:
    """处理单个 session，返回 JSONL 行列表。"""
    import whisper

    rows = []

    # ASR 转写
    audio_path = session / "audio.wav"
    segments = []
    if audio_path.exists():
        result = asr.transcribe(str(audio_path), language="zh", word_timestamps=True)
        segments = result.get("segments", [])

    # IMU token 序列
    imu_path = session / "imu.npy"
    imu_tokens = []
    if imu_path.exists() and imu_enc is not None:
        imu_data = np.load(imu_path).astype(np.float32)
        imu_data[:, :3] /= 20.0
        imu_data[:, 3:] /= 10.0
        imu_tokens = _encode_imu_sequence(imu_data, imu_enc, device)

    # 触觉 token 序列
    tac_path = session / "tactile.npy"
    tac_tokens = []
    if tac_path.exists() and tac_enc is not None:
        tac_data = np.load(tac_path).astype(np.float32) / 1023.0
        tac_tokens = _encode_tac_sequence(tac_data, tac_enc, device)

    # 视觉 token 序列（从帧目录）
    vis_tokens = []
    frames_dir = session / "frames"
    if frames_dir.exists() and vis_enc is not None:
        vis_tokens = _encode_vis_sequence(frames_dir, vis_enc, device)

    # 按 ASR 分段生成训练样本（每个语音片段对应一条样本）
    if segments:
        for seg in segments:
            rows.append({
                "vis_tokens": vis_tokens[:5] if vis_tokens else [],
                "aud_tokens": [],                # WavTokenizer 在运行时处理
                "imu_tokens": imu_tokens[:3] if imu_tokens else [],
                "tac_tokens": tac_tokens[:3] if tac_tokens else [],
                "text": seg["text"].strip(),
            })
    elif vis_tokens or imu_tokens:
        # 没有语音也生成样本（静默场景）
        rows.append({
            "vis_tokens": vis_tokens[:5] if vis_tokens else [],
            "aud_tokens": [],
            "imu_tokens": imu_tokens[:3] if imu_tokens else [],
            "tac_tokens": tac_tokens[:3] if tac_tokens else [],
            "text": "",
        })

    return rows


def _encode_imu_sequence(data: np.ndarray, enc, device: torch.device,
                          window=50, stride=50) -> list[int]:
    tokens = []
    for i in range(0, len(data) - window, stride):
        w = torch.from_numpy(data[i:i+window]).unsqueeze(0).to(device)
        tokens.append(enc.tokenize(w).item())
    return tokens


def _encode_tac_sequence(data: np.ndarray, enc, device: torch.device,
                          window=50, stride=50) -> list[int]:
    tokens = []
    for i in range(0, len(data) - window, stride):
        w = torch.from_numpy(data[i:i+window]).unsqueeze(0).to(device)
        tokens.append(enc.tokenize(w).item())
    return tokens


def _encode_vis_sequence(frames_dir: Path, enc, device: torch.device,
                          every_n: int = 9) -> list[int]:
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tokens = []
    files = sorted(frames_dir.glob("*.jpg"))[::every_n]
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        frame = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        t = transform(frame).unsqueeze(0).to(device)
        tokens.append(enc.tokenize(t).item())
    return tokens


# ——— 主入口 ———

def main() -> None:
    args = parse_args()
    sessions_dir = Path(args.sessions_dir)
    output_dir   = Path(args.output_dir)

    if args.mode in ("all", "stage1"):
        build_stage1(sessions_dir, output_dir)
    if args.mode in ("all", "stage2"):
        build_stage2(sessions_dir, output_dir)
    if args.mode in ("all", "llm"):
        build_llm_dataset(
            sessions_dir, output_dir,
            args.imu_ckpt, args.tactile_ckpt, args.visual_ckpt,
            args.device, args.whisper_model,
        )
    print("\n全部完成。")


if __name__ == "__main__":
    main()
