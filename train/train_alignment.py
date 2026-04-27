"""
跨模态对齐训练脚本 — Stage 2

加载 Stage 1 已训练好的各模态编码器，
用 ImageBind 作为语义锚，让我们的原子嵌入和 ImageBind 的嵌入空间对齐。

三路对齐（有 ImageBind 支持的模态）：
  视觉原子嵌入  ←→  ImageBind.encode_image()
  听觉原子嵌入  ←→  ImageBind.encode_audio()
  IMU 原子嵌入  ←→  ImageBind.encode_imu()

触觉用跨模态对比学习（ImageBind 无触觉）：
  TAC 原子嵌入  ←→  同时刻的 VIS 原子嵌入（拉近）
                ←→  不同时刻的 VIS 原子嵌入（推远）

数据要求：
  需要同步录制的多模态数据——同一时间戳下同时有视频帧、音频、IMU、触觉。
  对齐数据集目录结构：
    data/aligned/
      session_001/
        frames/      ← jpg 帧，文件名为时间戳
        audio.wav    ← 同步音频
        imu.npy      ← (N, 6) IMU 数据
        tactile.npy  ← (N, H, W) 触觉数据
        timestamps.npy ← (N,) 对齐时间戳

运行：
  python train/train_alignment.py \
    data_dir=data/aligned/ \
    imu_ckpt=checkpoints/imu/best.ckpt \
    tactile_ckpt=checkpoints/tactile/best.ckpt \
    visual_ckpt=checkpoints/visual/best.ckpt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from omegaconf import DictConfig, OmegaConf
import hydra

from sensorium.encoders import IMUEncoder, TactileEncoder, VisualEncoder
from sensorium.core.losses import CrossModalAlignLoss, ImageBindAlignLoss


# ——— 对齐数据集 ———

class AlignedMultiModalDataset(Dataset):
    """
    加载多个录制 session，每个样本是一个时间戳下的四模态快照。

    返回：
      frame:    (3, H, W) 视频帧
      imu:      (T_imu, 6) IMU 窗口
      tactile:  (T_tac, H, W) 触觉窗口
      audio_path + offset: 用于 ImageBind 加载音频（延迟加载）
    """

    IMU_WINDOW = 50      # IMU 窗口（0.5s @ 100Hz）
    TAC_WINDOW = 50      # 触觉窗口（0.5s @ 100Hz）

    def __init__(self, data_dir: str | Path, img_size: int = 224):
        from torchvision import transforms

        data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.samples: list[dict] = []

        for session in sorted(data_dir.iterdir()):
            if not session.is_dir():
                continue
            try:
                self._load_session(session)
            except Exception as e:
                print(f"跳过 {session.name}：{e}")

        print(f"[AlignedDataset] {len(self.samples)} 个对齐样本，来自 {data_dir}")

    def _load_session(self, session: Path) -> None:
        ts_path = session / "timestamps.npy"
        imu_path = session / "imu.npy"
        tac_path = session / "tactile.npy"
        frames_dir = session / "frames"

        if not all(p.exists() for p in [ts_path, imu_path, tac_path, frames_dir]):
            raise FileNotFoundError("session 数据不完整")

        timestamps = np.load(ts_path)                  # (N,)
        imu_data = np.load(imu_path).astype(np.float32)  # (N, 6)
        tac_data = np.load(tac_path).astype(np.float32)  # (N, H, W)

        # 归一化
        imu_data[:, :3] /= 20.0   # 加速度
        imu_data[:, 3:] /= 10.0   # 角速度
        tac_data /= 1023.0

        frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
        half = self.IMU_WINDOW // 2

        for i in range(half, len(timestamps) - half):
            frame_idx = min(i, len(frame_files) - 1)
            self.samples.append({
                "frame_path": str(frame_files[frame_idx]),
                "imu":  imu_data[i - half : i + half],         # (T_imu, 6)
                "tac":  tac_data[i - half : i + half],         # (T_tac, H, W)
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        import cv2
        frame = cv2.imread(s["frame_path"])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame_t = self.transform(frame)
        return {
            "frame": frame_t,
            "imu":   torch.from_numpy(s["imu"]),
            "tac":   torch.from_numpy(s["tac"]),
        }


# ——— Lightning 模块 ———

class AlignmentModule(L.LightningModule):
    """
    加载所有 Stage 1 编码器，用 ImageBind 作为锚点做对齐微调。
    只有 vq_head / imu encoder / tactile encoder 的参数参与训练。
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 加载 Stage 1 编码器
        self.vis_enc  = self._load_visual(cfg.visual_ckpt)
        self.imu_enc  = self._load_imu(cfg.imu_ckpt)
        self.tac_enc  = self._load_tactile(cfg.tactile_ckpt)

        # 对齐用的投影头（把我们的 latent_dim 投影到 ImageBind 的 1024 维）
        ib_dim = 1024
        self.vis_proj = nn.Linear(cfg.model.latent_dim, ib_dim)
        self.imu_proj = nn.Linear(cfg.model.latent_dim, ib_dim)

        # 损失函数
        self.ib_loss   = ImageBindAlignLoss()
        self.cont_loss = CrossModalAlignLoss(temperature=0.07)

        # ImageBind（冻结，只做推理）
        self._imagebind = None

    def _load_imagebind(self):
        if self._imagebind is None:
            import imagebind.models.imagebind_model as ib
            from imagebind.models.imagebind_model import ModalityType
            self._imagebind = ib.imagebind_huge(pretrained=True)
            self._imagebind.eval()
            for p in self._imagebind.parameters():
                p.requires_grad_(False)
            self._imagebind = self._imagebind.to(self.device)
            self.ModalityType = ModalityType
        return self._imagebind

    @staticmethod
    def _load_visual(ckpt: str) -> VisualEncoder:
        enc = VisualEncoder()
        if ckpt:
            state = torch.load(ckpt, map_location="cpu")
            enc.load_state_dict(state.get("state_dict", state))
        return enc

    @staticmethod
    def _load_imu(ckpt: str) -> IMUEncoder:
        enc = IMUEncoder()
        if ckpt:
            state = torch.load(ckpt, map_location="cpu")
            enc.load_state_dict(state.get("state_dict", state))
        return enc

    @staticmethod
    def _load_tactile(ckpt: str) -> TactileEncoder:
        enc = TactileEncoder()
        if ckpt:
            state = torch.load(ckpt, map_location="cpu")
            enc.load_state_dict(state.get("state_dict", state))
        return enc

    def _shared_step(self, batch: dict) -> dict[str, torch.Tensor]:
        frames = batch["frame"]   # (B, 3, H, W)
        imu    = batch["imu"]     # (B, T, 6)
        tac    = batch["tac"]     # (B, T, H, W)

        # ——— 我们的编码器输出 ———
        vis_enc = self.vis_enc
        vis_enc._load_backbone(frames.device)
        with torch.no_grad():
            cls = vis_enc.backbone.forward_features(frames)["x_norm_clstoken"]
        vis_z = vis_enc.vq_head(cls)                   # (B, latent_dim)

        imu_z  = self.imu_enc.encoder(imu.permute(0, 2, 1))  # (B, latent_dim)
        tac_z, _ = self.tac_enc.encode(tac)            # (B, latent_dim)

        # ——— ImageBind 锚点嵌入（冻结）———
        imagebind = self._load_imagebind()
        with torch.no_grad():
            ib_inputs = {
                self.ModalityType.VISION: frames,
                self.ModalityType.IMU:    imu,
            }
            ib_embs = imagebind(ib_inputs)
            ib_vis = ib_embs[self.ModalityType.VISION]   # (B, 1024)
            ib_imu = ib_embs[self.ModalityType.IMU]      # (B, 1024)

        # ——— 对齐损失 ———
        loss_vis = self.ib_loss(self.vis_proj(vis_z), ib_vis)
        loss_imu = self.ib_loss(self.imu_proj(imu_z), ib_imu)

        # 触觉用跨模态对比（TAC ↔ VIS，同 batch 内同索引为正样本）
        loss_tac = self.cont_loss(
            F.normalize(tac_z, dim=-1),
            F.normalize(vis_z.detach(), dim=-1),
        )

        loss = loss_vis + loss_imu + self.cfg.model.tac_weight * loss_tac
        return {
            "loss": loss,
            "loss_vis": loss_vis.detach(),
            "loss_imu": loss_imu.detach(),
            "loss_tac": loss_tac.detach(),
        }

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.log_dict({f"train/{k}": v for k, v in out.items()},
                      prog_bar=True, on_step=True, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.log_dict({f"val/{k}": v for k, v in out.items()}, prog_bar=True)

    def configure_optimizers(self):
        # 只训练投影头 + 各编码器的 VQ 量化器（backbone 和 DINOv2 保持冻结）
        params = (
            list(self.vis_proj.parameters())
            + list(self.imu_proj.parameters())
            + list(self.vis_enc.vq_head.parameters())
            + list(self.vis_enc.quantizer.parameters())
            + list(self.imu_enc.quantizer.parameters())
            + list(self.tac_enc.quantizer.parameters())
        )
        return torch.optim.AdamW(params, lr=self.cfg.optimizer.lr,
                                 weight_decay=self.cfg.optimizer.weight_decay)


# ——— Hydra 入口 ———

@hydra.main(config_path="../configs", config_name="alignment", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed)

    dataset = AlignedMultiModalDataset(data_dir=cfg.data.data_dir,
                                       img_size=cfg.model.img_size)
    n_val = int(len(dataset) * cfg.data.val_ratio)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_dl = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True,
                          num_workers=cfg.data.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.data.batch_size, shuffle=False,
                          num_workers=cfg.data.num_workers, pin_memory=True)

    model = AlignmentModule(cfg)

    callbacks = [
        ModelCheckpoint(dirpath="checkpoints/alignment",
                        filename="align-{epoch:03d}-{val/loss:.4f}",
                        monitor="val/loss", mode="min", save_top_k=3),
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_dl, val_dl)
    print(f"对齐模型已保存：{callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
