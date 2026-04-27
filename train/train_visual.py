"""
视觉编码器训练脚本 — Stage 1

DINOv2 backbone 完全冻结，只训练 VQ-GAN 量化头。
损失目标：让量化后的特征尽量接近原始 DINOv2 特征（特征重建）。

数据：从视频文件或图片目录加载，无需任何标注。

运行：
  python train/train_visual.py data_dir=data/video_recordings/
  python train/train_visual.py data_dir=data/frames/ trainer.max_epochs=50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from omegaconf import DictConfig, OmegaConf
import hydra

from sensorium.encoders.visual import VisualEncoder


# ——— 数据集 ———

class FrameDataset(Dataset):
    """
    从视频文件或图片目录加载帧。

    支持格式：
      - 目录下的 .mp4 / .avi / .mkv 视频文件（自动抽帧）
      - 目录下的 .jpg / .png 图片文件（直接读取）

    Args:
        data_dir:    数据目录
        frame_skip:  视频抽帧间隔（每 N 帧取 1 帧），默认 3（30fps→10fps）
        img_size:    输入图片尺寸（需为 14 的倍数，DINOv2 patch=14）
    """

    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        data_dir: str | Path,
        frame_skip: int = 3,
        img_size: int = 224,
    ):
        self.img_size = img_size
        data_dir = Path(data_dir)

        self.frames: list[np.ndarray] = []

        # 加载视频
        for ext in ("*.mp4", "*.avi", "*.mkv", "*.MOV"):
            for vf in sorted(data_dir.glob(ext)):
                self._extract_video(vf, frame_skip)

        # 加载图片
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for imgf in sorted(data_dir.glob(ext)):
                frame = cv2.imread(str(imgf))
                if frame is not None:
                    self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not self.frames:
            raise RuntimeError(f"在 {data_dir} 中没有找到有效的视频或图片文件")

        print(f"[FrameDataset] 共 {len(self.frames)} 帧")

    def _extract_video(self, path: Path, skip: int) -> None:
        cap = cv2.VideoCapture(str(path))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % skip == 0:
                self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frame = self.frames[idx]
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        tensor = self.TRANSFORM(frame)                 # (3, H, W)
        return {"frame": tensor}


# ——— Lightning 模块 ———

class VisualEncoderModule(L.LightningModule):
    """
    训练目标：量化后的特征 z_q 尽量接近 DINOv2 原始特征 z。
    等价于：用有损压缩（VQ）最大程度保留 DINOv2 的语义信息。
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = VisualEncoder(
            codebook_size=cfg.model.codebook_size,
            latent_dim=cfg.model.latent_dim,
            commitment_cost=cfg.model.commitment_cost,
        )

    def _shared_step(self, batch: dict) -> dict[str, torch.Tensor]:
        frames = batch["frame"]                             # (B, 3, H, W)

        # DINOv2 特征（冻结，不参与梯度）
        self.model._load_backbone(frames.device)
        with torch.no_grad():
            feats = self.model.backbone.forward_features(frames)
            cls_token = feats["x_norm_clstoken"]           # (B, 384)

        # VQ 量化头
        z = self.model.vq_head(cls_token)                  # (B, latent_dim)
        z_q, indices, vq_loss = self.model.quantizer(z)

        # 特征重建损失：量化后的特征应接近量化前的特征
        recon_loss = F.mse_loss(z_q, z.detach())

        return {
            "loss": recon_loss + vq_loss,
            "loss_recon": recon_loss.detach(),
            "loss_vq": vq_loss.detach(),
            "indices": indices,
            "codebook_usage": self.model.quantizer.usage_rate,
        }

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.log_dict({
            "train/loss":           out["loss"],
            "train/loss_recon":     out["loss_recon"],
            "train/loss_vq":        out["loss_vq"],
            "train/codebook_usage": out["codebook_usage"],
        }, prog_bar=True, on_step=True, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.log_dict({
            "val/loss":           out["loss"],
            "val/codebook_usage": out["codebook_usage"],
        }, prog_bar=True)

    def configure_optimizers(self):
        # 只优化 VQ 头，backbone 完全冻结
        params = list(self.model.vq_head.parameters()) + \
                 list(self.model.quantizer.parameters())
        opt = torch.optim.AdamW(params, lr=self.cfg.optimizer.lr,
                                weight_decay=self.cfg.optimizer.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.trainer.max_epochs
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}


# ——— Hydra 入口 ———

@hydra.main(config_path="../configs", config_name="visual", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed)

    dataset = FrameDataset(
        data_dir=cfg.data.data_dir,
        frame_skip=cfg.data.frame_skip,
        img_size=cfg.model.img_size,
    )
    n_val = int(len(dataset) * cfg.data.val_ratio)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_dl = DataLoader(train_ds, batch_size=cfg.data.batch_size,
                          shuffle=True, num_workers=cfg.data.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.data.batch_size,
                          shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True)

    model = VisualEncoderModule(cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/visual",
            filename="visual-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss", mode="min", save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_dl, val_dl)
    print(f"最优模型：{callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
