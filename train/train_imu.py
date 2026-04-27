"""
IMU 编码器训练脚本 — Stage 1

从原始 IMU 数据（CSV 或 ROS2 bag 转出的 npy 文件）自监督训练。
无需任何标注。

运行示例：
  python train/train_imu.py data_dir=data/imu_recordings/ trainer.max_epochs=100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra

from sensorium.encoders.imu import IMUEncoder


# ——— 数据集 ———

class IMUWindowDataset(Dataset):
    """
    从 .npy 文件加载 IMU 数据，切成滑动窗口。

    数据格式：npy 文件，shape (N_samples, 6)，列顺序 [ax, ay, az, gx, gy, gz]
    单位：加速度 m/s²，角速度 rad/s（训练时会归一化）

    Args:
        data_dir:    包含 .npy 文件的目录
        window_size: 每个窗口的采样点数（默认 50，即 100Hz × 0.5s）
        stride:      滑动步长（默认 25，50% 重叠）
        normalize:   是否做 per-channel 归一化
    """

    ACC_SCALE = 20.0   # ±20 m/s² 归一化范围
    GYRO_SCALE = 10.0  # ±10 rad/s 归一化范围

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 50,
        stride: int = 25,
        normalize: bool = True,
    ):
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在：{data_dir}")

        # 加载所有 npy 文件并拼接
        arrays = []
        for f in sorted(data_dir.glob("*.npy")):
            arr = np.load(f).astype(np.float32)  # (N, 6)
            if arr.ndim != 2 or arr.shape[1] != 6:
                continue
            arrays.append(arr)

        if not arrays:
            raise RuntimeError(f"在 {data_dir} 中没有找到有效的 .npy 文件")

        data = np.concatenate(arrays, axis=0)  # (N_total, 6)

        # 归一化
        if normalize:
            data[:, :3] /= self.ACC_SCALE
            data[:, 3:] /= self.GYRO_SCALE

        # 切窗口
        self.windows: list[np.ndarray] = []
        for start in range(0, len(data) - window_size, stride):
            self.windows.append(data[start : start + window_size])

        print(f"[IMUDataset] 加载 {len(arrays)} 个文件，"
              f"共 {len(data)} 个采样点，切出 {len(self.windows)} 个窗口")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window = torch.from_numpy(self.windows[idx])  # (T, 6)
        # 返回当前窗口和下一窗口（用于时序预测损失）
        next_idx = min(idx + 1, len(self.windows) - 1)
        next_window = torch.from_numpy(self.windows[next_idx])
        return {"x": window, "x_next": next_window}


# ——— Lightning 模块 ———

class IMUEncoderModule(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.model = IMUEncoder(
            codebook_size=cfg.model.codebook_size,
            latent_dim=cfg.model.latent_dim,
            window_size=cfg.model.window_size,
            commitment_cost=cfg.model.commitment_cost,
        )

    def forward(self, x, x_next=None):
        return self.model(x, x_next)

    def training_step(self, batch, batch_idx):
        out = self.model(batch["x"], batch["x_next"])
        self.log_dict({
            "train/loss":           out["loss"],
            "train/loss_recon":     out["loss_recon"],
            "train/loss_vq":        out["loss_vq"],
            "train/codebook_usage": out["codebook_usage"],
        }, prog_bar=True, on_step=True, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.model(batch["x"], batch["x_next"])
        self.log_dict({
            "val/loss":           out["loss"],
            "val/loss_recon":     out["loss_recon"],
            "val/codebook_usage": out["codebook_usage"],
        }, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.trainer.max_epochs
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}


# ——— Hydra 入口 ———

@hydra.main(config_path="../configs", config_name="imu", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed)

    # 数据
    dataset = IMUWindowDataset(
        data_dir=cfg.data.data_dir,
        window_size=cfg.model.window_size,
        stride=cfg.data.stride,
    )
    n_val = int(len(dataset) * cfg.data.val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True,
                          num_workers=cfg.data.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers, pin_memory=True)

    # 模型
    model = IMUEncoderModule(cfg)

    # 回调
    checkpoint = ModelCheckpoint(
        dirpath="checkpoints/imu",
        filename="imu-{epoch:03d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
    )

    # 日志（可选，注释掉则不用 wandb）
    logger = WandbLogger(project="sensorium", name="imu-encoder") if cfg.use_wandb else None

    # 训练
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint, RichProgressBar()],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_dl, val_dl)

    print(f"\n最优模型保存在：{checkpoint.best_model_path}")
    print(f"最终码本使用率：{model.model.quantizer.usage_rate:.1%}")


if __name__ == "__main__":
    main()
