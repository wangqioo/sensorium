"""
触觉编码器训练脚本 — Stage 1

从 FSR 阵列录制的压力地图序列自监督训练，无需任何标注。

数据格式：.npy 文件，shape (N_frames, H, W)，值域 0-1023（ADC 原始值）
          H=10，W=8（16×8 身体地图中实际有传感器的区域）

运行示例：
  python train/train_tactile.py data_dir=data/tactile_recordings/
  python train/train_tactile.py data_dir=data/tactile_recordings/ trainer.max_epochs=200
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, RichProgressBar, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra

from sensorium.encoders.tactile import TactileEncoder, BODY_MAP_H, BODY_MAP_W


# ——— 数据集 ———

class TactileWindowDataset(Dataset):
    """
    从 .npy 录制文件加载触觉压力地图序列，切成滑动时间窗口。

    数据格式：
      npy 文件，shape (N_frames, H, W) 或 (N_frames, N_sensors)
      - (N_frames, H, W)：已映射到身体地图（推荐）
      - (N_frames, N_sensors)：原始传感器排列，自动映射

    Args:
        data_dir:      包含 .npy 文件的目录
        window_frames: 窗口帧数，默认 50（100Hz × 0.5s）
        stride:        滑动步长，默认 25（50% 重叠）
        map_h:         身体地图高度
        map_w:         身体地图宽度
    """

    ADC_MAX = 1023.0  # 10bit ADC

    def __init__(
        self,
        data_dir: str | Path,
        window_frames: int = 50,
        stride: int = 25,
        map_h: int = BODY_MAP_H,
        map_w: int = BODY_MAP_W,
    ):
        self.window_frames = window_frames
        self.stride = stride
        self.map_h = map_h
        self.map_w = map_w

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在：{data_dir}")

        self.windows: list[np.ndarray] = []

        for f in sorted(data_dir.glob("*.npy")):
            arr = np.load(f).astype(np.float32)

            # 统一到 (N, H, W) 格式
            if arr.ndim == 2:
                n, last = arr.shape
                if last == map_h * map_w:
                    arr = arr.reshape(n, map_h, map_w)
                else:
                    print(f"  跳过 {f.name}：形状 {arr.shape} 不匹配")
                    continue
            elif arr.ndim == 3 and arr.shape[1:] == (map_h, map_w):
                pass
            else:
                print(f"  跳过 {f.name}：形状 {arr.shape} 不支持")
                continue

            # 归一化到 [0, 1]
            arr = arr / self.ADC_MAX
            arr = np.clip(arr, 0.0, 1.0)

            # 切窗口
            for start in range(0, len(arr) - window_frames, stride):
                self.windows.append(arr[start : start + window_frames])

        if not self.windows:
            raise RuntimeError(f"在 {data_dir} 中没有找到有效的触觉数据文件")

        print(
            f"[TactileDataset] 加载 {len(list(data_dir.glob('*.npy')))} 个文件，"
            f"切出 {len(self.windows)} 个窗口"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window = torch.from_numpy(self.windows[idx])  # (T, H, W)
        return {"x": window}


# ——— Lightning 模块 ———

class TactileEncoderModule(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.model = TactileEncoder(
            codebook_size=cfg.model.codebook_size,
            latent_dim=cfg.model.latent_dim,
            map_h=cfg.model.map_h,
            map_w=cfg.model.map_w,
            window_frames=cfg.model.window_frames,
            commitment_cost=cfg.model.commitment_cost,
        )

    def training_step(self, batch, batch_idx):
        out = self.model(batch["x"])
        self.log_dict(
            {
                "train/loss":           out["loss"],
                "train/loss_recon":     out["loss_recon"],
                "train/loss_vq":        out["loss_vq"],
                "train/codebook_usage": out["codebook_usage"],
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.model(batch["x"])
        self.log_dict(
            {
                "val/loss":           out["loss"],
                "val/loss_recon":     out["loss_recon"],
                "val/codebook_usage": out["codebook_usage"],
            },
            prog_bar=True,
        )

    def on_train_epoch_end(self):
        usage = self.model.quantizer.usage_rate
        # 码本利用率低于 50% 说明训练不稳定，提前预警
        if usage < 0.5:
            self.print(f"\n⚠️  码本利用率偏低：{usage:.1%}（建议降低学习率或调小码本）")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.cfg.trainer.max_epochs,
            eta_min=self.cfg.optimizer.lr * 0.01,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ——— Hydra 入口 ———

@hydra.main(config_path="../configs", config_name="tactile", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed)

    # 数据
    dataset = TactileWindowDataset(
        data_dir=cfg.data.data_dir,
        window_frames=cfg.model.window_frames,
        stride=cfg.data.stride_frames,
        map_h=cfg.model.map_h,
        map_w=cfg.model.map_w,
    )
    n_val = int(len(dataset) * cfg.data.val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # 模型
    model = TactileEncoderModule(cfg)

    # 回调
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/tactile",
            filename="tactile-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val/codebook_usage",
            patience=20,
            mode="max",        # 码本利用率越高越好
            min_delta=0.01,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    logger = WandbLogger(project="sensorium", name="tactile-encoder") if cfg.use_wandb else None

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_dl, val_dl)

    best = callbacks[0].best_model_path
    final_usage = model.model.quantizer.usage_rate
    print(f"\n最优模型：{best}")
    print(f"最终码本利用率：{final_usage:.1%}")

    # 训练完成后，打印码本摘要供反射引擎填入 atom_id
    _print_codebook_summary(model.model)


def _print_codebook_summary(model: TactileEncoder) -> None:
    """
    训练完成后，输出码本每个条目的激活频率和粗略描述。
    用于在反射引擎里手工填入 atom_id。
    """
    print("\n=== 码本摘要（用于填写反射引擎的 atom_id）===")
    usage = model.quantizer.ema_cluster_size
    total = usage.sum().item()
    top_k = usage.topk(20)
    for rank, (freq, idx) in enumerate(zip(top_k.values, top_k.indices)):
        pct = freq.item() / total * 100
        print(f"  Rank {rank+1:2d}  atom_id={idx.item():3d}  频率={pct:.1f}%")


if __name__ == "__main__":
    main()
