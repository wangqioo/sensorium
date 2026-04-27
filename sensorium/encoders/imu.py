"""
IMU 原子编码器

将六轴 IMU 信号（加速度 xyz + 角速度 xyz）的 0.5 秒窗口编码为离散 token。

架构：
  1D CNN encoder → VQ 量化 → 1D CNN decoder
  初始化权重来自 ImageBind 的 IMU encoder（跳过从零训练）

输出：2 个 IMU token/秒（每 0.5 秒一个窗口）
码本：256 个条目（运动状态种类有限）
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..core import VectorQuantizer, ReconLoss, TemporalPredLoss


class IMUEncoder(nn.Module):
    """
    Args:
        codebook_size:  码本大小，默认 256
        latent_dim:     潜变量维度，默认 256
        window_size:    输入窗口采样点数，默认 50（100Hz × 0.5s）
        commitment_cost: VQ commitment loss 权重
    """

    def __init__(
        self,
        codebook_size: int = 256,
        latent_dim: int = 256,
        window_size: int = 50,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.window_size = window_size

        # Encoder: (B, 6, T) → (B, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),   # 全局时序池化 → (B, 256, 1)
            nn.Flatten(),              # → (B, 256)
            nn.Linear(256, latent_dim),
        )

        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

        # Decoder: (B, latent_dim) → (B, T, 6)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4),
            nn.Unflatten(1, (256, 4)),                    # (B, 256, 4)
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # → (B, 128, 8)
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # → (B, 64, 16)
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),    # → (B, 32, 32)
            nn.GELU(),
            nn.Conv1d(32, 6, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(window_size),             # → (B, 6, T)
        )

        # 时序预测头：用当前窗口的 latent 预测下一窗口
        self.temporal_predictor = nn.Linear(latent_dim, latent_dim)

        # 损失函数
        self.recon_loss = ReconLoss()
        self.temporal_loss = TemporalPredLoss(latent_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: IMU 窗口，(B, T, 6)，T=window_size，值域建议归一化到 [-1, 1]
        Returns:
            z_q:     量化后的潜变量，(B, latent_dim)
            indices: 码本索引，(B,)，即 IMU token
        """
        z = self.encoder(x.permute(0, 2, 1))       # (B, T, 6) → (B, latent_dim)
        z_q, indices, _ = self.quantizer(z)
        return z_q, indices

    def decode(self, z_q: Tensor) -> Tensor:
        """从量化潜变量重建 IMU 序列，(B, latent_dim) → (B, T, 6)。"""
        return self.decoder(z_q).permute(0, 2, 1)   # (B, 6, T) → (B, T, 6)

    def forward(
        self,
        x: Tensor,
        x_next: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        训练前向传播。

        Args:
            x:      当前窗口，(B, T, 6)
            x_next: 下一窗口（可选），(B, T, 6)，用于计算时序预测损失
        Returns:
            包含各项损失和 token 索引的字典
        """
        x_perm = x.permute(0, 2, 1)                  # (B, 6, T)
        z = self.encoder(x_perm)                      # (B, latent_dim)
        z_q, indices, vq_loss = self.quantizer(z)
        x_recon = self.decoder(z_q).permute(0, 2, 1) # (B, T, 6)

        recon = self.recon_loss(x_recon, x)
        total = recon + vq_loss

        out = {
            "loss": total,
            "loss_recon": recon.detach(),
            "loss_vq": vq_loss.detach(),
            "indices": indices,
            "codebook_usage": self.quantizer.usage_rate,
        }

        if x_next is not None:
            x_next_perm = x_next.permute(0, 2, 1)
            z_next = self.encoder(x_next_perm).detach()  # 目标不参与反传
            t_loss = self.temporal_loss(z, z_next)
            out["loss"] = out["loss"] + 0.5 * t_loss
            out["loss_temporal"] = t_loss.detach()

        return out

    def tokenize(self, x: Tensor) -> Tensor:
        """推理接口：返回 token 索引序列。(B, T, 6) → (B,)"""
        with torch.no_grad():
            _, indices = self.encode(x)
        return indices
