"""
触觉原子编码器

将 FSR 压力传感器阵列的时空数据编码为离散 token。
传感器映射到 16×8 虚拟身体地图，每 0.5 秒一个窗口。

架构：
  空间 Conv2D（捕捉接触位置、面积）
  + 时间 Conv1D（捕捉抚摸/拍打/握持模式）
  → VQ 量化
  → 解码器重建

身体地图区域：
  HEAD:   rows 0-1,  cols 0-7   (2×8 = 16 点)
  BACK:   rows 2-5,  cols 0-7   (4×8 = 32 点)
  BELLY:  rows 6-9,  cols 2-5   (4×4 = 16 点，中间区域)
  SIDE_L: rows 6-9,  cols 0-1   (4×2 = 8 点)
  SIDE_R: rows 6-9,  cols 6-7   (4×2 = 8 点)

输出：5 个 TAC token/秒（每 0.2 秒一个窗口，步进 0.2 秒）
     或 2 个 TAC token/秒（0.5 秒窗口，推荐用于 LLM 集成）
码本：256 个条目
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core import VectorQuantizer, ReconLoss


# 身体区域定义（行范围, 列范围）
BODY_REGIONS = {
    "head":   ((0, 2),  (0, 8)),
    "back":   ((2, 6),  (0, 8)),
    "belly":  ((6, 10), (2, 6)),
    "side_l": ((6, 10), (0, 2)),
    "side_r": ((6, 10), (6, 8)),
}

BODY_MAP_H = 10  # 行数
BODY_MAP_W = 8   # 列数


class SpatialEncoder(nn.Module):
    """对单帧触觉地图做空间编码：(B, 1, H, W) → (B, C_spatial)"""

    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),   # (B, C, 2, 2)
            nn.Flatten(start_dim=1),         # (B, C*4)
        )
        self.out_dim = out_channels * 4

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TactileEncoder(nn.Module):
    """
    Args:
        codebook_size: 码本大小，默认 256
        latent_dim:    潜变量维度
        map_h:         身体地图高度（行数），默认 10
        map_w:         身体地图宽度（列数），默认 8
        window_frames: 时间窗口帧数（100Hz × 0.5s = 50 帧）
        commitment_cost: VQ commitment loss 权重
    """

    def __init__(
        self,
        codebook_size: int = 256,
        latent_dim: int = 256,
        map_h: int = BODY_MAP_H,
        map_w: int = BODY_MAP_W,
        window_frames: int = 50,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.map_h = map_h
        self.map_w = map_w
        self.window_frames = window_frames

        # Step 1：空间编码每一帧
        self.spatial_enc = SpatialEncoder(out_channels=64)
        spatial_out_dim = self.spatial_enc.out_dim  # 64*4 = 256

        # Step 2：时序编码帧序列
        # 输入：(B, spatial_out_dim, T)
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(spatial_out_dim, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
        )

        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

        # 解码器：从潜变量重建压力地图序列
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 2 * 2),
            nn.Unflatten(1, (256, 4)),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(64, map_h * map_w, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(window_frames),  # (B, H*W, T)
        )

        self.recon_loss = ReconLoss()

    def _encode_frames(self, x: Tensor) -> Tensor:
        """
        对时序中每帧做空间编码。

        Args:
            x: (B, T, H, W) 原始压力地图序列

        Returns:
            (B, spatial_dim, T)
        """
        B, T, H, W = x.shape
        frames = x.reshape(B * T, 1, H, W)              # (B*T, 1, H, W)
        spatial_feat = self.spatial_enc(frames)          # (B*T, spatial_dim)
        return spatial_feat.reshape(B, T, -1).permute(0, 2, 1)  # (B, spatial_dim, T)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, H, W)，值域 [0, 1]（ADC 读数归一化）
        Returns:
            z_q:     量化潜变量，(B, latent_dim)
            indices: 码本索引，(B,)，即触觉 token
        """
        spatial_seq = self._encode_frames(x)         # (B, spatial_dim, T)
        z = self.temporal_enc(spatial_seq)           # (B, latent_dim)
        z_q, indices, _ = self.quantizer(z)
        return z_q, indices

    def decode(self, z_q: Tensor) -> Tensor:
        """(B, latent_dim) → (B, T, H, W)"""
        B = z_q.size(0)
        out = self.decoder(z_q)                      # (B, H*W, T)
        return out.permute(0, 2, 1).reshape(B, self.window_frames, self.map_h, self.map_w)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Args:
            x: (B, T, H, W)
        """
        spatial_seq = self._encode_frames(x)
        z = self.temporal_enc(spatial_seq)
        z_q, indices, vq_loss = self.quantizer(z)
        x_recon = self.decode(z_q)

        recon = self.recon_loss(x_recon, x)

        return {
            "loss": recon + vq_loss,
            "loss_recon": recon.detach(),
            "loss_vq": vq_loss.detach(),
            "indices": indices,
            "codebook_usage": self.quantizer.usage_rate,
        }

    def tokenize(self, x: Tensor) -> Tensor:
        """推理接口：(B, T, H, W) → (B,)"""
        with torch.no_grad():
            _, indices = self.encode(x)
        return indices

    def region_activation(self, x: Tensor) -> dict[str, Tensor]:
        """
        计算各身体区域的平均激活强度，用于反射引擎判断接触位置。

        Args:
            x: 单帧压力地图，(H, W)
        Returns:
            各区域的平均压力值 {region_name: scalar_tensor}
        """
        result = {}
        for name, ((r0, r1), (c0, c1)) in BODY_REGIONS.items():
            region = x[r0:r1, c0:c1]
            result[name] = region.mean()
        return result
