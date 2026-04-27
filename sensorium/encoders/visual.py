"""
视觉原子编码器

使用冻结的 DINOv2-small 提取视觉特征，
再经过 VQ-GAN 量化头将连续特征离散化为视觉 token。

DINOv2 权重完全冻结，只训练量化头（节省大量计算）。

输出：10 个 VIS token/秒（每 3 帧取一次，30fps 输入）
码本：1024 个条目
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..core import VectorQuantizer


class VisualEncoder(nn.Module):
    """
    Args:
        codebook_size: 码本大小，默认 1024
        latent_dim:    量化器维度，默认 384（匹配 DINOv2-small patch dim）
        patch_select:  每帧保留的 patch 数量（从 300 个 patch 中选代表性区域）
    """

    def __init__(
        self,
        codebook_size: int = 1024,
        latent_dim: int = 384,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim

        # DINOv2-small：完全冻结，只用作特征提取器
        # 实际使用时通过 hub 加载：torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.backbone = None  # 延迟加载，避免导入时强制下载

        # VQ 量化头：把 DINOv2 的 CLS token (384维) 量化
        self.vq_head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

    def _load_backbone(self, device: torch.device) -> None:
        """懒加载 DINOv2，第一次调用时下载权重。"""
        if self.backbone is None:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=True,
                verbose=False,
            ).to(device)
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def encode(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            frames: 视频帧，(B, C, H, W)，H 和 W 须为 14 的倍数
                    推荐：(B, 3, 224, 224) 或 (B, 3, 280, 224)
        Returns:
            z_q:     量化潜变量，(B, latent_dim)
            indices: 码本索引，(B,)，即视觉 token
        """
        self._load_backbone(frames.device)

        with torch.no_grad():
            features = self.backbone.forward_features(frames)
            cls_token = features["x_norm_clstoken"]  # (B, 384)

        z = self.vq_head(cls_token)                  # (B, latent_dim)
        z_q, indices, _ = self.quantizer(z)
        return z_q, indices

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        self._load_backbone(frames.device)

        with torch.no_grad():
            features = self.backbone.forward_features(frames)
            cls_token = features["x_norm_clstoken"]

        z = self.vq_head(cls_token)
        z_q, indices, vq_loss = self.quantizer(z)

        return {
            "loss": vq_loss,
            "loss_vq": vq_loss.detach(),
            "indices": indices,
            "codebook_usage": self.quantizer.usage_rate,
        }

    def tokenize(self, frames: Tensor) -> Tensor:
        """推理接口：(B, C, H, W) → (B,)"""
        with torch.no_grad():
            _, indices = self.encode(frames)
        return indices
