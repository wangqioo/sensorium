"""
向量量化器 (Vector Quantizer with EMA)

所有感官编码器共用此模块将连续潜变量离散化为原子 token。

关键设计：
  - EMA 更新码本（比 loss 梯度更新稳定）
  - Straight-through estimator（梯度反传）
  - 死码重置（防止码本崩塌，这是 VQ 训练最常见的失败模式）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VectorQuantizer(nn.Module):
    """
    Args:
        codebook_size: 码本条目数量（原子数量）
        embedding_dim:  每个原子的维度
        commitment_cost: commitment loss 权重，控制 encoder 输出与码本的粘附强度
        decay:          EMA 衰减系数，越大码本更新越保守（推荐 0.95-0.99）
        dead_threshold: 低于此使用频率的码本条目视为"死码"并重置
    """

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        dead_threshold: float = 1.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.dead_threshold = dead_threshold

        # 码本本体，用 buffer 存（不参与梯度，通过 EMA 更新）
        embed = torch.randn(codebook_size, embedding_dim)
        self.register_buffer("codebook", F.normalize(embed, dim=-1))

        # EMA 统计量
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_avg", self.codebook.clone())

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            z: encoder 输出，shape 任意，最后一维必须是 embedding_dim
               例如 (B, D) / (B, T, D) / (B, H, W, D)

        Returns:
            z_q:     量化后的向量，shape 同 z（straight-through）
            indices: 码本索引，shape = z.shape[:-1]，dtype=long
            loss:    VQ loss（重建 + commitment），标量
        """
        orig_shape = z.shape
        flat = z.reshape(-1, self.embedding_dim)  # (N, D)

        # 计算 flat 到码本每个条目的 L2 距离
        # ||a - b||^2 = ||a||^2 - 2<a,b> + ||b||^2
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * (flat @ self.codebook.T)
            + self.codebook.pow(2).sum(1)
        )  # (N, codebook_size)

        indices = dist.argmin(dim=1)            # (N,)
        z_q_flat = self.codebook[indices]       # (N, D)

        if self.training:
            self._ema_update(flat, indices)

        # VQ loss = codebook loss + commitment loss
        # codebook loss: 让码本向 encoder 输出靠近（通过 EMA 实现，此项=0）
        # commitment loss: 让 encoder 输出向码本靠近
        loss = self.commitment_cost * F.mse_loss(z.detach(), z_q_flat.reshape(orig_shape))

        # Straight-through: 前向用量化值，反向梯度绕过量化直接到 encoder
        z_q = z + (z_q_flat.reshape(orig_shape) - z).detach()

        return z_q, indices.reshape(orig_shape[:-1]), loss

    def _ema_update(self, flat: Tensor, indices: Tensor) -> None:
        """用 EMA 更新码本，比梯度更新更稳定。"""
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.codebook_size).float()  # (N, K)

            # 更新各条目的使用频率
            batch_cluster_size = one_hot.sum(0)                        # (K,)
            self.ema_cluster_size = (
                self.decay * self.ema_cluster_size
                + (1 - self.decay) * batch_cluster_size
            )

            # 更新各条目的嵌入均值
            batch_embed_avg = one_hot.T @ flat                         # (K, D)
            self.ema_embed_avg = (
                self.decay * self.ema_embed_avg
                + (1 - self.decay) * batch_embed_avg
            )

            # Laplace 平滑后归一化更新码本
            total = self.ema_cluster_size.sum()
            smoothed = (
                (self.ema_cluster_size + 1e-5)
                / (total + self.codebook_size * 1e-5)
                * total
            )
            self.codebook = self.ema_embed_avg / smoothed.unsqueeze(1)

            # 死码重置：使用频率过低的条目随机重置到当前 batch 的样本上
            dead_mask = self.ema_cluster_size < self.dead_threshold
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                rand_idx = torch.randperm(flat.size(0), device=flat.device)[:n_dead]
                self.codebook[dead_mask] = F.normalize(flat[rand_idx].detach(), dim=-1)

    @property
    def usage_rate(self) -> float:
        """码本条目使用率，训练时监控此指标，应保持 > 80%。"""
        return (self.ema_cluster_size > self.dead_threshold).float().mean().item()
