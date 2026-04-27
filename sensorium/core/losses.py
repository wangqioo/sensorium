"""
训练损失函数集合

三类损失：
  ReconLoss:           重建损失，保证原子保留足够信息量
  TemporalPredLoss:    时序预测损失，让原子序列有因果关联
  CrossModalAlignLoss: 跨模态对齐损失，同一事件的多模态原子在语义上靠近
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReconLoss(nn.Module):
    """L1 + L2 混合重建损失，比纯 L2 对异常值更鲁棒。"""

    def __init__(self, l1_weight: float = 0.5):
        super().__init__()
        self.l1_weight = l1_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        l1 = F.l1_loss(pred, target)
        l2 = F.mse_loss(pred, target)
        return self.l1_weight * l1 + (1 - self.l1_weight) * l2


class TemporalPredLoss(nn.Module):
    """
    时序预测损失：用当前窗口的 token 预测下一窗口的潜变量。

    不预测原始信号（太容易过拟合噪声），而是预测 encoder 的输出，
    这样学到的是"有意义的动态规律"而非"噪声的自相关"。
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.predictor = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_current: Tensor, z_next: Tensor) -> Tensor:
        """
        Args:
            z_current: 当前窗口的 encoder 输出，(B, D)
            z_next:    下一窗口的 encoder 输出，(B, D)，作为预测目标
        """
        z_pred = self.predictor(z_current)
        return F.mse_loss(z_pred, z_next.detach())


class CrossModalAlignLoss(nn.Module):
    """
    跨模态对齐损失（对比学习风格）。

    同一时间戳的多模态原子嵌入应互相靠近（正样本），
    不同时间戳的应互相远离（负样本）。
    适用于视觉/听觉/IMU 之间，以及触觉与其他模态之间的对齐。
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a: Tensor, emb_b: Tensor) -> Tensor:
        """
        Args:
            emb_a: 模态 A 的原子嵌入，(B, D)，已 L2 归一化
            emb_b: 模态 B 的原子嵌入，(B, D)，已 L2 归一化
            同一 batch 内，索引相同的样本为正样本对（同时间戳）
        """
        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)

        logits = (emb_a @ emb_b.T) / self.temperature  # (B, B)
        labels = torch.arange(emb_a.size(0), device=emb_a.device)

        # 双向对比：A→B 和 B→A 各算一次
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) / 2


class ImageBindAlignLoss(nn.Module):
    """
    与 ImageBind 嵌入空间对齐的损失。

    ImageBind 已经免费把视觉/听觉/IMU 对齐到同一空间，
    用它作为"语义锚"来监督我们自己的原子嵌入。
    触觉模态不走此路，改用 CrossModalAlignLoss。
    """

    def forward(self, our_emb: Tensor, imagebind_emb: Tensor) -> Tensor:
        """
        Args:
            our_emb:        我们的编码器输出，(B, D_ours)
            imagebind_emb:  ImageBind 编码的对应样本，(B, D_ib)，预先提取好，不参与反传
        """
        our_norm = F.normalize(our_emb, dim=-1)
        ib_norm = F.normalize(imagebind_emb.detach(), dim=-1)

        # 如果维度不一致，用线性投影适配（在编码器里处理，这里只算对齐损失）
        return 1 - (our_norm * ib_norm).sum(dim=-1).mean()
