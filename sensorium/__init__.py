"""
Sensorium — 多模态原子感知系统

三阶段架构：
  Stage 1: 原子编码器  (各模态 VQ-VAE，自监督，无标注)
  Stage 2: 跨模态对齐 (ImageBind 锚点 + 对比损失)
  Stage 3: LLM 微调   (感官 token + 文字 token 联合训练)
"""

__version__ = "0.1.0"
