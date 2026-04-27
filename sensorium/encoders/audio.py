"""
听觉原子编码器

封装 WavTokenizer（ICLR 2025），直接使用预训练权重，无需训练。
WavTokenizer 生成 40-75 token/秒，此处下采样到 10 token/秒用于 LLM 集成。

WavTokenizer 已提供足够好的语义内容，能区分：
  人声 / 环境音 / 冲击声 / 音乐 / 静默 等类别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path


class AudioEncoder(nn.Module):
    """
    WavTokenizer 的封装器。

    Args:
        model_path:  WavTokenizer 权重路径（首次使用需从官方仓库下载）
        config_path: WavTokenizer 配置文件路径
        target_sr:   目标采样率，默认 24000Hz（WavTokenizer 要求）
        tokens_per_sec: 输出 token 密度，默认 10/秒
    """

    CODEBOOK_SIZE = 4096  # WavTokenizer 的码本大小（固定）

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        target_sr: int = 24000,
        tokens_per_sec: int = 10,
    ):
        super().__init__()
        self.target_sr = target_sr
        self.tokens_per_sec = tokens_per_sec
        self.model_path = model_path
        self.config_path = config_path
        self._model = None  # 懒加载

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self.model_path is None or self.config_path is None:
            raise RuntimeError(
                "需要提供 WavTokenizer 的模型权重和配置文件路径。\n"
                "下载地址：https://github.com/jishengpeng/WavTokenizer"
            )
        # WavTokenizer 的加载方式（参考官方 README）
        from encoder.utils import convert_audio
        from decoder.pretrained import WavTokenizer as _WavTokenizer
        self._model = _WavTokenizer.from_pretrained0802(self.config_path, self.model_path)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    def encode(self, waveform: Tensor, sr: int) -> Tensor:
        """
        Args:
            waveform: 原始音频波形，(1, num_samples) 或 (num_samples,)
            sr:       输入采样率

        Returns:
            token 序列，(T,)，T 取决于音频时长和 tokens_per_sec
        """
        self._load_model()
        from encoder.utils import convert_audio

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # 重采样到目标采样率
        if sr != self.target_sr:
            waveform = convert_audio(waveform, sr, self.target_sr, 1)

        with torch.no_grad():
            # WavTokenizer 输出：bandwidth_id=0 为最低比特率（约 40 token/秒）
            features, discrete_code = self._model.encode_infer(
                waveform, bandwidth_id=torch.tensor([0])
            )
            # discrete_code shape: (1, 1, T_wav)
            tokens = discrete_code.squeeze()  # (T_wav,)

        # 下采样到目标密度（每 N 个取 1 个，简单方式）
        step = max(1, tokens.shape[0] // (tokens.shape[0] * self.tokens_per_sec // 40))
        return tokens[::step]

    def tokenize(self, waveform: Tensor, sr: int) -> Tensor:
        """推理接口别名。"""
        return self.encode(waveform, sr)

    @property
    def codebook_size(self) -> int:
        return self.CODEBOOK_SIZE
