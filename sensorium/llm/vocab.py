"""
LLM 词汇表扩展

在 Qwen2.5-7B（或任意 HuggingFace 模型）的基础上，
追加感官 token 作为新词汇，让 LLM 能直接处理原子 token 流。

新增 token 设计：
  VIS_0000 ~ VIS_1023   → 1024 个视觉原子
  AUD_0000 ~ AUD_1023   → 1024 个听觉原子
  IMU_000  ~ IMU_255    →  256 个运动原子
  TAC_000  ~ TAC_255    →  256 个触觉原子
  特殊控制 token         →    6 个

训练策略：
  Stage A: 冻结 LLM，只训练新 token 的 embedding
  Stage B: LoRA 全量微调（rank=64，alpha=128）
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer, PreTrainedTokenizer


# 各模态码本大小（与编码器对齐）
CODEBOOK_SIZES = {
    "VIS": 1024,
    "AUD": 1024,
    "IMU": 256,
    "TAC": 256,
}

# 特殊控制 token
SPECIAL_TOKENS = [
    "<SENSOR_START>",
    "<SENSOR_END>",
    "<VIS_START>",
    "<AUD_START>",
    "<IMU_START>",
    "<TAC_START>",
]

# 各模态在新词汇表里的起始偏移（用于将原子 ID 转换为词汇表 ID）
MODALITY_OFFSETS: dict[str, int] = {}


def _build_sensor_tokens() -> list[str]:
    """生成所有感官 token 的字符串列表。"""
    tokens = list(SPECIAL_TOKENS)
    for modality, size in CODEBOOK_SIZES.items():
        pad = len(str(size - 1))
        for i in range(size):
            tokens.append(f"<{modality}_{i:0{pad}d}>")
    return tokens


def token_id_to_str(modality: str, atom_id: int) -> str:
    """将 (模态, 原子ID) 转换为 token 字符串。"""
    size = CODEBOOK_SIZES[modality]
    pad = len(str(size - 1))
    return f"<{modality}_{atom_id:0{pad}d}>"


class SensoriumTokenizer:
    """
    扩展了感官词汇的分词器。

    使用示例：
        st = SensoriumTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        ids = st.encode_multimodal(
            text="你好",
            vis_tokens=[437, 201],
            aud_tokens=[89],
            imu_tokens=[12],
            tac_tokens=[87],
        )

    Args:
        base_tokenizer: 原始 HuggingFace tokenizer
    """

    def __init__(self, base_tokenizer: PreTrainedTokenizer):
        self.tokenizer = base_tokenizer
        self._sensor_tokens = _build_sensor_tokens()
        self._extend_vocab()
        self._build_offset_map()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str | Path, **kwargs) -> "SensoriumTokenizer":
        tok = AutoTokenizer.from_pretrained(str(model_name_or_path), **kwargs)
        return cls(tok)

    def _extend_vocab(self) -> None:
        """把感官 token 加入原始词汇表。"""
        new_tokens = [t for t in self._sensor_tokens if t not in self.tokenizer.vocab]
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens, special_tokens=False)
            # 控制 token 单独标记为 special
            self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    def _build_offset_map(self) -> None:
        """建立 {模态: 在词汇表里的起始 ID} 映射。"""
        global MODALITY_OFFSETS
        for modality in CODEBOOK_SIZES:
            first_token = token_id_to_str(modality, 0)
            MODALITY_OFFSETS[modality] = self.tokenizer.convert_tokens_to_ids(first_token)

    def atom_to_vocab_id(self, modality: str, atom_id: int) -> int:
        """将原子 ID 转换为词汇表 ID。"""
        return MODALITY_OFFSETS[modality] + atom_id

    def vocab_id_to_atom(self, vocab_id: int) -> tuple[str, int] | None:
        """将词汇表 ID 反查为 (模态, 原子ID)，不是感官 token 则返回 None。"""
        for modality, offset in MODALITY_OFFSETS.items():
            size = CODEBOOK_SIZES[modality]
            if offset <= vocab_id < offset + size:
                return modality, vocab_id - offset
        return None

    def encode_multimodal(
        self,
        text: str = "",
        vis_tokens: Sequence[int] | None = None,
        aud_tokens: Sequence[int] | None = None,
        imu_tokens: Sequence[int] | None = None,
        tac_tokens: Sequence[int] | None = None,
    ) -> list[int]:
        """
        将多模态输入编码为词汇表 ID 序列。

        格式：
          <SENSOR_START>
          <VIS_START> VIS_xxx ...
          <AUD_START> AUD_xxx ...
          <IMU_START> IMU_xxx ...
          <TAC_START> TAC_xxx ...
          <SENSOR_END>
          文字 token...

        Returns:
            词汇表 ID 列表
        """
        ids: list[int] = []

        def add_tok(s: str) -> None:
            ids.append(self.tokenizer.convert_tokens_to_ids(s))

        has_sensor = any(t is not None for t in [vis_tokens, aud_tokens, imu_tokens, tac_tokens])

        if has_sensor:
            add_tok("<SENSOR_START>")

            if vis_tokens:
                add_tok("<VIS_START>")
                ids.extend(self.atom_to_vocab_id("VIS", a) for a in vis_tokens)

            if aud_tokens:
                add_tok("<AUD_START>")
                ids.extend(self.atom_to_vocab_id("AUD", a) for a in aud_tokens)

            if imu_tokens:
                add_tok("<IMU_START>")
                ids.extend(self.atom_to_vocab_id("IMU", a) for a in imu_tokens)

            if tac_tokens:
                add_tok("<TAC_START>")
                ids.extend(self.atom_to_vocab_id("TAC", a) for a in tac_tokens)

            add_tok("<SENSOR_END>")

        if text:
            text_ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.extend(text_ids)

        return ids

    def decode_multimodal(self, ids: list[int]) -> str:
        """将 ID 序列解码为可读字符串（感官 token 还原为符号表示）。"""
        parts = []
        i = 0
        while i < len(ids):
            atom = self.vocab_id_to_atom(ids[i])
            if atom:
                modality, aid = atom
                parts.append(f"[{modality}:{aid}]")
                i += 1
            else:
                # 找到连续的文字 token
                j = i + 1
                while j < len(ids) and self.vocab_id_to_atom(ids[j]) is None:
                    j += 1
                parts.append(self.tokenizer.decode(ids[i:j], skip_special_tokens=False))
                i = j
        return "".join(parts)

    def __len__(self) -> int:
        return len(self.tokenizer)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
