"""
滚动上下文窗口

管理过去 N 秒的多模态 token 历史，供 LLM 推理时构建输入序列。

设计原则：
  - 按时间戳自动淘汰过期条目
  - 对文字和传感器 token 分别维护，组装时按时序交织
  - 支持"当前传感器快照"插在上下文末尾（最新状态）
"""

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ContextEntry:
    """上下文中的一条记录，可以是传感器 token 或文字。"""
    timestamp: float
    modality: str            # "VIS" / "AUD" / "IMU" / "TAC" / "TEXT"
    tokens: list[int]        # 对文字，是 tokenizer 编码后的 ID 列表
    text: str = ""           # 仅 modality=="TEXT" 时有内容，便于调试


class ContextWindow:
    """
    滚动多模态上下文窗口。

    Args:
        max_seconds: 保留最近多少秒的历史，默认 30 秒
        max_entries: 最大条目数（防止极高频传感器撑爆内存）
    """

    def __init__(self, max_seconds: float = 30.0, max_entries: int = 2000):
        self.max_seconds = max_seconds
        self.max_entries = max_entries
        self._entries: deque[ContextEntry] = deque(maxlen=max_entries)

    def push_sensor(self, modality: str, tokens: list[int]) -> None:
        """推入一批传感器 token。"""
        self._entries.append(ContextEntry(
            timestamp=time.monotonic(),
            modality=modality,
            tokens=tokens,
        ))
        self._evict_expired()

    def push_text(self, text_ids: list[int], text: str = "") -> None:
        """推入一段文字（用户输入或上一轮 LLM 输出）。"""
        self._entries.append(ContextEntry(
            timestamp=time.monotonic(),
            modality="TEXT",
            tokens=text_ids,
            text=text,
        ))
        self._evict_expired()

    def build_sequence(
        self,
        current_snapshot: dict[str, list[int]] | None = None,
        tokenizer=None,
    ) -> list[int]:
        """
        把窗口内所有条目按时序组装成一个 token ID 列表。

        格式：
          [历史传感器+文字交织序列]
          <SENSOR_START>
          <VIS_START> VIS_xxx ...
          <AUD_START> AUD_xxx ...
          <IMU_START> IMU_xxx ...
          <TAC_START> TAC_xxx ...
          <SENSOR_END>
          （以上是"当前快照"，追加在历史末尾）

        Args:
            current_snapshot: 最新传感器快照 {modality: [atom_ids]}
            tokenizer:        SensoriumTokenizer，用于生成当前快照的 token ID

        Returns:
            完整的 token ID 列表，直接喂给 LLM
        """
        ids: list[int] = []

        # 历史序列（按时序排列）
        for entry in self._entries:
            ids.extend(entry.tokens)

        # 当前快照追加在末尾
        if current_snapshot and tokenizer:
            snapshot_ids = tokenizer.encode_multimodal(
                vis_tokens=current_snapshot.get("VIS"),
                aud_tokens=current_snapshot.get("AUD"),
                imu_tokens=current_snapshot.get("IMU"),
                tac_tokens=current_snapshot.get("TAC"),
            )
            ids.extend(snapshot_ids)

        return ids

    def _evict_expired(self) -> None:
        cutoff = time.monotonic() - self.max_seconds
        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.popleft()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def token_count(self) -> int:
        return sum(len(e.tokens) for e in self._entries)
