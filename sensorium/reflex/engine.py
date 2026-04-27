"""
反射引擎 — 第二层：快速通路

监听原子 token 流，匹配预设规则，触发立即动作。
完全不经过 LLM，端到端延迟 <50ms。

类比：脊髓反射弧，不需要皮层参与。

规则配置方式：
  原子 ID（整数）在 Stage 1 训练完成后，通过检查码本人工填入。
  每条规则绑定：{触发条件 → 动作回调 → 优先级 → 触发窗口}

触发逻辑：
  滑动窗口内检测到目标原子出现 N 次 → 触发
  高优先级规则抢占低优先级
  同一规则有冷却时间防止重复触发
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ReflexRule:
    """
    单条反射规则。

    Attributes:
        name:        规则名称（用于日志）
        modality:    感官模态，"VIS" / "AUD" / "IMU" / "TAC"
        atom_id:     触发的原子 token ID（Stage 1 训练后填入）
        action:      触发时调用的回调函数
        priority:    优先级，数字越大越高，高优先级抢占低优先级
        window_ms:   滑动检测窗口时长（毫秒）
        min_count:   窗口内最少出现次数才触发
        cooldown_ms: 触发后的冷却时间（毫秒），防止重复触发
    """
    name: str
    modality: str
    atom_id: int
    action: Callable[[], None]
    priority: int = 5
    window_ms: int = 500
    min_count: int = 1
    cooldown_ms: int = 1000

    _last_trigger: float = field(default=0.0, init=False, repr=False)

    def is_cooled_down(self) -> bool:
        return (time.monotonic() - self._last_trigger) * 1000 > self.cooldown_ms

    def mark_triggered(self) -> None:
        self._last_trigger = time.monotonic()


class ReflexEngine:
    """
    反射引擎主类。

    用法：
        engine = ReflexEngine()
        engine.add_rule(ReflexRule(
            name="头部抚摸",
            modality="TAC",
            atom_id=87,          # 训练后填入
            action=lambda: robot.wag(),
            priority=5,
        ))
        # 主循环里持续喂入 token
        engine.push("TAC", token_id=87)
    """

    def __init__(self):
        self._rules: list[ReflexRule] = []
        self._history: dict[str, deque] = {
            "VIS": deque(),
            "AUD": deque(),
            "IMU": deque(),
            "TAC": deque(),
        }
        self._lock = threading.Lock()

    def add_rule(self, rule: ReflexRule) -> None:
        with self._lock:
            self._rules.append(rule)
            self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, name: str) -> None:
        with self._lock:
            self._rules = [r for r in self._rules if r.name != name]

    def push(self, modality: str, token_id: int) -> str | None:
        """
        推入一个新的原子 token，检查是否触发任何规则。

        Args:
            modality: "VIS" / "AUD" / "IMU" / "TAC"
            token_id: 原子 token 的整数 ID

        Returns:
            被触发的规则名称，未触发则返回 None
        """
        now = time.monotonic()

        with self._lock:
            # 记录 token 历史
            hist = self._history[modality]
            hist.append((now, token_id))

            # 按优先级遍历规则，找到第一条满足条件的
            for rule in self._rules:
                if rule.modality != modality:
                    continue
                if not rule.is_cooled_down():
                    continue

                # 检查窗口内计数
                cutoff = now - rule.window_ms / 1000
                count = sum(
                    1 for ts, tid in hist
                    if ts >= cutoff and tid == rule.atom_id
                )

                if count >= rule.min_count:
                    rule.mark_triggered()
                    # 在新线程里执行动作，不阻塞 token 流
                    threading.Thread(target=rule.action, daemon=True).start()
                    return rule.name

        return None

    def push_batch(self, tokens: dict[str, list[int]]) -> list[str]:
        """
        批量推入多模态 token。

        Args:
            tokens: {"VIS": [437, 201], "AUD": [89], "IMU": [12], "TAC": [87]}
        Returns:
            所有触发的规则名称列表
        """
        triggered = []
        for modality, ids in tokens.items():
            for tid in ids:
                result = self.push(modality, tid)
                if result:
                    triggered.append(result)
        return triggered

    def _cleanup_history(self, max_age_ms: int = 5000) -> None:
        """清理过期的历史记录（可在后台定期调用）。"""
        cutoff = time.monotonic() - max_age_ms / 1000
        with self._lock:
            for hist in self._history.values():
                while hist and hist[0][0] < cutoff:
                    hist.popleft()


# ——— 预设规则工厂函数 ———
# 原子 ID 需要在 Stage 1 训练完成后填入实际值

def make_default_rules(robot) -> list[ReflexRule]:
    """
    创建默认反射规则集。
    robot 对象需要实现：turn_head(), wag(), flinch(), emergency_stop() 等接口。
    原子 ID 字段（atom_id）标注了 TODO，等码本训练完后填入。
    """
    return [
        ReflexRule(
            name="突然大声—转头",
            modality="AUD",
            atom_id=0,           # TODO: 填入"能量突增"对应的 AUD token ID
            action=robot.turn_head_to_sound,
            priority=10,
            window_ms=100,
            min_count=1,
            cooldown_ms=2000,
        ),
        ReflexRule(
            name="头部被抚摸—放松",
            modality="TAC",
            atom_id=0,           # TODO: 填入"头部轻抚"对应的 TAC token ID
            action=robot.enter_calm_mode,
            priority=5,
            window_ms=500,
            min_count=2,
            cooldown_ms=3000,
        ),
        ReflexRule(
            name="突然撞击—缩避",
            modality="TAC",
            atom_id=0,           # TODO: 填入"高能量撞击"对应的 TAC token ID
            action=robot.flinch_and_alert,
            priority=9,
            window_ms=100,
            min_count=1,
            cooldown_ms=1000,
        ),
        ReflexRule(
            name="节律性拍打—同频响应",
            modality="TAC",
            atom_id=0,           # TODO: 填入"间歇性拍打"对应的 TAC token ID
            action=robot.sync_wag_to_pat,
            priority=4,
            window_ms=2000,
            min_count=3,
            cooldown_ms=500,
        ),
        ReflexRule(
            name="跌倒检测—紧急停止",
            modality="IMU",
            atom_id=0,           # TODO: 填入"跌倒加速度模式"对应的 IMU token ID
            action=robot.emergency_stop,
            priority=10,
            window_ms=200,
            min_count=1,
            cooldown_ms=5000,
        ),
    ]
