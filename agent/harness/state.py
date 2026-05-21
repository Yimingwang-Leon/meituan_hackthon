from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DialogueStage(str, Enum):
    INIT = "INIT"
    IDENTITY_VERIFY = "IDENTITY_VERIFY"
    INTENT_EXPLAIN = "INTENT_EXPLAIN"
    PROBLEM_CONFIRM = "PROBLEM_CONFIRM"
    SOLUTION_PROVIDE = "SOLUTION_PROVIDE"
    FINISH = "FINISH"
    RULE_VIOLATION = "RULE_VIOLATION"
    TRANSFER_TO_HUMAN = "TRANSFER_TO_HUMAN"


REFUSAL_PATTERNS = ("不要", "不用", "不需要", "取消", "拒绝", "不想", "没空", "别打", "挂了")
HUMAN_REQUEST_PATTERNS = ("人工", "真人", "客服", "投诉", "经理", "上级")
IDENTITY_CONFIRM_PATTERNS = ("是我", "我是", "对", "嗯", "没错", "本人", "你说")
IDENTITY_ASK_PATTERNS = ("请问是", "是您", "本人", "确认一下")
TASK_DONE_PATTERNS = ("已确认", "完成", "已经处理", "帮您处理", "稍后回电", "转人工", "结束")
SENSITIVE_PATTERNS = (
    re.compile(r"(?<!\d)1\d{10}(?!\d)"),
    re.compile(r"(?<!\d)\d{15,19}(?!\d)"),
    re.compile(r"订单(?:金额|详情|地址|手机号|号码)"),
)


@dataclass
class DialogueState:
    case_id: str
    current_stage: DialogueStage = DialogueStage.INIT
    identity_verified: bool = False
    user_refusal_count: int = 0
    tool_called: list[str] = field(default_factory=list)
    sensitive_info_disclosed: bool = False
    task_completed: bool = False
    rule_violations: list[dict[str, Any]] = field(default_factory=list)
    max_turn_exceeded: bool = False
    human_requested: bool = False
    turn_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["current_stage"] = self.current_stage.value
        return data

    def diff(self, before: dict[str, Any]) -> dict[str, dict[str, Any]]:
        after = self.to_dict()
        return {
            key: {"before": before.get(key), "after": value}
            for key, value in after.items()
            if before.get(key) != value
        }

    def mark_violations(self, violations: list[dict[str, Any]]) -> None:
        known = {
            (v.get("rule_id"), v.get("turn_id"), v.get("message"))
            for v in self.rule_violations
        }
        for violation in violations:
            key = (
                violation.get("rule_id"),
                violation.get("turn_id"),
                violation.get("message"),
            )
            if key not in known:
                self.rule_violations.append(dict(violation))
                known.add(key)
        if violations:
            self.current_stage = DialogueStage.RULE_VIOLATION


def update_state_after_turn(
    state: DialogueState,
    user_message: str | None,
    agent_message: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    max_turns: int | None = None,
    previous_agent_message: str | None = None,
) -> dict[str, dict[str, Any]]:
    before = state.to_dict()
    state.turn_count += 1

    user_text = user_message or ""
    agent_text = agent_message or ""

    if user_text and any(token in user_text for token in REFUSAL_PATTERNS):
        state.user_refusal_count += 1
    if user_text and any(token in user_text for token in HUMAN_REQUEST_PATTERNS):
        state.human_requested = True
    identity_context = previous_agent_message or agent_text
    if (
        user_text
        and any(token in user_text for token in IDENTITY_CONFIRM_PATTERNS)
        and any(token in identity_context for token in IDENTITY_ASK_PATTERNS)
    ):
        state.identity_verified = True

    if agent_text:
        if any(token in agent_text for token in IDENTITY_ASK_PATTERNS):
            state.current_stage = DialogueStage.IDENTITY_VERIFY
        elif any(token in agent_text for token in ("来电", "通知", "提醒", "确认", "了解")):
            state.current_stage = DialogueStage.INTENT_EXPLAIN
        elif any(token in agent_text for token in ("原因", "问题", "情况", "方便说")):
            state.current_stage = DialogueStage.PROBLEM_CONFIRM
        elif any(token in agent_text for token in ("处理", "方案", "可以", "建议", "帮您")):
            state.current_stage = DialogueStage.SOLUTION_PROVIDE
        if any(token in agent_text for token in ("转人工", "人工处理", "同事回电")):
            state.current_stage = DialogueStage.TRANSFER_TO_HUMAN
        if any(token in agent_text for token in TASK_DONE_PATTERNS):
            state.task_completed = True
        if any(pattern.search(agent_text) for pattern in SENSITIVE_PATTERNS):
            state.sensitive_info_disclosed = True

    for call in tool_calls or []:
        name = str(call.get("name") or call.get("tool_name") or "")
        if name:
            state.tool_called.append(name)

    if max_turns is not None and state.turn_count >= max_turns:
        state.max_turn_exceeded = True

    if state.task_completed and not state.rule_violations:
        state.current_stage = DialogueStage.FINISH

    return state.diff(before)
