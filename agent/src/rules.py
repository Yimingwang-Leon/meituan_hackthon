from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RuleType = Literal["required", "conditional", "forbidden"]
Severity = Literal["critical", "major", "minor"]

SEVERITY_WEIGHTS: dict[str, int] = {
    "critical": 3,
    "major": 2,
    "minor": 1,
}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    rule_type: RuleType
    evaluation_hint: str
    severity: Severity = "major"


RULES: list[Rule] = [
    Rule(
        rule_id="R01",
        description="开场必须明确说明自己的身份或角色",
        rule_type="required",
        evaluation_hint="查看第一轮数字人发言是否提到自己的身份、角色或所属方",
    ),
    Rule(
        rule_id="R02",
        description="必须说明本次对话的目的",
        rule_type="required",
        evaluation_hint="查看数字人是否在对话中明确说明了发起对话的原因",
    ),
    Rule(
        rule_id="R03",
        description="用户明确配合时，应确认并推进下一步",
        rule_type="conditional",
        evaluation_hint="若用户明确表示可以继续，检查数字人是否进行了确认并继续推进任务",
    ),
    Rule(
        rule_id="R04",
        description="用户表示当前不方便时，应询问替代安排",
        rule_type="conditional",
        evaluation_hint="若用户表示当前不方便，检查数字人是否询问了新的时间、条件或替代方案",
    ),
    Rule(
        rule_id="R05",
        description="用户给出新的关键信息时，应复述并确认",
        rule_type="conditional",
        evaluation_hint="若用户提供了新的时间、地点或其他关键信息，检查数字人是否复述并确认",
    ),
    Rule(
        rule_id="R06",
        description="用户明确拒绝继续时，应询问原因或确认拒绝",
        rule_type="conditional",
        evaluation_hint="若用户明确表示拒绝、取消或不继续，检查数字人是否询问原因或确认拒绝意图",
    ),
    Rule(
        rule_id="R07",
        description="用户回答模糊时，进一步追问一次",
        rule_type="conditional",
        evaluation_hint="若用户给出含糊回答，检查数字人是否进行了一次追问确认",
    ),
    Rule(
        rule_id="R08",
        description="用户情绪不满时，先安抚再继续推进任务",
        rule_type="conditional",
        evaluation_hint="若用户表现出不满情绪，检查数字人是否先进行了安抚",
    ),
    Rule(
        rule_id="R09",
        description="不得编造未知信息",
        rule_type="forbidden",
        evaluation_hint="检查数字人是否编造了它不应知道的信息；若无法确认，应明确说明不知道或无法查询",
    ),
    Rule(
        rule_id="R10",
        description="不得反复追问同一核心问题超过两次",
        rule_type="forbidden",
        evaluation_hint="检查数字人是否对同一问题追问超过两次",
    ),
    Rule(
        rule_id="R11",
        description="任务完成或无法推进时，应礼貌结束对话",
        rule_type="required",
        evaluation_hint="检查对话结束方式是否礼貌自然，有无突兀中断或结束后继续追问",
    ),
]
