from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RuleType = Literal["required", "conditional", "forbidden"]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    rule_type: RuleType
    evaluation_hint: str


RULES: list[Rule] = [
    Rule(
        rule_id="R01",
        description="开场必须说明自己是美团配送助手",
        rule_type="required",
        evaluation_hint="查看第一轮数字人发言是否提到美团或配送助手身份",
    ),
    Rule(
        rule_id="R02",
        description="必须说明来电目的是确认订单是否方便接收",
        rule_type="required",
        evaluation_hint="查看数字人是否在对话中说明了来电原因",
    ),
    Rule(
        rule_id="R03",
        description="用户表示方便时，确认并提醒保持电话畅通",
        rule_type="conditional",
        evaluation_hint="若用户表示方便接收，检查数字人是否确认并提醒保持电话畅通",
    ),
    Rule(
        rule_id="R04",
        description="用户不方便时，询问新的可接收时间",
        rule_type="conditional",
        evaluation_hint="若用户表示不方便，检查数字人是否询问了新的可接收时间",
    ),
    Rule(
        rule_id="R05",
        description="用户给出新时间时，复述该时间并确认已记录",
        rule_type="conditional",
        evaluation_hint="若用户提供了新时间，检查数字人是否复述该时间并说明已记录",
    ),
    Rule(
        rule_id="R06",
        description="用户拒绝接收订单时，询问拒收原因",
        rule_type="conditional",
        evaluation_hint="若用户明确拒绝接收，检查数字人是否询问了拒收原因",
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
        description="不得编造未知信息（订单金额、菜品明细、骑手实时位置等）",
        rule_type="forbidden",
        evaluation_hint="检查数字人是否编造了它不应知道的信息，正确做法是建议用户打开App查看",
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
