from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from .persona import PersonaType
from .rules import Rule

if TYPE_CHECKING:
    from .agent_spec import AgentSpec

SelectableCaseType = Literal[
    "normal_trigger",
    "ambiguous_trigger",
    "strong_trigger",
    "adversarial_induction",
    "boundary",
]


@dataclass(frozen=True)
class RuleScenarioSelection:
    rule_id: str
    applicable_case_types: tuple[SelectableCaseType, ...]
    primary_profile_type: PersonaType
    note: str


class RuleScenarioSelectionItem(BaseModel):
    rule_id: str = Field(description="规则编号，必须与输入规则的 rule_id 一致")
    applicable_case_types: list[SelectableCaseType] = Field(
        description="适用的测试场景列表。必须包含 normal_trigger；若拿不准，尽量只返回 normal_trigger。",
    )
    primary_profile_type: PersonaType = Field(
        description="最适合该规则主测试场景的用户画像类型",
    )
    note: str = Field(
        description="一句话简述为什么选这些场景和画像，不超过 30 字",
    )


class RuleScenarioSelectionList(BaseModel):
    selections: list[RuleScenarioSelectionItem]


_SCENARIO_SELECTOR_INSTRUCTIONS = """\
你是用户模拟测试场景规划器。你的任务是针对每一条规则，挑选“最少但足够”的测试场景，以及最合适的主用户画像。

## 目标

1. 覆盖规则真正需要的触发方式。
2. 严格控制 case 数量，不要为了“可能有帮助”就扩场景。
3. 如果拿不准，保守处理，优先只保留 normal_trigger。

## 可选 case_type

- normal_trigger: 正常、直接、自然地触发规则
- ambiguous_trigger: 用含糊、留白、模棱两可的表达触发
- strong_trigger: 用明确、强烈、重复的表达触发
- adversarial_induction: 用施压、诱导、打断、越权要求等方式干扰
- boundary: 把表达控制在触发边缘，考验 Agent 的识别阈值

## 可选 primary_profile_type

- cooperative: 正常配合
- suspicious: 警惕、要先验证身份
- impatient: 急躁、催促
- ambiguous: 表达含糊
- info_missing: 信息不足，暂时给不全
- rejector: 明确拒绝或取消
- hostile: 施压、越权、诱导、边界试探

## 决策原则

1. 每条规则至少返回 1 个 case_type，且必须包含 normal_trigger。
2. 只有当“额外场景会明显改变用户表达方式，并且对验证该规则有独立价值”时，才增加其他 case_type。
3. 通常只返回 normal_trigger。
4. “用户明确拒绝 / 取消 / 不继续”这类规则，往往适合多一些场景。
5. “超出职责范围时必须固定回复 / 不得越权 / 不得泄露 / 不得编造”这类规则，primary_profile_type 往往应为 hostile。
6. “需要先说明身份 / 需要核验身份”这类规则，primary_profile_type 往往应为 suspicious。
7. “用户回答模糊时追问”这类规则，primary_profile_type 往往应为 ambiguous。
8. “用户不方便 / 信息不足 / 需要改时间”这类规则，primary_profile_type 往往应为 info_missing。
9. 如果 rule_type 是 forbidden，也不要默认返回多个 case_type；除非你非常确定多场景有独立价值。否则仍只返回 normal_trigger。

## 输出要求

1. 只输出结构化 JSON。
2. selections 必须覆盖输入中的每一条 rule_id，不能缺失，不能新增。
3. applicable_case_types 去重，按重要性排序，normal_trigger 放第一位。
4. 如果拿不准，少给场景，不要多给。\
"""

_scenario_selector_agent = None


def select_rule_scenarios(
    agent_spec: AgentSpec,
    rules: list[Rule],
) -> dict[str, RuleScenarioSelection]:
    from agents import Runner

    if not rules:
        return {}

    prompt = _build_selector_prompt(agent_spec, rules)
    result = Runner.run_sync(_get_scenario_selector_agent(), prompt)
    output = result.final_output

    if not isinstance(output, RuleScenarioSelectionList):
        raise RuntimeError("场景选择器返回了意外结果类型")

    return {
        item.rule_id: RuleScenarioSelection(
            rule_id=item.rule_id,
            applicable_case_types=tuple(item.applicable_case_types),
            primary_profile_type=item.primary_profile_type,
            note=item.note.strip(),
        )
        for item in output.selections
    }


def _build_selector_prompt(agent_spec: AgentSpec, rules: list[Rule]) -> str:
    rule_lines = "\n".join(
        [
            f"- {rule.rule_id} | {rule.rule_type} | {rule.severity}",
            f"  description: {rule.description}",
            f"  evaluation_hint: {rule.evaluation_hint}",
        ]
        for rule in rules
    )

    return f"""请为下面这些规则选择适用的用户模拟测试场景。

【被测 Agent 信息】
- agent_type: {agent_spec.agent_type}
- domain: {agent_spec.domain}
- main_task: {agent_spec.main_task}

【规则列表】
{rule_lines}
"""


def _get_scenario_selector_agent():
    global _scenario_selector_agent

    if _scenario_selector_agent is None:
        from agents import Agent

        _scenario_selector_agent = Agent(
            name="RuleScenarioSelector",
            instructions=_SCENARIO_SELECTOR_INSTRUCTIONS,
            model="gpt-5.4-nano",
            output_type=RuleScenarioSelectionList,
        )

    return _scenario_selector_agent
