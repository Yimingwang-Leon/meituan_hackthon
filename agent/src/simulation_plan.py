from __future__ import annotations

from dataclasses import dataclass

from .agent_spec import AgentSpec, parse_agent_spec
from .placeholders import (
    Placeholder,
    extract_placeholders,
    fill_placeholders,
)
from .rule_parser import parse_rules
from .rules import Rule
from .test_case_generator import SimulationCase, build_test_cases


@dataclass(frozen=True)
class SubPlan:
    """一组占位符填值对应的子计划。"""
    set_id: str
    label: str
    scenario_hint: str
    placeholder_values: dict[str, str]
    filled_instruction: str
    test_cases: list[SimulationCase]


@dataclass(frozen=True)
class SimulationPlan:
    original_instruction: str
    parsed_rules: list[Rule]
    agent_spec: AgentSpec
    placeholders: list[Placeholder]
    sub_plans: list[SubPlan]

    # 向后兼容：旧代码可能直接访问 .test_cases
    # 退化为第一个 sub_plan 的 test_cases
    @property
    def test_cases(self) -> list[SimulationCase]:
        return self.sub_plans[0].test_cases if self.sub_plans else []


def build_simulation_plan(instruction: str, num_sets: int = 3) -> SimulationPlan:
    parsed_rules = parse_rules(instruction)
    agent_spec = parse_agent_spec(instruction)
    extraction = extract_placeholders(instruction, agent_spec, num_sets=num_sets)
    test_cases = build_test_cases(agent_spec, parsed_rules)

    # 若 LLM 没有给出 sets（极少见的兜底）
    sets = extraction.sets or []
    if not sets:
        from .placeholders import PlaceholderSet
        sets = [PlaceholderSet(
            set_id="set_default",
            label="默认",
            scenario_hint="无占位符或自动生成失败的兜底",
            values=[],
        )]

    sub_plans = [
        SubPlan(
            set_id=s.set_id,
            label=s.label,
            scenario_hint=s.scenario_hint,
            placeholder_values=s.values_dict(),
            filled_instruction=fill_placeholders(instruction, s.values_dict()),
            test_cases=test_cases,
        )
        for s in sets
    ]
    return SimulationPlan(
        original_instruction=instruction,
        parsed_rules=parsed_rules,
        agent_spec=agent_spec,
        placeholders=extraction.placeholders,
        sub_plans=sub_plans,
    )
