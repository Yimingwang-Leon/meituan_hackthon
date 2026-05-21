from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .agent_spec import AgentSpec, parse_agent_spec
from .placeholders import (
    Placeholder,
    extract_placeholders,
    fill_placeholders,
)
from .rule_parser import parse_rules
from .rule_validation import (
    RuleValidationIssue,
    auto_fix_rules,
    print_validation_report,
    validate_rules,
)
from .rules import Rule
from .test_case_generator import SimulationCase, build_test_cases


ProgressCallback = Callable[[str], None]


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
    validation_issues: list[RuleValidationIssue] = field(default_factory=list)

    @property
    def test_cases(self) -> list[SimulationCase]:
        """向后兼容：退化为第一个 sub_plan 的 test_cases。"""
        return self.sub_plans[0].test_cases if self.sub_plans else []

    @property
    def validation_errors(self) -> list[RuleValidationIssue]:
        return [i for i in self.validation_issues if i.level == "error"]

    @property
    def validation_warnings(self) -> list[RuleValidationIssue]:
        return [i for i in self.validation_issues if i.level == "warning"]


def build_simulation_plan(
    instruction: str,
    num_sets: int = 3,
    auto_fix: bool = True,
    max_fix_attempts: int = 2,
    progress_callback: ProgressCallback | None = None,
) -> SimulationPlan:
    _emit_progress(progress_callback, "1/6 解析指令规则")
    parsed_rules = parse_rules(instruction)

    # 规则质量校验
    _emit_progress(progress_callback, "2/6 校验规则质量")
    issues = validate_rules(parsed_rules)
    print_validation_report(parsed_rules, issues)

    # 自动修复：若有 error 且开启 auto_fix，让 LLM 重写问题规则
    if auto_fix and any(i.level == "error" for i in issues):
        _emit_progress(progress_callback, "3/6 自动修复问题规则")
        parsed_rules, issues = auto_fix_rules(
            parsed_rules, issues, instruction, max_attempts=max_fix_attempts
        )
        print_validation_report(parsed_rules, issues)

    _emit_progress(progress_callback, "4/6 归纳 Agent 职责")
    agent_spec = parse_agent_spec(instruction)
    _emit_progress(progress_callback, "5/6 提取占位符并生成场景")
    extraction = extract_placeholders(instruction, agent_spec, num_sets=num_sets)
    _emit_progress(progress_callback, "6/6 生成目标测试用例")
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
        validation_issues=issues,
    )


def _emit_progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)
