from __future__ import annotations

from dataclasses import dataclass

from .agent_spec import AgentSpec, parse_agent_spec
from .rule_parser import parse_rules
from .rules import Rule
from .test_case_generator import SimulationCase, build_test_cases


@dataclass(frozen=True)
class SimulationPlan:
    parsed_rules: list[Rule]
    agent_spec: AgentSpec
    test_cases: list[SimulationCase]


def build_simulation_plan(instruction: str) -> SimulationPlan:
    parsed_rules = parse_rules(instruction)
    agent_spec = parse_agent_spec(instruction)
    test_cases = build_test_cases(agent_spec, parsed_rules)
    return SimulationPlan(
        parsed_rules=parsed_rules,
        agent_spec=agent_spec,
        test_cases=test_cases,
    )
