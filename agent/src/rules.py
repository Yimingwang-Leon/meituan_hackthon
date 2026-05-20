from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .deterministic_checks import DeterministicCheck


RuleType = Literal["required", "conditional", "forbidden"]
Severity = Literal["critical", "major", "minor"]

SEVERITY_WEIGHTS: dict[str, int] = {
    "critical": 3,
    "major": 2,
    "minor": 1,
}


@dataclass(frozen=True)
class Rule:
    """评测系统的核心契约。"""

    rule_id: str
    description: str
    rule_type: RuleType
    severity: Severity = "major"

    # 兼容旧链路：保留一段便于展示/归档的摘要字符串。
    evaluation_hint: str = ""

    # 结构化语义字段
    trigger_condition: str = ""
    expected_behavior: str = ""
    failure_criteria: list[str] = field(default_factory=list)
    evidence_requirement: str = ""

    # 可选的确定性检查器配置
    checks: list["DeterministicCheck"] = field(default_factory=list)


# 规则已改为从 instruction 动态解析。保留空列表给离线兜底路径使用。
RULES: list[Rule] = []
