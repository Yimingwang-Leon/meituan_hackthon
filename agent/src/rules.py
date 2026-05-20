from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

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
    """评测系统的核心契约。每条规则必须包含足够的结构化信息供下游使用。"""

    rule_id: str
    description: str
    rule_type: RuleType
    severity: Severity = "major"

    # 显式语义字段（替代旧的 evaluation_hint 自由文本）
    trigger_condition: str = ""        # 仅 conditional 必填；required/forbidden 空字符串
    expected_behavior: str = ""        # 触发后/全程 Agent 应该做什么
    failure_criteria: list[str] = field(default_factory=list)  # 至少 1 条；列举具体失败行为
    evidence_requirement: str = ""     # 提示 judge 应引用什么作为证据

    # 代码确定性检查（无则交给 LLM judge）
    checks: list["DeterministicCheck"] = field(default_factory=list)


# 旧的硬编码规则集已废弃。系统现在通过 parse_rules() 从 instruction 动态生成规则。
# 保留空列表作为 evaluator.py 的兜底，正常路径下不会用到。
RULES: list[Rule] = []
