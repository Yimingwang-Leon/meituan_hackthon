"""规则质量校验层。

对 parse_rules() 输出的规则做体检，暴露 LLM 拆解时可能漏掉的字段或质量问题。
本模块不修改规则，只产出 issue 列表供上层展示或决策。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .rules import Rule, Severity


IssueLevel = Literal["error", "warning"]


@dataclass(frozen=True)
class RuleValidationIssue:
    rule_id: str
    level: IssueLevel
    field: str
    message: str
    detail: str = ""

    def format_line(self) -> str:
        icon = "⛔" if self.level == "error" else "⚠️"
        return f"{icon} {self.level.upper():7} {self.rule_id} · {self.field}: {self.message}"


# 校验阈值
DESCRIPTION_MAX_LEN = 60
EVIDENCE_REQUIREMENT_MIN_LEN = 5
VALID_SEVERITIES: set[str] = {"critical", "major", "minor"}


def validate_rules(rules: list[Rule]) -> list[RuleValidationIssue]:
    """对一组 rule 做质量体检。返回所有 issue（error + warning）。"""
    issues: list[RuleValidationIssue] = []

    # 检查 rule_id 唯一性（先扫一遍）
    id_counter: dict[str, int] = {}
    for rule in rules:
        id_counter[rule.rule_id] = id_counter.get(rule.rule_id, 0) + 1
    for rule_id, count in id_counter.items():
        if count > 1:
            issues.append(RuleValidationIssue(
                rule_id=rule_id,
                level="error",
                field="rule_id",
                message=f"rule_id 重复出现 {count} 次",
            ))

    for rule in rules:
        # 1. description 非空
        if not rule.description.strip():
            issues.append(RuleValidationIssue(
                rule_id=rule.rule_id,
                level="error",
                field="description",
                message="规则描述为空",
            ))

        # 2. description 不超长（暗示合并）
        elif len(rule.description) > DESCRIPTION_MAX_LEN:
            issues.append(RuleValidationIssue(
                rule_id=rule.rule_id,
                level="warning",
                field="description",
                message=f"长度 {len(rule.description)} 字 > {DESCRIPTION_MAX_LEN}，可能合并了多个要求",
                detail=rule.description,
            ))

        # 3. conditional 必有 trigger_condition
        if rule.rule_type == "conditional" and not rule.trigger_condition.strip():
            issues.append(RuleValidationIssue(
                rule_id=rule.rule_id,
                level="error",
                field="trigger_condition",
                message="conditional 规则必须填 trigger_condition，否则无法定向触发",
            ))

        # 4. expected_behavior 非空
        if not rule.expected_behavior.strip():
            issues.append(RuleValidationIssue(
                rule_id=rule.rule_id,
                level="error",
                field="expected_behavior",
                message="缺少 expected_behavior，judge 不知道 Agent 该怎么做才算 pass",
            ))

        # 5. failure_criteria 至少 1 条
        if not rule.failure_criteria:
            issues.append(RuleValidationIssue(
                rule_id=rule.rule_id,
                level="error",
                field="failure_criteria",
                message="failure_criteria 列表为空，缺少具体失败标准",
            ))

        # 6. evidence_requirement 非空（warning，不致命）
        if len(rule.evidence_requirement.strip()) < EVIDENCE_REQUIREMENT_MIN_LEN:
            issues.append(RuleValidationIssue(
                rule_id=rule.rule_id,
                level="warning",
                field="evidence_requirement",
                message="缺少 evidence_requirement，judge 给的证据可能不规范",
            ))

        # 7. severity 合法
        if rule.severity not in VALID_SEVERITIES:
            issues.append(RuleValidationIssue(
                rule_id=rule.rule_id,
                level="error",
                field="severity",
                message=f"severity 值 '{rule.severity}' 不合法，应为 {VALID_SEVERITIES}",
            ))

    return issues


def print_validation_report(rules: list[Rule], issues: list[RuleValidationIssue]) -> None:
    """终端友好的报告打印。"""
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]

    print(f"\n========== 规则质量校验 ==========", flush=True)
    print(f"共 {len(rules)} 条规则，{len(errors)} 个 error，{len(warnings)} 个 warning", flush=True)

    if not issues:
        print("✅ 全部通过", flush=True)
        return

    for issue in issues:
        print(f"  {issue.format_line()}", flush=True)
        if issue.detail:
            print(f"     ↳ {issue.detail[:100]}", flush=True)


# ───────────────────────────────────────────────────────────────────────────
# 自动修复
# ───────────────────────────────────────────────────────────────────────────


def auto_fix_rules(
    rules: list[Rule],
    issues: list[RuleValidationIssue],
    instruction: str,
    max_attempts: int = 2,
) -> tuple[list[Rule], list[RuleValidationIssue]]:
    """对有 error 的规则，让 LLM 重新生成；warning 不修复。

    返回 (修复后的规则, 修复后仍存在的 issues)。最多尝试 max_attempts 次。
    """
    from agents import Agent, Runner
    from pydantic import BaseModel

    from .rule_parser import ParsedRule

    current_rules = list(rules)
    current_issues = list(issues)

    for attempt in range(max_attempts):
        error_issues = [i for i in current_issues if i.level == "error"]
        if not error_issues:
            break

        # 按 rule_id 聚合错误
        errors_by_id: dict[str, list[RuleValidationIssue]] = {}
        for issue in error_issues:
            errors_by_id.setdefault(issue.rule_id, []).append(issue)

        broken_ids = set(errors_by_id.keys())
        broken_rules = [r for r in current_rules if r.rule_id in broken_ids]
        if not broken_rules:
            break  # error 指向不存在的 rule_id，跳过

        print(
            f"\n========== 自动修复（第 {attempt + 1} 轮）==========",
            flush=True,
        )
        print(f"修复 {len(broken_rules)} 条规则", flush=True)

        fixed = _llm_fix_rules(broken_rules, errors_by_id, instruction)

        # 用修复后的规则替换原规则（按 rule_id 匹配）
        fixed_by_id = {r.rule_id: r for r in fixed}
        current_rules = [
            fixed_by_id.get(r.rule_id, r) for r in current_rules
        ]
        current_issues = validate_rules(current_rules)

    return current_rules, current_issues


def _llm_fix_rules(
    broken_rules: list[Rule],
    errors_by_id: dict[str, list[RuleValidationIssue]],
    instruction: str,
) -> list[Rule]:
    """让 LLM 重写一批规则，返回新版。"""
    from agents import Agent, Runner

    from .rule_parser import ParsedRule, ParsedRuleList, _build_evaluation_hint

    # 构造修复 prompt
    broken_summary = []
    for rule in broken_rules:
        issue_lines = "\n".join(
            f"    - {i.field}: {i.message}"
            for i in errors_by_id.get(rule.rule_id, [])
        )
        broken_summary.append(
            f"### {rule.rule_id} · {rule.rule_type} · {rule.severity}\n"
            f"  description: {rule.description}\n"
            f"  trigger_condition: {rule.trigger_condition!r}\n"
            f"  expected_behavior: {rule.expected_behavior!r}\n"
            f"  failure_criteria: {rule.failure_criteria}\n"
            f"  evidence_requirement: {rule.evidence_requirement!r}\n"
            f"  问题：\n{issue_lines}"
        )

    fix_prompt = (
        "以下规则在质量校验中有 error，请修复**所有列出的问题**后重新输出。\n\n"
        "## 原始指令（参考上下文）\n"
        f"{instruction[:2000]}\n\n"
        "## 待修复规则\n\n"
        + "\n\n".join(broken_summary)
        + "\n\n## 要求\n"
        "1. 保持每条规则的 rule_id / rule_type / severity 不变\n"
        "2. 补齐空字段：description、trigger_condition（仅 conditional）、expected_behavior、failure_criteria、evidence_requirement\n"
        "3. description 超长（>60 字）应拆分或精简\n"
        "4. failure_criteria 必须至少 1 条，写具体失败行为\n"
        "5. 只输出待修复的规则，不要返回未列出的规则"
    )

    fixer_agent = Agent(
        name="RuleFixer",
        instructions=(
            "你是规则修复专家。给定原始指令和一批有问题的规则，"
            "你需要补齐缺失字段、拆分过长描述，让每条规则通过质量校验。"
            "保持 rule_id / rule_type / severity 不变，只修复内容字段。"
        ),
        model="gpt-5.4-nano",
        output_type=ParsedRuleList,
    )

    result = Runner.run_sync(fixer_agent, fix_prompt)
    output = result.final_output

    if not isinstance(output, ParsedRuleList):
        return broken_rules

    fixed = [
        Rule(
            rule_id=r.rule_id,
            description=r.description,
            rule_type=r.rule_type,
            severity=r.severity,
            evaluation_hint=_build_evaluation_hint(r),
            trigger_condition=r.trigger_condition,
            expected_behavior=r.expected_behavior,
            failure_criteria=list(r.failure_criteria),
            evidence_requirement=r.evidence_requirement,
            checks=list(r.checks),
        )
        for r in output.rules
    ]
    return fixed
