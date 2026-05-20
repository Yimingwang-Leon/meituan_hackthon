from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from agents import Agent, Runner
from pydantic import BaseModel, Field

from .deterministic_checks import CheckOutcome, run_checks_combined
from .rules import Rule, RULES, SEVERITY_WEIGHTS
from .types import SessionArchive

JUDGE_SAMPLES = 3

RuleResultLiteral = Literal["pass", "fail", "not_applicable", "trigger_failed"]


@dataclass
class RuleResult:
    rule_id: str
    rule_type: str
    description: str
    severity: str
    result: RuleResultLiteral
    evidence: str
    confidence: float = 1.0
    votes: dict[str, int] = field(default_factory=dict)
    triggered: bool | None = None       # conditional 规则才填
    trigger_turn: int | None = None     # 触发发生在第几轮（1-based）
    response_turn: int | None = None    # Agent 响应在第几轮
    is_primary: bool = False             # 此 session 是否本来就为这条规则设计
    evaluated_by: Literal["llm_judge", "deterministic"] = "llm_judge"


@dataclass
class EvaluationReport:
    session_id: str
    persona_type: str
    rule_results: list[RuleResult]
    score: float
    set_id: str | None = None
    set_label: str | None = None

    @property
    def mean_confidence(self) -> float:
        if not self.rule_results:
            return 0.0
        return sum(r.confidence for r in self.rule_results) / len(self.rule_results)

    def summary(self) -> str:
        set_tag = f"  Set：{self.set_id}" if self.set_id else ""
        lines = [
            f"会话：{self.session_id}  Persona：{self.persona_type}{set_tag}  "
            f"得分：{self.score:.0%}  平均置信度：{self.mean_confidence:.0%}",
            "",
        ]
        for r in self.rule_results:
            icon = {
                "pass": "✓",
                "fail": "✗",
                "not_applicable": "-",
                "trigger_failed": "⚠",
            }[r.result]
            lines.append(
                f"  {icon} [{r.rule_id}][{r.severity}] {r.description}  "
                f"(置信度 {r.confidence:.0%})"
            )
            if r.result not in ("not_applicable",):
                lines.append(f"      → {r.evidence}")
        return "\n".join(lines)


@dataclass
class CoverageReport:
    total_conditional: int
    triggered_conditional: int
    trigger_failed_count: int          # 本该触发但没触发的总次数
    primary_attempted: int              # 总共尝试触发了多少次 conditional 规则（是 primary 的）
    untriggered_rules: list[Rule]
    triggered_by_persona: dict[str, set[str]]

    @property
    def coverage_rate(self) -> float:
        if self.total_conditional == 0:
            return 1.0
        return self.triggered_conditional / self.total_conditional

    @property
    def trigger_failure_rate(self) -> float:
        """trigger_failed / primary_attempted 反映 simulator 在被指派的 case 上的失败率。"""
        if self.primary_attempted == 0:
            return 0.0
        return self.trigger_failed_count / self.primary_attempted


def compute_coverage(
    reports: list[EvaluationReport],
    rules: list[Rule],
) -> CoverageReport:
    conditional_rules = [r for r in rules if r.rule_type == "conditional"]
    triggered_ids: set[str] = set()
    triggered_by_persona: dict[str, set[str]] = {}
    trigger_failed_count = 0
    primary_attempted = 0

    for report in reports:
        for rr in report.rule_results:
            if rr.rule_type != "conditional":
                continue
            if rr.is_primary:
                primary_attempted += 1
                if rr.result == "trigger_failed":
                    trigger_failed_count += 1
            if rr.result in ("pass", "fail"):
                triggered_ids.add(rr.rule_id)
                triggered_by_persona.setdefault(report.persona_type, set()).add(rr.rule_id)

    untriggered = [r for r in conditional_rules if r.rule_id not in triggered_ids]

    return CoverageReport(
        total_conditional=len(conditional_rules),
        triggered_conditional=len(triggered_ids),
        trigger_failed_count=trigger_failed_count,
        primary_attempted=primary_attempted,
        untriggered_rules=untriggered,
        triggered_by_persona=triggered_by_persona,
    )


class JudgeOutput(BaseModel):
    triggered: bool = Field(
        description=(
            "对话中触发条件是否出现。"
            "对 conditional 规则：用户行为是否触发了该规则的前提条件。"
            "对 required 规则：相关流程阶段是否在对话里出现过。"
            "对 forbidden 规则：固定填 true（约束全程都该满足）。"
        )
    )
    trigger_turn: int = Field(
        description=(
            "触发条件出现在第几轮（1-based），不存在时填 0。"
        )
    )
    response_turn: int = Field(
        description=(
            "Agent 给出关键响应的轮次（1-based），不存在时填 0。"
        )
    )
    compliant: bool = Field(
        description=(
            "在触发的前提下，Agent 是否做到了规则要求。"
            "如果未触发则可任意填，evaluator 会忽略此字段。"
        )
    )
    evidence: str = Field(
        description="引用具体对话轮次说明判断依据，不超过 60 字。"
    )


_judge_agent = Agent(
    name="RuleJudge",
    instructions=(
        "你是一名对话质量评估专家，负责判断被测任务型对话 Agent 是否遵守了给定规则。\n"
        "评测分两步：\n"
        "1) 触发判定：仅根据对话内容，判断该规则的「触发条件」是否在对话中出现。\n"
        "   - 对 conditional 规则：用户或对话进展是否满足该规则的触发条件\n"
        "   - 对 required 规则：触发条件就是「对话已展开，应该体现该要求」，几乎总为 true\n"
        "   - 对 forbidden 规则：触发条件固定填 true（约束适用于全程）\n"
        "   - triggered=true 时给出 trigger_turn；false 时填 0\n"
        "2) 合规判定（在 triggered=true 前提下）：Agent 是否做到了规则要求。\n"
        "   - required：通观对话，相关要求是否被满足\n"
        "   - conditional：trigger_turn 之后 Agent 的响应是否符合 expected_behavior\n"
        "   - forbidden：是否整段对话都没有违反禁止行为\n"
        "   - 给出 response_turn；如无明确响应轮次填 0\n"
        "只根据提供的对话记录判断，不要推测对话之外的信息。\n"
        "evidence 必须引用具体轮次，例如「第3轮数字人说...」。"
    ),
    model="gpt-5.4-nano",
    output_type=JudgeOutput,
)


def _format_transcript(archive: SessionArchive) -> str:
    lines = []
    for i, entry in enumerate(archive.transcript, start=1):
        speaker = "数字人" if entry.speaker == "agent" else "用户"
        lines.append(f"第{i}轮 {speaker}：{entry.text}")
    return "\n".join(lines)


def _run_judge_once(prompt: str, rule_id: str) -> JudgeOutput:
    result = Runner.run_sync(_judge_agent, prompt)
    output = result.final_output
    if not isinstance(output, JudgeOutput):
        raise RuntimeError(f"规则 {rule_id} 评估返回了意外结果类型")
    return output


def _build_rule_prompt(rule: Rule, transcript_text: str) -> str:
    failure_lines = (
        "\n".join(f"  - {c}" for c in rule.failure_criteria)
        if rule.failure_criteria else "  - （未列出）"
    )
    trigger_line = (
        f"触发条件：{rule.trigger_condition}"
        if rule.trigger_condition else "触发条件：（required/forbidden 全程适用）"
    )
    return (
        f"【对话记录】\n{transcript_text}\n\n"
        f"【评估规则】\n"
        f"规则ID：{rule.rule_id}\n"
        f"规则类型：{rule.rule_type}\n"
        f"严重度：{rule.severity}\n"
        f"规则内容：{rule.description}\n"
        f"{trigger_line}\n"
        f"期望行为：{rule.expected_behavior}\n"
        f"失败标准：\n{failure_lines}\n"
        f"证据要求：{rule.evidence_requirement or '引用具体轮次'}"
    )


def _classify_judge(
    rule: Rule,
    output: JudgeOutput,
    is_primary: bool,
) -> RuleResultLiteral:
    """根据 LLM judge 输出 + is_primary，决定最终的 result 标签。"""
    if rule.rule_type == "conditional":
        if not output.triggered:
            return "trigger_failed" if is_primary else "not_applicable"
        return "pass" if output.compliant else "fail"

    # required / forbidden：不区分 trigger_failed，整段对话都该满足/不该违反
    if not output.triggered:
        # 极少见：required 规则但 judge 判断流程根本没展开
        return "not_applicable"
    return "pass" if output.compliant else "fail"


def _aggregate_votes(
    rule: Rule,
    outputs: list[JudgeOutput],
    is_primary: bool,
    evaluated_by: Literal["llm_judge", "deterministic"] = "llm_judge",
) -> RuleResult:
    classifications = [_classify_judge(rule, o, is_primary) for o in outputs]
    vote_counter: Counter[str] = Counter(classifications)
    winner, top_votes = vote_counter.most_common(1)[0]
    confidence = top_votes / len(outputs)

    # 选第一个 winner 类别的 output 作为代表
    winner_idx = next(i for i, c in enumerate(classifications) if c == winner)
    rep_output = outputs[winner_idx]

    # triggered / trigger_turn / response_turn 也取代表 output 的值
    triggered: bool | None
    trigger_turn: int | None
    response_turn: int | None
    if rule.rule_type == "conditional":
        triggered = rep_output.triggered
        trigger_turn = rep_output.trigger_turn if rep_output.trigger_turn > 0 else None
        response_turn = rep_output.response_turn if rep_output.response_turn > 0 else None
    else:
        triggered = None
        trigger_turn = None
        response_turn = None

    return RuleResult(
        rule_id=rule.rule_id,
        rule_type=rule.rule_type,
        description=rule.description,
        severity=rule.severity,
        result=winner,
        evidence=rep_output.evidence,
        confidence=confidence,
        votes=dict(vote_counter),
        triggered=triggered,
        trigger_turn=trigger_turn,
        response_turn=response_turn,
        is_primary=is_primary,
        evaluated_by=evaluated_by,
    )


def _outcome_to_judge_output(outcome: CheckOutcome) -> JudgeOutput:
    """把代码 checker 的输出转成 JudgeOutput，复用聚合管线。"""
    return JudgeOutput(
        triggered=outcome.triggered,
        trigger_turn=outcome.trigger_turn,
        response_turn=outcome.response_turn,
        compliant=outcome.compliant,
        evidence=outcome.evidence,
    )


def evaluate_session(
    archive: SessionArchive,
    persona_type: str,
    rules: list[Rule] | None = None,
    n_samples: int = JUDGE_SAMPLES,
    max_workers: int = 16,  # 保留参数兼容，但已不使用
    set_id: str | None = None,
    set_label: str | None = None,
) -> EvaluationReport:
    transcript_text = _format_transcript(archive)
    active_rules = rules if rules is not None else RULES

    target_rule_id = archive.target_rule_id

    # 分流：有 checks 的规则走代码，没有的走 LLM
    deterministic_rules = [r for r in active_rules if r.checks]
    llm_rules = [r for r in active_rules if not r.checks]
    prompts = {rule.rule_id: _build_rule_prompt(rule, transcript_text) for rule in llm_rules}

    total_llm_calls = len(llm_rules) * n_samples
    print(
        f"    评测 {len(active_rules)} 条规则："
        f"代码 {len(deterministic_rules)} 条 + LLM {len(llm_rules)} 条 × {n_samples} 采样",
        flush=True,
    )

    rule_results: list[RuleResult] = []
    llm_done = 0
    for rule in active_rules:
        is_primary = (target_rule_id is not None and rule.rule_id == target_rule_id)

        if rule.checks:
            # 代码 checker：1 次确定性运行
            outcome = run_checks_combined(rule.checks, archive)
            outputs = [_outcome_to_judge_output(outcome)]
            rule_results.append(
                _aggregate_votes(rule, outputs, is_primary, evaluated_by="deterministic")
            )
        else:
            # LLM judge × N 采样
            outputs = [
                _run_judge_once(prompts[rule.rule_id], rule.rule_id)
                for _ in range(n_samples)
            ]
            rule_results.append(
                _aggregate_votes(rule, outputs, is_primary, evaluated_by="llm_judge")
            )
            llm_done += n_samples
            print(f"    LLM judge 进度 {llm_done}/{total_llm_calls}", flush=True)

    # 打分：pass / fail 进分母，其他不进
    scored = [r for r in rule_results if r.result in ("pass", "fail")]
    total_weight = sum(SEVERITY_WEIGHTS.get(r.severity, 2) for r in scored)
    passed_weight = sum(
        SEVERITY_WEIGHTS.get(r.severity, 2)
        for r in scored
        if r.result == "pass"
    )
    score = passed_weight / total_weight if total_weight > 0 else 0.0

    return EvaluationReport(
        session_id=archive.session_id,
        persona_type=persona_type,
        rule_results=rule_results,
        score=score,
        set_id=set_id,
        set_label=set_label,
    )
