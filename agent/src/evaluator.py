from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from agents import Agent, Runner
from pydantic import BaseModel, Field

from .rules import Rule, RULES, SEVERITY_WEIGHTS
from .types import SessionArchive

JUDGE_SAMPLES = 3


@dataclass
class RuleResult:
    rule_id: str
    rule_type: str
    description: str
    severity: str
    result: Literal["pass", "fail", "not_applicable"]
    evidence: str
    confidence: float = 1.0
    votes: dict[str, int] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    order_id: str
    persona_type: str
    rule_results: list[RuleResult]
    score: float

    @property
    def mean_confidence(self) -> float:
        if not self.rule_results:
            return 0.0
        return sum(r.confidence for r in self.rule_results) / len(self.rule_results)


@dataclass
class CoverageReport:
    total_conditional: int
    triggered_conditional: int
    untriggered_rules: list[Rule]
    triggered_by_persona: dict[str, set[str]]

    @property
    def coverage_rate(self) -> float:
        if self.total_conditional == 0:
            return 1.0
        return self.triggered_conditional / self.total_conditional


def compute_coverage(
    reports: list[EvaluationReport],
    rules: list[Rule],
) -> CoverageReport:
    conditional_rules = [r for r in rules if r.rule_type == "conditional"]
    triggered_ids: set[str] = set()
    triggered_by_persona: dict[str, set[str]] = {}

    for report in reports:
        for rr in report.rule_results:
            if rr.rule_type == "conditional" and rr.result != "not_applicable":
                triggered_ids.add(rr.rule_id)
                triggered_by_persona.setdefault(report.persona_type, set()).add(rr.rule_id)

    untriggered = [r for r in conditional_rules if r.rule_id not in triggered_ids]

    return CoverageReport(
        total_conditional=len(conditional_rules),
        triggered_conditional=len(triggered_ids),
        untriggered_rules=untriggered,
        triggered_by_persona=triggered_by_persona,
    )

    def summary(self) -> str:
        lines = [
            f"订单：{self.order_id}  Persona：{self.persona_type}  "
            f"得分：{self.score:.0%}  平均置信度：{self.mean_confidence:.0%}",
            "",
        ]
        for r in self.rule_results:
            icon = {"pass": "✓", "fail": "✗", "not_applicable": "-"}[r.result]
            lines.append(
                f"  {icon} [{r.rule_id}][{r.severity}] {r.description}  "
                f"(置信度 {r.confidence:.0%})"
            )
            if r.result != "not_applicable":
                lines.append(f"      → {r.evidence}")
        return "\n".join(lines)


class JudgeOutput(BaseModel):
    result: Literal["pass", "fail", "not_applicable"] = Field(
        description="pass=满足规则，fail=违反规则，not_applicable=条件规则的触发场景未出现"
    )
    evidence: str = Field(description="引用具体对话轮次作为判断依据，不超过50字")


_judge_agent = Agent(
    name="RuleJudge",
    instructions=(
        "你是一名对话质量评估专家，负责判断美团外呼数字人是否遵守了给定规则。"
        "只根据提供的对话记录进行判断，不要推测对话之外的信息。"
        "evidence 必须引用具体轮次，例如'第3轮数字人说...'。"
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
    return (
        f"【对话记录】\n{transcript_text}\n\n"
        f"【评估规则】\n"
        f"规则ID：{rule.rule_id}\n"
        f"规则类型：{rule.rule_type}\n"
        f"规则内容：{rule.description}\n"
        f"评估提示：{rule.evaluation_hint}"
    )


def _aggregate_votes(rule: Rule, outputs: list[JudgeOutput]) -> RuleResult:
    vote_counter: Counter[str] = Counter(o.result for o in outputs)
    winner, top_votes = vote_counter.most_common(1)[0]
    confidence = top_votes / len(outputs)
    winner_outputs = [o for o in outputs if o.result == winner]
    return RuleResult(
        rule_id=rule.rule_id,
        rule_type=rule.rule_type,
        description=rule.description,
        severity=rule.severity,
        result=winner,
        evidence=winner_outputs[0].evidence,
        confidence=confidence,
        votes=dict(vote_counter),
    )


def evaluate_session(
    archive: SessionArchive,
    persona_type: str,
    rules: list[Rule] | None = None,
    n_samples: int = JUDGE_SAMPLES,
    max_workers: int = 16,  # 保留参数兼容，但已不使用
) -> EvaluationReport:
    transcript_text = _format_transcript(archive)
    active_rules = rules if rules is not None else RULES

    prompts = {rule.rule_id: _build_rule_prompt(rule, transcript_text) for rule in active_rules}

    total_calls = len(active_rules) * n_samples
    print(f"    顺序评测 {total_calls} 个 judge 任务", flush=True)

    rule_results: list[RuleResult] = []
    for rule_idx, rule in enumerate(active_rules):
        outputs = [
            _run_judge_once(prompts[rule.rule_id], rule.rule_id)
            for _ in range(n_samples)
        ]
        rule_results.append(_aggregate_votes(rule, outputs))
        print(f"    judge 进度 {(rule_idx + 1) * n_samples}/{total_calls}", flush=True)

    scored = [r for r in rule_results if r.result != "not_applicable"]
    total_weight = sum(SEVERITY_WEIGHTS.get(r.severity, 2) for r in scored)
    passed_weight = sum(
        SEVERITY_WEIGHTS.get(r.severity, 2)
        for r in scored
        if r.result == "pass"
    )
    score = passed_weight / total_weight if total_weight > 0 else 0.0

    return EvaluationReport(
        order_id=archive.order_id,
        persona_type=persona_type,
        rule_results=rule_results,
        score=score,
    )
