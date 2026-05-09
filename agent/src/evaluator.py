from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agents import Agent, Runner
from pydantic import BaseModel, Field

from .rules import Rule, RULES
from .types import SessionArchive


@dataclass
class RuleResult:
    rule_id: str
    rule_type: str
    description: str
    result: Literal["pass", "fail", "not_applicable"]
    evidence: str


@dataclass
class EvaluationReport:
    order_id: str
    persona_type: str
    rule_results: list[RuleResult]
    score: float

    def summary(self) -> str:
        lines = [
            f"订单：{self.order_id}  Persona：{self.persona_type}  得分：{self.score:.0%}",
            "",
        ]
        for r in self.rule_results:
            icon = {"pass": "✓", "fail": "✗", "not_applicable": "-"}[r.result]
            lines.append(f"  {icon} [{r.rule_id}] {r.description}")
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


def _evaluate_rule(rule: Rule, transcript_text: str) -> RuleResult:
    prompt = (
        f"【对话记录】\n{transcript_text}\n\n"
        f"【评估规则】\n"
        f"规则ID：{rule.rule_id}\n"
        f"规则类型：{rule.rule_type}\n"
        f"规则内容：{rule.description}\n"
        f"评估提示：{rule.evaluation_hint}"
    )

    result = Runner.run_sync(_judge_agent, prompt)
    output = result.final_output

    if not isinstance(output, JudgeOutput):
        raise RuntimeError(f"规则 {rule.rule_id} 评估返回了意外结果类型")

    return RuleResult(
        rule_id=rule.rule_id,
        rule_type=rule.rule_type,
        description=rule.description,
        result=output.result,
        evidence=output.evidence,
    )


def evaluate_session(archive: SessionArchive, persona_type: str) -> EvaluationReport:
    transcript_text = _format_transcript(archive)
    rule_results: list[RuleResult] = []

    for rule in RULES:
        rule_result = _evaluate_rule(rule, transcript_text)
        rule_results.append(rule_result)

    scored = [r for r in rule_results if r.result != "not_applicable"]
    passed = sum(1 for r in scored if r.result == "pass")
    score = passed / len(scored) if scored else 0.0

    return EvaluationReport(
        order_id=archive.order_id,
        persona_type=persona_type,
        rule_results=rule_results,
        score=score,
    )
