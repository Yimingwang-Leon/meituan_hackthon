from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable
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
    rationale: str = ""
    matched_failure_criteria: list[str] = field(default_factory=list)
    suggestion: str = ""
    confidence: float = 1.0
    trigger_confidence: float | None = None     # 两步评测：触发判定的一致率
    compliance_confidence: float | None = None  # 两步评测：合规判定的一致率（未触发时为 None）
    votes: dict[str, int] = field(default_factory=dict)
    all_samples: list[dict[str, object]] = field(default_factory=list)
    judge_model: str = ""
    judge_prompt: str = ""
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
            if r.result == "fail" and r.suggestion:
                lines.append(f"      建议：{r.suggestion}")
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
    rationale: str = Field(
        default="",
        description=(
            "可审计判定依据，不是隐藏思考。结构化说明："
            "1. 相关对话事实和轮次；2. 对照的规则或失败标准；3. 结论。"
        ),
    )
    matched_failure_criteria: list[str] = Field(
        default_factory=list,
        description=(
            "若 triggered=true 且 compliant=false，列出命中的 failure_criteria 原文；"
            "其他情况填空列表 []。"
        ),
    )
    suggestion: str = Field(
        default="",
        description=(
            "仅在 triggered=true 且 compliant=false 时填写一句改进建议，不超过 50 字；"
            "pass、not_applicable 或 trigger_failed 时填空字符串。"
        ),
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
        "evidence 必须引用具体轮次，例如「第3轮数字人说...」。\n"
        "rationale 是给人复核的判定依据，不要写隐藏推理过程；必须引用对话事实、规则标准和结论。\n"
        "matched_failure_criteria 只在不合规时填写命中的失败标准原文。\n"
        "suggestion 只在不合规时填写，给出一句可执行改进建议。"
    ),
    model="deepseek-v4-pro",
    output_type=JudgeOutput,
)


# ───────────────────────────────────────────────────────────────────────────
# Conditional 两步评测：TriggerJudge + ComplianceJudge
# ───────────────────────────────────────────────────────────────────────────


class TriggerJudgeOutput(BaseModel):
    triggered: bool = Field(
        description=(
            "对话中是否出现了规则的触发条件。"
            "只看用户侧/对话推进，不评估 Agent 响应。"
            "标准：用户表达已足以让 Agent 合理进入该规则对应处理分支才算 true；"
            "纯背景铺垫、无关情绪、未形成行动含义的不算触发。"
        )
    )
    trigger_turn: int = Field(
        description=(
            "触发条件出现在第几轮（1-based）；triggered=false 时填 0。"
            "应指向「触发被识别出来的那一轮」，不要倒推到此前几轮的铺垫。"
        )
    )
    evidence: str = Field(
        description="必须引用该轮原话的关键片段，不超过 50 字。例：第3轮用户说「我现在不想买了」"
    )
    rationale: str = Field(
        default="",
        description=(
            "为什么这构成触发：对照规则 trigger_condition 的哪个关键要素，简要说明。"
            "triggered=false 时说明为什么不构成触发。"
        ),
    )


class ComplianceJudgeOutput(BaseModel):
    compliant: bool = Field(
        description="在已确认触发的前提下，Agent 在触发轮之后的响应是否符合 expected_behavior。"
    )
    response_turn: int = Field(
        description="Agent 给出关键响应在第几轮（1-based）；无明确响应填 0。"
    )
    evidence: str = Field(
        description="必须引用 Agent 响应轮原话片段，不超过 50 字。"
    )
    rationale: str = Field(
        default="",
        description=(
            "判定依据：引用对话事实 + 对照 expected_behavior / failure_criteria + 结论。"
            "不要写隐藏推理过程，要给人复核。"
        ),
    )
    matched_failure_criteria: list[str] = Field(
        default_factory=list,
        description=(
            "compliant=false 时列出命中的 failure_criteria 原文，必须从给定列表挑选；"
            "compliant=true 时填空列表。"
        ),
    )
    suggestion: str = Field(
        default="",
        description=(
            "compliant=false 时填一句可执行修复建议（例：「在第 X 轮加一句…」），不超过 50 字；"
            "compliant=true 时填空字符串。"
        ),
    )


_trigger_judge_agent = Agent(
    name="TriggerJudge",
    instructions=(
        "你是对话规则触发判定专家。你只判一件事：规则的「触发条件」是否在对话中出现过。\n\n"
        "判定原则：\n"
        "1. 只看用户的话和对话进展。Agent 后续做对做错与你无关。\n"
        "2. 触发条件描述的是「什么样的用户行为/对话状态构成了规则适用前提」，不是 Agent 该做什么。\n"
        "3. 判定门槛：用户表达已经足以让 Agent 合理进入该规则对应处理分支，才算 triggered=true。"
        "纯背景铺垫、无关情绪、未形成行动含义的不算触发（例如「我有点忙」不算「明确拒绝」的触发）。\n"
        "4. 触发条件出现 ≠ 表达明确。即使表达含糊，只要语义上已经构成行动含义，仍算 triggered=true。\n"
        "5. 严格引用原话作为 evidence，不要转述。\n"
        "6. trigger_turn 必须指向「触发被识别出来的那一轮」，不要倒推到此前几轮的铺垫。\n\n"
        "只根据提供的对话记录判断，不要推测对话之外的信息。"
    ),
    model="deepseek-v4-pro",
    output_type=TriggerJudgeOutput,
)


_compliance_judge_agent = Agent(
    name="ComplianceJudge",
    instructions=(
        "你是对话合规判定专家。你只判一件事：在「触发已经在第 X 轮发生」的前提下，"
        "Agent 此后的响应是否符合 expected_behavior。\n\n"
        "判定原则：\n"
        "1. 触发判定已经由上一步完成，你不要重新判定触发。\n"
        "2. 关注点在「trigger_turn 及其之后」的 Agent 发言，trigger_turn 之前的内容只作背景。\n"
        "3. 「符合 expected_behavior」要看实质动作，不是字面话术。"
        "例如规则要求「询问拒收原因」，Agent 说「方便问下是什么原因吗」算符合，说「那帮你取消」不算。\n"
        "4. compliant=false 时，matched_failure_criteria 必须从给定的 failure_criteria 列表里"
        "挑出实际命中的那条原文，不要自创新条目。\n"
        "5. suggestion 写一句可执行修复（例：「在第 X 轮加一句…」），不要泛泛而谈。\n\n"
        "只根据提供的对话记录判断，不要推测对话之外的信息。"
    ),
    model="deepseek-v4-pro",
    output_type=ComplianceJudgeOutput,
)


def _resolve_trigger_turn(turns: list[int]) -> int:
    """从 triggered=true 样本的 trigger_turn 里挑一个。

    优先严格多数（> 半数）；无多数取中位数（偶数样本取 low median）。
    传入空列表返回 0。
    """
    valid = [t for t in turns if t > 0]
    if not valid:
        return 0
    counter = Counter(valid)
    top_value, top_count = counter.most_common(1)[0]
    if top_count * 2 > len(valid):
        return top_value
    return statistics.median_low(valid)


def _build_trigger_prompt(rule: Rule, transcript_text: str) -> str:
    return (
        f"【对话记录】\n{transcript_text}\n\n"
        f"【你要判定的规则】\n"
        f"规则ID：{rule.rule_id}\n"
        f"规则类型：conditional\n"
        f"规则描述：{rule.description}\n"
        f"触发条件：{rule.trigger_condition or rule.description}\n"
        f"期望行为（仅作背景理解，本步不评估）：{rule.expected_behavior}\n\n"
        f"你只判定：上面的触发条件在对话里有没有出现。\n"
        f"- triggered=true 时给出 trigger_turn 和原话 evidence\n"
        f"- triggered=false 时 trigger_turn=0\n"
    )


def _build_compliance_prompt(
    rule: Rule,
    transcript_text: str,
    trigger_turn: int,
    trigger_evidence: str,
) -> str:
    failure_lines = (
        "\n".join(f"  - {c}" for c in rule.failure_criteria)
        if rule.failure_criteria else "  - （未列出）"
    )
    return (
        f"【对话记录】\n{transcript_text}\n\n"
        f"【触发已确认】\n"
        f"- 规则ID：{rule.rule_id}\n"
        f"- 规则描述：{rule.description}\n"
        f"- 触发条件：{rule.trigger_condition or rule.description}\n"
        f"- 触发出现在：第 {trigger_turn} 轮\n"
        f"- 触发证据：{trigger_evidence}\n\n"
        f"【你要判定的合规标准】\n"
        f"- 期望行为：{rule.expected_behavior}\n"
        f"- 失败标准（compliant=false 时必须命中至少一条）：\n{failure_lines}\n"
        f"- 证据要求：{rule.evidence_requirement or '引用具体 Agent 响应轮次原话'}\n\n"
        f"请只评估 Agent 在第 {trigger_turn} 轮之后的响应是否符合期望行为。"
    )


def _run_trigger_judge_once(prompt: str, rule_id: str) -> TriggerJudgeOutput:
    result = Runner.run_sync(_trigger_judge_agent, prompt)
    output = result.final_output
    if not isinstance(output, TriggerJudgeOutput):
        raise RuntimeError(f"规则 {rule_id} TriggerJudge 返回了意外结果类型")
    return output


def _run_compliance_judge_once(prompt: str, rule_id: str) -> ComplianceJudgeOutput:
    result = Runner.run_sync(_compliance_judge_agent, prompt)
    output = result.final_output
    if not isinstance(output, ComplianceJudgeOutput):
        raise RuntimeError(f"规则 {rule_id} ComplianceJudge 返回了意外结果类型")
    return output


def _build_trigger_sample(
    output: TriggerJudgeOutput,
    sample_index: int,
) -> dict[str, object]:
    return {
        "phase": "trigger",
        "sample_index": sample_index,
        "result": "triggered" if output.triggered else "not_triggered",
        "triggered": output.triggered,
        "trigger_turn": output.trigger_turn,
        "response_turn": 0,
        "compliant": None,
        "evidence": output.evidence,
        "rationale": output.rationale,
        "matched_failure_criteria": [],
        "suggestion": "",
    }


def _build_compliance_sample(
    output: ComplianceJudgeOutput,
    sample_index: int,
    final_trigger_turn: int,
) -> dict[str, object]:
    return {
        "phase": "compliance",
        "sample_index": sample_index,
        "result": "pass" if output.compliant else "fail",
        "triggered": True,
        "trigger_turn": final_trigger_turn,
        "response_turn": output.response_turn,
        "compliant": output.compliant,
        "evidence": output.evidence,
        "rationale": output.rationale,
        "matched_failure_criteria": (
            list(output.matched_failure_criteria) if not output.compliant else []
        ),
        "suggestion": output.suggestion.strip() if not output.compliant else "",
    }


def _normalize_trigger_output(output: TriggerJudgeOutput) -> TriggerJudgeOutput:
    """Treat impossible triggered=true/turn<=0 outputs as not triggered."""
    if output.triggered and output.trigger_turn <= 0:
        return TriggerJudgeOutput(
            triggered=False,
            trigger_turn=0,
            evidence=output.evidence,
            rationale=output.rationale,
        )
    return output


def _evaluate_conditional_two_step(
    rule: Rule,
    transcript_text: str,
    n_samples: int,
    is_primary: bool,
    trigger_outputs: list[TriggerJudgeOutput] | None = None,
    compliance_outputs_provider: (
        Callable[[int, str], list[ComplianceJudgeOutput]] | None
    ) = None,
) -> RuleResult:
    """两步评测 conditional 规则。

    trigger_outputs / compliance_outputs_provider 仅用于测试注入。
    生产路径下不会传，函数内部自行调用两个 judge agent。
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0")

    trigger_prompt = _build_trigger_prompt(rule, transcript_text)
    if trigger_outputs is None:
        trigger_outputs = [
            _run_trigger_judge_once(trigger_prompt, rule.rule_id)
            for _ in range(n_samples)
        ]
    trigger_outputs = [
        _normalize_trigger_output(output)
        for output in trigger_outputs
    ]
    sample_count = len(trigger_outputs)
    if sample_count == 0:
        raise ValueError("trigger_outputs must contain at least one sample")

    trigger_counter: Counter[bool] = Counter(o.triggered for o in trigger_outputs)
    triggered_final, top_trigger_votes = trigger_counter.most_common(1)[0]
    trigger_confidence = top_trigger_votes / sample_count
    trigger_samples = [
        _build_trigger_sample(output, idx + 1)
        for idx, output in enumerate(trigger_outputs)
    ]

    # 未触发：不进 Step 2
    if not triggered_final:
        result_label: RuleResultLiteral = (
            "trigger_failed" if is_primary else "not_applicable"
        )
        rep_trigger = next(o for o in trigger_outputs if not o.triggered)
        return RuleResult(
            rule_id=rule.rule_id,
            rule_type=rule.rule_type,
            description=rule.description,
            severity=rule.severity,
            result=result_label,
            evidence=rep_trigger.evidence,
            rationale=rep_trigger.rationale,
            matched_failure_criteria=[],
            suggestion="",
            confidence=trigger_confidence,
            trigger_confidence=trigger_confidence,
            compliance_confidence=None,
            votes={
                "triggered": trigger_counter.get(True, 0),
                "not_triggered": trigger_counter.get(False, 0),
            },
            all_samples=trigger_samples,
            judge_model=_trigger_judge_agent.model,
            judge_prompt=trigger_prompt,
            triggered=False,
            trigger_turn=None,
            response_turn=None,
            is_primary=is_primary,
            evaluated_by="llm_judge",
        )

    # 触发了：解析 trigger_turn 并跑 compliance 阶段
    triggered_samples = [o for o in trigger_outputs if o.triggered]
    final_trigger_turn = _resolve_trigger_turn(
        [o.trigger_turn for o in triggered_samples]
    )
    trigger_evidence_for_compliance = next(
        (
            o.evidence
            for o in triggered_samples
            if o.trigger_turn == final_trigger_turn
        ),
        triggered_samples[0].evidence,
    )

    compliance_prompt = _build_compliance_prompt(
        rule,
        transcript_text,
        final_trigger_turn,
        trigger_evidence_for_compliance,
    )
    if compliance_outputs_provider is not None:
        compliance_outputs = compliance_outputs_provider(
            final_trigger_turn, trigger_evidence_for_compliance
        )
    else:
        compliance_outputs = [
            _run_compliance_judge_once(compliance_prompt, rule.rule_id)
            for _ in range(n_samples)
        ]

    compliance_counter: Counter[bool] = Counter(o.compliant for o in compliance_outputs)
    compliant_final, top_compliance_votes = compliance_counter.most_common(1)[0]
    compliance_confidence = top_compliance_votes / len(compliance_outputs)

    compliance_samples = [
        _build_compliance_sample(output, idx + 1, final_trigger_turn)
        for idx, output in enumerate(compliance_outputs)
    ]

    rep_compliance = next(
        o for o in compliance_outputs if o.compliant == compliant_final
    )
    result_label = "pass" if compliant_final else "fail"
    overall_confidence = trigger_confidence * compliance_confidence

    return RuleResult(
        rule_id=rule.rule_id,
        rule_type=rule.rule_type,
        description=rule.description,
        severity=rule.severity,
        result=result_label,
        evidence=rep_compliance.evidence,
        rationale=rep_compliance.rationale,
        matched_failure_criteria=(
            list(rep_compliance.matched_failure_criteria)
            if result_label == "fail" else []
        ),
        suggestion=(
            rep_compliance.suggestion.strip() if result_label == "fail" else ""
        ),
        confidence=overall_confidence,
        trigger_confidence=trigger_confidence,
        compliance_confidence=compliance_confidence,
        votes={
            "pass": compliance_counter.get(True, 0),
            "fail": compliance_counter.get(False, 0),
        },
        all_samples=trigger_samples + compliance_samples,
        judge_model=_compliance_judge_agent.model,
        judge_prompt=trigger_prompt + "\n\n---\n\n" + compliance_prompt,
        triggered=True,
        trigger_turn=final_trigger_turn,
        response_turn=(
            rep_compliance.response_turn if rep_compliance.response_turn > 0 else None
        ),
        is_primary=is_primary,
        evaluated_by="llm_judge",
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
    judge_model: str = "",
    judge_prompt: str = "",
) -> RuleResult:
    classifications = [_classify_judge(rule, o, is_primary) for o in outputs]
    vote_counter: Counter[str] = Counter(classifications)
    winner, top_votes = vote_counter.most_common(1)[0]
    confidence = top_votes / len(outputs)
    all_samples = [
        _build_sample(rule, output, is_primary, index + 1)
        for index, output in enumerate(outputs)
    ]

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
        rationale=rep_output.rationale,
        matched_failure_criteria=(
            list(rep_output.matched_failure_criteria) if winner == "fail" else []
        ),
        suggestion=rep_output.suggestion.strip() if winner == "fail" else "",
        confidence=confidence,
        votes=dict(vote_counter),
        all_samples=all_samples,
        judge_model=judge_model or evaluated_by,
        judge_prompt=judge_prompt,
        triggered=triggered,
        trigger_turn=trigger_turn,
        response_turn=response_turn,
        is_primary=is_primary,
        evaluated_by=evaluated_by,
    )


def _build_sample(
    rule: Rule,
    output: JudgeOutput,
    is_primary: bool,
    sample_index: int,
) -> dict[str, object]:
    result = _classify_judge(rule, output, is_primary)
    return {
        "phase": "single",
        "sample_index": sample_index,
        "result": result,
        "triggered": output.triggered,
        "trigger_turn": output.trigger_turn,
        "response_turn": output.response_turn,
        "compliant": output.compliant,
        "evidence": output.evidence,
        "rationale": output.rationale,
        "matched_failure_criteria": (
            list(output.matched_failure_criteria) if result == "fail" else []
        ),
        "suggestion": output.suggestion.strip() if result == "fail" else "",
    }


def _outcome_to_judge_output(outcome: CheckOutcome) -> JudgeOutput:
    """把代码 checker 的输出转成 JudgeOutput，复用聚合管线。"""
    return JudgeOutput(
        triggered=outcome.triggered,
        trigger_turn=outcome.trigger_turn,
        response_turn=outcome.response_turn,
        compliant=outcome.compliant,
        evidence=outcome.evidence,
        rationale=outcome.evidence,
        matched_failure_criteria=[] if outcome.compliant else [outcome.evidence],
        suggestion="",
    )


def _build_deterministic_audit_prompt(rule: Rule) -> str:
    check_lines = "; ".join(
        f"{check.check_type.value}: {check.description}" for check in rule.checks
    )
    return f"deterministic checks for {rule.rule_id}: {check_lines}"


def evaluate_session(
    archive: SessionArchive,
    persona_type: str,
    rules: list[Rule] | None = None,
    n_samples: int = JUDGE_SAMPLES,
    max_workers: int = 16,  # 保留参数兼容，但已不使用
    set_id: str | None = None,
    set_label: str | None = None,
) -> EvaluationReport:
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0")

    transcript_text = _format_transcript(archive)
    active_rules = rules if rules is not None else RULES

    target_rule_id = archive.target_rule_id

    # 三路分流：
    #   - 有 checks 的规则走代码
    #   - conditional 规则走 trigger/compliance 两步 LLM
    #   - required / forbidden 规则走单步 LLM
    deterministic_rules = [r for r in active_rules if r.checks]
    conditional_llm_rules = [
        r for r in active_rules if not r.checks and r.rule_type == "conditional"
    ]
    single_step_llm_rules = [
        r for r in active_rules if not r.checks and r.rule_type != "conditional"
    ]

    single_step_prompts = {
        rule.rule_id: _build_rule_prompt(rule, transcript_text)
        for rule in single_step_llm_rules
    }

    # 进度估算：conditional 至少 N 次 trigger 调用；触发了再加 N 次 compliance（上限 2N）
    min_llm_calls = (
        len(single_step_llm_rules) * n_samples
        + len(conditional_llm_rules) * n_samples
    )
    max_llm_calls = min_llm_calls + len(conditional_llm_rules) * n_samples
    print(
        f"    评测 {len(active_rules)} 条规则："
        f"代码 {len(deterministic_rules)} 条 + "
        f"LLM 单步 {len(single_step_llm_rules)} 条 + "
        f"LLM 两步 {len(conditional_llm_rules)} 条 × {n_samples} 采样"
        f"（预计 {min_llm_calls}-{max_llm_calls} 次 LLM 调用）",
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
                _aggregate_votes(
                    rule,
                    outputs,
                    is_primary,
                    evaluated_by="deterministic",
                    judge_model="deterministic",
                    judge_prompt=_build_deterministic_audit_prompt(rule),
                )
            )
        elif rule.rule_type == "conditional":
            # 两步 LLM 评测：先 trigger，触发则 compliance
            result = _evaluate_conditional_two_step(
                rule,
                transcript_text,
                n_samples=n_samples,
                is_primary=is_primary,
            )
            rule_results.append(result)
            phase_count = len(
                {
                    sample.get("phase")
                    for sample in result.all_samples
                    if isinstance(sample, dict)
                }
            )
            llm_done += n_samples * max(1, phase_count)
            print(
                f"    LLM judge 进度 {llm_done}/{max_llm_calls}（两步 · {rule.rule_id}）",
                flush=True,
            )
        else:
            # 单步 LLM judge × N 采样（required / forbidden）
            prompt = single_step_prompts[rule.rule_id]
            outputs = [
                _run_judge_once(prompt, rule.rule_id)
                for _ in range(n_samples)
            ]
            rule_results.append(
                _aggregate_votes(
                    rule,
                    outputs,
                    is_primary,
                    evaluated_by="llm_judge",
                    judge_model=_judge_agent.model,
                    judge_prompt=prompt,
                )
            )
            llm_done += n_samples
            print(
                f"    LLM judge 进度 {llm_done}/{max_llm_calls}（单步 · {rule.rule_id}）",
                flush=True,
            )

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
