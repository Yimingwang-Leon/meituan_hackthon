from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from agents import Agent, Runner
from pydantic import BaseModel, Field


@dataclass
class SoftJudgeResult:
    score: float
    task_completion: float
    naturalness: float
    context_consistency: float
    business_tone: float
    explanation_quality: float
    emotion_handling: float
    reason: str
    uncertainty: float = 0.0
    model_name: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _SoftJudgeOutput(BaseModel):
    task_completion: float = Field(ge=0, le=1)
    naturalness: float = Field(ge=0, le=1)
    context_consistency: float = Field(ge=0, le=1)
    business_tone: float = Field(ge=0, le=1)
    explanation_quality: float = Field(ge=0, le=1)
    emotion_handling: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)
    reason: str
    uncertainty: float = Field(default=0.0, ge=0, le=1)


_judge_agent = Agent(
    name="HarnessSoftJudge",
    instructions=(
        "你是任务型外呼 Agent 的软指标评估专家。你只评估沟通质量和任务完成度，"
        "不要覆盖硬规则检查器的结论。\n"
        "评分维度：任务完成度、自然度、上下文一致性、业务语气、原因解释、用户情绪处理。\n"
        "请给 0-1 分，并用一两句话说明主要依据。只输出 JSON。"
    ),
    model="deepseek-v4-pro",
    output_type=_SoftJudgeOutput,
    request_timeout_seconds=float(os.getenv("HARNESS_JUDGE_TIMEOUT_SECONDS", "60")),
    max_retries=int(os.getenv("HARNESS_JUDGE_MAX_RETRIES", "1")),
    parse_max_retries=1,
)


class LLMJudge:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def evaluate(
        self,
        case: dict[str, Any],
        turns: list[dict[str, Any]],
        rule_check: dict[str, Any],
        state: dict[str, Any],
    ) -> SoftJudgeResult:
        if self.enabled and _has_api_key():
            try:
                return self._evaluate_with_llm(case, turns, rule_check, state)
            except Exception as exc:
                fallback = self._heuristic(case, turns, rule_check, state)
                fallback.reason = f"LLM Judge 失败，使用启发式评分：{type(exc).__name__}: {exc}"
                fallback.uncertainty = max(fallback.uncertainty, 0.5)
                return fallback
        return self._heuristic(case, turns, rule_check, state)

    def _evaluate_with_llm(
        self,
        case: dict[str, Any],
        turns: list[dict[str, Any]],
        rule_check: dict[str, Any],
        state: dict[str, Any],
    ) -> SoftJudgeResult:
        prompt = _build_prompt(case, turns, rule_check, state)
        result = Runner.run_sync(_judge_agent, prompt)
        output = result.final_output
        if not isinstance(output, _SoftJudgeOutput):
            raise RuntimeError("HarnessSoftJudge returned unexpected output")
        return SoftJudgeResult(
            score=output.overall_score,
            task_completion=output.task_completion,
            naturalness=output.naturalness,
            context_consistency=output.context_consistency,
            business_tone=output.business_tone,
            explanation_quality=output.explanation_quality,
            emotion_handling=output.emotion_handling,
            reason=output.reason,
            uncertainty=output.uncertainty,
            model_name=_judge_agent.model,
        )

    @staticmethod
    def _heuristic(
        case: dict[str, Any],
        turns: list[dict[str, Any]],
        rule_check: dict[str, Any],
        state: dict[str, Any],
    ) -> SoftJudgeResult:
        violations = rule_check.get("violations", [])
        high = sum(1 for v in violations if v.get("severity") == "high")
        medium = sum(1 for v in violations if v.get("severity") == "medium")
        agent_turns = [t for t in turns if t.get("role") == "agent"]
        ended = bool(agent_turns and any(token in agent_turns[-1].get("content", "") for token in ("再见", "感谢", "结束", "处理")))
        task_completion = 0.85 if state.get("task_completed") else 0.55
        if ended:
            task_completion = max(task_completion, 0.7)
        penalty = min(0.7, high * 0.35 + medium * 0.15)
        score = max(0.0, min(1.0, task_completion - penalty))
        return SoftJudgeResult(
            score=score,
            task_completion=task_completion,
            naturalness=0.75,
            context_consistency=0.7 if not violations else 0.55,
            business_tone=0.75,
            explanation_quality=0.65,
            emotion_handling=0.65,
            reason=(
                "启发式软评分：根据任务完成状态、结束方式和硬规则违规数量估算；"
                f"high={high}, medium={medium}。"
            ),
            uncertainty=0.35,
            model_name="heuristic",
        )


def _has_api_key() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _build_prompt(
    case: dict[str, Any],
    turns: list[dict[str, Any]],
    rule_check: dict[str, Any],
    state: dict[str, Any],
) -> str:
    transcript = "\n".join(
        f"{turn.get('turn_id')}. {turn.get('role')}: {turn.get('content')}"
        for turn in turns
    )
    return (
        f"【Case】\n{case}\n\n"
        f"【Transcript】\n{transcript}\n\n"
        f"【Hard Rule Check】\n{rule_check}\n\n"
        f"【Dialogue State】\n{state}\n\n"
        "请只评估软指标，不要因为硬规则直接一票否决；硬规则会由 harness 聚合。"
    )
