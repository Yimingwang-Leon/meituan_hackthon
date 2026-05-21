from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agents.base import AgentStepResult, BaseAgentAdapter
from agents.prompt_agent import PromptAgentAdapter
from evaluators.llm_judge import LLMJudge
from evaluators.rule_checker import run_rule_checks
from harness.recorder import TrajectoryRecorder
from harness.report import (
    aggregate_final_result,
    classify_failure_types,
    write_summary,
)
from harness.state import DialogueState, update_state_after_turn
from tools.registry import ToolRuntime, extract_tool_calls


DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass
class HarnessRunResult:
    case_id: str
    final_result: dict[str, Any]
    judge_result: dict[str, Any]
    rule_checker_result: dict[str, Any]
    failure_types: list[str]
    trajectory_path: str


class ScriptedMockAgentAdapter(BaseAgentAdapter):
    """No-LLM adapter for local harness smoke tests."""

    agent_version = "scripted_mock_agent"
    model_name = "none"
    prompt_version = "scripted"

    def reset(self, case: dict[str, Any]) -> None:
        self._responses = list(case.get("agent_responses") or [])
        self._idx = 0

    def step(self, user_message: str | None, state: DialogueState) -> AgentStepResult:
        if self._idx < len(self._responses):
            text = str(self._responses[self._idx])
        elif user_message and any(token in user_message for token in ("不要", "不用", "取消", "拒绝")):
            text = "好的，尊重您的决定，这边先不继续打扰。"
        else:
            text = "您好，我是美团客服，想和您确认一下当前订单情况。"
        self._idx += 1
        return AgentStepResult(reply_text=text, should_end=self._idx >= len(self._responses))


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    if isinstance(raw, dict) and isinstance(raw.get("cases"), list):
        return [dict(item) for item in raw["cases"]]
    raise ValueError(f"{path} must be a JSON list or an object with a cases list")


def build_agent_adapter(name: str, case: dict[str, Any]) -> BaseAgentAdapter:
    if name == "baseline_prompt_agent":
        return PromptAgentAdapter(
            instruction=case.get("instruction"),
            agent_version=str(case.get("agent_version") or "baseline_prompt_agent"),
            prompt_version=str(case.get("prompt_version") or "case_instruction"),
        )
    if name == "scripted_mock_agent":
        return ScriptedMockAgentAdapter()
    raise ValueError(f"unknown agent adapter: {name}")


class HarnessRunner:
    def __init__(
        self,
        agent_name: str,
        max_turns: int = 8,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        llm_judge_enabled: bool = False,
        tool_permission: str = "read",
        judge_threshold: float = 0.7,
    ) -> None:
        self.agent_name = agent_name
        self.max_turns = max_turns
        self.output_dir = output_dir
        self.llm_judge = LLMJudge(enabled=llm_judge_enabled)
        self.tool_permission = tool_permission
        self.judge_threshold = judge_threshold

    def run_cases(self, cases: list[dict[str, Any]]) -> list[HarnessRunResult]:
        return [self.run_case(case) for case in cases]

    def run_case(self, case: dict[str, Any]) -> HarnessRunResult:
        case_id = str(case.get("case_id") or case.get("id") or "case")
        max_turns = int(case.get("max_turns") or self.max_turns)
        adapter = build_agent_adapter(self.agent_name, case)
        adapter.reset(case)
        state = DialogueState(case_id=case_id)
        tool_runtime = ToolRuntime(permission_level=self.tool_permission)  # type: ignore[arg-type]
        recorder = TrajectoryRecorder(
            case_id=case_id,
            agent_version=adapter.agent_version,
            model_name=adapter.model_name,
            prompt_version=adapter.prompt_version,
            output_dir=self.output_dir / "trajectories",
        )

        turns: list[dict[str, Any]] = []
        tool_records: list[dict[str, Any]] = []
        last_agent_message: str | None = None
        turn_id = 1

        opening = adapter.step(None, state)
        self._record_agent_turn(
            recorder,
            turns,
            tool_records,
            state,
            tool_runtime,
            turn_id,
            opening,
            user_message=None,
            previous_agent_message=None,
            max_turns=max_turns,
        )
        last_agent_message = opening.reply_text
        turn_id += 1

        for user_message in list(case.get("user_messages") or []):
            if state.max_turn_exceeded or opening.should_end:
                break

            user_text = str(user_message)
            turns.append({"turn_id": turn_id, "role": "user", "content": user_text})
            recorder.add_turn(
                turn_id=turn_id,
                role="user",
                content=user_text,
                state=state.to_dict(),
                state_diff={},
            )
            turn_id += 1

            agent_result = adapter.step(user_text, state)
            self._record_agent_turn(
                recorder,
                turns,
                tool_records,
                state,
                tool_runtime,
                turn_id,
                agent_result,
                user_message=user_text,
                previous_agent_message=last_agent_message,
                max_turns=max_turns,
            )
            last_agent_message = agent_result.reply_text
            opening = agent_result
            turn_id += 1
            if agent_result.should_end:
                break

        if turn_id // 2 >= max_turns and not state.task_completed:
            state.max_turn_exceeded = True

        rule_check = run_rule_checks(
            state=state,
            turns=turns,
            tool_calls=tool_records,
            max_turns=max_turns,
        )
        rule_check_dict = rule_check.to_dict()
        state.mark_violations([v.to_dict() for v in rule_check.violations])

        judge = self.llm_judge.evaluate(
            case=case,
            turns=turns,
            rule_check=rule_check_dict,
            state=state.to_dict(),
        ).to_dict()
        final_result = aggregate_final_result(
            rule_check=rule_check_dict,
            judge_result=judge,
            judge_threshold=self.judge_threshold,
        )
        failure_types = classify_failure_types(
            rule_check=rule_check_dict,
            judge_result=judge,
            final_passed=bool(final_result["passed"]),
        )
        final_result["failure_types"] = failure_types

        trajectory_path = recorder.finish(
            rule_checker_result=rule_check_dict,
            judge_result=judge,
            final_result=final_result,
            failure_types=failure_types,
        )
        return HarnessRunResult(
            case_id=case_id,
            final_result=final_result,
            judge_result=judge,
            rule_checker_result=rule_check_dict,
            failure_types=failure_types,
            trajectory_path=str(trajectory_path),
        )

    def _record_agent_turn(
        self,
        recorder: TrajectoryRecorder,
        turns: list[dict[str, Any]],
        tool_records: list[dict[str, Any]],
        state: DialogueState,
        tool_runtime: ToolRuntime,
        turn_id: int,
        agent_result: AgentStepResult,
        user_message: str | None,
        previous_agent_message: str | None,
        max_turns: int,
    ) -> None:
        detected_tool_calls = [
            *agent_result.tool_calls,
            *extract_tool_calls(agent_result.reply_text),
        ]
        executed_tools = []
        for call in detected_tool_calls:
            record = tool_runtime.call(
                str(call.get("name")),
                dict(call.get("arguments") or {}),
                turn_id=turn_id,
            ).to_dict()
            executed_tools.append(record)
            tool_records.append(record)

        diff = update_state_after_turn(
            state,
            user_message=user_message,
            agent_message=agent_result.reply_text,
            tool_calls=executed_tools,
            max_turns=max_turns,
            previous_agent_message=previous_agent_message,
        )
        turns.append(
            {
                "turn_id": turn_id,
                "role": "agent",
                "content": agent_result.reply_text,
            }
        )
        incremental_rule_check = run_rule_checks(
            state=state,
            turns=turns,
            tool_calls=tool_records,
            max_turns=max_turns,
        ).to_dict()
        recorder.add_turn(
            turn_id=turn_id,
            role="agent",
            content=agent_result.reply_text,
            tool_calls=executed_tools,
            state=state.to_dict(),
            state_diff=diff,
            rule_check=incremental_rule_check,
        )


def replay(trajectory_path: Path) -> dict[str, Any]:
    return json.loads(trajectory_path.read_text(encoding="utf-8"))


def compare_versions(summary_a: Path, summary_b: Path) -> dict[str, Any]:
    a = json.loads(summary_a.read_text(encoding="utf-8"))
    b = json.loads(summary_b.read_text(encoding="utf-8"))
    return {
        "a_pass_rate": a.get("pass_rate"),
        "b_pass_rate": b.get("pass_rate"),
        "delta_pass_rate": round(float(b.get("pass_rate", 0)) - float(a.get("pass_rate", 0)), 4),
        "a_average_judge_score": a.get("average_judge_score"),
        "b_average_judge_score": b.get("average_judge_score"),
        "delta_average_judge_score": round(
            float(b.get("average_judge_score", 0)) - float(a.get("average_judge_score", 0)),
            4,
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    parser = argparse.ArgumentParser(description="Run the lightweight Agent Evaluation Harness.")
    parser.add_argument("--cases", required=False, default="cases/sample_cases.json")
    parser.add_argument("--agent", default="baseline_prompt_agent")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--llm-judge", action="store_true", help="Enable soft LLM judge when API key is available.")
    parser.add_argument("--tool-permission", default="read", choices=["read", "write", "handoff"])
    parser.add_argument("--replay", help="Print a saved trajectory JSON file and exit.")
    parser.add_argument("--compare", nargs=2, metavar=("SUMMARY_A", "SUMMARY_B"))
    args = parser.parse_args()

    if args.replay:
        print(json.dumps(replay(Path(args.replay)), ensure_ascii=False, indent=2))
        return
    if args.compare:
        print(json.dumps(compare_versions(Path(args.compare[0]), Path(args.compare[1])), ensure_ascii=False, indent=2))
        return

    if args.agent == "baseline_prompt_agent" and not _has_api_key():
        raise RuntimeError("baseline_prompt_agent 需要 DEEPSEEK_API_KEY 或 OPENAI_API_KEY；本地 smoke 可用 --agent scripted_mock_agent")

    cases = load_cases(Path(args.cases))
    runner = HarnessRunner(
        agent_name=args.agent,
        max_turns=args.max_turns,
        output_dir=Path(args.output_dir),
        llm_judge_enabled=args.llm_judge,
        tool_permission=args.tool_permission,
    )
    results = runner.run_cases(cases)
    case_results = [
        {
            "case_id": result.case_id,
            "final_result": result.final_result,
            "judge_result": result.judge_result,
            "rule_checker_result": result.rule_checker_result,
            "failure_types": result.failure_types,
            "trajectory_path": result.trajectory_path,
        }
        for result in results
    ]
    json_path, md_path = write_summary(case_results, Path(args.output_dir) / "reports")
    print(f"cases={len(case_results)} summary_json={json_path} summary_md={md_path}")


def _has_api_key() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
