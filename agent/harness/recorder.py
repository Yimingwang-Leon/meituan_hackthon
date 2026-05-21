from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrajectoryRecorder:
    case_id: str
    agent_version: str
    model_name: str = ""
    prompt_version: str = ""
    output_dir: Path = Path("outputs/trajectories")
    start_time: str = field(default_factory=utc_now)
    end_time: str = ""
    turns: list[dict[str, Any]] = field(default_factory=list)
    rule_checker_result: dict[str, Any] = field(default_factory=dict)
    judge_result: dict[str, Any] = field(default_factory=dict)
    final_result: dict[str, Any] = field(default_factory=dict)
    failure_types: list[str] = field(default_factory=list)

    def add_turn(
        self,
        turn_id: int,
        role: str,
        content: str,
        state: dict[str, Any],
        state_diff: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        rule_check: dict[str, Any] | None = None,
    ) -> None:
        self.turns.append(
            {
                "turn_id": turn_id,
                "role": role,
                "content": content,
                "tool_calls": tool_calls or [],
                "state": state,
                "state_diff": state_diff or {},
                "rule_check": rule_check or {},
            }
        )

    def finish(
        self,
        rule_checker_result: dict[str, Any],
        judge_result: dict[str, Any],
        final_result: dict[str, Any],
        failure_types: list[str],
    ) -> Path:
        self.end_time = utc_now()
        self.rule_checker_result = rule_checker_result
        self.judge_result = judge_result
        self.final_result = final_result
        self.failure_types = failure_types
        return self.write()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "agent_version": self.agent_version,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "turns": self.turns,
            "rule_checker_result": self.rule_checker_result,
            "judge_result": self.judge_result,
            "final_result": self.final_result,
            "failure_types": self.failure_types,
        }

    def write(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_case = _safe(self.case_id)
        safe_agent = _safe(self.agent_version)
        path = self.output_dir / f"{safe_case}_{safe_agent}.json"
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value)
