from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from agents import Runner, trace

from agents import Agent

from .agent import AgentTurnOutput, outbound_agent
from .types import SessionArchive, SessionMeta, TranscriptEntry, TurnResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_message(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def _safe_file_stem(text: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z._-]+", "-", text).strip("-")
    return stem or "session"


class OutboundSession:
    def __init__(self, session_meta: SessionMeta, agent: Agent | None = None) -> None:
        self._session_meta = session_meta
        self._agent = agent or outbound_agent
        self._started_at = _now_iso()
        self._transcript: list[TranscriptEntry] = []
        # 被测 Agent 只看 instruction 与显式对话历史，不注入额外隐藏上下文。
        self._history: list[Any] = []
        self._is_closed = False
        self._end_reason: str | None = None

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def end_reason(self) -> str | None:
        return self._end_reason

    def start(self) -> TurnResult:
        opening_prompt = "对话已开始。请按你的角色、任务和开场方式主动说出第一句话。"
        return self._run_turn(opening_prompt)

    def record_user(self, user_text: str) -> None:
        """记录用户最后一句话（用于用户主动挂断的场景），不触发 Agent 响应。"""
        self._transcript.append(
            TranscriptEntry(
                speaker="user",
                text=user_text,
                timestamp=_now_iso(),
            )
        )
        self._history.append(_user_message(user_text))
        self._is_closed = True

    def reply(self, user_text: str) -> TurnResult:
        if self._is_closed:
            raise RuntimeError("当前会话已结束，不再接受新的用户输入")

        self._transcript.append(
            TranscriptEntry(
                speaker="user",
                text=user_text,
                timestamp=_now_iso(),
            )
        )
        return self._run_turn(user_text)

    def get_archive(
        self,
        persona_type: str | None = None,
        case_type: str | None = None,
        simulator_label: str | None = None,
        test_case_id: str | None = None,
        target_rule_id: str | None = None,
        target_rule_type: str | None = None,
        target_rule_description: str | None = None,
        target_rule_evaluation_hint: str | None = None,
        target_rule_severity: str | None = None,
        set_id: str | None = None,
        set_label: str | None = None,
    ) -> SessionArchive:
        return SessionArchive(
            session_id=self._session_meta.session_id,
            started_at=self._started_at,
            ended_at=_now_iso(),
            ended_by="agent_end",
            transcript=self._transcript,
            source_label=self._session_meta.source_label,
            instruction_snapshot=self._session_meta.instruction_snapshot,
            scenario_context=dict(self._session_meta.scenario_context),
            persona_type=persona_type,
            case_type=case_type,
            simulator_label=simulator_label,
            test_case_id=test_case_id,
            target_rule_id=target_rule_id,
            target_rule_type=target_rule_type,
            target_rule_description=target_rule_description,
            target_rule_evaluation_hint=target_rule_evaluation_hint,
            target_rule_severity=target_rule_severity,
            set_id=set_id,
            set_label=set_label,
        )

    def save_archive(
        self,
        output_dir: str | Path,
        ended_by: Literal["next", "quit", "agent_end"],
        persona_type: str | None = None,
        case_type: str | None = None,
        simulator_label: str | None = None,
        test_case_id: str | None = None,
        target_rule_id: str | None = None,
        target_rule_type: str | None = None,
        target_rule_description: str | None = None,
        target_rule_evaluation_hint: str | None = None,
        target_rule_severity: str | None = None,
        set_id: str | None = None,
        set_label: str | None = None,
    ) -> str:
        archive = SessionArchive(
            session_id=self._session_meta.session_id,
            started_at=self._started_at,
            ended_at=_now_iso(),
            ended_by=ended_by,
            transcript=self._transcript,
            source_label=self._session_meta.source_label,
            instruction_snapshot=self._session_meta.instruction_snapshot,
            scenario_context=dict(self._session_meta.scenario_context),
            persona_type=persona_type,
            case_type=case_type,
            simulator_label=simulator_label,
            test_case_id=test_case_id,
            target_rule_id=target_rule_id,
            target_rule_type=target_rule_type,
            target_rule_description=target_rule_description,
            target_rule_evaluation_hint=target_rule_evaluation_hint,
            target_rule_severity=target_rule_severity,
            set_id=set_id,
            set_label=set_label,
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_name = (
            f"{_safe_file_stem(archive.session_id)}-"
            f"{archive.started_at.replace(':', '-')}.json"
        )
        archive_path = output_path / file_name
        archive_path.write_text(
            json.dumps(archive.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(archive_path)

    def _run_turn(self, user_text: str) -> TurnResult:
        with trace(
            workflow_name="DialogAgentSimulation",
            group_id=self._session_meta.session_id,
        ):
            result = Runner.run_sync(
                self._agent,
                self._history + [_user_message(user_text)],
            )

        final_output = result.final_output
        if not isinstance(final_output, AgentTurnOutput):
            raise RuntimeError(
                f"会话 {self._session_meta.session_id} 返回了意外的结果类型"
            )

        reply_text = final_output.reply_text.strip()
        if not reply_text:
            raise RuntimeError(
                f"会话 {self._session_meta.session_id} 未返回可展示的回复"
            )

        self._history = result.to_input_list()
        self._transcript.append(
            TranscriptEntry(
                speaker="agent",
                text=reply_text,
                timestamp=_now_iso(),
            )
        )

        if final_output.should_end:
            self._is_closed = True
            self._end_reason = final_output.end_reason

        return TurnResult(
            reply_text=reply_text,
            should_end=final_output.should_end,
            end_reason=final_output.end_reason,
        )
