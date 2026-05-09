from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from agents import Runner, trace

from .agent import AgentTurnOutput, build_order_context, outbound_agent
from .types import LoadedOrder, SessionArchive, TranscriptEntry, TurnResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_message(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


class OutboundSession:
    def __init__(self, loaded_order: LoadedOrder) -> None:
        self._loaded_order = loaded_order
        self._started_at = _now_iso()
        self._transcript: list[TranscriptEntry] = []
        self._history: list[Any] = [_user_message(build_order_context(loaded_order.order))]
        self._is_closed = False
        self._end_reason: str | None = None

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def end_reason(self) -> str | None:
        return self._end_reason

    def start(self) -> TurnResult:
        opening_prompt = (
            "现在开始一通新的订单外呼模拟。"
            "你必须主动先说第一句话，完成开场问候、身份说明和来电目的说明。"
        )
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
            raise RuntimeError("当前订单会话已结束，不再接受新的用户输入")

        self._transcript.append(
            TranscriptEntry(
                speaker="user",
                text=user_text,
                timestamp=_now_iso(),
            )
        )
        return self._run_turn(user_text)

    def save_archive(
        self, output_dir: str | Path, ended_by: Literal["next", "quit", "agent_end"]
    ) -> str:
        archive = SessionArchive(
            order_id=self._loaded_order.order.order_id,
            user_name=self._loaded_order.order.user_name,
            source_file=self._loaded_order.file_name,
            started_at=self._started_at,
            ended_at=_now_iso(),
            ended_by=ended_by,
            transcript=self._transcript,
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_name = f"{archive.order_id}-{archive.started_at.replace(':', '-')}.json"
        archive_path = output_path / file_name
        archive_path.write_text(
            json.dumps(archive.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(archive_path)

    def _run_turn(self, user_text: str) -> TurnResult:
        with trace(
            workflow_name="MeituanOrderOutboundSimulation",
            group_id=self._loaded_order.order.order_id,
        ):
            result = Runner.run_sync(
                outbound_agent,
                self._history + [_user_message(user_text)],
            )

        final_output = result.final_output
        if not isinstance(final_output, AgentTurnOutput):
            raise RuntimeError(
                f"订单 {self._loaded_order.order.order_id} 返回了意外的结果类型"
            )

        reply_text = final_output.reply_text.strip()
        if not reply_text:
            raise RuntimeError(
                f"订单 {self._loaded_order.order.order_id} 未返回可展示的回复"
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
