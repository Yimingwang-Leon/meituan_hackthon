from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SessionMeta:
    session_id: str
    source_label: str | None = None
    instruction_snapshot: str | None = None
    scenario_context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TranscriptEntry:
    speaker: Literal["agent", "user"]
    text: str
    timestamp: str


@dataclass(frozen=True)
class SessionArchive:
    session_id: str
    started_at: str
    ended_at: str
    ended_by: Literal["next", "quit", "agent_end"]
    transcript: list[TranscriptEntry]
    source_label: str | None = None
    instruction_snapshot: str | None = None
    scenario_context: dict[str, str] = field(default_factory=dict)
    persona_type: str | None = None
    case_type: str | None = None
    simulator_label: str | None = None
    test_case_id: str | None = None
    target_rule_id: str | None = None
    target_rule_type: str | None = None
    target_rule_description: str | None = None
    target_rule_evaluation_hint: str | None = None
    target_rule_severity: str | None = None
    set_id: str | None = None
    set_label: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TurnResult:
    reply_text: str
    should_end: bool
    end_reason: str | None = None
