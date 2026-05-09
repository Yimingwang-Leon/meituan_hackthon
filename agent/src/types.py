from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class OrderRecord:
    user_name: str
    order_id: str
    eta: str
    address: str
    notes: str | None = None


@dataclass(frozen=True)
class LoadedOrder:
    file_path: str
    file_name: str
    order: OrderRecord


@dataclass(frozen=True)
class TranscriptEntry:
    speaker: Literal["agent", "user"]
    text: str
    timestamp: str


@dataclass(frozen=True)
class SessionArchive:
    order_id: str
    user_name: str
    source_file: str
    started_at: str
    ended_at: str
    ended_by: Literal["next", "quit", "agent_end"]
    transcript: list[TranscriptEntry]
    persona_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TurnResult:
    reply_text: str
    should_end: bool
    end_reason: str | None = None
