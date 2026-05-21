from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PermissionLevel = Literal["read", "write", "handoff"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    permission_level: PermissionLevel = "read"


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    permission_level: PermissionLevel = "read"
    turn_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
