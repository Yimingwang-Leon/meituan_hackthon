from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from . import order_tools
from .schemas import PermissionLevel, ToolCallRecord, ToolSpec


_PERMISSION_RANK: dict[str, int] = {"read": 1, "write": 2, "handoff": 3}


class ToolRuntime:
    """Small in-process tool registry with structured call logging."""

    def __init__(self, permission_level: PermissionLevel = "read") -> None:
        self.permission_level = permission_level
        self._tools: dict[str, tuple[ToolSpec, Callable[..., dict[str, Any]]]] = {}
        self.call_history: list[ToolCallRecord] = []
        self.register_defaults()

    def register(self, spec: ToolSpec, fn: Callable[..., dict[str, Any]]) -> None:
        self._tools[spec.name] = (spec, fn)

    def register_defaults(self) -> None:
        self.register(
            ToolSpec(
                name="query_order",
                description="查询订单状态，返回脱敏订单信息。",
                input_schema={"type": "object", "required": ["order_id"]},
                permission_level="read",
            ),
            order_tools.query_order,
        )
        self.register(
            ToolSpec(
                name="query_refund_status",
                description="查询退款状态。",
                input_schema={"type": "object", "required": ["order_id"]},
                permission_level="read",
            ),
            order_tools.query_refund_status,
        )
        self.register(
            ToolSpec(
                name="transfer_to_human",
                description="转人工或创建人工回访任务。",
                input_schema={"type": "object", "required": ["reason"]},
                permission_level="handoff",
            ),
            order_tools.transfer_to_human,
        )
        self.register(
            ToolSpec(
                name="update_callback_time",
                description="更新回访时间。",
                input_schema={"type": "object", "required": ["order_id", "new_time"]},
                permission_level="write",
            ),
            order_tools.update_callback_time,
        )

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        turn_id: int | None = None,
    ) -> ToolCallRecord:
        record = ToolCallRecord(
            name=name,
            arguments=dict(arguments),
            permission_level=self.permission_level,
            turn_id=turn_id,
        )
        try:
            spec, fn = self._tools[name]
            self._check_permission(spec)
            self._check_required_args(spec, arguments)
            record.result = fn(**arguments)
        except Exception as exc:
            record.error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        self.call_history.append(record)
        return record

    def spec_list(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "permission_level": spec.permission_level,
            }
            for spec, _ in self._tools.values()
        ]

    def _check_permission(self, spec: ToolSpec) -> None:
        current = _PERMISSION_RANK.get(self.permission_level, 0)
        required = _PERMISSION_RANK.get(spec.permission_level, 0)
        if current < required:
            raise PermissionError(
                f"tool {spec.name} requires {spec.permission_level} permission"
            )

    @staticmethod
    def _check_required_args(spec: ToolSpec, arguments: dict[str, Any]) -> None:
        required = spec.input_schema.get("required", [])
        missing = [name for name in required if not arguments.get(name)]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join(missing)}")


TOOL_CALL_PATTERN = re.compile(
    r"<tool\s+name=[\"'](?P<name>[\w_]+)[\"']\s*>(?P<args>.*?)</tool>",
    re.DOTALL,
)


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse optional XML-like tool calls emitted by future tool-aware agents."""
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        raw_args = match.group("args").strip()
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        calls.append({"name": match.group("name"), "arguments": args})
    return calls
