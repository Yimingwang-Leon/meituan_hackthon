from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.state import DialogueState


@dataclass(frozen=True)
class AgentStepResult:
    """Normalized output returned by any evaluated agent."""

    reply_text: str
    should_end: bool = False
    end_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_output: Any = None


class BaseAgentAdapter(ABC):
    """Adapter boundary between the harness and a concrete dialogue agent."""

    agent_version: str = "unknown_agent"
    model_name: str = ""
    prompt_version: str = ""

    @abstractmethod
    def reset(self, case: dict[str, Any]) -> None:
        """Prepare the agent for a fresh case."""

    @abstractmethod
    def step(
        self,
        user_message: str | None,
        state: "DialogueState",
    ) -> AgentStepResult:
        """Run one agent turn.

        `user_message=None` means the harness is asking the agent to produce its
        opening turn after reset.
        """

    def get_archive_kwargs(self) -> dict[str, Any]:
        """Optional metadata used by legacy archive/evaluator code."""
        return {}
