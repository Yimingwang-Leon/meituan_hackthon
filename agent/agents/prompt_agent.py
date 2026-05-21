from __future__ import annotations

from typing import Any

from src.agent import OUTBOUND_INSTRUCTIONS, make_outbound_agent
from src.session import OutboundSession
from src.types import SessionMeta

from .base import AgentStepResult, BaseAgentAdapter


class PromptAgentAdapter(BaseAgentAdapter):
    """Wrap the existing prompt-only outbound agent behind the harness API."""

    def __init__(
        self,
        instruction: str | None = None,
        agent_version: str = "baseline_prompt_agent",
        prompt_version: str = "inline",
    ) -> None:
        self._base_instruction = instruction or OUTBOUND_INSTRUCTIONS
        self.agent_version = agent_version
        self.prompt_version = prompt_version
        self.model_name = "deepseek-v4-flash"
        self._session: OutboundSession | None = None
        self._started = False
        self._case: dict[str, Any] = {}

    def reset(self, case: dict[str, Any]) -> None:
        self._case = dict(case)
        instruction = str(case.get("instruction") or self._base_instruction)
        self.prompt_version = str(case.get("prompt_version") or self.prompt_version)
        self.agent_version = str(case.get("agent_version") or self.agent_version)

        agent = make_outbound_agent(instruction)
        self.model_name = agent.model
        self._session = OutboundSession(
            SessionMeta(
                session_id=str(case.get("case_id") or "case"),
                source_label="harness",
                instruction_snapshot=instruction,
                scenario_context=dict(case.get("scenario_context") or {}),
            ),
            agent=agent,
        )
        self._started = False

    def step(self, user_message: str | None, state) -> AgentStepResult:
        if self._session is None:
            raise RuntimeError("PromptAgentAdapter.reset(case) must be called before step().")

        if not self._started:
            result = self._session.start()
            self._started = True
        else:
            if user_message is None:
                raise ValueError("user_message is required after the opening turn")
            result = self._session.reply(user_message)

        return AgentStepResult(
            reply_text=result.reply_text,
            should_end=result.should_end,
            end_reason=result.end_reason,
            raw_output=result,
        )

    def get_archive_kwargs(self) -> dict[str, Any]:
        return {
            "persona_type": self._case.get("persona_type"),
            "case_type": self._case.get("case_type"),
            "simulator_label": self._case.get("label") or self._case.get("case_id"),
            "test_case_id": self._case.get("case_id"),
            "target_rule_id": self._case.get("target_rule_id"),
            "set_id": self._case.get("set_id"),
            "set_label": self._case.get("set_label"),
        }

    def archive(self):
        if self._session is None:
            return None
        return self._session.get_archive(**self.get_archive_kwargs())
