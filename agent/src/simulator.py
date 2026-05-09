from __future__ import annotations

from typing import Any

from agents import Agent, Runner, trace
from pydantic import BaseModel, Field

from .persona import UserPersona, build_persona_instructions
from .types import LoadedOrder, TurnResult


class SimulatorTurnOutput(BaseModel):
    reply_text: str = Field(description="用户说的一句自然中文口语回复")
    should_end: bool = Field(description="用户是否想结束对话（挂断或强烈拒绝不想继续）")


class UserSimulator:
    def __init__(self, loaded_order: LoadedOrder, persona: UserPersona) -> None:
        self._order_id = loaded_order.order.order_id
        self._history: list[Any] = []
        self._is_closed = False

        instructions = build_persona_instructions(persona, loaded_order.order.eta)
        self._agent = Agent(
            name=f"UserSimulator-{persona.persona_type}",
            instructions=instructions,
            model="gpt-5.4-nano",
            output_type=SimulatorTurnOutput,
        )

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def reply(self, agent_text: str) -> TurnResult:
        if self._is_closed:
            raise RuntimeError("用户模拟器已结束对话")

        with trace(
            workflow_name="MeituanUserSimulation",
            group_id=self._order_id,
        ):
            result = Runner.run_sync(
                self._agent,
                self._history + [{"role": "user", "content": agent_text}],
            )

        final_output = result.final_output
        if not isinstance(final_output, SimulatorTurnOutput):
            raise RuntimeError("用户模拟器返回了意外的结果类型")

        self._history = result.to_input_list()

        if final_output.should_end:
            self._is_closed = True

        return TurnResult(
            reply_text=final_output.reply_text.strip(),
            should_end=final_output.should_end,
        )
