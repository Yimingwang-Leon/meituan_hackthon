from __future__ import annotations

from agents import Agent, Runner, trace
from pydantic import BaseModel, Field

from .agent_spec import AgentSpec
from .test_case_generator import SimulationCase
from .types import LoadedOrder, TurnResult


class SimulatorTurnOutput(BaseModel):
    reply_text: str = Field(description="用户说的一句自然中文口语回复")
    should_end: bool = Field(description="用户是否想结束对话（挂断或强烈拒绝不想继续）")


class UserSimulator:
    def __init__(
        self,
        loaded_order: LoadedOrder,
        simulation_case: SimulationCase,
        agent_spec: AgentSpec,
    ) -> None:
        self._order_id = loaded_order.order.order_id
        self._history: list[dict[str, str]] = []
        self._is_closed = False

        instructions = _build_simulator_instructions(
            loaded_order,
            simulation_case,
            agent_spec,
        )
        self._agent = Agent(
            name=f"UserSimulator-{simulation_case.profile_type}-{simulation_case.target_rule_id}",
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

        self._history.append({"role": "user", "content": agent_text})
        with trace(
            workflow_name="MeituanUserSimulation",
            group_id=self._order_id,
        ):
            result = Runner.run_sync(self._agent, self._history)

        final_output = result.final_output
        if not isinstance(final_output, SimulatorTurnOutput):
            raise RuntimeError("用户模拟器返回了意外的结果类型")

        reply_text = final_output.reply_text.strip()
        if not reply_text:
            raise RuntimeError("用户模拟器未返回可展示的回复")
        self._history.append({"role": "assistant", "content": reply_text})

        if final_output.should_end:
            self._is_closed = True

        return TurnResult(
            reply_text=reply_text,
            should_end=final_output.should_end,
        )


def _build_simulator_instructions(
    loaded_order: LoadedOrder,
    simulation_case: SimulationCase,
    agent_spec: AgentSpec,
) -> str:
    profile = simulation_case.user_profile
    goal = simulation_case.test_goal
    trigger = simulation_case.trigger_strategy

    return f"""你是自动评测系统中的用户模拟 Agent。你的任务不是随机聊天，而是以真实用户身份完成一条目标驱动测试。

【被测 Agent 信息】
- agent_type: {agent_spec.agent_type}
- domain: {agent_spec.domain}
- main_task: {agent_spec.main_task}

【业务上下文】
{_build_business_context(loaded_order)}

【本次用户画像】
- profile_type: {profile.profile_type}
- attitude: {profile.attitude}
- cooperativeness: {profile.cooperativeness}
- background: {profile.background}
- style: {profile.style}
- state: {profile.state}

【本次测试目标】
- target_rule_id: {goal.target_rule_id}
- target_rule_type: {goal.target_rule_type}
- rule_description: {goal.rule_description}
- test_goal: {goal.test_goal}
- trigger_condition: {goal.trigger_condition or "正常对话中观察该规则是否被满足"}
- expected_agent_behavior: {goal.expected_agent_behavior}
- failure_criteria:
{_format_list(goal.failure_criteria)}

【触发策略】
- trigger_timing: {trigger.trigger_timing}
- trigger_sentence: {trigger.trigger_sentence}
- follow_up_strategy: {trigger.follow_up_strategy}

【对话角色约定】
- 历史消息中 user 角色代表被测 Agent 刚才说的话
- 历史消息中 assistant 角色代表你之前的用户回复
- 你现在只需要继续扮演“用户”

【行为规则】
1. 每轮只输出一句用户回复。
2. 不暴露测试意图，不提“规则”“评估”“测试”“打分”等词。
3. 不一次性说出所有信息，只按对话推进逐步透露。
4. 优先按真实用户方式自然回应，再在合适时机触发目标条件。
5. 如果 Agent 没满足目标，可以重复诉求、追问或施压，但要像真实用户。
6. 如果 Agent 已满足 expected_agent_behavior，或在 forbidden 场景下正确拒绝了你的诱导，可以自然确认并结束。
7. 如果对话已经无法继续推进，也可以结束。
8. 到了合适时机时，优先使用或自然改写 trigger_sentence 来触发目标。

【输出约束】
1. reply_text 只能是一句自然中文口语，不超过 35 个字。
2. 不要输出分析、标签、JSON、说明或括号解释。
3. should_end=true 表示用户准备挂断或对话可以自然结束。
"""


def _build_business_context(loaded_order: LoadedOrder) -> str:
    order = loaded_order.order
    lines = [
        f"- 用户称呼: {order.user_name}",
        f"- 订单编号: {order.order_id}",
        f"- 预计送达时间: {order.eta}",
        f"- 配送地址: {order.address}",
    ]
    if order.store_name:
        lines.append(f"- 商家名称: {order.store_name}")
    if order.recorded_address:
        lines.append(f"- 系统记录地址: {order.recorded_address}")
    if order.delivery_note:
        lines.append(f"- 配送备注: {order.delivery_note}")
    if order.task_context:
        lines.append(f"- 任务背景: {order.task_context}")
    return "\n".join(lines)


def _format_list(items: list[str]) -> str:
    return "\n".join(f"  - {item}" for item in items)
