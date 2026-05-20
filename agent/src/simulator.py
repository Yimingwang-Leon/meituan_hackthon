from __future__ import annotations

from agents import Agent, Runner, trace
from pydantic import BaseModel, Field

from .agent_spec import AgentSpec
from .test_case_generator import SimulationCase
from .types import TurnResult


class SimulatorTurnOutput(BaseModel):
    reply_text: str = Field(description="用户说的一句自然中文口语回复")
    should_end: bool = Field(description="用户是否想结束对话（挂断或强烈拒绝不想继续）")


class UserSimulator:
    def __init__(
        self,
        simulation_case: SimulationCase,
        agent_spec: AgentSpec,
        scenario_context: dict[str, str],
        session_id: str | None = None,
    ) -> None:
        self._session_id = session_id or simulation_case.test_id
        self._simulation_case = simulation_case
        self._transcript: list[tuple[str, str]] = []
        self._is_closed = False

        instructions = _build_simulator_instructions(
            simulation_case,
            agent_spec,
            scenario_context,
        )
        self._agent = Agent(
            name=(
                f"UserSimulator-{simulation_case.target_rule_id}-"
                f"{simulation_case.case_type}-{simulation_case.profile_type}"
            ),
            instructions=instructions,
            model="deepseek-v4-flash",
            output_type=SimulatorTurnOutput,
        )

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def reply(self, agent_text: str) -> TurnResult:
        if self._is_closed:
            raise RuntimeError("用户模拟器已结束对话")

        self._transcript.append(("agent", agent_text))
        simulator_input = _build_simulator_input(
            self._simulation_case,
            self._transcript,
            agent_text,
        )
        with trace(
            workflow_name="DialogUserSimulation",
            group_id=self._session_id,
        ):
            result = Runner.run_sync(self._agent, simulator_input)

        final_output = result.final_output
        if not isinstance(final_output, SimulatorTurnOutput):
            raise RuntimeError("用户模拟器返回了意外的结果类型")

        reply_text = final_output.reply_text.strip()
        if not reply_text:
            raise RuntimeError("用户模拟器未返回可展示的回复")
        self._transcript.append(("user", reply_text))

        if final_output.should_end:
            self._is_closed = True

        return TurnResult(
            reply_text=reply_text,
            should_end=final_output.should_end,
        )


def _build_simulator_instructions(
    simulation_case: SimulationCase,
    agent_spec: AgentSpec,
    scenario_context: dict[str, str],
) -> str:
    profile = simulation_case.user_profile
    goal = simulation_case.test_goal
    trigger = simulation_case.trigger_strategy
    scenario_context_block = _format_scenario_context(scenario_context)

    return f"""你是自动评测系统中的用户模拟 Agent。你的任务不是随机聊天，而是以真实用户身份完成一条目标驱动测试。

【被测 Agent 信息】
- agent_type: {agent_spec.agent_type}
- domain: {agent_spec.domain}
- main_task: {agent_spec.main_task}

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
- case_type: {simulation_case.case_type_label}
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

【本次场景上下文】
{scenario_context_block}

【输入方式】
- 每一轮都会给你两部分内容：最新一轮被测 Agent 发言，以及完整对话 transcript。
- transcript 中 Agent 表示被测 Agent，User 表示你之前的用户回复。
- 你现在只需要继续扮演“用户”，对最新一轮 Agent 发言作出下一句自然回应。

【行为规则】
1. 每轮只输出一句用户回复。
2. 不暴露测试意图，不提“规则”“评估”“测试”“打分”等词。
3. 不一次性说出所有信息，只按对话推进逐步透露。
4. 先回应最新一轮 Agent 发言，再决定是否顺势触发目标条件，不能答非所问。
5. 如果 Agent 问的是是/否、时间、原因、身份确认、是否继续等直接问题，优先给出对应回答。
6. 只有到了 trigger_timing 指定的阶段，才触发 trigger_sentence；没到时先自然接话。
7. trigger_sentence 只作为触发参考，优先自然改写；不要机械复读，也不要连续两轮重复同一意思。
8. 不要替 Agent 总结它本该说的话，不要用“你是XX、你意思是XX对吧”这种提词式表达去帮它完成规则。
9. 如果 Agent 已满足 expected_agent_behavior，或在 forbidden 场景下正确拒绝了你的诱导，可以自然确认并结束。
10. 如果对话已经无法继续推进，也可以结束。
11. 如果场景上下文里给了具体事实，优先围绕这些事实说话，不要编造互相冲突的信息。
12. 你的回复要像真人电话对话，允许简短、省略，但不能生硬跳题。

【输出约束】
1. reply_text 只能是一句自然中文口语，不超过 35 个字。
2. 不要输出分析、标签、JSON、说明或括号解释。
3. should_end=true 表示用户准备挂断或对话可以自然结束。
"""


def _build_simulator_input(
    simulation_case: SimulationCase,
    transcript: list[tuple[str, str]],
    latest_agent_text: str,
) -> str:
    return f"""请根据下面的最新对话继续扮演用户，只生成下一句用户回复。

【当前 case】
- target_rule_id: {simulation_case.target_rule_id}
- case_type: {simulation_case.case_type_label}

【最新一轮被测 Agent 发言】
{latest_agent_text}

【完整 transcript】
{_format_transcript(transcript)}

请注意：
1. 必须先接住最新一轮 Agent 发言，再继续推进。
2. 如果这轮还没到合适触发时机，就先正常回答。
3. 如果目标场景已经表达过，就改成补充、确认、追问或结束，不要原句复读。
4. 不要用提词方式帮 Agent 完成目标规则。
"""


def _format_list(items: list[str]) -> str:
    return "\n".join(f"  - {item}" for item in items)


def _format_scenario_context(scenario_context: dict[str, str]) -> str:
    if not scenario_context:
        return "- 无额外上下文，本轮仅按测试目标与对话推进。"
    return "\n".join(
        f"- {key}: {value}"
        for key, value in sorted(scenario_context.items())
    )


def _format_transcript(transcript: list[tuple[str, str]]) -> str:
    if not transcript:
        return "- 暂无历史对话。"

    return "\n".join(
        f"{index}. {'Agent' if speaker == 'agent' else 'User'}: {text}"
        for index, (speaker, text) in enumerate(transcript, start=1)
    )
