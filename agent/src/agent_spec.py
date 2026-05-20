from __future__ import annotations

from agents import Agent, Runner
from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    agent_type: str = Field(description="被测 Agent 的类型名称")
    domain: str = Field(description="业务领域，例如外卖履约、客服、营销")
    main_task: str = Field(description="该 Agent 的主要任务")
    workflow_rules: list[str] = Field(
        default_factory=list,
        description="流程型规则，按业务推进顺序列出",
    )
    condition_rules: list[str] = Field(
        default_factory=list,
        description="条件触发规则，写明条件和对应动作",
    )
    prohibited_rules: list[str] = Field(
        default_factory=list,
        description="禁止行为规则",
    )
    style_rules: list[str] = Field(
        default_factory=list,
        description="风格、语气、礼貌、共情类规则",
    )
    required_information: list[str] = Field(
        default_factory=list,
        description="完成任务通常需要获取或确认的信息",
    )
    termination_conditions: list[str] = Field(
        default_factory=list,
        description="允许或要求结束对话的条件",
    )


_AGENT_SPEC_INSTRUCTIONS = """\
你是一名对话测试架构师。你的任务是把任意对话 Agent 的 system prompt 解析成统一的 AgentSpec。

输出要求：
1. 只基于给定 prompt 本身归纳，不补充外部知识。
2. workflow_rules / condition_rules / prohibited_rules / style_rules 都应去重、短句化。
3. condition_rules 必须写清楚“什么条件下，Agent 应该怎么做”。
4. required_information 填 Agent 需要确认、收集、核实的信息。
5. termination_conditions 填什么情况下应结束对话。
6. agent_type / domain / main_task 不能为空。
"""

_agent_spec_agent = Agent(
    name="AgentSpecParser",
    instructions=_AGENT_SPEC_INSTRUCTIONS,
    model="deepseek-v4-flash",
    output_type=AgentSpec,
)


def parse_agent_spec(instruction: str) -> AgentSpec:
    result = Runner.run_sync(_agent_spec_agent, instruction)
    output = result.final_output

    if not isinstance(output, AgentSpec):
        raise RuntimeError("AgentSpec 解析返回了意外结果类型")

    return output
