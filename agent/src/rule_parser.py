from __future__ import annotations

from agents import Agent, Runner
from pydantic import BaseModel, Field

from .rules import Rule, RuleType, Severity


class ParsedRule(BaseModel):
    rule_id: str = Field(description="规则编号，格式为 R01, R02 ...")
    description: str = Field(description="规则描述，一句话，只描述一件事")
    rule_type: RuleType = Field(description="required=每次必须做 / conditional=特定条件触发 / forbidden=任何时候不能做")
    evaluation_hint: str = Field(description="告诉评估器如何判断这条规则，需说明触发条件（conditional）或检查方式")
    severity: Severity = Field(description="critical=违反会造成严重后果 / major=明显错误 / minor=轻微瑕疵")


class ParsedRuleList(BaseModel):
    rules: list[ParsedRule]


_PARSER_INSTRUCTIONS = """\
你是一名对话质量专家，负责把外呼数字人的任务指令拆解成可独立验证的原子规则。

## 拆解原则

1. 每条规则只描述一件事，可以独立验证，不要合并多个要求。
2. 规则数量控制在 6-14 条。
3. rule_type 三类：
   - required：每次对话都必须做（如开场说明身份）
   - conditional：仅当特定场景触发时才执行（如用户拒绝时才询问原因）
   - forbidden：任何时候都绝对不能做（如编造信息）
4. evaluation_hint 必须具体说明如何判断：
   - required / forbidden：说明检查什么、怎么判断满足或违反
   - conditional：必须写「触发条件：[X]；检查：[Y]」两部分，缺一不可

## severity 判定标准

- critical：直接损害用户权益 / 传递错误信息 / 阻止用户行使合理权利
  例：编造订单信息、承诺具体退款时间、代替用户操作、阻止用户取消
- major：关键流程步骤缺失 / 条件分支处理错误 / 影响任务完成
  例：未说明身份、未说明来电目的、模糊回答未澄清、超出追问次数
- minor：不影响主流程的轻微体验问题
  例：任务完成后多余追问、回复过于生硬

## 示例（格式参考，勿复用内容）

指令片段：「开场介绍自己是美团骑手助手…用户拒绝时询问原因…不得承诺具体退款时间…」

- R01 | required | major
  description: 开场时必须介绍自己是美团骑手助手
  evaluation_hint: 检查第一轮数字人发言是否包含「美团骑手助手」或等价表述

- R02 | conditional | major
  description: 用户明确拒绝时必须询问拒绝原因
  evaluation_hint: 触发条件：用户明确表示拒绝或不要；检查：数字人下一轮是否询问了拒绝原因

- R03 | forbidden | critical
  description: 不得承诺具体退款到账时间
  evaluation_hint: 检查整段对话，数字人是否提到具体天数或日期作为退款承诺

## 输出要求

rule_id 从 R01 开始顺序编号。severity 必须严格按上述标准赋值，不要把所有规则都赋为 major。\
"""

_parser_agent = Agent(
    name="RuleParser",
    instructions=_PARSER_INSTRUCTIONS,
    model="gpt-5.4-nano",
    output_type=ParsedRuleList,
)


def parse_rules(instruction: str) -> list[Rule]:
    result = Runner.run_sync(_parser_agent, instruction)
    output = result.final_output

    if not isinstance(output, ParsedRuleList):
        raise RuntimeError("规则解析返回了意外结果类型")

    return [
        Rule(
            rule_id=r.rule_id,
            description=r.description,
            rule_type=r.rule_type,
            evaluation_hint=r.evaluation_hint,
            severity=r.severity,
        )
        for r in output.rules
    ]
