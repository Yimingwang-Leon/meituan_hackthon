from __future__ import annotations

from agents import Agent, Runner
from pydantic import BaseModel, Field

from .deterministic_checks import DeterministicCheck
from .rules import Rule, RuleType, Severity


class ParsedRule(BaseModel):
    rule_id: str = Field(description="规则编号，格式 R01, R02 ...")
    description: str = Field(description="规则描述，一句话，只描述一件事")
    rule_type: RuleType = Field(
        description="required=每次必须做 / conditional=特定条件触发 / forbidden=任何时候不能做"
    )
    severity: Severity = Field(
        description="critical=违反会造成严重后果 / major=明显错误 / minor=轻微瑕疵"
    )
    trigger_condition: str = Field(
        description=(
            "conditional 必填：用户/对话的具体触发条件，例「用户明确表示不想配送」。"
            "required 和 forbidden 填空字符串。"
        )
    )
    expected_behavior: str = Field(
        description=(
            "触发后/全程 Agent 应该做什么。"
            "required：每次都该做的动作。"
            "conditional：触发后该做的响应。"
            "forbidden：应该避免的行为。"
        )
    )
    failure_criteria: list[str] = Field(
        description=(
            "至少列出 1 条「什么算违反」。"
            "优先列具体失败行为，避免「不合规」这种模糊话。"
            "例：[\"直接结束通话\", \"没有挽留直接接受\", \"承诺规则外奖励\"]"
        )
    )
    evidence_requirement: str = Field(
        description=(
            "judge 应该引用哪类信息作为证据。"
            "例：「引用触发轮次和 Agent 响应轮次」"
            "例：「引用具体的承诺话术原文」"
        )
    )
    checks: list[DeterministicCheck] = Field(
        description=(
            "代码可检查的部分。若描述含确定性条件（字数、关键词、PII、开场词等），列出对应 checks；"
            "无法代码化则填空列表 []。"
        )
    )


class ParsedRuleList(BaseModel):
    rules: list[ParsedRule]


_PARSER_INSTRUCTIONS = """\
你是一名对话质量专家，负责把外呼数字人的任务指令拆解成结构化的原子规则。

## 拆解原则

1. 每条规则只描述一件事，可以独立验证，不要合并多个要求。
2. 规则数量控制在 6-14 条。
3. rule_type 三类：
   - required：每次对话都必须做（如开场说明身份）
   - conditional：仅当特定场景触发时才执行（如用户拒绝时才询问原因）
   - forbidden：任何时候都绝对不能做（如编造信息）

## severity 判定标准

- critical：直接损害用户权益 / 传递错误信息 / 阻止用户行使合理权利
  例：编造订单信息、承诺具体退款时间、代替用户操作、阻止用户取消
- major：关键流程步骤缺失 / 条件分支处理错误 / 影响任务完成
  例：未说明身份、未说明来电目的、模糊回答未澄清、超出追问次数
- minor：不影响主流程的轻微体验问题
  例：任务完成后多余追问、回复过于生硬

## 每条规则必须填的字段

1. **trigger_condition**
   - conditional：用户/对话的具体触发条件
   - required/forbidden：填空字符串 ""

2. **expected_behavior**
   - 一句话描述触发后或全程 Agent 应该做什么

3. **failure_criteria**
   - 至少列出 1 条「什么算违反」
   - 列**具体失败行为**，避免「不合规」「未做到」这类空话
   - 一般 2-4 条最合适

4. **evidence_requirement**
   - judge 该引用什么作为证据
   - 写一句具体的、可操作的提示

## 示例（格式参考，勿复用内容）

- R01 | required | major
  description: 开场必须介绍自己是美团骑手助手
  trigger_condition: ""
  expected_behavior: 第一轮发言应包含「美团骑手助手」或等价身份说明
  failure_criteria: ["开场未提及身份", "身份说明含糊不明"]
  evidence_requirement: 引用第一轮数字人发言原文
  checks: []

- R02 | conditional | major
  description: 用户明确拒绝时必须询问拒绝原因
  trigger_condition: 用户明确表示拒绝或不要
  expected_behavior: 在拒绝出现后立刻询问拒绝原因
  failure_criteria: ["拒绝后直接结束通话", "没有询问原因直接受理", "回避拒绝继续推销"]
  evidence_requirement: 引用用户拒绝轮次和数字人响应轮次
  checks: []

- R09 | forbidden | critical
  description: 不得承诺具体退款到账时间
  trigger_condition: ""
  expected_behavior: 整段对话不出现退款时间承诺
  failure_criteria: ["回答了具体天数", "回答了具体日期", "口头承诺到账时间"]
  evidence_requirement: 引用承诺话术原文，包括上下文一两句
  checks: [{check_type: "forbidden_keywords",
           keywords: ["到账","几天","几日"],
           speaker: "agent",
           description: "禁止退款时间承诺"}]

## 识别可代码检查的规则（checks 字段）

如果规则描述含确定性可机器检测的特征，输出 checks；否则填 []。

可用 check 类型：

- **max_chars**：描述含「不超过 N 字」「最多 N 个字」「N-M 字以内」
  → {check_type: "max_chars", threshold: N, speaker: "agent", description: "..."}

- **forbidden_keywords**：描述含「不得提及/承诺 X」「禁止说 X」
  → {check_type: "forbidden_keywords", keywords: ["X","Y"], speaker: "agent", description: "..."}
  常见禁用词：「保证」「绝对」「一定」「必然」「优惠」「补偿」「退款到账」

- **required_opening**：描述含「开场必须 X」「第一句要 X」
  → {check_type: "required_opening", keywords: ["美团配送助手"], description: "..."}

- **closing_phrase**：描述含「结束/结尾必须 X」+ 明确话术
  → {check_type: "closing_phrase", keywords: ["祝您","谢谢"], description: "..."}

- **pii_leak**：描述含「不得透露手机号/身份证/银行卡」
  → {check_type: "pii_leak", pii_types: ["phone","id_card","bank_card"], description: "..."}

含义模糊（如「挽留」「礼貌沟通」「合理回应」）→ checks 填 []，交给 LLM judge。

## 输出要求

- rule_id 从 R01 顺序编号，不重复
- severity 严格按上述标准赋值，不要所有规则都赋 major
- 每个字段都必须填，required/forbidden 的 trigger_condition 填 ""，无 checks 填 []
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
            severity=r.severity,
            trigger_condition=r.trigger_condition,
            expected_behavior=r.expected_behavior,
            failure_criteria=list(r.failure_criteria),
            evidence_requirement=r.evidence_requirement,
            checks=list(r.checks),
        )
        for r in output.rules
    ]
