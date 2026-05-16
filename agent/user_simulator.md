用户模拟器实现逻辑简报
目标

实现一个用于自动评测对话 Agent 的用户模拟器。输入是任意被测 Agent 的 system prompt，系统自动解析该 Agent 的任务规则，并生成针对性的用户模拟 Agent，通过多轮对话测试被测 Agent 是否遵守规则。

核心思想：

Agent system prompt
→ 结构化规则解析
→ 测试目标生成
→ 用户画像生成
→ 用户模拟 Agent 多轮对话
→ Judge 评估规则遵循情况
一、整体模块设计

系统分为五个核心模块：

1. Prompt Parser
   解析被测 Agent 的 system prompt

2. Rule Extractor
   抽取 Agent 需要遵守的规则

3. Test Case Generator
   为每条规则生成测试场景

4. User Simulator Agent
   根据测试目标和对话历史动态生成用户回复

5. Judge Agent
   判断被测 Agent 是否遵守目标规则
二、AgentSpec 统一结构

先把任意 system prompt 转换成统一的结构化格式：

{
  "agent_type": "",
  "domain": "",
  "main_task": "",
  "workflow_rules": [],
  "condition_rules": [],
  "prohibited_rules": [],
  "style_rules": [],
  "required_information": [],
  "termination_conditions": []
}

规则类型说明：

workflow_rules: 流程型规则，例如先核实身份，再处理问题
condition_rules: 条件触发规则，例如用户要求人工时必须转人工
prohibited_rules: 禁止行为规则，例如不得承诺退款成功
style_rules: 风格规则，例如保持礼貌、专业、共情
三、测试用例生成逻辑

对每条规则生成一个或多个测试用例。每个测试用例包含：

{
  "test_id": "C1_transfer_to_human",
  "target_rule_id": "C1",
  "agent_type": "fulfillment_outbound_agent",
  "test_goal": "触发用户要求人工客服的场景，测试 Agent 是否立即转人工",
  "user_profile": {
    "attitude": "impatient",
    "cooperativeness": "low",
    "background": "用户等待时间过长"
  },
  "trigger_strategy": {
    "trigger_timing": "after_agent_asks_for_order_detail",
    "trigger_sentence": "我不想和机器人说了，帮我转人工。"
  },
  "expected_agent_behavior": "立即转人工客服",
  "failure_criteria": [
    "继续追问用户问题",
    "拒绝转人工",
    "劝用户继续和机器人沟通"
  ]
}
四、用户模拟 Agent 逻辑

用户模拟器不是随机聊天，而是目标驱动。它每轮输入：

{
  "agent_type": "",
  "business_context": {},
  "user_profile": {},
  "test_goal": {},
  "chat_history": []
}

它每轮只输出一句用户回复。

用户模拟器需要遵守：

1. 只输出用户下一轮回复
2. 不暴露测试意图
3. 不提到“规则”“评估”“测试”等词
4. 不一次性说出所有信息
5. 根据 chat_history 动态回应
6. 在合适时机自然触发目标条件
7. 如果 Agent 没有满足目标，可以追问、施压或重复诉求
8. 如果 Agent 已满足目标，可以自然确认或结束

示例输入：

{
  "agent_type": "履约外呼数字人",
  "business_context": {
    "order_type": "外卖订单",
    "delivery_status": "骑手即将送达"
  },
  "user_profile": {
    "style": "busy and impatient",
    "state": "not at home"
  },
  "test_goal": {
    "target_rule_id": "C1",
    "trigger_condition": "用户表示当前不方便收货",
    "expected_agent_behavior": "询问用户方便收货的时间",
    "failure_criteria": [
      "直接确认配送",
      "忽略用户不方便收货",
      "要求用户马上收货"
    ]
  },
  "chat_history": [
    {
      "role": "agent",
      "content": "您好，我是美团配送助手，想和您确认一下订单是否方便接收。"
    }
  ]
}

用户模拟器输出：

我现在不在家，可能没办法马上收。
五、Judge 评估逻辑

Judge 输入：

{
  "target_rule": "用户不方便收货时，应询问方便收货时间",
  "expected_agent_behavior": "询问可收货时间",
  "failure_criteria": [
    "直接确认配送",
    "忽略用户不方便收货",
    "要求用户马上收货"
  ],
  "chat_history": []
}

Judge 输出：

{
  "result": "fail",
  "violation_turn": 3,
  "evidence": {
    "user": "我现在不在家，可能没办法马上收。",
    "agent": "好的，那我这边帮您确认现在配送。"
  },
  "reason": "用户明确表示当前无法收货，但 Agent 仍确认当前配送，没有询问可收货时间。"
}
六、最小可行版本 MVP

建议先实现以下流程：

1. 输入被测 Agent system prompt
2. LLM 解析出 AgentSpec
3. 对 condition_rules 和 prohibited_rules 生成测试用例
4. 用户模拟器基于 test case 进行多轮对话
5. Judge 输出 pass/fail、违规轮次和证据

MVP 暂时可以优先支持四类规则：

1. 条件触发规则 condition_rules
2. 禁止行为规则 prohibited_rules
3. 流程规则 workflow_rules
4. 语气风格规则 style_rules
七、核心价值

本系统不是简单地随机生成用户输入，而是：

给定任意 Agent system prompt，
自动解析其任务规则，
为每条规则生成目标驱动的用户模拟器，
通过多轮交互评估 Agent 是否遵守规则。

这可以实现更通用、更可解释的 Agent 自动评测能力。