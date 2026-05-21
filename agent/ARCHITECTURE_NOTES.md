# 项目架构笔记

> 更新时间：2026-05-21
>
> 这份笔记按当前代码实现整理。历史上的“订单驱动”链路已经移除，当前主路径是
> `instruction -> SimulationPlan -> session -> evaluation -> memory/report`。
> 如果 README、旧注释或历史 session 与本文冲突，以 `app.py` 和 `src/*.py` 为准。

## 1. 当前系统到底在做什么

这是一个面向任务型对话 Agent 的自动评测框架。

输入是一段 system prompt / task instruction，系统会：

1. 自动拆出可评测的原子规则
2. 归纳被测 Agent 的职责结构
3. 识别 prompt 中的占位符并生成多组填充值
4. 为每条规则生成目标驱动的测试 case
5. 运行“被测 Agent vs 用户模拟器”对话
6. 用确定性 checker 或 LLM Judge 对结果打分
7. 把 transcript、评测结果和聚合报告落盘

当前最重要的边界仍然成立：

- 被测 Agent 只看到原始 instruction，或占位符填值后的 instruction
- 解析出来的规则只给测试计划生成器和评测器使用
- 不把评测规则直接喂给被测 Agent，避免“对着答案答题”

## 2. 一张图看主链路

```text
instruction
  -> build_simulation_plan(...)
     -> parse_rules(...)
     -> validate_rules(...)
     -> auto_fix_rules(...)          # 只有 error 时才触发
     -> parse_agent_spec(...)
     -> extract_placeholders(...)
     -> build_test_cases(...)
  -> SimulationPlan
     -> sub_plans                    # 每组占位符填值一个 SubPlan
     -> test_cases                   # 每条规则派生出的 SimulationCase
  -> run session
     -> make_outbound_agent(filled_instruction)
     -> UserSimulator(case, agent_spec, scenario_context)
     -> OutboundSession.start()/reply()
  -> SessionArchive
  -> evaluate_session(...)
     -> deterministic_checks / LLM judge
  -> EvaluationReport
  -> sessions/*.json + memory/evaluation_memory.jsonl + reports/report-*.json
```

当前真正的执行单元是：

`SubPlan × SimulationCase`

而不是历史上的 `order × persona`。

## 3. 分层理解

### 3.1 运行时封装层

`agent/agents.py` 是当前运行时的关键底座。

虽然依赖里有 `openai-agents`，但代码实际导入的是项目内这个同名模块。它只模拟了一个很小的接口面：

- `Agent`
- `Runner.run_sync(...)`
- `trace(...)`（当前是 no-op）

这个封装负责：

- 把 `Agent.instructions` 包成 system prompt
- 通过 `openai.OpenAI().chat.completions.create(...)` 发请求
- 对结构化输出做 JSON 解析和自动修复重试
- 在对话类输出上做 plain-text -> `{reply_text, should_end}` 的兜底转换
- 读取 `DEEPSEEK_* / OPENAI_* / LLM_*` 环境变量

这意味着当前项目并没有真正依赖 `openai-agents` 的 tracing、tool calling 或多 agent runtime；它只复用了一个接近的 API 形状。

### 3.2 测试计划生成层

核心文件：

- `src/simulation_plan.py`
- `src/rule_parser.py`
- `src/rule_validation.py`
- `src/agent_spec.py`
- `src/placeholders.py`
- `src/scenario_selector.py`
- `src/test_case_generator.py`

职责拆分：

- `rule_parser.py`
  - 把 instruction 解析成结构化 `Rule`
  - 规则字段不只包含 `description`，还包含 `trigger_condition`、`expected_behavior`、`failure_criteria`、`checks`
- `rule_validation.py`
  - 对 LLM 产出的规则做体检
  - `error` 会触发自动修复，`warning` 只提示
- `agent_spec.py`
  - 把 prompt 归纳成统一 `AgentSpec`
- `placeholders.py`
  - 先用正则做占位符发现
  - 再让 LLM 补语义和生成多组 `PlaceholderSet`
  - 当前“源码里实际出现的占位符”是权威来源，用来压制 LLM 幻觉
- `scenario_selector.py`
  - 为每条规则选“最少但足够”的 case 类型和主画像
- `test_case_generator.py`
  - 把 `Rule + AgentSpec` 变成可执行 `SimulationCase`

### 3.3 对话执行层

核心文件：

- `src/agent.py`
- `src/simulator.py`
- `src/session.py`

职责拆分：

- `agent.py`
  - 定义被测 Agent 的结构化输出类型 `AgentTurnOutput`
  - `make_outbound_agent(instructions)` 会在业务 instruction 后追加输出协议
- `simulator.py`
  - 用 `SimulationCase` 驱动用户模拟器
  - 它不是“泛聊天 persona”，而是“围绕某条目标规则出招的测试执行器”
- `session.py`
  - 封装一次真实对话
  - 维护 transcript 和模型输入 history
  - 提供 `start()`、`reply()`、`record_user()`、`save_archive()`、`get_archive()`

### 3.4 评估层

核心文件：

- `src/evaluator.py`
- `src/deterministic_checks.py`
- `src/rules.py`

职责拆分：

- `rules.py`
  - 定义 `Rule` 契约和严重度权重
  - `RULES` 现在是空列表，只给旧链路兜底
- `deterministic_checks.py`
  - 提供可代码判断的 rule checks：
    - `max_chars`
    - `forbidden_keywords`
    - `required_opening`
    - `closing_phrase`
    - `pii_leak`
- `evaluator.py`
  - 对有 `checks` 的规则走确定性判断
  - 其余规则走 LLM Judge
  - 对同一条规则可做多次采样投票
  - 额外产出 `compute_coverage(...)` 统计 conditional 规则触发情况

### 3.5 展示与持久化层

核心文件：

- `app.py`
- `src/memory.py`
- `auto_main.py`
- `evaluate_main.py`
- `main.py`

其中：

- `app.py` 是当前最完整、最可信的主入口
- `src/memory.py` 负责把评测结果追加写入 `memory/evaluation_memory.jsonl`
- `auto_main.py` 负责批量生成 session
- `evaluate_main.py` 负责离线重评 session
- `main.py` / `src/index.py` 是手动 CLI

## 4. 目录地图

```text
agent/
├── app.py                     # Streamlit UI + 执行编排 + 仪表板
├── auto_main.py               # 批量生成 session
├── evaluate_main.py           # 批量离线评估 session
├── main.py                    # 手动 CLI 入口（转发到 src/index.py）
├── agents.py                  # 本地 LLM runtime shim
├── instructions/              # 示例 instruction JSON
├── src/
│   ├── agent.py               # 被测 Agent 封装
│   ├── agent_spec.py          # instruction -> AgentSpec
│   ├── deterministic_checks.py# 确定性评估器
│   ├── evaluator.py           # 规则评估 + coverage
│   ├── index.py               # 手动 CLI 主流程
│   ├── memory.py              # evaluation_memory.jsonl 写入
│   ├── persona.py             # Persona 类型定义；主流程里主要用于筛选
│   ├── placeholders.py        # 占位符发现、归一化、填值
│   ├── rule_parser.py         # instruction -> Rule[]
│   ├── rule_validation.py     # rule 质量校验与自动修复
│   ├── rules.py               # Rule 契约与 severity weight
│   ├── scenario_selector.py   # 每条 rule 选最少测试场景
│   ├── session.py             # 单次对话执行与归档
│   ├── simulation_plan.py     # 计划装配总入口
│   ├── simulator.py           # 用户模拟器
│   ├── test_case_generator.py # Rule -> SimulationCase
│   └── types.py               # SessionArchive 等共用 dataclass
├── test_agents.py
├── test_evaluator.py
├── test_memory.py
├── test_parser.py             # 需要真实 API key 的半手工脚本
├── test_placeholders.py
├── test_simulation_cases.py
└── ARCHITECTURE_NOTES.md
```

运行后通常还会出现这些目录：

- `sessions/`
- `reports/`
- `memory/`

它们是运行产物，不是当前仓库的静态代码结构核心。

## 5. 核心数据对象

### `Rule`

定义在 `src/rules.py`。

关键字段：

- `rule_id`
- `description`
- `rule_type`：`required | conditional | forbidden`
- `severity`：`critical | major | minor`
- `trigger_condition`
- `expected_behavior`
- `failure_criteria`
- `evidence_requirement`
- `checks`

`checks` 是否为空，决定这条规则是走代码评估还是 LLM Judge。

### `AgentSpec`

定义在 `src/agent_spec.py`。

关键字段：

- `agent_type`
- `domain`
- `main_task`
- `workflow_rules`
- `condition_rules`
- `prohibited_rules`
- `style_rules`
- `required_information`
- `termination_conditions`

### `Placeholder` / `PlaceholderSet`

定义在 `src/placeholders.py`。

- `Placeholder` 描述单个占位符
- `PlaceholderSet` 描述一组场景化填值

常见占位符语法：

- `${name}`
- `**X**` / `**X 单**`
- `[name]`
- `{{name}}`
- `<name>`

### `SimulationCase`

定义在 `src/test_case_generator.py`。

这是当前“用户模拟任务”的核心对象，而不是旧版 `UserPersona`。

关键字段：

- `test_id`
- `case_type`
- `user_profile`
- `test_goal`
- `trigger_strategy`

其中 `case_type` 可能是：

- `normal_trigger`
- `ambiguous_trigger`
- `strong_trigger`
- `adversarial_induction`
- `boundary`

### `SubPlan` / `SimulationPlan`

定义在 `src/simulation_plan.py`。

- `SimulationPlan` 是对一段 instruction 的完整测试计划
- `SubPlan` 是某一组占位符填值展开后的子计划

关键字段：

- `parsed_rules`
- `agent_spec`
- `placeholders`
- `sub_plans`
- `validation_issues`

每个 `SubPlan` 包含：

- `set_id`
- `label`
- `scenario_hint`
- `placeholder_values`
- `filled_instruction`
- `test_cases`

### `SessionArchive`

定义在 `src/types.py`。

当前 archive 已经能保存足够多的重评上下文：

- `session_id`
- `source_label`
- `instruction_snapshot`
- `scenario_context`
- `persona_type`
- `case_type`
- `simulator_label`
- `test_case_id`
- `target_rule_id`
- `target_rule_type`
- `target_rule_description`
- `target_rule_evaluation_hint`
- `target_rule_severity`
- `set_id`
- `set_label`
- `transcript`

### `RuleResult` / `EvaluationReport`

定义在 `src/evaluator.py`。

`RuleResult.result` 目前有四种值：

- `pass`
- `fail`
- `not_applicable`
- `trigger_failed`

其中 `trigger_failed` 只用于“这次 session 本来就是为了触发某个 conditional 规则，但模拟器没把条件演出来”的情况。

## 6. 测试计划是怎么生成的

`build_simulation_plan(...)` 当前固定走 6 步：

1. `parse_rules(instruction)`
2. `validate_rules(parsed_rules)`
3. `auto_fix_rules(...)`（仅当存在 error）
4. `parse_agent_spec(instruction)`
5. `extract_placeholders(instruction, agent_spec, num_sets=...)`
6. `build_test_cases(agent_spec, parsed_rules)`

几个关键事实：

- 规则只对原始 instruction 解析一次
- `AgentSpec` 只归纳一次
- `SimulationCase` 只生成一次
- placeholder set 只影响 `filled_instruction`
- 不会对每个 `SubPlan` 重新拆规则或重新生成 case

所以 placeholder set 的作用是：

`同一份测试计划下的参数化 prompt 变体`

而不是“每组参数一套新规则”。

## 7. 占位符链路的真实实现

`src/placeholders.py` 不是纯 LLM 方案，而是“规则发现 + LLM 补全”的混合实现。

### 7.1 先做确定性发现

代码先用正则扫描 instruction 中真实出现的占位符模式，得到 `_PlaceholderOccurrence`。

这一层的作用是：

- 决定有哪些 identifier 是可信的
- 避免 LLM 凭空脑补不存在的变量
- 保留原文语法形态供后续替换

### 7.2 再让 LLM 补语义和生成值

之后才把“已确定占位符清单”发给 LLM，让它：

- 补 `semantic`
- 推断 `value_type`
- 生成 `set_1 ~ set_n`

### 7.3 最后再归一化

`_normalize_extraction(...)` 会再次约束：

- 去重
- 修正 identifier
- 删除源码里不存在的 hallucinated placeholder
- 确保每个 set 都覆盖全部 identifier
- 无占位符时补 `set_default`

这是目前代码里一个很重要的稳态设计。

## 8. Session 是怎么跑的

### 8.1 被测 Agent

`make_outbound_agent(instructions)` 会把业务 instruction 与统一输出协议拼接，要求模型始终返回：

- `reply_text`
- `should_end`
- `end_reason`

默认模型当前写死为 `deepseek-v4-flash`。

### 8.2 用户模拟器

`UserSimulator` 初始化时会拿到：

- `SimulationCase`
- `AgentSpec`
- `scenario_context`
- `session_id`

它的 prompt 会显式包含：

- 当前规则目标
- 用户画像
- 触发策略
- failure criteria
- 当前 placeholder 填值上下文

因此它的目标不是“像某类 persona 一样随便聊”，而是“自然地把目标场景演出来”。

### 8.3 会话控制

`OutboundSession` 当前行为：

- `start()` 先给被测 Agent 一个固定开场提示
- `reply(user_text)` 追加用户发言并驱动下一轮模型输出
- `record_user(user_text)` 用于“用户说完最后一句就挂断”
- `save_archive(...)` / `get_archive(...)` 负责产出 `SessionArchive`

当前 session 内部没有注入任何隐藏业务对象或订单上下文，核心上下文就是：

- `filled_instruction`
- 外显对话历史

## 9. 评估是怎么做的

### 9.1 Hybrid evaluator

`evaluate_session(...)` 会把规则分成两类：

- `rule.checks` 非空 -> 确定性检查
- `rule.checks` 为空 -> LLM Judge

这就是当前的 hybrid evaluator。

### 9.2 LLM Judge

Judge 输出的结构包括：

- `triggered`
- `trigger_turn`
- `response_turn`
- `compliant`
- `evidence`
- `rationale`
- `matched_failure_criteria`
- `suggestion`

对同一规则可采样 `n_samples` 次，然后用 `_aggregate_votes(...)` 做多数投票。

### 9.3 结果标签与打分

当前打分逻辑：

- `pass` / `fail` 进入分母
- `not_applicable` / `trigger_failed` 不进入分母

严重度权重：

- `critical = 3`
- `major = 2`
- `minor = 1`

### 9.4 Streamlit 当前评估的是“目标规则”

这是一个很关键的实现细节。

在 `app.py` 中，每个自动生成的 session 只针对一个 `SimulationCase` 运行，并且评估时传入：

`rules=[target_rule]`

所以：

- 单个 session 的 `EvaluationReport` 通常只包含一条规则结果
- 页面上的平均分，本质上是“目标规则通过率”的聚合
- `compute_coverage(...)` 再把这些单规则 session 拼回全局 coverage 视角

这和“每个 session 跑完整规则集”的评估口径不同。

## 10. 四个入口分别做什么

### `app.py`

当前最完整的主流程。

它负责：

- 读取页面输入的 instruction
- 生成 `SimulationPlan`
- 用侧边栏 persona 勾选结果过滤 `SimulationCase.profile_type`
- 按侧边栏并行度执行所有 `SubPlan × SimulationCase`
- 归档 session
- 评估目标规则
- 追加 memory
- 生成可视化仪表板

另外两个重要事实：

- `app.py` 既是 UI 层，也是执行编排层，没有单独的 service/orchestrator 模块
- `app.py` 通过 `ThreadPoolExecutor` 支持 1 / 2 / 4 / 8 / 16 路并行，每个 worker 跑完整的“对话 + 目标规则评估”
- Streamlit UI 更新和 `memory/evaluation_memory.jsonl` 写入仍在主线程完成

### `auto_main.py`

批量 session 生成器。

它支持：

- 默认 instruction
- `instructions/*.json`
- 直接传入一段 inline instruction

流程是：

1. 读取 instruction
2. `build_simulation_plan(...)`
3. 遍历所有 `sub_plan.test_cases`
4. 跑 session
5. 写 `sessions/*.json`

它不负责汇总和报表。

### `evaluate_main.py`

离线重评入口。

流程是：

1. 读取 `sessions/*.json`
2. 反序列化为 `SessionArchive`
3. 优先从 archive 里恢复 `target_rule_*` 字段构造 `Rule`
4. 调 `evaluate_session(...)`
5. 追加 memory
6. 生成 `reports/report-*.json`

注意点：

- 对当前新生成的 archive，它通常能恢复“这次 session 的目标规则”
- 如果 archive 是老格式，缺少 `target_rule_*`，才会退回 `RULES`
- 而当前 `RULES` 是空列表，所以旧归档在这条链路下可能评估不到任何规则

### `main.py` / `src/index.py`

手动 CLI。

它会：

- 读取 instruction
- 启动 `OutboundSession`
- 人工输入用户回复
- 在 `/quit`、EOF 或 Agent 自然结束时归档

这条链路当前仍可用，但不包含自动测试计划和自动评估。

## 11. 当前持久化格式

### `instructions/*.json`

这是示例输入，不是运行时必须格式。

当前样本包含：

- `id`
- `scenario`
- `instruction`
- `success_criteria`
- `failure_criteria`

其中 `success_criteria` / `failure_criteria` 更像人工参考。主流程并不会直接读取它们来打分，而是重新对 `instruction` 调 `parse_rules(...)`。

### `sessions/*.json`

保存一次真实对话的 archive。

对当前代码生成的新 archive，已经会带：

- `instruction_snapshot`
- `scenario_context`
- `target_rule_*`
- `set_id` / `set_label`

所以它比旧文档里描述的格式完整得多。

### `memory/evaluation_memory.jsonl`

这是最完整的逐次评测明细。

每条记录除了 archive 基本信息，还会保存：

- `score`
- `mean_confidence`
- `has_violation`
- `violation_count`
- `trigger_failed_count`
- `judge_results`
- `all_samples`
- `judge_model`
- `judge_prompt`

如果要做事后分析，memory 通常比 session archive 更适合作为事实来源。

### `reports/report-*.json`

这是 `evaluate_main.py` 产出的聚合报告，包含：

- 总 session 数
- overall score
- by_persona 平均分
- rule fail rate
- 每个 session 的规则详情

## 12. Persona 在当前代码中的真实角色

`src/persona.py` 里仍然保留了 7 个 `UserPersona` 常量，但它们不再直接驱动主流程里的用户模拟器。

当前它的作用主要有两个：

1. 作为 persona 类型枚举与展示标签来源
2. 给 Streamlit 侧边栏提供勾选项

真正控制模拟器行为的是 `SimulationCase.user_profile`，它由 `test_case_generator.py` 生成。

所以“选 persona”在现在的 UI 里本质上是：

`筛掉哪些自动生成 case 要执行`

而不是“把这个 persona prompt 直接送进模拟器”。

## 13. 测试面

当前离线可跑的单测主要覆盖：

- `test_agents.py`
  - 结构化输出修复
  - plain-text 对话输出兜底
- `test_placeholders.py`
  - 占位符替换与归一化
  - 反 hallucination
- `test_simulation_cases.py`
  - rule -> case 的映射逻辑
- `test_evaluator.py`
  - 多次 judge 投票聚合
- `test_memory.py`
  - memory 序列化

`test_parser.py` 不是纯单测，它会真实调用模型，更多是半手工验证脚本。

## 14. 当前限制与注意事项

### 14.1 `app.py` 很重

`app.py` 接近 2000 行，同时承担：

- 样式
- 输入控制
- 执行编排
- 实时状态管理
- 指标计算
- 可视化渲染

它是当前最权威的主入口，但也意味着 UI 和 orchestration 还没有拆层。

### 14.2 整条链路高度依赖 LLM

以下步骤都依赖模型：

- `parse_rules`
- `auto_fix_rules`
- `parse_agent_spec`
- `extract_placeholders`
- `scenario_selector`
- `UserSimulator`
- `LLM Judge`

虽然 `agents.py` 做了重试和 JSON 修复，但根本波动仍来自模型与网络。

### 14.3 Streamlit 支持 case 级并行

Streamlit 主流程现在可以按用户选择的并行度执行测试 case：

- 可选并行度：1 / 2 / 4 / 8 / 16
- 并行单位：`SubPlan × SimulationCase`
- 每个 worker 内部完整执行对话生成、session 归档、目标规则评估
- 主线程负责收集结果、写 memory、更新页面

`evaluate_session(...)` 内部仍是顺序评估传入的规则和 Judge 采样；当前并行发生在 case 层，不发生在单个 case 内部。

性能基本随这个量级线性增长：

`ceil((placeholder_set 数 × 测试 case 数) / 并行度) × judge 采样数`

并行度越高越容易触发模型接口限流或连接抖动，真实最佳值取决于 API 配额和网络稳定性。

### 14.4 placeholder set 不会改变规则集

这一点再强调一次：

- 改的是 `filled_instruction`
- 不改 `parsed_rules`
- 不改 `test_cases`

如果以后出现“规则本身依赖参数值变化”的需求，需要重构 `SimulationPlan` 生成边界。

### 14.5 离线重放仍不是完整复现

虽然 archive 里已经保存了目标规则元数据和 instruction 快照，但离线链路仍没有持久化整个 `SimulationPlan`。

现在能稳定重放的是：

- 这次 session 的 transcript
- 这次 session 的目标规则评估

还不能完整重放的是：

- 当时全部 `parsed_rules`
- 全量 `sub_plans`
- 全部 `SimulationCase`

### 14.6 本地 `agents.py` 是一个重要隐含前提

如果后续有人以为这里真的在用 `openai-agents` 全家桶，很多行为会判断错。

当前项目真正依赖的是：

- `openai` SDK
- 本地 `agents.py` 里的兼容接口

这件事需要一直记住。

## 15. 建议阅读顺序

如果要重新进入代码，按这个顺序最快：

1. `README.md`
2. `agents.py`
3. `src/types.py`
4. `src/rules.py`
5. `src/simulation_plan.py`
6. `src/rule_parser.py`
7. `src/rule_validation.py`
8. `src/placeholders.py`
9. `src/agent_spec.py`
10. `src/scenario_selector.py`
11. `src/test_case_generator.py`
12. `src/agent.py`
13. `src/simulator.py`
14. `src/session.py`
15. `src/evaluator.py`
16. `app.py`

## 16. 后续改动时建议守住的边界

- 不要让被测 Agent 直接看到解析后的规则
- 保持 `Rule` 结构化字段完整，不要退回只靠一句 `description`
- 占位符发现继续以源码扫描为权威，避免纯 LLM 识别
- 如果扩展 deterministic checker，优先把可代码化规则塞进 `checks`
- 如果要做完整离线复现，优先持久化整个 `SimulationPlan`
