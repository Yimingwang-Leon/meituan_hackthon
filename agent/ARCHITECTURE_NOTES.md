# 项目框架笔记

> 更新（2026-05-20）：
> 当前代码已经完成一轮“去订单化”重构。`orders/` 与 `src/orders.py` 已移除，执行主维度从
> `订单 × placeholder_set × SimulationCase`
> 改为
> `placeholder_set × SimulationCase`。
> 旧笔记里凡是提到 `LoadedOrder / OrderRecord / order_id / 订单选择` 的部分，现都应理解为历史设计，不再代表当前实现。

## 1. 项目定位

这是一个“对话 Agent 自动评测框架”，当前业务样例是“美团外呼数字人”，但现在的实现已经不只是固定场景脚本，而是开始支持**从任意 system prompt / task instruction 自动生成测试计划**。

当前最准确的主链路是：

`system prompt -> SimulationPlan -> 对话执行 -> evaluator -> report / memory`

其中 `SimulationPlan` 内部又包含一层更细的生成链路：

`instruction -> parse_rules -> parse_agent_spec -> extract_placeholders -> build_test_cases -> sub_plans`

也就是说，这个项目现在的核心已经不是“写死几类 Persona 去聊一聊”，而是：

1. 从 prompt 自动拆出可评测规则
2. 从 prompt 自动归纳被测 Agent 的职责结构
3. 从 prompt 自动识别占位符并生成多组测试值
4. 从规则自动派生目标导向的测试用例
5. 运行多轮对话并用 LLM judge 逐条打分

一个始终保留的关键边界是：**被测 Agent 只拿业务指令，不拿拆出来的评测规则**。规则只服务测试计划生成和评估，避免“按答案作答”。

## 2. 技术栈与运行形态

- 语言：Python
- LLM Agent 框架：`openai-agents`
- UI：`Streamlit`
- 配置：`python-dotenv`
- 数据建模：`dataclasses` + `pydantic`
- 持久化：本地 JSON / JSONL（`sessions/`、`reports/`、`memory/`）

模型调用目前基本都写死在代码里，主要使用 `gpt-5.4-nano`：

- 规则拆解：`src/rule_parser.py`
- Agent 结构归纳：`src/agent_spec.py`
- 占位符提取：`src/placeholders.py`
- 被测数字人：`src/agent.py`
- 用户模拟器：`src/simulator.py`
- 规则评判器：`src/evaluator.py`

当前运行形态主要有 4 条：

1. Streamlit 实时评测
2. `auto_main.py` 批量生成会话
3. `evaluate_main.py` 批量离线评测
4. `main.py` / `src/index.py` 手动 CLI 对话

## 3. 主要入口

### 推荐入口

- `app.py`
  - Streamlit 页面
  - 支持直接粘贴任意 system prompt / task instruction
  - 调 `build_simulation_plan(...)` 生成完整测试计划
  - 支持占位符场景组、多规则自动测试用例、实时评测、可视化
  - 当前最可信的主流程

### 批量入口

- `auto_main.py`
  - 批量跑订单
  - 根据 `order.scenario` 读取 `instructions/*.json`
  - 对每份 instruction 构建并缓存 `SimulationPlan`
  - 遍历 `sub_plans × test_cases × orders`
  - 只负责生成和归档会话，不做汇总可视化

- `evaluate_main.py`
  - 读取 `sessions/*.json`
  - 反序列化为 `SessionArchive`
  - 批量调用 `evaluate_session(...)`
  - 生成 `reports/report-*.json`
  - 追加写入 `memory/evaluation_memory.jsonl`

### 手动入口

- `main.py`
  - 转发到 `src/index.py`
  - 是人工逐轮输入的 CLI 入口
  - 当前看起来比较陈旧，见下面“注意事项”

## 4. 目录理解

```text
agent/
├── app.py                  # Streamlit 主入口，当前最完整主流程
├── auto_main.py            # 批量生成会话
├── evaluate_main.py        # 批量评估已归档 session
├── main.py                 # 手动 CLI 入口
├── instructions/           # 场景级 instruction 样本
├── memory/                 # 单次评测记忆（JSONL）
├── orders/                 # 待测订单样本
├── reports/                # 评估报告输出
├── sessions/               # 会话归档输出
└── src/
    ├── agent.py            # 被测数字人 Agent
    ├── agent_spec.py       # 从 prompt 归纳 AgentSpec
    ├── evaluator.py        # 规则评估器
    ├── index.py            # 手动 CLI 流程
    ├── memory.py           # 评测记忆落盘
    ├── orders.py           # 订单加载
    ├── persona.py          # Persona 类型定义与旧画像预设
    ├── placeholders.py     # 占位符提取、填充、校验
    ├── rule_parser.py      # 指令拆规则
    ├── rules.py            # Rule 结构、静态 RULES、权重
    ├── session.py          # 单会话编排与归档
    ├── simulation_plan.py  # 测试计划总装配
    ├── simulator.py        # 用户模拟器
    ├── test_case_generator.py # 从规则自动生成测试用例
    ├── types.py            # 共用数据类型
    └── __init__.py
```

## 5. 实际执行链路

### 5.1 总体视角

如果只记一件事，现在最应该记的是：

`app.py / auto_main.py` 不再直接把“订单 × 固定 Persona”送进对话，而是先把 instruction 变成一个 `SimulationPlan`，再从 plan 里展开真正要跑的任务。

更接近实际的执行单元是：

`订单 × placeholder_set × SimulationCase`

其中：

- `placeholder_set` 表示一组占位符填值后的 prompt 场景
- `SimulationCase` 表示针对某条规则自动生成的一条目标导向测试用例

### 5.2 Streamlit 链路

`app.py` 的实际流程是：

1. 从页面文本框拿到 `instructions`
2. 调 `build_simulation_plan(instructions, num_sets=...)`
3. 这一步内部会依次做：
   - `parse_rules(instructions)`
   - `parse_agent_spec(instructions)`
   - `extract_placeholders(instructions, agent_spec, num_sets=...)`
   - `build_test_cases(agent_spec, parsed_rules)`
4. 拿到 `SimulationPlan` 后，读取：
   - `parsed_rules`
   - `agent_spec`
   - `placeholders`
   - `sub_plans`
5. 根据侧边栏勾选的 Persona 类型做过滤
   - 这里的 `selected_personas` 现在本质上是**测试用例筛选器**
   - 不是直接把 `UserPersona` 对象送进模拟器
6. 将 `sub_plans × test_cases` 展平为待执行任务
7. 每个任务里创建：
   - `make_outbound_agent(sub_plan.filled_instruction)`
   - `OutboundSession(order, agent=...)`
   - `UserSimulator(order, case, agent_spec)`
8. 先 `outbound.start()`，让数字人主动开场
9. 进入最多 `MAX_TURNS=15` 轮对话
10. 对话结束后：
   - `save_archive(...)`
   - `get_archive(...)`
   - `evaluate_session(archive, ..., rules=parsed_rules, set_id=..., set_label=...)`
   - `append_evaluation_memory(...)`
11. 把结果写进 `st.session_state`
12. 页面展示统计卡片、热力图、覆盖率、单会话卡片

这里的评估规则是**运行时动态解析出的 `parsed_rules`**，不是静态常量。

### 5.3 当 system prompt 里有占位符时，怎么处理

当前实现会走“多场景 prompt 展开”流程。

#### 第一步：识别占位符

`src/placeholders.py` 会尝试识别这些形式：

- `${name}`
- `**X**`、`**X 单**`、`**Y 天**`
- `[name]`
- `{{name}}`
- `<name>`

每个占位符会被抽成一个 `Placeholder`，包含：

- `raw_pattern`
- `identifier`
- `semantic`
- `value_type`
- `unit`
- `confidence`

#### 第二步：生成多组测试值

提取器会按 `num_sets` 生成多组 `PlaceholderSet`：

- `set_1`：标准场景
- `set_2`：边界场景
- `set_3`：高压场景

每组场景都要给全部占位符填值。

#### 第三步：构造 `SubPlan`

`build_simulation_plan(...)` 会把每个 `PlaceholderSet` 变成一个 `SubPlan`，其中包含：

- `set_id`
- `label`
- `scenario_hint`
- `placeholder_values`
- `filled_instruction`
- `test_cases`

这里有一个很重要的现状：

- `parsed_rules` 只对**原始 instruction**解析一次
- `agent_spec` 也只归纳一次
- `test_cases` 也只生成一次
- 不会针对每组填值后的 prompt 重新拆规则或重新生成 case

所以当前 placeholder set 的作用是：**改变被测 Agent 实际拿到的 prompt 内容**，而不是改变规则集或 case 集。

#### 第四步：运行时展开会话

真正执行时，会按每个 `sub_plan` 单独创建：

- 一份填充后的 outbound agent prompt
- 一批复用的 `SimulationCase`

因此总会话数会被放大为：

`订单数 × placeholder_set 数 × 命中的 test_case 数`

### 5.4 当 system prompt 里没有占位符时，怎么处理

当前实现不会走特殊分支，而是退化成统一流程里的“单场景”版本。

处理方式是：

1. `extract_placeholders(...)` 返回：
   - `placeholders = []`
   - 通常会带一组默认 `set_default`
2. 即使提取器没返回 `sets`，`build_simulation_plan(...)` 里也有兜底，会自动补一个默认场景组
3. 这个默认 `SubPlan` 的特征是：
   - `placeholder_values = {}`
   - `filled_instruction == original_instruction`
4. 后续仍然走同一条链路：
   - 规则照常解析
   - AgentSpec 照常归纳
   - 测试用例照常生成
   - 只是 `sub_plans` 只有 1 组

所以“无占位符 prompt”的实际效果可以理解为：

`单一 instruction + 单一 sub_plan + 多条规则对应的自动测试用例`

也就是说，占位符只是对测试计划做“横向场景扩展”，不是整个框架成立的前提。

### 5.5 批量 CLI 链路

`auto_main.py` 的流程现在是：

1. 读取订单
2. 根据 `order.scenario` 从 `instructions/*.json` 找对应 instruction
3. 对同一份 instruction 做 `plan_cache`
4. 首次遇到某份 instruction 时，调用 `build_simulation_plan(...)`
5. 遍历：
   - `plan.sub_plans`
   - `sub_plan.test_cases`
   - 当前订单
6. 对每个任务创建：
   - `make_outbound_agent(sub_plan.filled_instruction)`
   - `OutboundSession(...)`
   - `UserSimulator(order, simulation_case, plan.agent_spec)`
7. 生成 transcript 并写入 `sessions/`

这条链路现在更像“离线批量生成测试会话”。

### 5.6 批量评估链路

`evaluate_main.py` 的流程是：

1. 读取 `sessions/*.json`
2. 反序列化为 `SessionArchive`
3. 对每个 session 调 `evaluate_session(...)`
4. 追加写入 `memory/evaluation_memory.jsonl`
5. 打印汇总
6. 生成 `reports/report-*.json`

但这里有一个关键限制：

- 它现在没有从 session 恢复运行时的 `parsed_rules`
- 也没有恢复 `set_id` / `set_label`
- 默认会退回 `src/rules.py` 里的静态 `RULES`

所以它只能算“离线重评近似版”，还不能完整复现 Streamlit 当次运行时的评估口径。

## 6. 核心模块职责

### `src/simulation_plan.py`

- 当前测试计划总装配器
- `build_simulation_plan(...)` 负责把一段 instruction 变成统一的 `SimulationPlan`
- 输出里同时包含：
  - `parsed_rules`
  - `agent_spec`
  - `placeholders`
  - `sub_plans`

这是新增架构里的核心中枢。

### `src/rule_parser.py`

- 用 LLM 把 instruction 拆成 `ParsedRuleList`
- 输出要求比较严格：
  - 原子化
  - 6 到 14 条
  - 三种类型：`required` / `conditional` / `forbidden`
  - 必须补 `evaluation_hint`
  - 必须给 `severity`

这部分是整个评测的“规则生成层”。

### `src/agent_spec.py`

- 用 LLM 把任意对话 Agent prompt 归纳成 `AgentSpec`
- 目前主要提取：
  - `agent_type`
  - `domain`
  - `main_task`
  - `workflow_rules`
  - `condition_rules`
  - `prohibited_rules`
  - `style_rules`
  - `required_information`
  - `termination_conditions`

这部分是“把具体 prompt 抽象成统一测试语义”的一层。

### `src/placeholders.py`

- 负责识别 instruction 里的占位符
- 负责为占位符生成多组场景化测试值
- 提供：
  - `extract_placeholders(...)`
  - `fill_placeholders(...)`
  - `validate_placeholder_value(...)`

这部分让测试计划可以从“单 prompt”扩成“多组参数化 prompt”。

### `src/test_case_generator.py`

- 负责从 `AgentSpec + list[Rule]` 自动生成 `SimulationCase`
- 每条规则会被包装成一条目标导向测试用例
- 自动推导内容包括：
  - 用户画像类型
  - 测试目标
  - 触发条件
  - 预期 Agent 行为
  - 失败标准
  - 触发话术
  - 跟进策略

这是这轮新增内容里最关键的一层：**测试案例不再主要靠手工穷举，而是从规则自动派生。**

### `src/agent.py`

- 定义被测数字人的默认指令 `OUTBOUND_INSTRUCTIONS`
- `make_outbound_agent(instructions)` 会在业务指令后追加统一的结构化输出约束
- 返回的 Agent 输出是 `AgentTurnOutput`

注意：当前 `session` 不再往历史里补订单上下文；被测 Agent 看到的核心上下文，默认来自传入的完整 instruction。

### `src/session.py`

- `OutboundSession` 负责真实对话编排
- `start()` 会额外给模型一个“电话已接通”的开场提示
- `reply()` 记录用户输入并驱动下一轮 Agent 输出
- `record_user()` 用于“用户说完最后一句就挂断”的场景
- `save_archive()` / `get_archive()` 负责生成归档

当前一个很重要的实现变化是：

- `_history` 默认从空列表开始
- **不再**在内部自动注入订单上下文
- 归档里的 `transcript` 只保存外显对话

### `src/simulator.py`

- `UserSimulator` 也是一个 LLM agent
- 但它现在不是“泛 Persona 聊天器”，而是**目标驱动测试执行器**
- 初始化输入是：
  - `loaded_order`
  - `simulation_case`
  - `agent_spec`
- prompt 里会明确带入：
  - 当前规则目标
  - 触发策略
  - 失败标准
  - 用户画像

这意味着模拟器当前是围绕“验证某条规则”来出招，而不是单纯扮演某个脾气类型。

### `src/evaluator.py`

- `evaluate_session(...)` 会把 transcript 格式化成“第 N 轮 数字人/用户：...”
- 对每条规则跑 N 次 judge
- 多数投票得到最终 `RuleResult`
- `confidence = 最高票数 / 采样数`
- `not_applicable` 不参与总分分母
- `compute_coverage(...)` 用于统计 conditional 规则是否被触发

当前实现是顺序跑 judge，不走并发。

### `src/memory.py`

- 负责把单次评测结果追加写入 `memory/evaluation_memory.jsonl`
- 会保存：
  - session 基本信息
  - transcript
  - `simulator_label`
  - `test_case_id`
  - `target_rule_id`
  - `set_id` / `set_label`
  - 每条规则的 `judger_result`
  - `is_violation`
  - `confidence`
  - `votes`

### `src/orders.py`

- 把 `orders/*.json` 解析成 `LoadedOrder`
- 字段命名会从 JSON 的 camelCase 转成 dataclass 的 snake_case

### `src/persona.py`

- 仍然定义了 7 种 Persona 类型：
  - `cooperative`
  - `suspicious`
  - `impatient`
  - `ambiguous`
  - `info_missing`
  - `rejector`
  - `hostile`
- 但当前主流程里，它更像：
  - Persona 类型枚举来源
  - UI 里的筛选项来源
  - 旧版画像预设保留

现在的 `app.py` / `auto_main.py` 不再直接把 `UserPersona` 塞进 `UserSimulator`；真正驱动模拟器的是 `SimulationCase`。

### `src/rules.py`

- 定义 `Rule` 数据结构
- 提供静态 `RULES`
- 提供 `SEVERITY_WEIGHTS`

这里的 `RULES` 现在主要是离线评估兜底，不是推荐主路径。

### `src/types.py`

- 定义共用 dataclass：
  - `OrderRecord`
  - `LoadedOrder`
  - `TranscriptEntry`
  - `SessionArchive`
  - `TurnResult`

## 7. 关键数据结构

### 规则：`Rule`

关键字段：

- `rule_id`
- `description`
- `rule_type`
- `evaluation_hint`
- `severity`

### Agent 规格：`AgentSpec`

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

### 占位符：`Placeholder` / `PlaceholderSet`

- `Placeholder` 描述单个占位符的语义和类型
- `PlaceholderSet` 描述一组测试填值

关键字段：

- `raw_pattern`
- `identifier`
- `semantic`
- `value_type`
- `set_id`
- `label`
- `scenario_hint`
- `values`

### 测试用例：`SimulationCase`

它不是普通 Persona，而是一条针对单规则的测试任务。

关键字段：

- `test_id`
- `label`
- `user_profile`
- `test_goal`
- `trigger_strategy`

### 测试计划：`SubPlan` / `SimulationPlan`

- `SimulationPlan` 是一次 instruction 解析后的完整计划
- `SubPlan` 是某一组占位符填值对应的子计划

关键字段：

- `parsed_rules`
- `agent_spec`
- `placeholders`
- `sub_plans`
- `set_id`
- `placeholder_values`
- `filled_instruction`
- `test_cases`

### 会话归档：`SessionArchive`

关键字段：

- `order_id`
- `user_name`
- `source_file`
- `started_at`
- `ended_at`
- `ended_by`
- `transcript`
- `persona_type`
- `simulator_label`
- `test_case_id`
- `target_rule_id`

### 评估结果：`RuleResult` / `EvaluationReport`

- `RuleResult` 保存单条规则的判定结果、证据、置信度、投票分布
- `EvaluationReport` 保存单个 session 的总分、规则结果，以及可选的 `set_id` / `set_label`

## 8. 数据契约

### `instructions/*.json`

每个场景文件至少包含：

- `id`
- `scenario`
- `instruction`
- `success_criteria`
- `failure_criteria`

这里的 `success_criteria` / `failure_criteria` 仍然更像人工参考，不是当前主流程直接执行的规则源。主流程还是会重新调用 `rule_parser` 生成 `parsed_rules`。

### `orders/*.json`

每个订单样本至少包含：

- `userName`
- `orderId`
- `eta`
- `address`

可选字段：

- `scenario`
- `storeName`
- `recordedAddress`
- `deliveryNote`
- `taskContext`

`scenario` 很关键，它决定 `auto_main.py` 选哪份 instruction。

### `sessions/*.json`

新归档里通常至少包含：

- `order_id`
- `user_name`
- `source_file`
- `started_at`
- `ended_at`
- `ended_by`
- `transcript`
- `persona_type`
- `simulator_label`
- `test_case_id`
- `target_rule_id`

注意：当前 session 归档**不保存**：

- `set_id`
- `set_label`
- `parsed_rules`
- `filled_instruction`

这些信息只会在运行时评测结果或 memory 里出现。

### `memory/evaluation_memory.jsonl`

每条 memory 会额外保存：

- `set_id`
- `set_label`
- `score`
- `mean_confidence`
- `judge_results`
- `has_violation`
- `violation_count`

所以如果要回溯“某次 session 在哪个 placeholder set 下被判错”，memory 比 session archive 更完整。

## 9. 当前代码层面的注意事项

### 9.1 `app.py` 是最可信的当前主流程

后面要改功能时，优先以 `app.py` 为准。README 和部分 CLI 还没完全追上这轮测试计划架构。

### 9.2 `evaluate_main.py` 仍然默认走静态 `RULES`

这是当前最重要的不一致点之一：

- `app.py` 评估时传入的是运行时 `parsed_rules`
- `evaluate_main.py` 没传 `rules`，会退回 `src/rules.py` 里的静态 `RULES`

这意味着：

- 离线重评不一定和当次 prompt 对齐
- 对 `cancel / delay / address / review` 这类场景，静态规则可能不精确
- 离线评估也丢失了 placeholder set 维度

### 9.3 `selected_personas` 现在是“筛选器”，不是“直接输入”

`app.py` 里侧边栏勾选的 Persona，本质上是筛选哪些 `SimulationCase.profile_type` 要执行。

真正驱动模拟器的是自动生成的 `SimulationCase`，不是旧版 `UserPersona` prompt。

### 9.4 Streamlit 仍然会对所有选中订单复用同一段 instruction

`app.py` 的文本框只有一份 `instructions`，运行时会把它复用到当前选中的所有订单。

这意味着：

- 它适合测“同一类 prompt”在不同订单上的表现
- 如果同时选了多个不同 `scenario` 的订单，但没同步改 prompt，就可能不匹配

### 9.5 placeholder set 当前只改变 prompt 填值，不重建规则和测试用例

当前 `build_simulation_plan(...)` 的行为是：

- 规则只解析一次
- AgentSpec 只归纳一次
- 测试用例只生成一次
- 每个 `SubPlan` 只换 `filled_instruction`

这很实用，但也意味着：

- 如果某些规则本应随占位符值变化而变化，当前架构还没覆盖
- placeholder set 更像“同一测试计划下的参数化 prompt 变体”

### 9.6 `session.py` 已经不再注入订单上下文

旧版理解里 `_history` 会先塞订单背景，这现在已经不成立。

当前现状是：

- `_history` 初始为空
- 被测 Agent 主要依赖传入的 instruction
- `transcript` 只保存外显对话

### 9.7 session 归档的追踪信息比以前多，但仍不足以完整重放

现在 `SessionArchive` 已经多了：

- `persona_type`
- `simulator_label`
- `test_case_id`
- `target_rule_id`

但如果想完整复现一次运行，还缺：

- instruction 快照
- `parsed_rules`
- `set_id`
- `set_label`

### 9.8 旧归档和新归档可能混在一起

`sessions/` 里已有一些历史文件，可能：

- 缺少 `persona_type`
- 缺少 `test_case_id`
- 命名风格与当前流程不同

看历史数据时要区分新旧口径。

### 9.9 手动 CLI 入口疑似陈旧

`src/index.py` 里 `_print_order_header(...)` 访问了 `order.order.notes`，但 `OrderRecord` 已经没有 `notes` 字段。

说明：

- `main.py` 这条人工对话入口没有跟上数据结构
- 真要用手动 CLI，需要先修这处字段漂移

### 9.10 评估器现在是顺序执行

`src/evaluator.py` 虽然保留了 `max_workers` 参数，也还 import 了 `ThreadPoolExecutor`，但当前实现是顺序跑 judge。

这意味着：

- 单次评估逻辑更稳定
- 但总耗时会随 `规则数 × 采样数 × 会话数` 线性增长

## 10. 建议阅读顺序

如果后面要快速重新进入状态，按这个顺序读会更快：

1. `README.md`
2. `src/types.py`
3. `src/rules.py`
4. `src/simulation_plan.py`
5. `src/rule_parser.py`
6. `src/agent_spec.py`
7. `src/placeholders.py`
8. `src/test_case_generator.py`
9. `src/agent.py`
10. `src/simulator.py`
11. `src/session.py`
12. `src/evaluator.py`
13. `app.py`

## 11. 后续改动时优先守住的边界

### 评测边界

- 不要让被测 Agent 直接看到解析后的规则
- judge 仍应只基于 `transcript` 打分
- 测试计划生成层和执行层应保持可替换

### 场景扩展边界

新增场景时，至少要同步看这几层：

1. `instructions/*.json`
2. `orders/*.json`
3. `rule_parser` 是否能拆出稳定规则
4. `test_case_generator` 的画像映射是否还能覆盖
5. 是否需要支持新的占位符语法

### 数据闭环边界

如果后面要做“离线可重放、可重评、可比较”，优先补这些持久化：

1. instruction 快照
2. `parsed_rules`
3. `set_id` / `set_label`
4. placeholder values

### 性能边界

当前最慢的部分不是 UI，而是：

1. `parse_rules(...)`
2. `parse_agent_spec(...)`
3. `extract_placeholders(...)`
4. 多会话对话生成
5. 每条规则的多次 judge

总耗时大致会随下面这项一起放大：

`订单数 × placeholder_set 数 × test_case 数 × rule_judge_samples`

真正要提速，重点不是样式层，而是：

- 评估器并发
- 会话执行并发
- 计划缓存
- 动态规则持久化后减少重复解析
