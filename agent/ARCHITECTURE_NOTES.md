# 项目框架笔记

## 1. 项目定位

这是一个针对“美团外呼数字人”做自动评测的 Python 项目。核心不是单纯生成对话，而是把一段业务指令拆成可验证规则，跑模拟对话，再用 LLM 逐条打分。

实际代码里的主链路是：

`任务指令 -> rule_parser -> 原子规则 -> 数字人/用户模拟对话 -> evaluator -> 会话报告/可视化`

一个关键设计点是：**被测外呼 agent 只拿到业务指令，不拿到拆出来的评测规则**，避免“按答案作答”。

## 2. 技术栈与运行形态

- 语言：Python
- LLM Agent 框架：`openai-agents`
- UI：`Streamlit`
- 配置：`python-dotenv`
- 数据建模：`dataclasses` + `pydantic`
- 持久化：本地 JSON 文件（`sessions/`、`reports/`）

模型调用目前主要都写死在代码里，默认使用 `gpt-5.4-nano`：

- 规则拆解：`src/rule_parser.py`
- 外呼数字人：`src/agent.py`
- 用户模拟器：`src/simulator.py`
- 规则评判器：`src/evaluator.py`

## 3. 主要入口

### 推荐入口

- `app.py`
  - Streamlit 页面
  - 支持自定义任务指令
  - 会动态调用 `parse_rules(...)`
  - 会按选中的订单和 Persona 跑完整评测
  - 评测时使用的是“本次解析出的规则”

### 命令行入口

- `auto_main.py`
  - 批量跑订单 × Persona 对话
  - 根据订单里的 `scenario` 去 `instructions/*.json` 取场景指令
  - 只负责生成和归档会话，不负责汇总可视化

- `evaluate_main.py`
  - 读取 `sessions/*.json`
  - 批量做规则评估并输出 `reports/report-*.json`

- `main.py`
  - 手动对话入口，实际转发到 `src/index.py`
  - 这条链路目前看起来比较陈旧，见下面“注意事项”

## 4. 目录理解

```text
agent/
├── app.py                  # Streamlit 主入口，最完整的运行链路
├── auto_main.py            # 批量模拟对话
├── evaluate_main.py        # 批量评估已归档 session
├── main.py                 # 手动 CLI 入口
├── instructions/           # 场景级任务指令与 success/failure criteria
├── memory/                 # 逐次评测记忆（JSONL，记录 transcript + judge 结论）
├── orders/                 # 待测订单样本
├── reports/                # 评估报告输出
├── sessions/               # 会话归档输出
└── src/
    ├── agent.py            # 被测外呼 agent
    ├── evaluator.py        # 规则评估器
    ├── memory.py           # 评测记忆落盘
    ├── orders.py           # 订单加载
    ├── persona.py          # Persona 定义
    ├── rule_parser.py      # 指令拆规则
    ├── rules.py            # Rule 数据结构 + 静态规则
    ├── session.py          # 单会话编排与归档
    ├── simulator.py        # 用户模拟器
    ├── types.py            # 共用数据类型
    └── index.py            # 手动 CLI 流程
```

## 5. 实际执行链路

### 5.1 Streamlit 链路

`app.py` 的完整流程最值得记：

1. 从页面文本框拿到任务指令
2. 调 `parse_rules(instructions)` 解析为 `list[Rule]`
3. 根据侧边栏选择的订单和 Persona 组成任务集合
4. 每个任务里创建：
   - `OutboundSession(order, agent=make_outbound_agent(instructions))`
   - `UserSimulator(order, persona)`
5. 先 `outbound.start()` 让数字人主动开场
6. 进入最多 `MAX_TURNS=15` 轮对话
7. 对话结束后保存 archive，并调用 `evaluate_session(...)`
8. 把结果写进 `st.session_state`
9. 页面展示仪表板、热力图、覆盖率、会话归档

这里的评估规则是**动态解析结果**，不是静态常量。

### 5.2 批量 CLI 链路

`auto_main.py` 的流程是：

1. 读取订单
2. 根据 `order.scenario` 从 `instructions/*.json` 里找对应 instruction
3. 创建定制版 outbound agent
4. 遍历 `ALL_PERSONAS`
5. 生成会话并存入 `sessions/`

这条链路更像“离线生成测试数据”。

### 5.3 批量评估链路

`evaluate_main.py` 的流程是：

1. 读取 `sessions/*.json`
2. 反序列化为 `SessionArchive`
3. 对每个 session 调 `evaluate_session(...)`
4. 打印汇总
5. 生成 `reports/report-*.json`

此外，评测完成后会追加写入 `memory/evaluation_memory.jsonl`，用于保存单次 session 的对话和 judge 违规判断。

## 6. 核心模块职责

### `src/rule_parser.py`

- 用 LLM 把 instruction 拆成 `ParsedRuleList`
- 输出要求比较严格：
  - 原子化
  - 6 到 14 条
  - 三种类型：`required` / `conditional` / `forbidden`
  - 必须补 `evaluation_hint`
  - 必须给 `severity`

这部分是整个评测的“规则生成层”。

### `src/agent.py`

- 定义被测数字人的默认指令 `OUTBOUND_INSTRUCTIONS`
- `make_outbound_agent(instructions)` 可按场景或页面输入生成新 agent
- `build_order_context(order)` 会把订单字段拼成上下文文本

注意：订单上下文不是直接展示给用户的话术，而是作为模型输入背景。

### `src/session.py`

- `OutboundSession` 是真实对话编排器
- 初始化时会把 `build_order_context(order)` 作为第一条“user message”塞进 `_history`
- `start()` 会额外给模型一个开场提示，强制数字人先说第一句话
- `reply()` 记录用户输入并驱动下一轮 agent 输出
- `save_archive()` 把 transcript 落盘到 JSON

一个容易忽略的点：**归档里的 transcript 只有可见对话，不包含内部注入的订单上下文。**

### `src/simulator.py`

- `UserSimulator` 本质上也是一个 LLM agent
- Persona 指令由 `build_persona_instructions(...)` 动态拼出来
- 模拟器只掌握被允许暴露的信息
- 当 `should_end=True` 时，表示用户想挂断或拒绝继续

### `src/evaluator.py`

- `evaluate_session(...)` 会把 transcript 格式化成“第 N 轮 数字人/用户：...”
- 对每条规则跑 N 次 judge
- 多数投票得到最终 `RuleResult`
- `confidence = 最高票数 / 采样数`
- `not_applicable` 不参与总分分母
- `compute_coverage(...)` 用于统计 conditional 规则是否被触发

### `src/memory.py`

- 负责把单次评测结果追加写入 `memory/evaluation_memory.jsonl`
- 每条 memory 会保存：
  - session 基本信息
  - transcript
  - 每条规则的 `judger_result`
  - `is_violation`
  - evidence / confidence / votes

### `src/orders.py`

- 把 `orders/*.json` 解析成 `LoadedOrder`
- 字段命名会从 JSON 的 camelCase 转成 dataclass 的 snake_case

### `src/persona.py`

- 预置 7 种 Persona：
  - `cooperative`
  - `suspicious`
  - `impatient`
  - `ambiguous`
  - `info_missing`
  - `rejector`
  - `hostile`

Persona 的核心控制参数是：

- `available_info`
- `mood`
- `rejection_threshold`
- `ambiguity_rate`

## 7. 关键数据结构

### 订单：`OrderRecord`

关键字段：

- `user_name`
- `order_id`
- `eta`
- `address`
- `store_name`
- `recorded_address`
- `delivery_note`
- `task_context`
- `scenario`

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

### 规则：`Rule`

关键字段：

- `rule_id`
- `description`
- `rule_type`
- `evaluation_hint`
- `severity`

### 评估结果：`RuleResult` / `EvaluationReport`

- `RuleResult` 保存单条规则结果、证据、置信度、投票分布
- `EvaluationReport` 保存单个 session 的总分与规则结果列表

## 8. 数据契约

### `instructions/*.json`

每个场景文件至少包含：

- `id`
- `scenario`
- `instruction`
- `success_criteria`
- `failure_criteria`

这里的 `failure_criteria` 更像人工参考标准，不是程序直接执行的规则源。程序在 Streamlit 路径里还是会重新调用 `rule_parser` 生成规则。

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

## 9. 当前代码层面的注意事项

### 9.1 `app.py` 是最可信的“当前主流程”

如果后面要改功能，优先以 `app.py` 的链路为准。README 描述和 CLI 都基本围绕它，但不一定和所有分支完全同步。

### 9.2 `evaluate_main.py` 当前默认走静态 `RULES`

这是一个很重要的现状：

- `app.py` 评估时传入的是 `parsed_rules`
- `evaluate_main.py` 没传 `rules`，会退回 `src/rules.py` 里的静态 `RULES`

这意味着：

- CLI 评估不一定和场景指令完全对齐
- 对 `cancel / delay / address / review` 这类场景，静态规则可能不够精确
- `review` 场景里甚至把身份写成“美团客服助手”，而静态规则更偏“配送助手”

如果后面要统一评估口径，这里应该优先收敛。

### 9.3 Streamlit 会对所有选中订单复用同一段 instruction

`app.py` 的文本框只有一份 `instructions`，运行时会把它复用到当前选中的所有订单。

这意味着：

- `app.py` 更适合评测“同一类任务指令”在不同订单/Persona 上的表现
- 如果同时选了多个不同 `scenario` 的订单，但没有同步改 prompt，就可能出现“订单场景”和“任务指令”不匹配

而 `auto_main.py` 不是这样，它会按 `order.scenario` 自动去找对应的场景 instruction。

### 9.4 手动 CLI 入口疑似陈旧

`src/index.py` 里 `_print_order_header(...)` 访问了 `order.order.notes`，但 `OrderRecord` 已经没有 `notes` 字段了。

说明：

- `main.py` 这条人工对话入口可能没跟着数据结构一起维护
- 真要用手动 CLI，需要先修这处字段漂移

### 9.5 评估器现在是顺序执行

`src/evaluator.py` 里虽然保留了 `max_workers` 参数，也 import 了 `ThreadPoolExecutor`，但当前实现已经改成顺序跑 judge。

这和 README 里的说明一致，原因是避免事件循环/线程冲突。

### 9.6 `session` 的内部 history 和归档 transcript 不是一回事

- `_history` 里包含系统给模型看的订单上下文
- `transcript` 只保存外显的用户/数字人对话

评估器是基于 `transcript` 打分的，不会看到内部注入背景。

### 9.7 旧归档和新归档可能混在一起

`sessions/` 里已经有一些历史文件，命名风格和现在的订单样本不完全一致，也可能缺少 `persona_type`。

所以看历史数据时要区分：

- 旧样本：`unknown` persona 较多
- 新样本：通常会带 `persona_type`

### 9.8 `evaluator.py` 里有一段死代码味道

`compute_coverage(...)` 下面缩进了一段 `summary(self)`，但位置在函数体里且前面已经 `return`。

这不影响主流程，但说明这个文件被改过几轮，后续重构时可以顺手清理。

## 10. 建议阅读顺序

如果后面要快速重新进入状态，按这个顺序读会更快：

1. `README.md`
2. `src/types.py`
3. `src/rules.py`
4. `src/persona.py`
5. `src/agent.py`
6. `src/session.py`
7. `src/simulator.py`
8. `src/rule_parser.py`
9. `src/evaluator.py`
10. `app.py`

## 11. 后续改动时优先守住的边界

### 评测边界

- 不要让被测 agent 直接看到解析后的规则
- 规则生成和规则评估应保持可替换

### 场景扩展边界

新增场景时至少要同步三处：

1. `instructions/*.json`
2. `orders/*.json`
3. 评估口径是否仍能复用当前规则体系

### 性能边界

当前最慢的部分不是 UI，而是：

1. `parse_rules(...)`
2. 多 Persona 对话
3. 每条规则的多次 judge

真正要提速，重点在评估器并发和整体 async 化。
