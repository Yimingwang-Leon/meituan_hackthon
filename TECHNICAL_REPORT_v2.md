# 技术报告：对话 Agent 指令遵循自动评测系统

## 1. 项目背景

任务型对话 Agent（如外呼数字人、客服 bot）的核心质量指标是"指令遵循率"：Agent 是否按业务 prompt 中规定的流程、约束、禁止项执行对话。当前业界对这类系统的评测主要依赖三种方式，各有局限：

- **人工抽检**：准确但成本高、抽样小、判断主观
- **写死的规则脚本**：稳定但需手工维护，prompt 改一版脚本就要改一版
- **直接 LLM 打分**：易实施但结果不稳定、无证据、不可复核

本系统目标是构建一个**自动化、可解释、可复核**的评测闭环：输入业务 prompt，输出每条规则的判定结果、判定证据、Judge 一致率、改进建议。

## 2. 系统架构

### 2.1 整体数据流

```
业务指令 (instruction)
       │
       ▼
[规则解析层]
  ├ rule_parser          指令 → 6-14 条原子规则
  ├ rule_validation      规则结构体检 + 自动修复
  ├ agent_spec           归纳 Agent 职责
  ├ placeholders         占位符抽取 + 多组取值
  ├ scenario_selector    为每条规则挑选测试场景
  └ test_case_generator  派生 SimulationCase
       │
       ▼
[对话执行层]
  ├ session (被测 Agent)        ┐
  └ simulator (用户模拟器)        │ 多轮对话
                                  ▼
[评测层]
  ├ deterministic_checks   可机器判定的规则（字数 / 关键词 / PII / 开场 / 结尾）
  ├ TriggerJudge (LLM)     conditional 规则的触发判定
  └ ComplianceJudge (LLM)  触发后的合规判定
       │
       ▼
[报告层]
  ├ Streamlit 仪表板
  ├ reports/report-*.json
  └ memory/evaluation_memory.jsonl
```

### 2.2 关键设计原则

- **被测 Agent 与评测规则隔离**：Agent 只看原始指令，不接触解析后的规则，避免"对着答案答题"
- **规则一等公民**：规则作为可序列化、可版本化、可复核的数据契约存在
- **混合判定**：可代码判定的规则走 deterministic，避免无意义的 LLM 调用
- **判定过程可审计**：每条规则保留完整的 N 次采样输出、prompt 快照、模型版本

## 3. 评测方法

### 3.1 规则三分类

规则被自动拆为三种类型，对应不同评测语义：

- **required**：每次对话都应满足（例：开场必须说明身份）
- **conditional**：特定触发条件下才评估（例：用户拒收时必须挽留）
- **forbidden**：全程不得违反（例：不得编造信息、不得越权承诺）

### 3.2 严重度分级

- **关键 (critical)**：直接损害用户权益 / 传递错误信息
- **重要 (major)**：关键流程缺失 / 条件分支处理错误
- **一般 (minor)**：不影响主流程的轻微体验问题

总分按严重度加权：

```
通过率 = Σ(通过规则的等级权重) / Σ(适用规则的等级权重)
```

未触发的 conditional 规则不计入分母，避免把没出现的场景误算为失败。

### 3.3 混合评测：deterministic + LLM

系统在解析规则时同步生成 deterministic check 规格（如果适用）。评测时按如下路径分流：

| 规则特征 | 评测路径 | 调用次数 |
|---|---|---|
| 含 deterministic check（字数 / 关键词 / PII / 开场 / 结尾） | 代码 checker | 1 次（确定性） |
| conditional 且无 check | 两阶段 LLM Judge | N + N 次（触发后才跑合规） |
| required / forbidden 且无 check | 单阶段 LLM Judge | N 次 |

这样设计的收益：可机器判定的规则避免随机性和成本，语义类规则才使用 LLM。

### 3.4 Conditional 两阶段 Judge

对 conditional 规则，单次 LLM 调用同时判"是否触发"和"触发后是否合规"会让两个判断的错误相乘。系统将其拆为两个独立 agent：

**Stage 1 · TriggerJudge**
- 输入：完整对话 + 规则的 trigger_condition + expected_behavior（仅作背景）
- 输出：triggered (bool) + trigger_turn (int) + evidence + rationale
- 只关注用户侧表达和对话进展，不评估 Agent 响应

**Stage 2 · ComplianceJudge**（仅在 triggered_final=true 时执行）
- 输入：完整对话 + 规则的 expected_behavior + failure_criteria + 上一步确认的 trigger_turn + trigger_evidence
- 输出：compliant (bool) + response_turn + evidence + rationale + matched_failure_criteria + suggestion
- 注意力锚定在 trigger_turn 之后的 Agent 响应

**判定汇总**：

- triggered_final：N 次采样的多数投票
- trigger_turn_final：在 triggered=true 的样本中取多数轮次；无多数则取中位数
- compliant_final：N 次合规采样的多数投票
- 最终结果：未触发 → trigger_failed（is_primary）或 not_applicable；触发后 → pass / fail
- 总置信度：`trigger_confidence × compliance_confidence`

**异常兜底**：若 LLM 输出 `triggered=true` 但 `trigger_turn=0` 这种自相矛盾的结果，强制降级为未触发，避免合规阶段基于错误轮次继续判断。

### 3.5 多采样与可复核

每条 LLM 规则可独立运行 N 次采样（默认 N=3），多数投票决定结果，并输出 Judge 一致率：

```
Judge 一致率 = max(投票数) / N
```

报告同时展示三类置信度：
- 触发一致率（trigger_confidence）
- 合规一致率（compliance_confidence）
- 总 Judge 一致率（confidence）

每次采样的完整输出都被保留：

- result（pass / fail / not_applicable / trigger_failed）
- triggered / trigger_turn / response_turn
- evidence（引用对话轮次原话）
- rationale（结构化判定依据）
- matched_failure_criteria（fail 时从给定列表挑出命中条目，不允许自创）
- suggestion（fail 时给一句可执行修复建议）

这使评测不仅有最终 pass/fail，也能看到分歧发生在哪一步。

## 4. 用户模拟器

### 4.1 7 种用户画像

| Persona | 特征 |
|---|---|
| cooperative | 配合，直接确认 |
| suspicious | 警惕陌生来电，需要验证身份 |
| impatient | 忙碌，希望快速结束 |
| ambiguous | 回答含糊，经常不确定 |
| info_missing | 不在家，无法提供关键信息 |
| rejector | 明确拒收，测试拒收分支 |
| hostile | 试图套取内部信息或诱导 Agent 越权操作 |

### 4.2 5 种触发强度

每条规则根据语义自动选择适用的触发强度组合：

- normal_trigger：正常、直接、自然地触发
- ambiguous_trigger：含糊、留白的表达
- strong_trigger：明确、强烈、重复的表达
- adversarial_induction：施压、打断、诱导
- boundary：刚好踩在触发边缘

简单规则只用 normal_trigger，高风险规则（如拒收、越权诱导）覆盖多种强度。

### 4.3 触发覆盖率

系统输出两个指标量化模拟器充分性：

- **coverage_rate**：多少 conditional 规则被任何 session 实际触发到
- **trigger_failure_rate**：主任务里多少次模拟器没把规则演出来

这两个指标用于区分"Agent 失败"和"模拟器失败"，避免错误归因。

## 5. 可靠性设计

| 维度 | 实现 |
|---|---|
| 可解释 | 每条规则独立判断，输出 pass / fail / not_applicable + 引用具体轮次的 evidence + rationale |
| 可量化 | 严重度加权打分 + 触发覆盖率 + Judge 一致率 |
| 可重复 | N 采样多数投票 + Judge 一致率，分歧时人工抽查 |
| 可复核 | 保存 transcript / 规则快照 / 每次 sample / judge_prompt / 模型名 |
| 错误隔离 | conditional 两步 judge + 混合 evaluator（确定性归代码） |
| 容错 | LLM 调用层指数退避重试 + JSON 输出修复循环 + plain-reply 降级 |

## 6. 当前能力边界

- **无人工标注 ground truth**：N 采样一致率仅衡量 self-consistency，不能衡量 judge 准确率
- **规则解析依赖 LLM 稳定性**：同一段 prompt 多次解析可能产生略不同的规则集（已通过 auto_fix 兜底）
- **顺序规划阶段**：build_simulation_plan 内的 4 个 LLM 步骤仍顺序执行，单 prompt 准备 30-60 秒
- **触发句模板部分硬编码**：test_case_generator 针对几类典型 trigger 写了固定模板，新业务领域会退化到通用模板
- **模型耦合**：默认 DeepSeek，模型常量散落多处文件

## 7. 后续方向

按 ROI 排序：

1. 人工标注 ground truth + Cohen's κ：把"judge 比 baseline 准多少"从断言变成数字
2. 跨模型一致性：用第二个模型跑同一批规则，输出 inter-model agreement
3. parsed_rules 持久化到 SessionArchive：让离线评测与在线评测口径完全一致
4. 失败规则聚类：把多 session 共同 fail 的规则聚成"业务问题类型"，输出诊断摘要
5. 规则集版本锁定：给规则集打 hash，不同评测报告可横向比较

## 8. 关键源码索引

| 模块 | 文件 | 职责 |
|---|---|---|
| 规则数据 | `agent/src/rules.py` | Rule 数据结构 + 等级权重 |
| 规则解析 | `agent/src/rule_parser.py` | 指令 → 结构化规则 |
| 规则校验 | `agent/src/rule_validation.py` | 结构体检 + 自动修复 |
| 占位符 | `agent/src/placeholders.py` | 占位符抽取 + 多场景取值 |
| 测试规划 | `agent/src/scenario_selector.py` + `test_case_generator.py` | 为规则派生测试用例 |
| 对话执行 | `agent/src/session.py` + `simulator.py` + `agent.py` | 多轮对话编排 |
| 评测 | `agent/src/evaluator.py` | 混合 evaluator + 两步 Judge |
| 代码 checker | `agent/src/deterministic_checks.py` | 5 类确定性检查 |
| 持久化 | `agent/src/memory.py` | 评测记忆 JSONL |
| LLM 客户端 | `agent/agents.py` | 重试 / JSON 修复 / plain-reply 降级 |
| UI | `agent/app.py` | Streamlit 仪表板 |
