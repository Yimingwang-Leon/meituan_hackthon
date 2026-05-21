# 技术报告：任务型对话 Agent 自动评测系统

## 1. 项目目标

本项目面向任务型外呼 Agent 的指令遵循评测。目标不是训练一个新的对话模型，而是构建一个可解释、可量化、可复核的评测系统，用于回答：

- Agent 是否按业务指令完成必要流程？
- 在拒收、质疑、信息缺失、越权诱导等场景下是否仍然合规？
- 评测结果是否能被人工复查，而不是只给一个黑盒分数？

系统输入一段业务 prompt，自动生成规则、测试场景、模拟对话，并输出逐条规则的判定证据、Judge 一致率、失败原因和改进建议。

## 2. 系统架构

```text
Instruction
  -> Rule Parser
  -> Placeholder Extractor
  -> Simulation Plan
  -> Outbound Agent + User Simulator
  -> Deterministic Checker / LLM Judge
  -> Streamlit Report + Evaluation Memory
```

核心模块位于 `agent/src/`：

- `rule_parser.py`：将自然语言指令拆成 required / conditional / forbidden 规则。
- `placeholders.py`：识别 `${name}`、`**X 单**` 等占位符并生成多组取值。
- `test_case_generator.py`：为每条规则生成正常触发、模糊触发、强触发、边界等测试场景。
- `simulator.py`：根据用户画像和目标场景生成用户回复。
- `evaluator.py`：执行确定性检查与 LLM Judge，并聚合多次采样结果。
- `memory.py`：保存 transcript、规则结果、Judge sample 和报告字段，支持后续复核。

被测 Agent 只看到原始 instruction，不接触解析后的规则，避免“对着答案答题”。

## 3. 评测方法

### 3.1 规则拆解

系统将指令拆为三类规则：

- `required`：每次对话都应该满足，例如开场身份说明。
- `conditional`：特定条件触发后才评估，例如用户拒绝时必须挽留。
- `forbidden`：全程不得出现，例如编造信息或越权承诺。

规则解析后会进入质量校验，检查字段完整性、条件规则是否有触发条件、失败标准是否具体等。warning 用于提示人工复核，error 可触发自动修复。

### 3.2 混合评测

系统不把所有规则都交给 LLM 判断，而是按可判定性分流：

- 确定性规则：字数、关键词、PII、开场固定信息等，用代码 checker 判断。
- 语义规则：挽留、解释、安抚、越权边界等，用 LLM Judge 判断。

这样可以降低 LLM 不稳定性，也能让可机器判定的规则保持确定性。

### 3.3 Conditional 两阶段 Judge

conditional 规则被拆成两步：

1. `TriggerJudge`：判断规则前提是否在对话中出现。
2. `ComplianceJudge`：在触发已确认的前提下，判断 Agent 后续响应是否合规。

这样可以区分两种失败来源：

- 场景没有演出来，或 Judge 对触发条件有分歧。
- 场景已经触发，但 Agent 的响应不满足规则。

报告中分别展示：

- `触发一致率`
- `合规一致率`
- 最终 `Judge一致率`

如果触发轮次无效，例如模型输出 `triggered=true` 但 `trigger_turn=0`，系统会降级为未触发，避免第二阶段基于错误轮次继续判断。

### 3.4 多采样与可复核

LLM Judge 支持 1 次或 3 次采样。3 次采样通过多数投票得到最终结果，同时保存每次 sample：

- result
- trigger_turn / response_turn
- evidence
- rationale
- matched_failure_criteria
- suggestion

这使得评测不仅有最终 pass/fail，也能看到 Judge 分歧发生在哪一步。

## 4. 报告与指标

Streamlit 报告包含：

- 评测结论摘要
- 规则通过率
- 条件规则触发率
- Judge 一致率
- 目标场景未触发率
- 用户类型表现对比
- 逐条对话和规则评测明细

评分采用规则等级加权：

```text
规则通过率 = 通过规则权重之和 / 适用规则权重之和
```

未触发的规则不计入通过率分母，避免把没有出现的场景误算为失败。

## 5. 可靠性设计

当前系统通过以下方式提升可靠性：

- 规则质量校验：减少解析出的规则缺字段或粒度过粗。
- deterministic checker：把可代码判断的规则从 LLM Judge 中剥离。
- 多采样投票：暴露 Judge 分歧，而不是隐藏不确定性。
- 两阶段 Judge：把“是否触发”和“是否合规”拆开。
- 持久化 replay 信息：保存 transcript、规则快照、prompt、模型名和 sample。
- 失败建议：每条 fail 规则给出可执行修复建议，便于 prompt 迭代。

## 6. 使用注意与后续方向

当前版本已经具备完整自动评测闭环，正式使用时建议关注以下事项：

- 对较长或包含多个业务要求的规则，建议结合规则质量校验结果确认规则粒度。
- 当目标场景未触发率较高时，应优先检查测试场景和用户模拟器行为，而不是直接归因于 Agent 失败。
- Judge 一致率较低的规则应优先复核，因为这通常意味着触发条件或合规标准存在边界情况。
- 并行度过高时可能遇到模型接口限流或连接抖动，正式评测建议从较低并行度开始。

后续可继续增强：

- 锁定规则集版本和 hash，使不同评测报告可横向比较。
- 增加负向对照样本，验证 evaluator 能否稳定抓住典型违规。
- 将失败规则聚类成业务问题类型，输出更面向业务方的诊断摘要。
