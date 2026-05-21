# 对话 Agent 评测系统

面向任务型外呼 Agent 的自动化评测工具。系统输入一段业务指令，自动拆解规则、生成测试场景、模拟用户对话，并用可复核的 Judge 结果输出评测报告。

主项目代码位于 [`agent/`](./agent)。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 规则解析 | 将原始 prompt 拆成 required / conditional / forbidden 三类可评测规则 |
| 场景生成 | 自动识别占位符并生成多组取值场景，覆盖不同用户类型 |
| 用户模拟 | 通过配合型、拒收型、警惕型、急躁型等用户画像生成对话 |
| 混合评测 | 确定性规则用代码检查，语义规则用 LLM Judge |
| 两阶段 Judge | conditional 规则先判断“是否触发”，再判断“触发后是否合规” |
| 可复核报告 | 保存 transcript、证据、判定依据、改进建议和每次 Judge sample |

## 系统流程

```text
任务指令
  -> 规则解析
  -> 占位符取值与测试场景生成
  -> 被测 Agent 与用户模拟器对话
  -> Deterministic Checker / LLM Judge
  -> Streamlit 评测报告 + JSONL 评测记忆
```

关键设计原则：被测 Agent 只看到原始 instruction，不接触评测规则，避免“对着答案答题”。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r agent/requirements.txt

cp agent/.env.example agent/.env
# 编辑 agent/.env，填写 DEEPSEEK_API_KEY

.venv/bin/streamlit run agent/app.py
```

## 项目结构

```text
agent/
├── app.py                 # Streamlit 可视化入口
├── agents.py              # OpenAI-compatible LLM 调用封装
├── evaluate_main.py       # 命令行评测入口
├── src/
│   ├── rule_parser.py     # 指令拆解为结构化规则
│   ├── simulation_plan.py # 评测计划构建
│   ├── test_case_generator.py
│   ├── simulator.py       # 用户模拟器
│   ├── evaluator.py       # 混合评测与两阶段 Judge
│   └── memory.py          # 评测记忆持久化
└── README.md              # 详细运行和开发说明
```

## 文档

- [详细运行说明](./agent/README.md)
- [技术报告](./TECHNICAL_REPORT.md)
- [架构备注](./agent/ARCHITECTURE_NOTES.md)

## 使用注意

- 并行度建议按模型接口稳定性调整。正式评测建议先用 1-4 路并行验证，再根据接口稳定性提高并行度。
- 规则解析结果建议快速复核。系统会进行规则质量校验和自动修复，但较长或包含多个业务要求的规则仍建议确认粒度是否符合预期。
- LLM Judge 结果会保存证据、判定依据和每次 sample，建议优先复核 Judge 一致率较低的规则。
