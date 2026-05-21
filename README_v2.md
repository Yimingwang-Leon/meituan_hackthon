# 对话 Agent 评测系统

一个针对任务型外呼对话 Agent 的自动化评测工具。给定一段业务指令（prompt），系统自动拆解评测规则、模拟多种用户对话、产出可解释的评测报告。

## 功能

- **自动规则解析**：从业务 prompt 拆出 required / conditional / forbidden 三类原子规则
- **多场景用户模拟**：内置 7 种用户画像（配合、警惕、急躁、模糊、缺信息、拒收、对抗）
- **混合评测引擎**：可机器判定的规则走代码 checker，语义规则走 LLM Judge
- **两阶段条件规则评测**：先判触发是否出现，再判触发后是否合规
- **多采样 + 一致率**：每条规则可独立运行多次 Judge，多数投票决定结果
- **可视化报告**：Streamlit 仪表板包含规则通过率、触发覆盖率、Judge 一致率、用户类型对比
- **完整审计**：保留每次 Judge 的 evidence、rationale、failure_criteria 命中、改进建议、prompt 快照

## 系统流程

```
业务指令 (instruction)
       │
       ▼
  规则解析 + 占位符抽取 + 测试场景生成
       │
       ▼
  被测 Agent  ⇄  用户模拟器     ← 多轮对话
       │
       ▼
  Deterministic Checker  +  LLM Judge
       │
       ▼
  评测报告 (Streamlit + JSON + JSONL)
```

被测 Agent 只看到原始 instruction，不接触解析后的评测规则。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r agent/requirements.txt

cp agent/.env.example agent/.env
# 编辑 agent/.env，填写 DEEPSEEK_API_KEY
```

## 快速开始

启动可视化界面：

```bash
.venv/bin/streamlit run agent/app.py
```

在页面中：
1. 粘贴业务指令
2. 选择用户画像、采样次数、并行度
3. 点击运行，等待评测完成
4. 查看仪表板和逐 session 详情

命令行批量评测：

```bash
cd agent
python auto_main.py instructions/cancel.json    # 批量生成对话
python evaluate_main.py                         # 评测已生成对话
```

端到端 smoke test（验证两步 Judge 真实跑通）：

```bash
cd agent
python smoke_test_e2e.py cancel
```

## 项目结构

```
.
├── README.md                 # 项目首页（业务向）
├── TECHNICAL_REPORT.md       # 技术方案说明
└── agent/
    ├── app.py                # Streamlit 可视化入口
    ├── agents.py             # LLM 调用封装（重试 / JSON 修复 / 容错）
    ├── auto_main.py          # 命令行批量对话
    ├── evaluate_main.py      # 命令行批量评测
    ├── smoke_test_e2e.py     # 端到端 smoke test
    ├── instructions/         # 业务指令样例（cancel / address / confirm / delay / review）
    ├── sessions/             # 对话归档
    ├── reports/              # 评测报告 JSON
    ├── memory/               # 评测记忆 JSONL
    └── src/
        ├── rule_parser.py    # 指令 → 原子规则
        ├── rule_validation.py # 规则质量校验 + 自动修复
        ├── agent_spec.py     # Agent 职责归纳
        ├── placeholders.py   # 占位符抽取 + 多场景取值
        ├── scenario_selector.py # 测试场景规划
        ├── test_case_generator.py # 测试用例派生
        ├── persona.py        # 用户画像定义
        ├── agent.py          # 被测 Agent
        ├── simulator.py      # 用户模拟器
        ├── session.py        # 单次对话会话管理
        ├── evaluator.py      # 混合评测 + 两步 Judge
        ├── deterministic_checks.py # 代码 checker（PII / 字数 / 关键词等）
        ├── memory.py         # 评测记忆持久化
        ├── rules.py          # Rule 数据结构 + 等级权重
        └── types.py          # 共用数据类型
```

## 评分机制

规则按等级加权：

- 关键 (critical) = 3
- 重要 (major) = 2
- 一般 (minor) = 1

```
规则通过率 = 通过规则的权重之和 / 适用规则的权重之和
```

未触发的条件规则（result = not_applicable）不计入分母。

## Judge 一致率

每条 LLM 评测规则可独立运行 N 次（默认 N=3），多数投票决定结果：

```
Judge 一致率 = max(投票数) / N
```

- 100%：全部一致，结果可信
- 67%：2-1 分歧，建议人工抽查
- 33%：严重分歧，必须人工复核

## 配置项

`agent/.env` 支持的环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 必填 | DeepSeek API 密钥 |
| `OPENAI_API_KEY` | 可选 | 兼容 OpenAI 接口的备用密钥 |
| `LLM_REQUEST_TIMEOUT_SECONDS` | 180 | 单次 LLM 请求超时 |
| `LLM_MAX_RETRIES` | 2 | 网络抖动后的最大重试次数 |
| `LLM_RETRY_DELAY_SECONDS` | 1 | 重试间隔基数（指数退避） |
| `LLM_PARSE_MAX_RETRIES` | 1 | JSON 输出异常后的修复重试次数 |
| `DEEPSEEK_REASONING_EFFORT` | low | 推理强度（low / medium / high） |
| `DEEPSEEK_THINKING_ENABLED` | false | 是否启用 thinking 模式 |

## 测试

```bash
cd agent
python -m unittest test_agents test_evaluator test_evaluate_main test_memory test_placeholders test_simulation_cases
```

当前 34 个单元测试，覆盖 LLM 客户端容错、规则聚合、两步 Judge 流程、占位符 normalize、测试用例派生、JSON 修复管线。运行时间约 10ms（全部 mock，不调真实 LLM）。

## 文档

- 技术方案与设计推导：`TECHNICAL_REPORT.md`
- 详细运行说明：`agent/README.md`
- 源码深度阅读笔记：`agent/ARCHITECTURE_NOTES.md`
