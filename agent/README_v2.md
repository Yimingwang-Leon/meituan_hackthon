# agent · 对话评测系统主代码

`agent/` 是对话 Agent 评测系统的主代码目录。本文档说明本目录的运行方式、组件构成和开发约定。

## 环境准备

```bash
# 在仓库根目录
python3 -m venv .venv
source .venv/bin/activate
pip install -r agent/requirements.txt

cp agent/.env.example agent/.env
# 编辑 agent/.env 填写 DEEPSEEK_API_KEY
```

## 运行方式

### 1. Streamlit 可视化界面（推荐）

```bash
.venv/bin/streamlit run agent/app.py
```

界面包含 8 个章节：

1. 任务指令输入
2. 解析规则（按 required / conditional / forbidden 分组）
3. 实时执行（进度条 + 单 session 即时展示）
4. 评测仪表板（规则通过率 / 触发覆盖率 / Judge 一致率）
5. 规则表现热力图（按用户类型）
6. 用户类型通过率对比
7. 模拟器触发情况
8. 逐条测试记录

侧边栏可控制：用户类型勾选、Judge 采样次数（1 或 3）、占位符取值组数、并行测试数（1 / 2 / 4 / 8 / 16）。

### 2. 命令行批量

```bash
cd agent

# 仅批量生成对话（不评测）
python auto_main.py
python auto_main.py instructions/confirm.json 3   # 指定 instruction + 占位符组数
python auto_main.py confirm 2                     # 简写
python auto_main.py "你是一名..."                   # 内联 prompt

# 评测已生成的对话
python evaluate_main.py
```

### 3. 端到端 smoke test

验证两步 Judge 真实端到端跑通（含真实 LLM 调用，约 5 分钟）：

```bash
cd agent
python smoke_test_e2e.py cancel
python smoke_test_e2e.py confirm
```

成功时输出 `✅ PASS`；失败时打印诊断信息并 exit 1。

### 4. 单元测试

```bash
cd agent
python -m unittest test_agents test_evaluator test_evaluate_main test_memory test_placeholders test_simulation_cases
```

34 个测试，约 10ms，全部 mock 不调真实 LLM。

## 目录结构

```
agent/
├── app.py                       # Streamlit 可视化入口
├── agents.py                    # LLM 调用封装（重试 / JSON 修复 / 容错降级）
├── auto_main.py                 # 批量对话生成
├── evaluate_main.py             # 批量离线评测
├── main.py                      # 手动 CLI 对话
├── smoke_test_e2e.py            # 端到端 smoke test
├── instructions/                # 业务指令样例（cancel / address / confirm / delay / review）
├── sessions/                    # 对话归档（JSON）
├── reports/                     # 评测报告（JSON）
├── memory/                      # 评测记忆（JSONL，每行一条评测结果）
├── test_agents.py               # LLM 客户端容错测试
├── test_evaluator.py            # 评测聚合 + 两步 Judge 测试
├── test_evaluate_main.py        # 报告序列化测试
├── test_memory.py               # 记忆持久化测试
├── test_placeholders.py         # 占位符 normalize / fill 测试
├── test_simulation_cases.py     # 测试用例派生测试
└── src/
    ├── rules.py                 # Rule 数据结构 + 严重度权重
    ├── rule_parser.py           # 指令 → 结构化规则
    ├── rule_validation.py       # 规则质量校验 + 自动修复
    ├── agent_spec.py            # Agent 职责归纳
    ├── placeholders.py          # 占位符抽取 + 多场景取值
    ├── scenario_selector.py     # 为规则挑选测试场景
    ├── test_case_generator.py   # SimulationCase 派生
    ├── persona.py               # 7 种用户画像定义
    ├── agent.py                 # 被测 Agent 封装
    ├── simulator.py             # 用户模拟器
    ├── session.py               # 单次对话会话管理
    ├── simulation_plan.py       # 评测计划总编排
    ├── evaluator.py             # 混合评测 + 两步 Judge
    ├── deterministic_checks.py  # 5 类代码 checker
    ├── memory.py                # 评测记忆 JSONL 写入
    └── types.py                 # 共用 dataclass
```

## 环境变量

`agent/.env` 支持的所有变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 必填 | DeepSeek API 密钥 |
| `OPENAI_API_KEY` | 可选 | 兼容 OpenAI 接口的备用密钥 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 自建 / 代理时覆盖 |
| `OPENAI_BASE_URL` | 同上 | DeepSeek base URL 的别名 |
| `LLM_REQUEST_TIMEOUT_SECONDS` | 180 | 单次 LLM 请求超时 |
| `LLM_MAX_RETRIES` | 2 | 网络抖动后的最大重试次数 |
| `LLM_RETRY_DELAY_SECONDS` | 1 | 重试间隔基数（指数退避） |
| `LLM_PARSE_MAX_RETRIES` | 1 | 结构化 JSON 输出异常后的修复重试次数 |
| `DEEPSEEK_REASONING_EFFORT` | low | 推理强度（low / medium / high） |
| `DEEPSEEK_THINKING_ENABLED` | false | 是否启用 thinking 模式 |

## 评分机制

规则按严重度加权：

- 关键 (critical) = 3
- 重要 (major) = 2
- 一般 (minor) = 1

```
规则通过率 = 通过规则的权重之和 / 适用规则的权重之和
```

未触发的条件规则不计入分母。

## Judge 一致率

每条 LLM 评测规则可独立运行 N 次（侧边栏滑块控制，1 或 3），多数投票决定最终结果：

```
Judge 一致率 = max(投票数) / N
```

显示约定：

- 100%：全部一致
- ≥ 66%：部分分歧，建议复核
- < 66%：严重分歧，必须人工复核

对 conditional 规则，UI 同时展示触发一致率和合规一致率，便于定位分歧来源。

## 用户画像（Persona）

| 类型 | 中文 | 特征 |
|---|---|---|
| cooperative | 配合型 | 正常配合，直接确认 |
| suspicious | 警惕型 | 警惕陌生来电，需要验证身份 |
| impatient | 急躁型 | 忙碌，希望快速结束 |
| ambiguous | 模糊型 | 回答含糊，经常不确定 |
| info_missing | 缺信息型 | 不在家，无法提供关键信息 |
| rejector | 拒收型 | 明确拒收，测试拒收分支 |
| hostile | 对抗型 | 试图套取内部信息或诱导 Agent 越权操作 |

## 数据契约

### `instructions/*.json`（输入）

```jsonc
{
  "id": "cancel",
  "scenario": "订单取消确认",
  "instruction": "...",              // 业务 prompt 原文
  "success_criteria": [...],         // 人工参考，主流程不使用
  "failure_criteria": [               // 人工参考，主流程会重新调 rule_parser 生成规则
    {"description": "...", "severity": "major"}
  ]
}
```

### `sessions/*.json`（对话归档）

由 `OutboundSession` 写入。包含 session_id / transcript / persona_type / test_case_id / target_rule_id 等元数据。

### `reports/report-*.json`（评测报告）

由 `evaluate_main._save_report` 或 `app.py` 写入。结构：

```jsonc
{
  "summary": {
    "total_sessions": ...,
    "overall_score": ...,
    "by_persona": {...},
    "rule_fail_rate": {...}
  },
  "sessions": [
    {
      "session_id": ...,
      "persona_type": ...,
      "score": ...,
      "rules": [
        {
          "rule_id": ...,
          "result": ...,
          "evidence": ...,
          "rationale": ...,
          "matched_failure_criteria": [...],
          "suggestion": ...,
          "confidence": ...,
          "trigger_confidence": ...,
          "compliance_confidence": ...,
          "all_samples": [...]
        }
      ]
    }
  ]
}
```

### `memory/evaluation_memory.jsonl`（评测记忆）

每行一条 JSON，记录单次评测的完整快照。可用于跨次评测对比 / 失败案例回溯。

## 性能与并行度

- 默认 Streamlit 侧边栏选择 1 / 2 / 4 / 8 / 16 路并行
- 单 session 评测耗时约 30 秒（14 规则 × 3 采样，含两步 conditional ×2 调用）
- DeepSeek 接口在 4-8 路并行下稳定；16 路偶发限流（已有 retry 兜底）
- build_simulation_plan 阶段（rule_parser + agent_spec + placeholders + scenario_selector）仍是顺序 LLM 调用，单 prompt 准备约 30-60 秒

## 常见问题

| 现象 | 处理 |
|---|---|
| `RuleParser 调用 ... 失败或超时` | 检查网络；调高 `LLM_REQUEST_TIMEOUT_SECONDS` 到 180 或更高 |
| `模型输出不是合法 JSON` | 系统会自动按 `LLM_PARSE_MAX_RETRIES` 重发修复请求；若仍失败检查 prompt 长度 |
| 对话型 Agent 直接输出自然语言 | 系统会自动降级到 plain-reply 解析，无需手动处理 |
| Conditional 规则全部 `trigger_failed` | 模拟器没把规则触发起来；检查 `coverage_report.trigger_failure_rate`；可能需要换 case_type 或 persona |
| 并行度太高频繁限流 | 降到 4-8 路；retry 兜底但响应变慢 |

## 添加自定义内容

### 添加新的 deterministic check

1. 在 `src/deterministic_checks.py` 的 `CheckType` enum 加新值
2. 写检查函数（参考 `_check_required_opening`）
3. 注册到 `_DISPATCH` 字典
4. 在 `src/rule_parser.py` 的 `_PARSER_INSTRUCTIONS` 提示中说明新 check 的触发条件

### 添加新的用户画像

1. `src/persona.py` 的 `PersonaType` Literal 加新类型
2. 添加 `UserPersona` 实例到 `ALL_PERSONAS`
3. 更新 `src/test_case_generator.py` 的 `_select_profile_type` / `_build_profile` 加映射
4. 在 `src/scenario_selector.py` 的 prompt 中加入新画像描述

### 切换 LLM 提供商

1. 调整 `agents.py:_get_base_url` / `_get_api_key` 读取新的 env vars
2. 全局替换 8 处 `model="deepseek-..."` 硬编码
3. 检查目标模型是否支持 `reasoning_effort` 参数（OpenAI / Claude 不支持，需要在 `_chat_completion` 加 provider 判断剥离）
