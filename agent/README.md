# 对话 Agent 评测系统

针对任务型对话 Agent 指令遵循效果的自动评估系统。输入一条任务指令，系统自动拆解原子规则、模拟多种用户场景对话、多采样 LLM 评判，输出**可解释、可量化、可衡量可靠性**的评测报告。

## 系统流程

```
任务指令（instruction）
      ↓
  rule_parser          ← 自动拆解成原子规则（required / conditional / forbidden）
      ↓
  list[Rule]  ──────────────────────────────────→  evaluator（LLM Judge × N 采样）
                                                          ↑
  outbound agent（只看原始 instruction）→ 对话记录 ───────┘
      ↑
  user simulator（7 种 Persona）
```

**关键设计原则**：被测 agent 只看原始 instruction，不接触评测规则，避免"对着答案答题"。

## 评测能力矩阵

| 维度 | 实现方式 |
|------|---------|
| 可解释 | 每条规则独立判断，输出 pass / fail / not_applicable + 引用具体轮次的 evidence |
| 可量化 | 按 severity（critical/major/minor）加权打分；触发条件未出现的规则不计入分母 |
| 可靠性 | 每条规则多采样投票，输出**置信度**（多次 judge 一致率） |
| 模拟器充分性 | 7 种 Persona 覆盖主要分支；输出**条件规则触发覆盖率**，量化哪些分支被实际测试到 |

## 目录结构

```
agent/
├── instructions/    # 示例任务指令 JSON
├── memory/          # 逐次评测记忆（transcript + judger 判断）
├── sessions/        # 自动生成的对话归档
├── src/
│   ├── rule_parser.py   # 指令 → 原子规则（LLM 拆解）
│   ├── agent.py         # 被测对话 Agent 封装
│   ├── simulator.py     # 用户模拟器
│   ├── persona.py       # 7 种 Persona 定义
│   ├── evaluator.py     # 多采样 LLM Judge，按 severity 加权 + 触发覆盖率
│   ├── memory.py        # 评测记忆落盘（JSONL）
│   ├── session.py       # 单次对话会话管理
│   ├── rules.py         # Rule 数据类 + severity 权重
│   └── types.py         # 共用数据类型
├── app.py           # Streamlit 可视化界面（推荐入口）
├── auto_main.py     # 命令行批量跑对话
├── evaluate_main.py # 命令行批量评估
├── test_parser.py   # 测试 rule_parser 在各场景的拆解效果
└── main.py          # 手动对话（人工扮演用户）
```

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖（在项目根目录）
python3 -m venv .venv
source .venv/bin/activate
pip install -r agent/requirements.txt

# 2. 配置 API Key
cp agent/.env.example agent/.env
# 编辑 agent/.env，填写 DEEPSEEK_API_KEY
# DeepSeek 兼容 OpenAI 接口，base URL 使用 https://api.deepseek.com
```

## 运行方式

**方式一：可视化界面（推荐）**

```bash
.venv/bin/streamlit run agent/app.py
```

页面包含 8 个编号章节：

1. **任务指令** — 粘贴任意任务型对话 Agent prompt
2. **解析规则** — Bento 卡片展示规则总数 / 必做 / 条件 / 禁止
3. **实时执行** — 进度条 + 每个 session 完成后立即展示对话和评测
4. **评测仪表板** — 综合得分 / 条件覆盖率 / 置信度 / 会话数
5. **规则 × 角色热力图** — 红=低通过率（指令薄弱点）/ 绿=高通过率
6. **角色得分排名** — 哪类用户最容易暴露问题
7. **模拟器覆盖度** — 未触发的条件规则 + 各角色触发了哪些规则
8. **会话归档** — 逐 session 的对话记录 + 规则评测明细

侧边栏可控制：角色勾选 / 评测深度（1-3 次采样）/ 占位符场景数。

**方式二：命令行**

```bash
cd agent

# 跑对话
python auto_main.py
python auto_main.py instructions/confirm.json 3
python auto_main.py confirm 2
python auto_main.py "你是一名..."

# 评估所有已保存的对话
python evaluate_main.py
```

**方式三：验证 rule_parser 效果**

```bash
cd agent
python test_parser.py
```

对比输出规则与 `instructions/*.json` 中的 `failure_criteria`，人工判断拆解质量。

## 输入来源

系统不再依赖订单数据集。测试输入来自三部分：

1. 原始 instruction
2. instruction 中自动识别并填充的 placeholder sets
3. 根据解析规则自动生成的 `SimulationCase`

如果 prompt 中存在占位符，系统会自动生成多组 `scenario_context`，并同时提供给被测 Agent 与用户模拟器。

## 评分机制

规则按 severity 加权：critical=3 / major=2 / minor=1

```
得分 = 通过规则的权重之和 / 适用规则的权重之和
```

`not_applicable` 的规则（触发条件未出现）不计入分母。

## 多采样置信度

为了保证评估**可靠性**，每条规则可独立运行 N 次 LLM judge，按多数投票决定最终结果，并输出置信度：

```
置信度 = max(投票数) / N
```

- 🟢 全部一致（置信度 100%）
- 🟡 部分分歧（置信度 ≥ 66%）
- 🔴 严重分歧（置信度 < 66%，需人工复核）

侧边栏「评测深度」滑块控制 N（1=快速、3=可靠）。

## Persona 说明

| 类型 | 中文 | 特征 |
|---------|------|------|
| cooperative | 配合型 | 配合，直接确认 |
| suspicious | 警惕型 | 警惕陌生来电，需要验证身份 |
| impatient | 急躁型 | 忙碌，希望快速结束 |
| ambiguous | 模糊型 | 回答含糊，经常不确定 |
| info_missing | 缺信息型 | 不在家，无法提供关键信息 |
| rejector | 拒收型 | 明确拒收，测试拒收分支 |
| hostile | 对抗型 | 试图套取内部信息或诱导数字人执行越权操作 |

## 已知限制

- **评测器目前顺序执行**（避免 `Runner.run_sync` 在多线程下与 OpenAI 客户端 asyncio 事件循环冲突）。彻底加速需要把整条链路改成 async + `asyncio.gather`，后续优化方向。
- **rule_parser 偶发原子性问题**：少数情况会把两个独立检查合并到一条规则里（如「最多追问两次且不透露内部状态」），可加 post-processing 检测「且/同时」类合并词来拆分。

## 后续一些想法/待做

- 微调角色模型
- 对话角色性格拓展/补全
- 整体改成 async 架构以并发加速
- 加 rule_parser 输出后的原子性后处理校验
- 用真实脱敏数据替换自造测试集
