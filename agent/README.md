# 美团外呼数字人评测系统

针对外呼数字人任务指令遵循效果的自动评估系统。输入一条任务指令，系统自动拆解规则、模拟多种用户场景下的对话，并输出可解释、可量化的评测报告。

## 系统流程

```
任务指令（instruction）
      ↓
  rule_parser          ← 自动拆解成原子规则（required / conditional / forbidden）
      ↓
  list[Rule]  ─────────────────────────────────→  evaluator（LLM Judge）
                                                         ↑
  outbound agent（只看原始 instruction）→ 对话记录 ──────┘
      ↑
  user simulator（6 种 Persona）
```

**关键设计原则**：呼出 agent 只看原始 instruction，不接触评测规则，避免"对着答案答题"。

## 目录结构

```
agent/
├── orders/          # 待测订单 JSON（8 条，覆盖 5 种场景）
├── instructions/    # 各场景的任务指令 + 人工验证用 criteria（5 个场景）
├── sessions/        # 自动生成的对话归档
├── src/
│   ├── rule_parser.py   # 指令 → 原子规则（LLM 拆解）
│   ├── agent.py         # 外呼数字人（被测对象）
│   ├── simulator.py     # 用户模拟器
│   ├── persona.py       # 6 种 Persona 定义
│   ├── evaluator.py     # LLM Judge，逐条规则打分
│   ├── session.py       # 单次对话会话管理
│   ├── orders.py        # 订单加载
│   ├── rules.py         # Rule 数据类 + severity 权重
│   └── types.py         # 共用数据类型
├── app.py           # Streamlit 可视化界面（推荐入口）
├── auto_main.py     # 命令行批量跑对话
├── evaluate_main.py # 命令行批量评估
├── test_parser.py   # 测试 rule_parser 在各场景的拆解效果
└── main.py          # 手动对话（人工扮演用户）
```

## 核心模块

| 模块 | 说明 |
|------|------|
| `rule_parser.py` | 把任意外呼 instruction 拆成原子规则，含 rule_type / severity / evaluation_hint |
| `agent.py` | 外呼数字人，`make_outbound_agent(instruction)` 接收任意指令 |
| `simulator.py` | 用户模拟器，每个 Persona 有独立性格、情绪、信息量 |
| `persona.py` | 6 种 Persona：cooperative / suspicious / impatient / ambiguous / info_missing / rejector |
| `evaluator.py` | LLM Judge，逐条规则输出 pass / fail / not_applicable + 证据；按 severity 加权计分 |

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖（在项目根目录）
python3 -m venv .venv
source .venv/bin/activate
pip install -r agent/requirements.txt

# 2. 配置 API Key
cp agent/.env.example agent/.env
# 编辑 agent/.env，填写 OPENAI_API_KEY
```

## 运行方式

**方式一：可视化界面（推荐）**

```bash
.venv/bin/streamlit run agent/app.py
```

在「输入」页粘贴任务指令，选择订单和 Persona，点击「开始评测」。结果边跑边展示，左侧对话记录、右侧规则评测。

**方式二：命令行**

```bash
cd agent

# 跑对话（可指定单条订单）
python auto_main.py                  # 全部订单 × 6 Persona
python auto_main.py confirm_001      # 仅指定订单

# 评估所有已保存的对话
python evaluate_main.py
```

**方式三：验证 rule_parser 效果**

```bash
cd agent
python test_parser.py
```

对比输出规则与 `instructions/*.json` 中的 `failure_criteria`，人工判断拆解质量。

## 订单格式

```json
{
  "userName": "王先生",
  "orderId": "confirm_001",
  "eta": "2026-05-11 19:30",
  "address": "上海市杨浦区某小区",
  "scenario": "confirm",
  "storeName": "老上海本帮菜",
  "deliveryNote": "无门铃，请电话联系"
}
```

`scenario` 字段对应 `instructions/` 下的同名 JSON，命令行模式下用于加载对应指令。

## 评分机制

规则按 severity 加权：critical=3 / major=2 / minor=1

```
得分 = 通过规则的权重之和 / 适用规则的权重之和
```

`not_applicable` 的规则（触发条件未出现）不计入分母。

## Persona 说明

| Persona | 特征 |
|---------|------|
| cooperative | 配合，直接确认 |
| suspicious | 警惕陌生来电，需要验证身份 |
| impatient | 忙碌，希望快速结束 |
| ambiguous | 回答含糊，经常不确定 |
| info_missing | 不在家，无法提供关键信息 |
| rejector | 明确拒收，测试拒收分支 |

## 后续一些想法/待做

- 微调角色模型
- 对话角色性格拓展/补全

