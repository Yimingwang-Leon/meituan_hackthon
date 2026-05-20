from __future__ import annotations

import re
from enum import Enum

from agents import Agent, Runner
from pydantic import BaseModel, Field

from .agent_spec import AgentSpec


class PlaceholderType(str, Enum):
    NAME = "name"           # 人名 / 机构名
    INTEGER = "integer"     # 整数（单数、天数）
    TIME = "time"           # 时间点 23:00
    DURATION = "duration"   # 时长
    AMOUNT = "amount"       # 金额
    DATE = "date"           # 日期
    LOCATION = "location"   # 地址 / 位置
    OTHER = "other"


class Placeholder(BaseModel):
    raw_pattern: str = Field(
        description="原文中出现的完整模式，含装饰符。例 '${rider_name}' 或 '**X 单**'"
    )
    identifier: str = Field(
        description="标识符（去除装饰）。例 'rider_name' 或 'X'"
    )
    semantic: str = Field(
        description="结合上下文推断的语义说明。例 '骑手姓名' 或 '单日合同每天单数要求'"
    )
    value_type: PlaceholderType
    unit: str | None = Field(
        default=None, description="单位。例 '单'、'天'、'元'"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="对 identifier 与 semantic 推断的置信度"
    )


class PlaceholderValue(BaseModel):
    identifier: str = Field(description="对应 Placeholder 的 identifier")
    value: str = Field(description="该 identifier 在本组场景下的填充值")


class PlaceholderSet(BaseModel):
    set_id: str = Field(description="set_1, set_2, set_3 ...")
    label: str = Field(description="简短场景标签，例 '正常骑手 · 单日合同'")
    scenario_hint: str = Field(
        description="该组测试值的目标，一句话说明"
    )
    values: list[PlaceholderValue] = Field(
        description="所有 placeholders 的填充值列表"
    )

    def values_dict(self) -> dict[str, str]:
        return {v.identifier: v.value for v in self.values}


class PlaceholderExtraction(BaseModel):
    placeholders: list[Placeholder]
    sets: list[PlaceholderSet]


_EXTRACTOR_INSTRUCTIONS = """\
你是 prompt 占位符分析专家。

# 任务
给定一段对话 Agent 的指令文本，识别其中的占位符，并为它们生成多组合理的测试值。

# 阶段 1 · 识别占位符

常见占位符格式：
- ${name}                变量语法
- **X**, **X 单**, **Y 天**  markdown 加粗的单字母（常带单位，不同单位的相同符号算一个变量）
- [name], {{name}}, <name>   方括号/双花括号/尖括号

每个占位符必须给出：
- raw_pattern: 原文中完整匹配模式，保留装饰符（** $ {} 等）
- identifier: 净化标识符（去除装饰符）
- semantic: 结合上下文推断的语义，避免空泛
- value_type: 选择最匹配的类型枚举
- unit: 单位（如有）
- confidence: 0-1，结合上下文能多确定推断结果

# 阶段 2 · 生成测试场景

为占位符生成 N 组场景化测试值，每组 set_id 形如 set_1 / set_2 / set_3。

场景设计原则（按 set_id 顺序对应）：
- set_1 · 标准：业务最常见的中位数值，主流程覆盖
- set_2 · 边界：让某些条件分支被激活的极端值（刚好踩阈值）
- set_3 · 高压：偏大或偏紧的值，易触发 Agent 越权或失误

值的硬约束：
- 每组 values 是一个列表，元素是 {identifier, value} 结构
- 列表必须覆盖所有 placeholders 的 identifier（一一对应）
- 三组之间 value 要有显著差异
- INTEGER 类型只能用纯数字字符串（如 "20"）
- TIME 类型必须 HH:MM 格式（如 "23:00"）
- DATE 类型必须 YYYY-MM-DD 格式
- NAME 类型给真实风格的中文名 / 机构名
- AMOUNT 类型给纯数字字符串，单位放到 unit 字段

# 没有占位符时

如果指令完全没有占位符（如部分 SaaS 通知 prompt），返回：
- placeholders: []
- sets: 一组 set_id="set_default"，values={}
"""

_extractor_agent = Agent(
    name="PlaceholderExtractor",
    instructions=_EXTRACTOR_INSTRUCTIONS,
    model="gpt-5.4-nano",
    output_type=PlaceholderExtraction,
)


def extract_placeholders(
    instruction: str,
    agent_spec: AgentSpec | None = None,
    num_sets: int = 3,
) -> PlaceholderExtraction:
    """识别 instruction 中的占位符，并生成 num_sets 组测试值。"""
    parts = [f"【指令文本】\n{instruction}"]
    if agent_spec is not None:
        parts.append(
            "\n【业务上下文】\n"
            f"agent_type: {agent_spec.agent_type}\n"
            f"domain: {agent_spec.domain}\n"
            f"main_task: {agent_spec.main_task}"
        )
    parts.append(f"\n请按要求生成 {num_sets} 组测试场景。")

    result = Runner.run_sync(_extractor_agent, "\n".join(parts))
    output = result.final_output
    if not isinstance(output, PlaceholderExtraction):
        raise RuntimeError("占位符提取返回了意外结果类型")
    return _normalize_extraction(output)


def fill_placeholders(instruction: str, values: dict[str, str]) -> str:
    """根据 identifier → value 字典，替换 instruction 中的占位符。

    支持的占位符语法：
      ${name}      变量
      **X**, **X 单**, **X 天**   markdown 加粗的单字母+可选单位
      [name]       方括号
      {{name}}     双花括号
      <name>       尖括号
    """
    result = instruction

    # 按 key 长度倒序，避免短前缀先匹配影响长 key（如 X 在 XY 之前替换）
    sorted_items = sorted(values.items(), key=lambda kv: -len(kv[0]))

    for key, val in sorted_items:
        result = result.replace(f"${{{key}}}", val)
        result = result.replace(f"[{key}]", val)
        result = result.replace(f"{{{{{key}}}}}", val)
        result = result.replace(f"<{key}>", val)
        # ** 后紧跟 key 且 key 后是单词边界（空格或 *）
        pattern = rf"\*\*({re.escape(key)})\b"
        result = re.sub(pattern, f"**{val}", result)

    return result


def validate_placeholder_value(p: Placeholder, value: str) -> bool:
    """轻量类型校验。返回 True 表示值符合声明的类型。"""
    if not value:
        return False
    if p.value_type == PlaceholderType.INTEGER:
        return value.lstrip("-").isdigit()
    if p.value_type == PlaceholderType.TIME:
        return bool(re.match(r"^\d{1,2}:\d{2}$", value))
    if p.value_type == PlaceholderType.DATE:
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", value))
    if p.value_type == PlaceholderType.AMOUNT:
        return value.replace(".", "", 1).lstrip("-").isdigit()
    return True


def _normalize_extraction(extraction: PlaceholderExtraction) -> PlaceholderExtraction:
    placeholders = _dedupe_placeholders(extraction.placeholders)
    ordered_ids = [placeholder.identifier for placeholder in placeholders]
    normalized_sets = [
        PlaceholderSet(
            set_id=placeholder_set.set_id,
            label=placeholder_set.label,
            scenario_hint=placeholder_set.scenario_hint,
            values=_dedupe_placeholder_values(placeholder_set.values, ordered_ids),
        )
        for placeholder_set in extraction.sets
    ]
    return PlaceholderExtraction(
        placeholders=placeholders,
        sets=normalized_sets,
    )


def _dedupe_placeholders(placeholders: list[Placeholder]) -> list[Placeholder]:
    deduped: dict[str, Placeholder] = {}
    ordered_ids: list[str] = []

    for placeholder in placeholders:
        identifier = placeholder.identifier.strip()
        if not identifier:
            continue
        if identifier not in deduped:
            deduped[identifier] = placeholder
            ordered_ids.append(identifier)
            continue
        if placeholder.confidence > deduped[identifier].confidence:
            deduped[identifier] = placeholder

    return [deduped[identifier] for identifier in ordered_ids]


def _dedupe_placeholder_values(
    values: list[PlaceholderValue],
    ordered_ids: list[str],
) -> list[PlaceholderValue]:
    deduped: dict[str, PlaceholderValue] = {}
    extras_in_order: list[str] = []

    for value in values:
        identifier = value.identifier.strip()
        if not identifier:
            continue
        if identifier not in deduped:
            if identifier not in ordered_ids:
                extras_in_order.append(identifier)
            deduped[identifier] = value

    identifiers = ordered_ids + [identifier for identifier in extras_in_order if identifier not in ordered_ids]
    return [deduped[identifier] for identifier in identifiers if identifier in deduped]
