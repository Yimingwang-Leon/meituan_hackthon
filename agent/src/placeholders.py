from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
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


@dataclass(frozen=True)
class _PlaceholderOccurrence:
    raw_pattern: str
    identifier: str
    unit: str | None
    value_type: PlaceholderType
    start: int


_EXTRACTOR_INSTRUCTIONS = """\
你是 prompt 占位符分析专家。

# 任务
给定一段对话 Agent 的指令文本，识别其中的占位符，并为它们生成多组合理的测试值。

如果输入中提供了“已确定占位符清单”，该清单就是唯一可信列表：
- 只能为清单中的 identifier 生成语义和测试值
- 不要新增、删除、改名任何 identifier
- raw_pattern / identifier 必须照抄清单

# 阶段 1 · 识别占位符

常见占位符格式：
- ${name}                变量语法
- **X**, **X 单**, **Y 天**  markdown 加粗的单字母（常带单位，不同单位的相同符号算一个变量）
- [name], {{name}}, <name>   方括号/双花括号/尖括号

不要把裸货币符号识别为占位符：
- "$"、"¥"、"￥" 只是金额符号，不是变量名
- 例如 "+$ 元" 不应生成 identifier="$"
- 只有 "${amount}"、"**A 元**" 这类明确变量语法才算占位符

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

_extractor_agent = None


def extract_placeholders(
    instruction: str,
    agent_spec: AgentSpec | None = None,
    num_sets: int = 3,
) -> PlaceholderExtraction:
    """识别 instruction 中的占位符，并生成 num_sets 组测试值。"""
    from agents import Runner

    detected_placeholders = _placeholders_from_occurrences(
        _discover_placeholder_occurrences(instruction)
    )
    if not detected_placeholders:
        return _build_no_placeholder_extraction()

    parts = [f"【指令文本】\n{instruction}"]
    parts.append(
        "\n【已确定占位符清单】\n"
        + "\n".join(
            (
                f"- raw_pattern={placeholder.raw_pattern} | "
                f"identifier={placeholder.identifier} | "
                f"value_type={placeholder.value_type.value} | "
                f"unit={placeholder.unit or ''}"
            )
            for placeholder in detected_placeholders
        )
    )
    if agent_spec is not None:
        parts.append(
            "\n【业务上下文】\n"
            f"agent_type: {agent_spec.agent_type}\n"
            f"domain: {agent_spec.domain}\n"
            f"main_task: {agent_spec.main_task}"
        )
    parts.append(f"\n请按要求生成 {num_sets} 组测试场景。")

    result = Runner.run_sync(_get_extractor_agent(), "\n".join(parts))
    output = result.final_output
    if not isinstance(output, PlaceholderExtraction):
        raise RuntimeError("占位符提取返回了意外结果类型")
    return _normalize_extraction(
        output,
        source_instruction=instruction,
        num_sets=num_sets,
        detected_placeholders=detected_placeholders,
    )


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
        result = _replace_markdown_placeholder(result, key, val)

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


def _discover_placeholder_occurrences(instruction: str) -> list[_PlaceholderOccurrence]:
    occurrences: list[_PlaceholderOccurrence] = []
    occupied: list[tuple[int, int]] = []

    def add_occurrence(
        start: int,
        end: int,
        raw_pattern: str,
        identifier: str,
        unit: str | None = None,
    ) -> None:
        if not _is_valid_placeholder_identifier(identifier):
            return
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            return
        occurrences.append(
            _PlaceholderOccurrence(
                raw_pattern=raw_pattern,
                identifier=identifier,
                unit=unit,
                value_type=_infer_placeholder_type(identifier, raw_pattern, unit),
                start=start,
            )
        )
        occupied.append((start, end))

    variable_patterns = [
        re.compile(r"\$\{\s*([A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*)\s*\}"),
        re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*)\s*\}\}"),
        re.compile(r"(?<!\!)\[\s*([A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*)\s*\](?!\()"),
        re.compile(r"<\s*([A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*)\s*>"),
    ]
    for pattern in variable_patterns:
        for match in pattern.finditer(instruction):
            add_occurrence(
                match.start(),
                match.end(),
                match.group(0),
                match.group(1).strip(),
            )

    for match in re.finditer(r"\*\*\s*(.*?)\s*\*\*", instruction):
        parsed = _parse_bold_placeholder_content(match.group(1))
        if parsed is None:
            continue
        identifier, unit = parsed
        add_occurrence(match.start(), match.end(), match.group(0), identifier, unit)

    deduped: dict[str, _PlaceholderOccurrence] = {}
    ordered_ids: list[str] = []
    for occurrence in sorted(occurrences, key=lambda item: item.start):
        if occurrence.identifier in deduped:
            continue
        deduped[occurrence.identifier] = occurrence
        ordered_ids.append(occurrence.identifier)
    return [deduped[identifier] for identifier in ordered_ids]


def _parse_bold_placeholder_content(content: str) -> tuple[str, str | None] | None:
    normalized = content.strip()
    if not normalized:
        return None
    token_match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)", normalized)
    if not token_match:
        return None

    token = token_match.group(1)
    if not _is_bold_placeholder_token(token):
        return None

    rest = normalized[token_match.end():].strip()
    if rest and re.search(r"[A-Za-z0-9_]", rest):
        return None
    unit = _normalize_unit(rest)
    return token, unit


def _is_bold_placeholder_token(token: str) -> bool:
    if len(token) == 1 and token.isupper():
        return True
    return "_" in token or any(char.isdigit() for char in token)


def _normalize_unit(text: str) -> str | None:
    cleaned = text.strip(" ：:，,。；;（）()[]【】")
    return cleaned or None


def _placeholders_from_occurrences(
    occurrences: list[_PlaceholderOccurrence],
) -> list[Placeholder]:
    return [
        Placeholder(
            raw_pattern=occurrence.raw_pattern,
            identifier=occurrence.identifier,
            semantic=f"根据上下文填写 {occurrence.identifier}",
            value_type=occurrence.value_type,
            unit=occurrence.unit,
            confidence=1.0,
        )
        for occurrence in occurrences
    ]


def _infer_placeholder_type(
    identifier: str,
    raw_pattern: str,
    unit: str | None,
) -> PlaceholderType:
    haystack = f"{identifier} {raw_pattern} {unit or ''}".lower()
    if any(token in haystack for token in ["name", "姓名", "名字", "rider_name", "user_name"]):
        return PlaceholderType.NAME
    if any(token in haystack for token in ["date", "日期"]):
        return PlaceholderType.DATE
    if any(token in haystack for token in ["time", "时间", "点", "时刻"]):
        return PlaceholderType.TIME
    if any(token in haystack for token in ["金额", "价格", "元", "¥", "$", "amount", "price"]):
        return PlaceholderType.AMOUNT
    if any(token in haystack for token in ["地址", "位置", "城市", "location", "address", "city"]):
        return PlaceholderType.LOCATION
    if any(token in haystack for token in ["天", "小时", "分钟", "时长", "duration"]):
        return PlaceholderType.INTEGER
    if unit or len(identifier) == 1 or any(char.isdigit() for char in identifier):
        return PlaceholderType.INTEGER
    return PlaceholderType.OTHER


def _build_no_placeholder_extraction() -> PlaceholderExtraction:
    return PlaceholderExtraction(
        placeholders=[],
        sets=[
            PlaceholderSet(
                set_id="set_default",
                label="默认场景",
                scenario_hint="原始指令没有可替换占位符",
                values=[],
            )
        ],
    )


def _normalize_extraction(
    extraction: PlaceholderExtraction,
    source_instruction: str | None = None,
    num_sets: int | None = None,
    detected_placeholders: list[Placeholder] | None = None,
) -> PlaceholderExtraction:
    authoritative_placeholders = detected_placeholders
    if authoritative_placeholders is None and source_instruction is not None:
        authoritative_placeholders = _placeholders_from_occurrences(
            _discover_placeholder_occurrences(source_instruction)
        )

    if authoritative_placeholders is not None:
        if not authoritative_placeholders:
            return _build_no_placeholder_extraction()
        placeholders, identifier_aliases = _merge_detected_placeholders(
            authoritative_placeholders,
            extraction.placeholders,
            source_instruction=source_instruction,
        )
    else:
        placeholders, identifier_aliases = _dedupe_placeholders(
            extraction.placeholders,
            source_instruction=source_instruction,
        )
    ordered_ids = [placeholder.identifier for placeholder in placeholders]
    raw_sets = list(extraction.sets[:num_sets]) if num_sets else list(extraction.sets)
    target_set_count = num_sets or len(raw_sets) or 1
    while len(raw_sets) < target_set_count:
        set_index = len(raw_sets) + 1
        raw_sets.append(
            PlaceholderSet(
                set_id=f"set_{set_index}",
                label=_fallback_set_label(set_index),
                scenario_hint=_fallback_set_hint(set_index),
                values=[],
            )
        )

    normalized_sets = [
        PlaceholderSet(
            set_id=placeholder_set.set_id,
            label=placeholder_set.label,
            scenario_hint=placeholder_set.scenario_hint,
            values=_dedupe_placeholder_values(
                placeholder_set.values,
                ordered_ids,
                identifier_aliases,
                placeholders,
                set_index,
            ),
        )
        for set_index, placeholder_set in enumerate(raw_sets, start=1)
    ]
    return PlaceholderExtraction(
        placeholders=placeholders,
        sets=normalized_sets,
    )


def _replace_markdown_placeholder(text: str, identifier: str, value: str) -> str:
    """替换 markdown 粗体中的占位符标识符，保留原单位和空格。

    支持：
    - **X**
    - **X 单**
    - **X单**
    - **name**

    只替换粗体内容起始处的 identifier，自身后的单位/说明原样保留。
    """
    pattern = re.compile(
        rf"(\*\*\s*){re.escape(identifier)}(?![A-Za-z0-9_])",
    )
    return pattern.sub(lambda match: f"{match.group(1)}{value}", text)


def _get_extractor_agent():
    global _extractor_agent

    if _extractor_agent is None:
        from agents import Agent

        _extractor_agent = Agent(
            name="PlaceholderExtractor",
            instructions=_EXTRACTOR_INSTRUCTIONS,
            model="deepseek-v4-flash",
            output_type=PlaceholderExtraction,
        )

    return _extractor_agent


def _dedupe_placeholders(
    placeholders: list[Placeholder],
    source_instruction: str | None = None,
) -> tuple[list[Placeholder], dict[str, str]]:
    deduped: dict[str, Placeholder] = {}
    ordered_ids: list[str] = []
    identifier_aliases: dict[str, str] = {}

    for placeholder in placeholders:
        raw_identifier = placeholder.identifier.strip()
        identifier = _resolve_placeholder_identifier(placeholder, source_instruction)
        if not _is_valid_placeholder_identifier(identifier):
            continue
        if source_instruction and not _is_supported_placeholder_occurrence(
            identifier,
            source_instruction,
        ):
            continue
        if raw_identifier:
            identifier_aliases[raw_identifier] = identifier
        normalized_placeholder = _copy_placeholder_with_identifier(
            placeholder,
            identifier,
        )
        if identifier not in deduped:
            deduped[identifier] = normalized_placeholder
            ordered_ids.append(identifier)
            continue
        if normalized_placeholder.confidence > deduped[identifier].confidence:
            deduped[identifier] = normalized_placeholder

    return [deduped[identifier] for identifier in ordered_ids], identifier_aliases


def _merge_detected_placeholders(
    detected_placeholders: list[Placeholder],
    llm_placeholders: list[Placeholder],
    source_instruction: str | None = None,
) -> tuple[list[Placeholder], dict[str, str]]:
    llm_by_id: dict[str, Placeholder] = {}
    identifier_aliases: dict[str, str] = {}
    for placeholder in llm_placeholders:
        raw_identifier = placeholder.identifier.strip()
        identifier = _resolve_placeholder_identifier(placeholder, source_instruction)
        if not identifier:
            continue
        if raw_identifier:
            identifier_aliases[raw_identifier] = identifier
        if identifier not in llm_by_id or placeholder.confidence > llm_by_id[identifier].confidence:
            llm_by_id[identifier] = placeholder

    merged: list[Placeholder] = []
    for detected in detected_placeholders:
        llm_placeholder = llm_by_id.get(detected.identifier)
        if llm_placeholder is None:
            merged.append(detected)
            continue
        merged.append(
            Placeholder(
                raw_pattern=detected.raw_pattern,
                identifier=detected.identifier,
                semantic=llm_placeholder.semantic or detected.semantic,
                value_type=(
                    detected.value_type
                    if detected.value_type != PlaceholderType.OTHER
                    else llm_placeholder.value_type
                ),
                unit=detected.unit or llm_placeholder.unit,
                confidence=max(detected.confidence, llm_placeholder.confidence),
            )
        )

    for placeholder in detected_placeholders:
        identifier_aliases.setdefault(placeholder.identifier, placeholder.identifier)
    return merged, identifier_aliases


def _dedupe_placeholder_values(
    values: list[PlaceholderValue],
    ordered_ids: list[str],
    identifier_aliases: dict[str, str] | None = None,
    placeholders: list[Placeholder] | None = None,
    set_index: int = 1,
) -> list[PlaceholderValue]:
    deduped: dict[str, PlaceholderValue] = {}
    aliases = identifier_aliases or {}
    placeholder_by_id = {
        placeholder.identifier: placeholder
        for placeholder in (placeholders or [])
    }

    for value in values:
        raw_identifier = value.identifier.strip()
        identifier = aliases.get(raw_identifier, raw_identifier)
        if not identifier or identifier not in ordered_ids:
            continue
        if identifier not in deduped:
            deduped[identifier] = _copy_placeholder_value_with_identifier(
                value,
                identifier,
            )

    for identifier in ordered_ids:
        if identifier in deduped:
            continue
        placeholder = placeholder_by_id.get(identifier)
        if placeholder is None:
            continue
        deduped[identifier] = PlaceholderValue(
            identifier=identifier,
            value=_fallback_placeholder_value(placeholder, set_index),
        )

    return [deduped[identifier] for identifier in ordered_ids if identifier in deduped]


def _resolve_placeholder_identifier(
    placeholder: Placeholder,
    source_instruction: str | None,
) -> str:
    raw_identifier = placeholder.identifier.strip()
    if source_instruction:
        canonical = _identifier_from_raw_pattern(placeholder.raw_pattern)
        if canonical and _is_supported_placeholder_occurrence(canonical, source_instruction):
            return canonical
    return raw_identifier


def _fallback_set_label(set_index: int) -> str:
    return {
        1: "标准场景",
        2: "边界场景",
        3: "高压场景",
    }.get(set_index, f"场景 {set_index}")


def _fallback_set_hint(set_index: int) -> str:
    return {
        1: "使用常见中位数值覆盖主流程",
        2: "使用偏边界值覆盖临界条件",
        3: "使用偏高压值覆盖鲁棒性",
    }.get(set_index, "补齐缺失测试值")


def _fallback_placeholder_value(placeholder: Placeholder, set_index: int) -> str:
    variant = max(1, set_index)
    value_type = placeholder.value_type
    unit = placeholder.unit or ""

    if value_type == PlaceholderType.NAME:
        return ["张师傅", "李师傅", "王师傅"][(variant - 1) % 3]
    if value_type == PlaceholderType.TIME:
        return ["22:00", "21:00", "23:00"][(variant - 1) % 3]
    if value_type == PlaceholderType.DATE:
        return ["2026-06-01", "2026-06-02", "2026-06-03"][(variant - 1) % 3]
    if value_type == PlaceholderType.AMOUNT:
        return ["1", "2", "5"][(variant - 1) % 3]
    if value_type == PlaceholderType.LOCATION:
        return ["望京站", "中关村站", "国贸站"][(variant - 1) % 3]
    if value_type in {PlaceholderType.INTEGER, PlaceholderType.DURATION}:
        if "单" in unit:
            return ["10", "1", "30"][(variant - 1) % 3]
        if "天" in unit:
            return ["5", "1", "7"][(variant - 1) % 3]
        if "点" in unit:
            return ["22", "21", "23"][(variant - 1) % 3]
        return ["3", "1", "10"][(variant - 1) % 3]
    return ["标准值", "边界值", "高压值"][(variant - 1) % 3]


def _identifier_from_raw_pattern(raw_pattern: str) -> str | None:
    raw = raw_pattern.strip()
    wrappers = [
        (r"^\$\{\s*([A-Za-z0-9_\u4e00-\u9fff]+)\s*\}$", 1),
        (r"^\{\{\s*([A-Za-z0-9_\u4e00-\u9fff]+)\s*\}\}$", 1),
        (r"^\[\s*([A-Za-z0-9_\u4e00-\u9fff]+)\s*\]$", 1),
        (r"^<\s*([A-Za-z0-9_\u4e00-\u9fff]+)\s*>$", 1),
    ]
    for pattern, group_index in wrappers:
        match = re.match(pattern, raw)
        if match:
            return match.group(group_index).strip()

    bold_match = re.match(r"^\*\*\s*(.*?)\s*\*\*$", raw)
    if not bold_match:
        return None

    content = bold_match.group(1).strip()
    ascii_match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)(?=$|\s|[\u4e00-\u9fff])", content)
    if ascii_match:
        return ascii_match.group(1)

    chinese_match = re.match(r"^([\u4e00-\u9fff]+)(?=$|\s)", content)
    if chinese_match:
        return chinese_match.group(1)
    return None


def _copy_placeholder_with_identifier(
    placeholder: Placeholder,
    identifier: str,
) -> Placeholder:
    if hasattr(placeholder, "model_copy"):
        return placeholder.model_copy(update={"identifier": identifier})
    return placeholder.copy(update={"identifier": identifier})


def _copy_placeholder_value_with_identifier(
    value: PlaceholderValue,
    identifier: str,
) -> PlaceholderValue:
    if hasattr(value, "model_copy"):
        return value.model_copy(update={"identifier": identifier})
    return value.copy(update={"identifier": identifier})


def _is_valid_placeholder_identifier(identifier: str) -> bool:
    if not identifier:
        return False
    if identifier in {"$", "¥", "￥"}:
        return False
    if re.fullmatch(r"[$¥￥+\-.,，。:：;；/\\|()\[\]{}<>]+", identifier):
        return False
    return bool(re.search(r"[A-Za-z0-9_\u4e00-\u9fff]", identifier))


def _is_supported_placeholder_occurrence(identifier: str, instruction: str) -> bool:
    escaped = re.escape(identifier)
    supported_patterns = [
        rf"\$\{{\s*{escaped}\s*\}}",
        rf"\{{\{{\s*{escaped}\s*\}}\}}",
        rf"\[\s*{escaped}\s*\]",
        rf"<\s*{escaped}\s*>",
        rf"\*\*\s*{escaped}(?![A-Za-z0-9_])[^*]*\*\*",
    ]
    return any(re.search(pattern, instruction) for pattern in supported_patterns)
