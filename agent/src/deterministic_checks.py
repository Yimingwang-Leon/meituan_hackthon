"""确定性检查器。

设计原则：
- checker 实现是写死的有限库（max_chars / forbidden_keywords / ...）
- 每条规则用哪个 checker + 什么参数由 LLM 动态生成
- 没有匹配 checker 的规则 → checks 留空，evaluator 自动 fallback 到 LLM judge
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from .types import SessionArchive


class CheckType(str, Enum):
    MAX_CHARS = "max_chars"
    FORBIDDEN_KEYWORDS = "forbidden_keywords"
    REQUIRED_OPENING = "required_opening"
    CLOSING_PHRASE = "closing_phrase"
    PII_LEAK = "pii_leak"


class PiiType(str, Enum):
    PHONE = "phone"
    ID_CARD = "id_card"
    BANK_CARD = "bank_card"


# 预定义 PII 正则
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "phone": re.compile(r"(?<!\d)1\d{10}(?!\d)"),                    # 11 位手机号
    "id_card": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),             # 18 位身份证
    "bank_card": re.compile(r"(?<!\d)\d{15,19}(?!\d)"),              # 15-19 位银行卡
}


class DeterministicCheck(BaseModel):
    check_type: CheckType = Field(description="检查类型")
    description: str = Field(description="人类可读的检查目标说明")
    speaker: Literal["agent", "user", "both"] = Field(
        default="agent", description="检查哪一方的发言（默认 agent）"
    )

    # 不同 check 用其中一部分字段，未用字段填默认值
    threshold: int = Field(default=0, description="阈值（max_chars 用，0 表示未设置）")
    keywords: list[str] = Field(
        default_factory=list,
        description="关键词列表（forbidden_keywords / required_opening / closing_phrase 用）",
    )
    pii_types: list[PiiType] = Field(
        default_factory=list,
        description="PII 检查类型（pii_leak 用）",
    )


class CheckOutcome(BaseModel):
    """单次 checker 运行的结果。结构对齐 evaluator.JudgeOutput 以复用聚合逻辑。"""

    triggered: bool
    trigger_turn: int
    response_turn: int
    compliant: bool
    evidence: str


def _iter_turns(archive: SessionArchive, speaker: str):
    """生成 (turn_idx, entry) 序列，按 speaker 过滤。"""
    for i, entry in enumerate(archive.transcript, start=1):
        if speaker == "both" or entry.speaker == speaker:
            yield i, entry


def _check_max_chars(check: DeterministicCheck, archive: SessionArchive) -> CheckOutcome:
    limit = check.threshold
    for i, entry in _iter_turns(archive, check.speaker):
        n = len(entry.text)
        if n > limit:
            return CheckOutcome(
                triggered=True,
                trigger_turn=i,
                response_turn=i,
                compliant=False,
                evidence=f"第{i}轮{entry.speaker}回复 {n} 字，超过 {limit} 字限制",
            )
    return CheckOutcome(
        triggered=True,
        trigger_turn=0,
        response_turn=0,
        compliant=True,
        evidence=f"所有{check.speaker}回复均在 {limit} 字以内",
    )


def _check_forbidden_keywords(check: DeterministicCheck, archive: SessionArchive) -> CheckOutcome:
    for i, entry in _iter_turns(archive, check.speaker):
        for kw in check.keywords:
            if kw in entry.text:
                return CheckOutcome(
                    triggered=True,
                    trigger_turn=i,
                    response_turn=i,
                    compliant=False,
                    evidence=f"第{i}轮{entry.speaker}出现禁用词「{kw}」",
                )
    return CheckOutcome(
        triggered=True,
        trigger_turn=0,
        response_turn=0,
        compliant=True,
        evidence=f"未检测到禁用词：{', '.join(check.keywords)}",
    )


def _check_required_opening(check: DeterministicCheck, archive: SessionArchive) -> CheckOutcome:
    # 取第一轮 agent 发言（必定是开场）
    first_agent = next(
        ((i, e) for i, e in enumerate(archive.transcript, start=1) if e.speaker == "agent"),
        None,
    )
    if first_agent is None:
        return CheckOutcome(
            triggered=False,
            trigger_turn=0,
            response_turn=0,
            compliant=False,
            evidence="对话中没有数字人发言",
        )
    i, entry = first_agent
    for phrase in check.keywords:
        if phrase in entry.text:
            return CheckOutcome(
                triggered=True,
                trigger_turn=i,
                response_turn=i,
                compliant=True,
                evidence=f"第{i}轮开场包含「{phrase}」",
            )
    return CheckOutcome(
        triggered=True,
        trigger_turn=i,
        response_turn=i,
        compliant=False,
        evidence=f"开场未包含必需话术：{', '.join(check.keywords)}",
    )


def _check_closing_phrase(check: DeterministicCheck, archive: SessionArchive) -> CheckOutcome:
    last_agent = None
    last_idx = 0
    for i, entry in enumerate(archive.transcript, start=1):
        if entry.speaker == "agent":
            last_agent = entry
            last_idx = i
    if last_agent is None:
        return CheckOutcome(
            triggered=False,
            trigger_turn=0,
            response_turn=0,
            compliant=False,
            evidence="对话中没有数字人发言",
        )
    for phrase in check.keywords:
        if phrase in last_agent.text:
            return CheckOutcome(
                triggered=True,
                trigger_turn=last_idx,
                response_turn=last_idx,
                compliant=True,
                evidence=f"第{last_idx}轮结束语包含「{phrase}」",
            )
    return CheckOutcome(
        triggered=True,
        trigger_turn=last_idx,
        response_turn=last_idx,
        compliant=False,
        evidence=f"结束语未包含礼貌话术：{', '.join(check.keywords)}",
    )


def _check_pii_leak(check: DeterministicCheck, archive: SessionArchive) -> CheckOutcome:
    types_to_check = check.pii_types or list(PiiType)
    for i, entry in _iter_turns(archive, check.speaker):
        for pii_type in types_to_check:
            pattern = _PII_PATTERNS.get(pii_type.value)
            if pattern is None:
                continue
            m = pattern.search(entry.text)
            if m:
                return CheckOutcome(
                    triggered=True,
                    trigger_turn=i,
                    response_turn=i,
                    compliant=False,
                    evidence=f"第{i}轮{entry.speaker}泄露 {pii_type.value}：{m.group()[:4]}***",
                )
    return CheckOutcome(
        triggered=True,
        trigger_turn=0,
        response_turn=0,
        compliant=True,
        evidence=f"未检测到 PII 泄露（{', '.join(t.value for t in types_to_check)}）",
    )


_DISPATCH = {
    CheckType.MAX_CHARS: _check_max_chars,
    CheckType.FORBIDDEN_KEYWORDS: _check_forbidden_keywords,
    CheckType.REQUIRED_OPENING: _check_required_opening,
    CheckType.CLOSING_PHRASE: _check_closing_phrase,
    CheckType.PII_LEAK: _check_pii_leak,
}


def run_check(check: DeterministicCheck, archive: SessionArchive) -> CheckOutcome:
    """跑单个 checker。"""
    fn = _DISPATCH[check.check_type]
    return fn(check, archive)


def run_checks_combined(
    checks: list[DeterministicCheck], archive: SessionArchive
) -> CheckOutcome:
    """跑一组 checker，任一 fail 即整体 fail。证据合并成一段。"""
    outcomes = [run_check(c, archive) for c in checks]

    # 任一不合规 → 整体不合规
    failed = [o for o in outcomes if not o.compliant]
    if failed:
        first = failed[0]
        evidences = "；".join(o.evidence for o in failed)
        return CheckOutcome(
            triggered=True,
            trigger_turn=first.trigger_turn,
            response_turn=first.response_turn,
            compliant=False,
            evidence=evidences[:200],
        )

    # 全部合规
    evidences = "；".join(o.evidence for o in outcomes)
    return CheckOutcome(
        triggered=True,
        trigger_turn=0,
        response_turn=0,
        compliant=True,
        evidence=evidences[:200],
    )
