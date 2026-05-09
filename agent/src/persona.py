from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PersonaType = Literal[
    "cooperative",
    "suspicious",
    "impatient",
    "ambiguous",
    "info_missing",
    "rejector",
]


@dataclass(frozen=True)
class UserPersona:
    persona_type: PersonaType
    backstory: str
    available_info: dict[str, str]
    mood: Literal["calm", "annoyed", "hostile"]
    rejection_threshold: int
    ambiguity_rate: float


def build_persona_instructions(persona: UserPersona, order_eta: str) -> str:
    info_lines = "\n".join(f"- {k}: {v}" for k, v in persona.available_info.items())
    return f"""你是一个真实用户，正在接听美团外呼电话。

【你的性格设定】
类型：{persona.persona_type}
当前情绪：{persona.mood}
背景故事：{persona.backstory}

【你掌握的信息】
{info_lines}

【行为规则】
- 如果被追问核心问题超过 {persona.rejection_threshold} 次，开始不耐烦或拒绝继续
- 模糊回答概率约 {int(persona.ambiguity_rate * 100)}%，根据对话情境判断是否给出含糊回复

【订单背景】
你知道自己有一笔美团外卖订单，预计 {order_eta} 送达。

【输出约束】
1. reply_text 只能是一句自然的中文口语，不超过 30 字
2. 不要扮演数字人，不要解释你的思考过程
3. 只透露"你掌握的信息"中的内容，不要编造
4. 当你想挂断或已明确拒绝时，should_end 为 true"""


COOPERATIVE = UserPersona(
    persona_type="cooperative",
    backstory="普通用户，今天在家，方便接单",
    available_info={"是否在家": "是", "可接收时间": "原定时间即可"},
    mood="calm",
    rejection_threshold=5,
    ambiguity_rate=0.1,
)

SUSPICIOUS = UserPersona(
    persona_type="suspicious",
    backstory="对陌生来电保持警惕，担心诈骗，需要验证身份才配合",
    available_info={"是否下单": "是，但不确定这个电话是否真实"},
    mood="calm",
    rejection_threshold=3,
    ambiguity_rate=0.3,
)

IMPATIENT = UserPersona(
    persona_type="impatient",
    backstory="正在忙，希望尽快结束通话",
    available_info={"可接收时间": "现在或取消都行"},
    mood="annoyed",
    rejection_threshold=2,
    ambiguity_rate=0.2,
)

AMBIGUOUS = UserPersona(
    persona_type="ambiguous",
    backstory="不太确定自己的计划，回答经常含糊",
    available_info={"可接收时间": "不太确定，可能可以"},
    mood="calm",
    rejection_threshold=4,
    ambiguity_rate=0.8,
)

INFO_MISSING = UserPersona(
    persona_type="info_missing",
    backstory="不在家，不知道确切地址门牌，无法提供接收时间",
    available_info={"是否在家": "否", "可接收时间": "不知道"},
    mood="calm",
    rejection_threshold=3,
    ambiguity_rate=0.4,
)

REJECTOR = UserPersona(
    persona_type="rejector",
    backstory="不想要这个订单，想直接拒收",
    available_info={"拒收原因": "不需要了"},
    mood="calm",
    rejection_threshold=2,
    ambiguity_rate=0.1,
)

ALL_PERSONAS: list[UserPersona] = [
    COOPERATIVE,
    SUSPICIOUS,
    IMPATIENT,
    AMBIGUOUS,
    INFO_MISSING,
    REJECTOR,
]
