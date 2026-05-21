from __future__ import annotations

import unittest

from src.placeholders import (
    Placeholder,
    PlaceholderExtraction,
    PlaceholderSet,
    PlaceholderType,
    PlaceholderValue,
    fill_placeholders,
    _normalize_extraction,
)


class PlaceholderFillTest(unittest.TestCase):
    def test_fill_same_identifier_across_markdown_unit_variants(self) -> None:
        instruction = (
            "单日 **X 单**；多日 **X天**；兜底 **X**；"
            "姓名 ${rider_name}；别名 [rider_name]。"
        )

        filled = fill_placeholders(
            instruction,
            {
                "X": "20",
                "rider_name": "张家豪",
            },
        )

        self.assertIn("**20 单**", filled)
        self.assertIn("**20天**", filled)
        self.assertIn("**20**", filled)
        self.assertIn("姓名 张家豪", filled)
        self.assertIn("别名 张家豪", filled)

    def test_fill_does_not_replace_longer_identifier_prefix(self) -> None:
        instruction = "阈值 **X**，周期 **X_days**。"

        filled = fill_placeholders(
            instruction,
            {
                "X": "15",
                "X_days": "7",
            },
        )

        self.assertIn("**15**", filled)
        self.assertIn("**7**", filled)
        self.assertNotIn("**15_days**", filled)

    def test_normalize_drops_bare_currency_and_canonicalizes_raw_pattern(self) -> None:
        instruction = (
            "你好，请问是${rider_name}吗？"
            "单日合同每天至少完成 **X 单**；"
            "多日合同每天至少完成 **Y 单**。"
            "必须在前一天 **Z 点之前**取消；"
            "连续完成 **W 天**多日合同，"
            "每单多 +$ 元。"
        )
        extraction = PlaceholderExtraction(
            placeholders=[
                Placeholder(
                    raw_pattern="${rider_name}",
                    identifier="rider_name",
                    semantic="骑手姓名",
                    value_type=PlaceholderType.NAME,
                    unit=None,
                    confidence=0.95,
                ),
                Placeholder(
                    raw_pattern="**X 单**",
                    identifier="X",
                    semantic="单日合同单量",
                    value_type=PlaceholderType.INTEGER,
                    unit="单",
                    confidence=0.9,
                ),
                Placeholder(
                    raw_pattern="**Y 单**",
                    identifier="Y",
                    semantic="多日合同每天单量",
                    value_type=PlaceholderType.INTEGER,
                    unit="单",
                    confidence=0.9,
                ),
                Placeholder(
                    raw_pattern="**Z 点之前**",
                    identifier="deadline_time",
                    semantic="退出飞毛腿取消截止时间",
                    value_type=PlaceholderType.TIME,
                    unit="点",
                    confidence=0.9,
                ),
                Placeholder(
                    raw_pattern="**W 天**",
                    identifier="W",
                    semantic="奖励要求连续完成天数",
                    value_type=PlaceholderType.INTEGER,
                    unit="天",
                    confidence=0.9,
                ),
                Placeholder(
                    raw_pattern="+$ 元",
                    identifier="$",
                    semantic="金额符号",
                    value_type=PlaceholderType.AMOUNT,
                    unit="元",
                    confidence=0.8,
                ),
                Placeholder(
                    raw_pattern="+$ 元",
                    identifier="reward_amount",
                    semantic="额外奖励金额",
                    value_type=PlaceholderType.AMOUNT,
                    unit="元",
                    confidence=0.8,
                ),
            ],
            sets=[
                PlaceholderSet(
                    set_id="set_1",
                    label="标准",
                    scenario_hint="标准测试",
                    values=[
                        PlaceholderValue(identifier="rider_name", value="张师傅"),
                        PlaceholderValue(identifier="X", value="20"),
                        PlaceholderValue(identifier="Y", value="10"),
                        PlaceholderValue(identifier="deadline_time", value="22:00"),
                        PlaceholderValue(identifier="W", value="7"),
                        PlaceholderValue(identifier="$", value="1"),
                        PlaceholderValue(identifier="reward_amount", value="1"),
                    ],
                )
            ],
        )

        normalized = _normalize_extraction(extraction, source_instruction=instruction)

        self.assertEqual(
            [placeholder.identifier for placeholder in normalized.placeholders],
            ["rider_name", "X", "Y", "Z", "W"],
        )
        self.assertEqual(
            [value.identifier for value in normalized.sets[0].values],
            ["rider_name", "X", "Y", "Z", "W"],
        )
        self.assertEqual(normalized.sets[0].values[3].value, "22:00")

    def test_normalize_uses_source_syntax_as_authority_when_llm_misses_items(self) -> None:
        instruction = (
            "你好，请问是${rider_name}吗？"
            "单日合同每天至少完成 **X 单**；"
            "多日合同每天至少完成 **Y 单**。"
            "必须在前一天 **Z 点之前**取消；"
            "连续完成 **W 天**多日合同，每单多 +$ 元。"
        )
        extraction = PlaceholderExtraction(
            placeholders=[
                Placeholder(
                    raw_pattern="${rider_name}",
                    identifier="rider_name",
                    semantic="骑手姓名",
                    value_type=PlaceholderType.NAME,
                    unit=None,
                    confidence=0.9,
                ),
                Placeholder(
                    raw_pattern="+$ 元",
                    identifier="bonus_amount",
                    semantic="模型脑补金额变量",
                    value_type=PlaceholderType.AMOUNT,
                    unit="元",
                    confidence=0.9,
                ),
            ],
            sets=[
                PlaceholderSet(
                    set_id="set_1",
                    label="标准",
                    scenario_hint="标准测试",
                    values=[
                        PlaceholderValue(identifier="rider_name", value="张师傅"),
                        PlaceholderValue(identifier="bonus_amount", value="1"),
                    ],
                )
            ],
        )

        normalized = _normalize_extraction(
            extraction,
            source_instruction=instruction,
            num_sets=1,
        )

        self.assertEqual(
            [placeholder.identifier for placeholder in normalized.placeholders],
            ["rider_name", "X", "Y", "Z", "W"],
        )
        self.assertEqual(
            [value.identifier for value in normalized.sets[0].values],
            ["rider_name", "X", "Y", "Z", "W"],
        )
        self.assertNotIn(
            "bonus_amount",
            [value.identifier for value in normalized.sets[0].values],
        )

    def test_bold_regular_emphasis_is_not_a_placeholder(self) -> None:
        instruction = "请保持**礼貌**，不要把 +$ 元当变量，也不要误判普通加粗。"
        extraction = PlaceholderExtraction(
            placeholders=[
                Placeholder(
                    raw_pattern="**礼貌**",
                    identifier="礼貌",
                    semantic="普通强调文本",
                    value_type=PlaceholderType.OTHER,
                    unit=None,
                    confidence=0.9,
                )
            ],
            sets=[],
        )

        normalized = _normalize_extraction(extraction, source_instruction=instruction)

        self.assertEqual(normalized.placeholders, [])
        self.assertEqual(normalized.sets[0].set_id, "set_default")


if __name__ == "__main__":
    unittest.main()
