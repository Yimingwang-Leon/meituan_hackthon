from __future__ import annotations

import unittest

from src.placeholders import fill_placeholders


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


if __name__ == "__main__":
    unittest.main()
