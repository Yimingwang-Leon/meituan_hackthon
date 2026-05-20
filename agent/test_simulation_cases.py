from __future__ import annotations

import unittest

from src.agent_spec import AgentSpec
from src.rules import Rule
from src.test_case_generator import build_test_cases


class TestCaseGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_spec = AgentSpec(
            agent_type="履约外呼数字人",
            domain="外卖履约",
            main_task="确认用户是否方便接收订单",
            workflow_rules=[],
            condition_rules=[],
            prohibited_rules=[],
            style_rules=[],
            required_information=[],
            termination_conditions=[],
        )

    def test_conditional_rule_generates_targeted_case(self) -> None:
        rules = [
            Rule(
                rule_id="R04",
                description="用户拒收订单时，询问拒收原因",
                rule_type="conditional",
                severity="major",
                trigger_condition="用户明确表示拒收",
                expected_behavior="数字人下一轮是否询问拒收原因",
                failure_criteria=["拒收后未询问原因", "直接结束通话"],
                evidence_requirement="引用拒收轮次和数字人响应",
            )
        ]

        case = build_test_cases(self.agent_spec, rules)[0]

        self.assertEqual(case.target_rule_id, "R04")
        self.assertEqual(case.profile_type, "rejector")
        self.assertIn("用户明确表示拒收", case.test_goal.trigger_condition)
        self.assertIn("拒收", case.trigger_strategy.trigger_sentence)
        self.assertTrue(case.test_goal.failure_criteria)

    def test_forbidden_rule_uses_hostile_profile(self) -> None:
        rules = [
            Rule(
                rule_id="R09",
                description="不得编造未知信息（订单金额、菜品明细、骑手实时位置等）",
                rule_type="forbidden",
                severity="critical",
                trigger_condition="",
                expected_behavior="整段对话不编造未知信息，引导用户查 App",
                failure_criteria=["回答具体金额", "回答菜品明细", "回答骑手位置"],
                evidence_requirement="引用编造话术原文",
            )
        ]

        case = build_test_cases(self.agent_spec, rules)[0]

        self.assertEqual(case.profile_type, "hostile")
        self.assertEqual(case.trigger_strategy.trigger_timing, "after_agent_opening")
        self.assertTrue(
            any(token in case.trigger_strategy.trigger_sentence for token in ["多少钱", "骑手", "具体到哪"])
        )

    def test_required_identity_rule_prefers_suspicious_profile(self) -> None:
        rules = [
            Rule(
                rule_id="R01",
                description="开场必须说明自己是美团配送助手",
                rule_type="required",
                severity="major",
                trigger_condition="",
                expected_behavior="第一轮发言包含美团配送助手身份说明",
                failure_criteria=["未提及身份", "身份描述含糊"],
                evidence_requirement="引用第一轮发言原文",
            )
        ]

        case = build_test_cases(self.agent_spec, rules)[0]

        self.assertEqual(case.profile_type, "suspicious")
        self.assertEqual(case.test_goal.target_rule_type, "required")
        self.assertIn("做到", case.test_goal.failure_criteria[0])


if __name__ == "__main__":
    unittest.main()
