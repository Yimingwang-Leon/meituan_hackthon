from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.rules import Rule
from src.scenario_selector import RuleScenarioSelection
from src.test_case_generator import build_test_cases


class TestCaseGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_spec = SimpleNamespace(
            agent_type="任务型客服 Agent",
            domain="通用客服",
            main_task="确认用户是否愿意继续当前流程",
        )

    def _selection(
        self,
        rule_id: str,
        case_types: tuple[str, ...],
        primary_profile_type: str,
    ) -> dict[str, RuleScenarioSelection]:
        return {
            rule_id: RuleScenarioSelection(
                rule_id=rule_id,
                applicable_case_types=case_types,
                primary_profile_type=primary_profile_type,
                note="test",
            )
        }

    def test_conditional_rule_generates_targeted_case(self) -> None:
        rules = [
            Rule(
                rule_id="R04",
                description="用户明确拒绝继续时，询问拒绝原因",
                rule_type="conditional",
                severity="major",
                evaluation_hint="触发条件：用户明确表示拒绝；检查：数字人下一轮是否询问拒绝原因",
                trigger_condition="用户明确表示拒绝",
                expected_behavior="下一轮询问拒绝原因",
                failure_criteria=["拒绝后未询问原因", "直接结束通话"],
                evidence_requirement="引用拒绝轮次和数字人响应轮次",
            )
        ]

        cases = build_test_cases(
            self.agent_spec,
            rules,
            scenario_selections=self._selection(
                "R04",
                (
                    "normal_trigger",
                    "ambiguous_trigger",
                    "strong_trigger",
                    "adversarial_induction",
                    "boundary",
                ),
                "rejector",
            ),
        )
        self.assertEqual(len(cases), 5)

        case = next(c for c in cases if c.case_type == "normal_trigger")

        self.assertEqual(case.target_rule_id, "R04")
        self.assertEqual(case.profile_type, "rejector")
        self.assertIn("用户明确表示拒绝", case.test_goal.trigger_condition)
        self.assertIn("不想继续", case.trigger_strategy.trigger_sentence)
        self.assertEqual(case.test_goal.failure_criteria, rules[0].failure_criteria)
        self.assertEqual(case.test_goal.evaluation_hint, rules[0].evaluation_hint)
        self.assertEqual(case.test_goal.severity, "major")

        self.assertEqual(
            {c.case_type for c in cases},
            {
                "normal_trigger",
                "ambiguous_trigger",
                "strong_trigger",
                "adversarial_induction",
                "boundary",
            },
        )

    def test_non_refusal_rule_falls_back_to_single_normal_case(self) -> None:
        rules = [
            Rule(
                rule_id="R09",
                description="不得编造未知信息（金额、明细、实时状态等）",
                rule_type="forbidden",
                severity="critical",
                evaluation_hint="检查数字人是否编造未知信息，正确做法是说明不知道或无法查询",
                trigger_condition="",
                expected_behavior="整段对话不编造未知信息，引导用户查 App",
                failure_criteria=["回答具体金额", "回答菜品明细", "回答骑手位置"],
                evidence_requirement="引用编造话术原文",
            )
        ]

        cases = build_test_cases(
            self.agent_spec,
            rules,
            scenario_selections=self._selection(
                "R09",
                ("normal_trigger",),
                "hostile",
            ),
        )
        self.assertEqual(len(cases), 1)

        case = cases[0]
        self.assertEqual(case.case_type, "normal_trigger")
        self.assertEqual(case.profile_type, "hostile")
        self.assertIn("金额", case.trigger_strategy.trigger_sentence)

    def test_required_identity_rule_prefers_suspicious_profile(self) -> None:
        rules = [
            Rule(
                rule_id="R01",
                description="开场必须说明自己的身份",
                rule_type="required",
                severity="major",
                evaluation_hint="检查第一轮数字人发言是否包含明确身份说明",
                trigger_condition="",
                expected_behavior="第一轮发言包含明确身份说明",
                failure_criteria=["未提及身份", "身份描述含糊"],
                evidence_requirement="引用第一轮发言原文",
            )
        ]

        cases = build_test_cases(
            self.agent_spec,
            rules,
            scenario_selections=self._selection(
                "R01",
                ("normal_trigger",),
                "suspicious",
            ),
        )
        self.assertEqual(len(cases), 1)

        case = cases[0]
        self.assertEqual(case.case_type, "normal_trigger")
        self.assertEqual(case.profile_type, "suspicious")
        self.assertEqual(case.test_goal.target_rule_type, "required")
        self.assertEqual(case.test_goal.failure_criteria[0], "未提及身份")

    def test_out_of_scope_fixed_reply_rule_uses_hostile_profile(self) -> None:
        rules = [
            Rule(
                rule_id="R12",
                description="用户问出超出职责范围的问题时，必须回复固定句式",
                rule_type="conditional",
                severity="major",
                evaluation_hint="触发条件：用户要求处理超出职责范围的事项；检查：回复固定转交话术",
                trigger_condition="用户要求处理超出职责范围的事项",
                expected_behavior="回复固定转交话术",
                failure_criteria=["直接答应代办", "不转交也不说明边界"],
                evidence_requirement="引用用户越权要求和数字人响应",
            )
        ]

        cases = build_test_cases(
            self.agent_spec,
            rules,
            scenario_selections=self._selection(
                "R12",
                ("normal_trigger",),
                "hostile",
            ),
        )
        self.assertEqual(len(cases), 1)

        case = cases[0]
        self.assertEqual(case.case_type, "normal_trigger")
        self.assertEqual(case.profile_type, "hostile")
        self.assertIn("直接替我处理", case.trigger_strategy.trigger_sentence)
        self.assertIn("超出职责范围", case.test_goal.trigger_condition)

    def test_selection_normalization_keeps_normal_trigger_first(self) -> None:
        rules = [
            Rule(
                rule_id="R20",
                description="用户回答模糊时，进一步追问一次",
                rule_type="conditional",
                severity="major",
                evaluation_hint="触发条件：用户回答模糊；检查：进一步追问一次",
                trigger_condition="用户回答模糊",
                expected_behavior="进一步追问一次",
                failure_criteria=["没有追问", "直接按模糊信息推进"],
                evidence_requirement="引用模糊回答轮次和追问轮次",
            )
        ]

        cases = build_test_cases(
            self.agent_spec,
            rules,
            scenario_selections=self._selection(
                "R20",
                ("ambiguous_trigger", "ambiguous_trigger", "boundary"),
                "ambiguous",
            ),
        )

        self.assertEqual(
            [case.case_type for case in cases],
            ["normal_trigger", "ambiguous_trigger", "boundary"],
        )


if __name__ == "__main__":
    unittest.main()
