from __future__ import annotations

import unittest

from src.evaluator import (
    ComplianceJudgeOutput,
    JudgeOutput,
    TriggerJudgeOutput,
    _aggregate_votes,
    _evaluate_conditional_two_step,
    _resolve_trigger_turn,
)
from src.rules import Rule


class EvaluatorAggregationTest(unittest.TestCase):
    def test_aggregate_votes_persists_samples_and_fail_suggestion(self) -> None:
        rule = Rule(
            rule_id="R02",
            description="用户拒收时必须询问拒收原因",
            rule_type="conditional",
            severity="major",
            trigger_condition="用户明确拒收",
            expected_behavior="询问一次拒收原因",
            failure_criteria=["拒收后未询问原因", "直接结束通话"],
            evidence_requirement="引用用户拒收和数字人响应轮次",
        )
        outputs = [
            JudgeOutput(
                triggered=True,
                trigger_turn=2,
                response_turn=3,
                compliant=False,
                evidence="第3轮数字人直接结束，未询问原因",
                rationale="第2轮用户拒收，第3轮数字人未询问原因，命中失败标准，结论 fail。",
                matched_failure_criteria=["拒收后未询问原因"],
                suggestion="用户拒收后先询问一次原因，再记录并结束。",
            ),
            JudgeOutput(
                triggered=True,
                trigger_turn=2,
                response_turn=3,
                compliant=False,
                evidence="第3轮没有追问拒收原因",
                rationale="拒收条件已触发，数字人没有追问原因，结论 fail。",
                matched_failure_criteria=["拒收后未询问原因"],
                suggestion="补一句询问拒收原因的话术。",
            ),
            JudgeOutput(
                triggered=True,
                trigger_turn=2,
                response_turn=3,
                compliant=True,
                evidence="第3轮表达尊重用户决定",
                rationale="第3轮回应可理解为接受用户决定，结论 pass。",
                matched_failure_criteria=["拒收后未询问原因"],
                suggestion="这条建议不应进入 pass 样本。",
            ),
        ]

        result = _aggregate_votes(
            rule,
            outputs,
            is_primary=True,
            judge_model="deepseek-v4-pro",
            judge_prompt="judge prompt snapshot",
        )

        self.assertEqual(result.result, "fail")
        self.assertEqual(result.confidence, 2 / 3)
        self.assertEqual(result.votes, {"fail": 2, "pass": 1})
        self.assertEqual(result.evidence, "第3轮数字人直接结束，未询问原因")
        self.assertEqual(result.matched_failure_criteria, ["拒收后未询问原因"])
        self.assertEqual(result.suggestion, "用户拒收后先询问一次原因，再记录并结束。")
        self.assertEqual(result.judge_model, "deepseek-v4-pro")
        self.assertEqual(result.judge_prompt, "judge prompt snapshot")
        self.assertEqual(len(result.all_samples), 3)
        self.assertEqual(result.all_samples[0]["result"], "fail")
        self.assertEqual(result.all_samples[1]["suggestion"], "补一句询问拒收原因的话术。")
        self.assertEqual(result.all_samples[2]["result"], "pass")
        self.assertEqual(result.all_samples[2]["matched_failure_criteria"], [])
        self.assertEqual(result.all_samples[2]["suggestion"], "")

    def test_non_fail_results_do_not_surface_suggestions(self) -> None:
        rule = Rule(
            rule_id="R03",
            description="用户拒收时必须询问拒收原因",
            rule_type="conditional",
            severity="major",
            trigger_condition="用户明确拒收",
            expected_behavior="询问一次拒收原因",
            failure_criteria=["拒收后未询问原因"],
            evidence_requirement="引用用户拒收和数字人响应轮次",
        )

        pass_result = _aggregate_votes(
            rule,
            [
                JudgeOutput(
                    triggered=True,
                    trigger_turn=2,
                    response_turn=3,
                    compliant=True,
                    evidence="第3轮询问了拒收原因",
                    rationale="触发后已追问原因，结论 pass。",
                    matched_failure_criteria=["拒收后未询问原因"],
                    suggestion="不应出现",
                )
            ],
            is_primary=True,
        )
        self.assertEqual(pass_result.result, "pass")
        self.assertEqual(pass_result.suggestion, "")
        self.assertEqual(pass_result.matched_failure_criteria, [])

        not_applicable_result = _aggregate_votes(
            rule,
            [
                JudgeOutput(
                    triggered=False,
                    trigger_turn=0,
                    response_turn=0,
                    compliant=False,
                    evidence="用户没有拒收",
                    rationale="触发条件未出现，结论 not_applicable。",
                    matched_failure_criteria=["拒收后未询问原因"],
                    suggestion="不应出现",
                )
            ],
            is_primary=False,
        )
        self.assertEqual(not_applicable_result.result, "not_applicable")
        self.assertEqual(not_applicable_result.suggestion, "")
        self.assertEqual(not_applicable_result.matched_failure_criteria, [])


def _make_conditional_rule(rule_id: str = "R10") -> Rule:
    return Rule(
        rule_id=rule_id,
        description="用户拒收时必须询问拒收原因",
        rule_type="conditional",
        severity="major",
        trigger_condition="用户明确拒收",
        expected_behavior="询问一次拒收原因",
        failure_criteria=["拒收后未询问原因", "直接结束通话"],
        evidence_requirement="引用用户拒收和数字人响应轮次",
    )


class ResolveTriggerTurnTest(unittest.TestCase):
    def test_returns_zero_for_empty_or_all_zero(self) -> None:
        self.assertEqual(_resolve_trigger_turn([]), 0)
        self.assertEqual(_resolve_trigger_turn([0, 0, 0]), 0)

    def test_returns_strict_majority(self) -> None:
        self.assertEqual(_resolve_trigger_turn([3, 3, 5]), 3)
        self.assertEqual(_resolve_trigger_turn([5, 3, 3]), 3)

    def test_no_majority_falls_back_to_median_low(self) -> None:
        # 三个不同轮次 → 中位数 4
        self.assertEqual(_resolve_trigger_turn([2, 4, 6]), 4)

    def test_two_way_tie_uses_median_low(self) -> None:
        # 两个轮次各占一半，median_low 取较小的
        self.assertEqual(_resolve_trigger_turn([2, 5]), 2)


class ConditionalTwoStepTriggerFailedTest(unittest.TestCase):
    def test_n_samples_zero_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_samples"):
            _evaluate_conditional_two_step(
                _make_conditional_rule(),
                transcript_text="transcript",
                n_samples=0,
                is_primary=True,
                trigger_outputs=[],
            )

    def test_all_trigger_false_primary_yields_trigger_failed(self) -> None:
        rule = _make_conditional_rule()
        trigger_outputs = [
            TriggerJudgeOutput(
                triggered=False,
                trigger_turn=0,
                evidence="用户全程未明确拒收",
                rationale="对话中没有出现行动含义层面的拒收。",
            )
            for _ in range(3)
        ]

        result = _evaluate_conditional_two_step(
            rule,
            transcript_text="transcript",
            n_samples=3,
            is_primary=True,
            trigger_outputs=trigger_outputs,
        )

        self.assertEqual(result.result, "trigger_failed")
        self.assertEqual(result.trigger_confidence, 1.0)
        self.assertIsNone(result.compliance_confidence)
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.votes, {"triggered": 0, "not_triggered": 3})
        self.assertEqual(len(result.all_samples), 3)
        self.assertTrue(
            all(s["phase"] == "trigger" for s in result.all_samples)
        )
        self.assertFalse(result.triggered)
        self.assertIsNone(result.trigger_turn)

    def test_all_trigger_false_non_primary_yields_not_applicable(self) -> None:
        rule = _make_conditional_rule()
        trigger_outputs = [
            TriggerJudgeOutput(
                triggered=False, trigger_turn=0, evidence="无", rationale="无"
            )
            for _ in range(3)
        ]

        result = _evaluate_conditional_two_step(
            rule,
            transcript_text="transcript",
            n_samples=3,
            is_primary=False,
            trigger_outputs=trigger_outputs,
        )

        self.assertEqual(result.result, "not_applicable")
        self.assertEqual(result.suggestion, "")

    def test_invalid_trigger_turn_is_downgraded_and_skips_compliance(self) -> None:
        rule = _make_conditional_rule()
        trigger_outputs = [
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=0,
                evidence="第3轮用户疑似拒收",
                rationale="模型误填了触发轮次 0",
            )
            for _ in range(3)
        ]

        def provider(_turn: int, _evidence: str) -> list[ComplianceJudgeOutput]:
            raise AssertionError("invalid trigger_turn should not enter compliance")

        result = _evaluate_conditional_two_step(
            rule,
            transcript_text="transcript",
            n_samples=3,
            is_primary=True,
            trigger_outputs=trigger_outputs,
            compliance_outputs_provider=provider,
        )

        self.assertEqual(result.result, "trigger_failed")
        self.assertFalse(result.triggered)
        self.assertIsNone(result.trigger_turn)
        self.assertIsNone(result.compliance_confidence)
        self.assertEqual(result.votes, {"triggered": 0, "not_triggered": 3})
        self.assertTrue(
            all(sample["result"] == "not_triggered" for sample in result.all_samples)
        )


class ConditionalTwoStepPassFailTest(unittest.TestCase):
    def test_triggered_and_compliant_yields_pass(self) -> None:
        rule = _make_conditional_rule()
        trigger_outputs = [
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=3,
                evidence="第3轮用户「我不要了」",
                rationale="形成明确拒绝意图",
            )
            for _ in range(3)
        ]
        compliance_outputs = [
            ComplianceJudgeOutput(
                compliant=True,
                response_turn=4,
                evidence="第4轮数字人询问了拒收原因",
                rationale="触发后第4轮已追问原因，符合期望行为",
            )
            for _ in range(3)
        ]

        result = _evaluate_conditional_two_step(
            rule,
            transcript_text="transcript",
            n_samples=3,
            is_primary=True,
            trigger_outputs=trigger_outputs,
            compliance_outputs_provider=lambda _turn, _ev: compliance_outputs,
        )

        self.assertEqual(result.result, "pass")
        self.assertEqual(result.trigger_confidence, 1.0)
        self.assertEqual(result.compliance_confidence, 1.0)
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.suggestion, "")
        self.assertEqual(result.matched_failure_criteria, [])
        self.assertEqual(result.trigger_turn, 3)
        self.assertEqual(result.response_turn, 4)
        self.assertEqual(result.votes, {"pass": 3, "fail": 0})
        # all_samples 应同时包含 trigger + compliance 阶段
        phases = {s["phase"] for s in result.all_samples}
        self.assertEqual(phases, {"trigger", "compliance"})
        self.assertEqual(len(result.all_samples), 6)

    def test_triggered_but_non_compliant_yields_fail_with_diagnostics(self) -> None:
        rule = _make_conditional_rule()
        trigger_outputs = [
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=3,
                evidence="第3轮用户拒收",
                rationale="明确拒收",
            )
            for _ in range(3)
        ]
        compliance_outputs = [
            ComplianceJudgeOutput(
                compliant=False,
                response_turn=4,
                evidence="第4轮数字人直接帮取消",
                rationale="未询问拒收原因，直接进入取消流程",
                matched_failure_criteria=["拒收后未询问原因"],
                suggestion="在第4轮加一句「方便了解下拒收原因吗」",
            ),
            ComplianceJudgeOutput(
                compliant=False,
                response_turn=4,
                evidence="第4轮没追问原因",
                rationale="同上",
                matched_failure_criteria=["拒收后未询问原因"],
                suggestion="在确认取消前补一句询问原因",
            ),
            ComplianceJudgeOutput(
                compliant=True,
                response_turn=4,
                evidence="第4轮可视为尊重用户决定",
                rationale="把这看作合规的极端宽松解读",
            ),
        ]

        result = _evaluate_conditional_two_step(
            rule,
            transcript_text="transcript",
            n_samples=3,
            is_primary=True,
            trigger_outputs=trigger_outputs,
            compliance_outputs_provider=lambda _turn, _ev: compliance_outputs,
        )

        self.assertEqual(result.result, "fail")
        self.assertEqual(result.trigger_confidence, 1.0)
        self.assertAlmostEqual(result.compliance_confidence, 2 / 3)
        self.assertAlmostEqual(result.confidence, 2 / 3)
        self.assertEqual(result.matched_failure_criteria, ["拒收后未询问原因"])
        self.assertTrue(result.suggestion.startswith("在第4轮加一句"))
        self.assertEqual(result.votes, {"pass": 1, "fail": 2})
        # 代表样本应来自 compliance 阶段的 winner（fail），不是 pass 那一条
        self.assertEqual(result.evidence, "第4轮数字人直接帮取消")
        # all_samples 中 pass 样本不应携带 matched_failure_criteria / suggestion
        pass_compliance_samples = [
            s for s in result.all_samples
            if s["phase"] == "compliance" and s["result"] == "pass"
        ]
        self.assertEqual(len(pass_compliance_samples), 1)
        self.assertEqual(pass_compliance_samples[0]["matched_failure_criteria"], [])
        self.assertEqual(pass_compliance_samples[0]["suggestion"], "")


class ConditionalTwoStepTriggerTurnResolutionTest(unittest.TestCase):
    def test_majority_trigger_turn_wins_when_one_sample_drifts(self) -> None:
        rule = _make_conditional_rule()
        # 两个样本说第3轮触发，一个说第5轮 → 应取多数 3
        trigger_outputs = [
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=3,
                evidence="第3轮用户「我不要了」",
                rationale="明确拒绝",
            ),
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=3,
                evidence="第3轮用户拒收",
                rationale="明确拒绝",
            ),
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=5,
                evidence="第5轮用户重申拒收",
                rationale="同样的拒收意图",
            ),
        ]
        compliance_outputs = [
            ComplianceJudgeOutput(
                compliant=True,
                response_turn=4,
                evidence="第4轮已询问原因",
                rationale="符合期望",
            )
            for _ in range(3)
        ]

        captured: dict[str, object] = {}

        def provider(turn: int, evidence: str) -> list[ComplianceJudgeOutput]:
            captured["turn"] = turn
            captured["evidence"] = evidence
            return compliance_outputs

        result = _evaluate_conditional_two_step(
            rule,
            transcript_text="transcript",
            n_samples=3,
            is_primary=True,
            trigger_outputs=trigger_outputs,
            compliance_outputs_provider=provider,
        )

        self.assertEqual(captured["turn"], 3)
        # trigger evidence 透传给 compliance 时，应取 turn=3 那条的原话
        self.assertEqual(captured["evidence"], "第3轮用户「我不要了」")
        self.assertEqual(result.trigger_turn, 3)

    def test_no_majority_falls_back_to_median_trigger_turn(self) -> None:
        rule = _make_conditional_rule()
        # 三个互不相同 → 中位数 4
        trigger_outputs = [
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=2,
                evidence="第2轮用户拒收",
                rationale="较早的拒绝表达",
            ),
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=4,
                evidence="第4轮用户拒收",
                rationale="中间的拒绝表达",
            ),
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=6,
                evidence="第6轮用户拒收",
                rationale="较晚的拒绝表达",
            ),
        ]
        compliance_outputs = [
            ComplianceJudgeOutput(
                compliant=True, response_turn=5, evidence="第5轮已追问", rationale="符合"
            )
            for _ in range(3)
        ]

        captured: dict[str, object] = {}

        def provider(turn: int, evidence: str) -> list[ComplianceJudgeOutput]:
            captured["turn"] = turn
            captured["evidence"] = evidence
            return compliance_outputs

        result = _evaluate_conditional_two_step(
            rule,
            transcript_text="transcript",
            n_samples=3,
            is_primary=True,
            trigger_outputs=trigger_outputs,
            compliance_outputs_provider=provider,
        )

        self.assertEqual(captured["turn"], 4)
        self.assertEqual(captured["evidence"], "第4轮用户拒收")
        self.assertEqual(result.trigger_turn, 4)

    def test_invalid_trigger_turn_samples_do_not_participate_in_turn_resolution(self) -> None:
        rule = _make_conditional_rule()
        trigger_outputs = [
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=3,
                evidence="第3轮用户拒收",
                rationale="有效触发",
            ),
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=5,
                evidence="第5轮用户重申拒收",
                rationale="有效触发",
            ),
            TriggerJudgeOutput(
                triggered=True,
                trigger_turn=0,
                evidence="模型判断触发但漏填轮次",
                rationale="无效触发轮次",
            ),
        ]
        compliance_outputs = [
            ComplianceJudgeOutput(
                compliant=True,
                response_turn=6,
                evidence="第6轮已询问原因",
                rationale="符合",
            )
            for _ in range(3)
        ]

        captured: dict[str, object] = {}

        def provider(turn: int, evidence: str) -> list[ComplianceJudgeOutput]:
            captured["turn"] = turn
            captured["evidence"] = evidence
            return compliance_outputs

        result = _evaluate_conditional_two_step(
            rule,
            transcript_text="transcript",
            n_samples=3,
            is_primary=True,
            trigger_outputs=trigger_outputs,
            compliance_outputs_provider=provider,
        )

        # 无效 turn=0 样本被降级为 not_triggered；有效样本 3/5 用 median_low 取 3。
        self.assertEqual(captured["turn"], 3)
        self.assertEqual(captured["evidence"], "第3轮用户拒收")
        self.assertEqual(result.trigger_turn, 3)
        self.assertAlmostEqual(result.trigger_confidence, 2 / 3)
        self.assertEqual(result.votes, {"pass": 3, "fail": 0})


class ConditionalTwoStepConfidenceProductTest(unittest.TestCase):
    def test_overall_confidence_is_trigger_times_compliance(self) -> None:
        rule = _make_conditional_rule()
        # trigger: 2 true + 1 false → trigger_conf = 2/3
        trigger_outputs = [
            TriggerJudgeOutput(
                triggered=True, trigger_turn=3, evidence="第3轮拒收", rationale=""
            ),
            TriggerJudgeOutput(
                triggered=True, trigger_turn=3, evidence="第3轮拒收", rationale=""
            ),
            TriggerJudgeOutput(
                triggered=False, trigger_turn=0, evidence="没看到拒收", rationale=""
            ),
        ]
        # compliance: 3 都 compliant → compliance_conf = 1.0
        compliance_outputs = [
            ComplianceJudgeOutput(
                compliant=True, response_turn=4, evidence="第4轮已询问", rationale=""
            )
            for _ in range(3)
        ]

        result = _evaluate_conditional_two_step(
            rule,
            transcript_text="transcript",
            n_samples=3,
            is_primary=True,
            trigger_outputs=trigger_outputs,
            compliance_outputs_provider=lambda _t, _e: compliance_outputs,
        )

        self.assertAlmostEqual(result.trigger_confidence, 2 / 3)
        self.assertAlmostEqual(result.compliance_confidence, 1.0)
        self.assertAlmostEqual(result.confidence, 2 / 3 * 1.0)


if __name__ == "__main__":
    unittest.main()
