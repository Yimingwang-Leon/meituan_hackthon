from __future__ import annotations

import unittest

from src.evaluator import JudgeOutput, _aggregate_votes
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


if __name__ == "__main__":
    unittest.main()
