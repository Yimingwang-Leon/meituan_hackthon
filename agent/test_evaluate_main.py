from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluate_main import _save_report
from src.evaluator import EvaluationReport, RuleResult


class EvaluateMainReportTest(unittest.TestCase):
    def test_save_report_includes_two_step_confidence_fields(self) -> None:
        report = EvaluationReport(
            session_id="session-1",
            persona_type="cooperative",
            score=1.0,
            rule_results=[
                RuleResult(
                    rule_id="R02",
                    rule_type="conditional",
                    description="用户拒收时必须询问原因",
                    severity="major",
                    result="pass",
                    evidence="第4轮数字人询问原因",
                    confidence=2 / 3,
                    trigger_confidence=2 / 3,
                    compliance_confidence=1.0,
                    votes={"pass": 3, "fail": 0},
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = _save_report([report], Path(temp_dir))
            payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

        rule_payload = payload["sessions"][0]["rules"][0]
        self.assertAlmostEqual(rule_payload["trigger_confidence"], 2 / 3)
        self.assertAlmostEqual(rule_payload["compliance_confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
