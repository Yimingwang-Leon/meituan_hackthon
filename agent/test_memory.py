from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from src.memory import append_evaluation_memory
from src.types import SessionArchive, TranscriptEntry


@dataclass
class FakeRuleResult:
    rule_id: str
    rule_type: str
    description: str
    severity: str
    result: str
    evidence: str
    confidence: float = 1.0
    votes: dict[str, int] = field(default_factory=dict)


@dataclass
class FakeEvaluationReport:
    session_id: str
    persona_type: str
    rule_results: list[FakeRuleResult]
    score: float

    @property
    def mean_confidence(self) -> float:
        if not self.rule_results:
            return 0.0
        return sum(r.confidence for r in self.rule_results) / len(self.rule_results)


class EvaluationMemoryTest(unittest.TestCase):
    def test_append_evaluation_memory_persists_transcript_and_judger_results(self) -> None:
        archive = SessionArchive(
            session_id="set_default:R01_cooperative",
            started_at="2026-05-15T00:00:00+00:00",
            ended_at="2026-05-15T00:01:00+00:00",
            ended_by="agent_end",
            transcript=[
                TranscriptEntry(
                    speaker="agent",
                    text="王先生您好，我是美团配送助手。",
                    timestamp="2026-05-15T00:00:10+00:00",
                ),
                TranscriptEntry(
                    speaker="user",
                    text="可以。",
                    timestamp="2026-05-15T00:00:20+00:00",
                ),
            ],
            source_label="unit-test",
            instruction_snapshot="你是一名客服助手。",
            scenario_context={"customer_name": "王先生"},
            persona_type="cooperative",
            case_type="normal_trigger",
            target_rule_id="R01",
            target_rule_type="required",
            target_rule_description="开场必须说明自己的身份",
            target_rule_evaluation_hint="检查第一轮数字人发言是否包含明确身份说明",
            target_rule_severity="major",
            set_id="set_default",
            set_label="默认",
        )
        report = FakeEvaluationReport(
            session_id="set_default:R01_cooperative",
            persona_type="cooperative",
            score=0.5,
            rule_results=[
                FakeRuleResult(
                    rule_id="R01",
                    rule_type="required",
                    description="开场必须说明自己是美团配送助手",
                    severity="major",
                    result="pass",
                    evidence="第1轮数字人说明了身份",
                    confidence=1.0,
                    votes={"pass": 1},
                ),
                FakeRuleResult(
                    rule_id="R02",
                    rule_type="forbidden",
                    description="不得编造未知信息",
                    severity="critical",
                    result="fail",
                    evidence="第3轮数字人编造了未知信息",
                    confidence=1.0,
                    votes={"fail": 1},
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            memory_file = append_evaluation_memory(archive, report, temp_dir)

            data = Path(memory_file).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(data), 1)

            entry = json.loads(data[0])
            self.assertEqual(entry["session_key"], "set_default:R01_cooperative")
            self.assertEqual(entry["archive_session_id"], "set_default:R01_cooperative")
            self.assertEqual(entry["persona_type"], "cooperative")
            self.assertEqual(entry["case_type"], "normal_trigger")
            self.assertTrue(entry["has_violation"])
            self.assertEqual(entry["violation_count"], 1)
            self.assertEqual(entry["scenario_context"]["customer_name"], "王先生")
            self.assertEqual(entry["target_rule_id"], "R01")
            self.assertEqual(entry["target_rule_severity"], "major")
            self.assertEqual(entry["transcript"][0]["speaker"], "agent")
            self.assertEqual(entry["transcript"][1]["text"], "可以。")
            self.assertEqual(entry["judge_results"][0]["judger_result"], "pass")
            self.assertFalse(entry["judge_results"][0]["is_violation"])
            self.assertIn("rationale", entry["judge_results"][0])
            self.assertIn("suggestion", entry["judge_results"][0])
            self.assertIn("matched_failure_criteria", entry["judge_results"][0])
            self.assertIn("all_samples", entry["judge_results"][0])
            self.assertIn("judge_model", entry["judge_results"][0])
            self.assertEqual(entry["judge_results"][0]["rationale"], "")
            self.assertEqual(entry["judge_results"][0]["suggestion"], "")
            self.assertEqual(entry["judge_results"][0]["all_samples"], [])
            self.assertEqual(entry["judge_results"][1]["judger_result"], "fail")
            self.assertTrue(entry["judge_results"][1]["is_violation"])


if __name__ == "__main__":
    unittest.main()
