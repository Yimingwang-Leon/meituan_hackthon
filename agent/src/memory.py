from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .types import SessionArchive, TranscriptEntry

if TYPE_CHECKING:
    from .evaluator import EvaluationReport, RuleResult


MEMORY_FILE_NAME = "evaluation_memory.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_transcript_entry(entry: TranscriptEntry) -> dict[str, str]:
    return {
        "speaker": entry.speaker,
        "text": entry.text,
        "timestamp": entry.timestamp,
    }


def _serialize_rule_result(rule_result: RuleResult) -> dict[str, object]:
    applicable = rule_result.result != "not_applicable"
    is_violation = rule_result.result == "fail"
    return {
        "rule_id": rule_result.rule_id,
        "rule_type": rule_result.rule_type,
        "description": rule_result.description,
        "severity": rule_result.severity,
        "judger_result": rule_result.result,
        "applicable": applicable,
        "is_violation": is_violation,
        "evidence": rule_result.evidence,
        "confidence": rule_result.confidence,
        "votes": rule_result.votes,
    }


def build_evaluation_memory_entry(
    archive: SessionArchive,
    report: EvaluationReport,
) -> dict[str, object]:
    recorded_at = _now_iso()
    session_id = f"{report.order_id}:{report.persona_type}:{archive.started_at}"
    violation_count = sum(1 for rr in report.rule_results if rr.result == "fail")
    applicable_rule_count = sum(
        1 for rr in report.rule_results if rr.result != "not_applicable"
    )

    return {
        "memory_version": 1,
        "memory_id": f"{session_id}:{recorded_at}",
        "session_id": session_id,
        "recorded_at": recorded_at,
        "order_id": report.order_id,
        "persona_type": report.persona_type,
        "simulator_label": archive.simulator_label,
        "test_case_id": archive.test_case_id,
        "target_rule_id": archive.target_rule_id,
        "source_file": archive.source_file,
        "session_started_at": archive.started_at,
        "session_ended_at": archive.ended_at,
        "session_ended_by": archive.ended_by,
        "score": report.score,
        "mean_confidence": report.mean_confidence,
        "has_violation": violation_count > 0,
        "violation_count": violation_count,
        "applicable_rule_count": applicable_rule_count,
        "transcript": [
            _serialize_transcript_entry(entry) for entry in archive.transcript
        ],
        "judge_results": [
            _serialize_rule_result(rule_result)
            for rule_result in report.rule_results
        ],
    }


def append_evaluation_memory(
    archive: SessionArchive,
    report: EvaluationReport,
    output_dir: str | Path,
    file_name: str = MEMORY_FILE_NAME,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    memory_entry = build_evaluation_memory_entry(archive, report)
    memory_file = output_path / file_name
    with memory_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(memory_entry, ensure_ascii=False) + "\n")

    return str(memory_file)
