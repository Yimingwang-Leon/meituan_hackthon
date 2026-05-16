from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.evaluator import EvaluationReport, evaluate_session
from src.memory import append_evaluation_memory
from src.types import SessionArchive, TranscriptEntry


def _load_archive(path: Path) -> SessionArchive:
    raw = json.loads(path.read_text(encoding="utf-8"))
    transcript = [
        TranscriptEntry(
            speaker=entry["speaker"],
            text=entry["text"],
            timestamp=entry["timestamp"],
        )
        for entry in raw["transcript"]
    ]
    return SessionArchive(
        order_id=raw["order_id"],
        user_name=raw["user_name"],
        source_file=raw["source_file"],
        started_at=raw["started_at"],
        ended_at=raw["ended_at"],
        ended_by=raw["ended_by"],
        transcript=transcript,
        persona_type=raw.get("persona_type"),
        simulator_label=raw.get("simulator_label"),
        test_case_id=raw.get("test_case_id"),
        target_rule_id=raw.get("target_rule_id"),
    )


def _save_report(reports: list[EvaluationReport], output_dir: Path) -> str:
    by_persona: dict[str, list[float]] = {}
    for r in reports:
        key = r.persona_type or "unknown"
        by_persona.setdefault(key, []).append(r.score)

    fail_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for r in reports:
        for rr in r.rule_results:
            if rr.result == "not_applicable":
                continue
            total_counts[rr.rule_id] = total_counts.get(rr.rule_id, 0) + 1
            if rr.result == "fail":
                fail_counts[rr.rule_id] = fail_counts.get(rr.rule_id, 0) + 1

    report_data = {
        "summary": {
            "total_sessions": len(reports),
            "overall_score": sum(r.score for r in reports) / len(reports) if reports else 0,
            "by_persona": {
                persona: round(sum(scores) / len(scores), 4)
                for persona, scores in sorted(by_persona.items())
            },
            "rule_fail_rate": {
                rule_id: round(fail_counts.get(rule_id, 0) / total, 4)
                for rule_id, total in sorted(total_counts.items())
            },
        },
        "sessions": [
            {
                "order_id": r.order_id,
                "persona_type": r.persona_type,
                "score": round(r.score, 4),
                "rules": [
                    {
                        "rule_id": rr.rule_id,
                        "rule_type": rr.rule_type,
                        "description": rr.description,
                        "result": rr.result,
                        "evidence": rr.evidence,
                    }
                    for rr in r.rule_results
                ],
            }
            for r in reports
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_path = output_dir / f"report-{ts}.json"
    output_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(output_path)


def _print_summary(reports: list[EvaluationReport]) -> None:
    print("\n" + "=" * 60)
    print("评测汇总")
    print("=" * 60)

    # 按 persona 分组
    by_persona: dict[str, list[float]] = {}
    for r in reports:
        key = r.persona_type or "unknown"
        by_persona.setdefault(key, []).append(r.score)

    print("\n【按 Persona 平均分】")
    for persona, scores in sorted(by_persona.items()):
        avg = sum(scores) / len(scores)
        print(f"  {persona:<15} {avg:.0%}")

    # 按规则统计 fail 率
    fail_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for r in reports:
        for rr in r.rule_results:
            if rr.result == "not_applicable":
                continue
            total_counts[rr.rule_id] = total_counts.get(rr.rule_id, 0) + 1
            if rr.result == "fail":
                fail_counts[rr.rule_id] = fail_counts.get(rr.rule_id, 0) + 1

    print("\n【按规则 fail 率（从高到低）】")
    rule_fail_rates = [
        (rule_id, fail_counts.get(rule_id, 0) / total)
        for rule_id, total in total_counts.items()
    ]
    for rule_id, rate in sorted(rule_fail_rates, key=lambda x: -x[1]):
        print(f"  {rule_id}  {rate:.0%}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，请在 agent/.env 中填写")

    sessions_dir = project_root / "sessions"
    memory_dir = project_root / "memory"
    session_files = sorted(sessions_dir.glob("*.json"))

    if not session_files:
        print(f"未在 {sessions_dir} 中找到 session 文件。")
        return

    reports: list[EvaluationReport] = []

    for path in session_files:
        print(f"\n评估：{path.name}")
        archive = _load_archive(path)
        persona_type = archive.persona_type or "unknown"
        report = evaluate_session(archive, persona_type)
        append_evaluation_memory(archive, report, memory_dir)
        print(report.summary())
        reports.append(report)

    _print_summary(reports)

    reports_dir = project_root / "reports"
    output_path = _save_report(reports, reports_dir)
    print(f"\n报告已保存：{output_path}")


if __name__ == "__main__":
    main()
