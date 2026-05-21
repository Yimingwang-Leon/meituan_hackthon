from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


FAILURE_TYPES = [
    "RULE_VIOLATION",
    "GOAL_FAILURE",
    "TOOL_ERROR",
    "CONTEXT_ERROR",
    "OVER_PERSUASION",
    "PRIVACY_RISK",
    "HALLUCINATION",
    "JUDGE_UNCERTAINTY",
    "MAX_TURN_EXCEEDED",
]


def classify_failure_types(
    rule_check: dict[str, Any],
    judge_result: dict[str, Any],
    final_passed: bool,
) -> list[str]:
    failure_types: set[str] = set()
    for violation in rule_check.get("violations", []):
        failure_types.add("RULE_VIOLATION")
        if violation.get("failure_type"):
            failure_types.add(str(violation["failure_type"]))
    if judge_result.get("uncertainty", 0) >= 0.5:
        failure_types.add("JUDGE_UNCERTAINTY")
    if not final_passed and judge_result.get("task_completion", 1) < 0.65:
        failure_types.add("GOAL_FAILURE")
    return sorted(ft for ft in failure_types if ft in FAILURE_TYPES)


def aggregate_final_result(
    rule_check: dict[str, Any],
    judge_result: dict[str, Any],
    judge_threshold: float = 0.7,
) -> dict[str, Any]:
    violations = rule_check.get("violations", [])
    high_violations = [v for v in violations if v.get("severity") == "high"]
    judge_score = float(judge_result.get("score", 0.0))
    passed = not high_violations and judge_score >= judge_threshold
    if high_violations:
        reason = "存在 high severity 硬规则违规，最终不通过。"
    elif judge_score < judge_threshold:
        reason = f"无严重硬规则违规，但软指标分数 {judge_score:.2f} 低于阈值 {judge_threshold:.2f}。"
    else:
        reason = "无严重硬规则违规，软指标达到阈值。"
    return {
        "passed": passed,
        "score": round(judge_score, 4),
        "judge_threshold": judge_threshold,
        "high_severity_rule_violations": len(high_violations),
        "reason": reason,
    }


def write_summary(
    case_results: list[dict[str, Any]],
    output_dir: Path = Path("outputs/reports"),
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(case_results)
    passed = sum(1 for item in case_results if item["final_result"]["passed"])
    avg_judge = (
        sum(float(item["judge_result"].get("score", 0)) for item in case_results) / total
        if total else 0.0
    )
    failure_counter = Counter(
        failure_type
        for item in case_results
        for failure_type in item.get("failure_types", [])
    )
    high_rule_violations = sum(
        int(item["final_result"].get("high_severity_rule_violations", 0))
        for item in case_results
    )
    worst_cases = sorted(
        case_results,
        key=lambda item: float(item["judge_result"].get("score", 0)),
    )[:5]

    summary = {
        "total_cases": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "average_judge_score": round(avg_judge, 4),
        "failure_type_counts": dict(sorted(failure_counter.items())),
        "high_severity_rule_violation_count": high_rule_violations,
        "cases": case_results,
        "worst_cases": [
            {
                "case_id": item["case_id"],
                "score": item["judge_result"].get("score"),
                "failure_types": item.get("failure_types", []),
                "trajectory_path": item.get("trajectory_path"),
            }
            for item in worst_cases
        ],
    }

    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_summary_markdown(summary), encoding="utf-8")
    return json_path, md_path


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Agent Evaluation Harness Summary",
        "",
        f"- Total cases: {summary['total_cases']}",
        f"- Pass rate: {summary['pass_rate']:.0%}",
        f"- Average judge score: {summary['average_judge_score']:.2f}",
        f"- High severity rule violations: {summary['high_severity_rule_violation_count']}",
        "",
        "## Failure Types",
    ]
    if summary["failure_type_counts"]:
        for failure_type, count in summary["failure_type_counts"].items():
            lines.append(f"- {failure_type}: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "## Cases", ""])
    for item in summary["cases"]:
        status = "PASS" if item["final_result"]["passed"] else "FAIL"
        lines.append(
            f"- `{item['case_id']}` [{status}] score={item['judge_result'].get('score')} "
            f"failures={', '.join(item.get('failure_types', [])) or '-'} "
            f"trajectory={item.get('trajectory_path')}"
        )
    lines.append("")
    return "\n".join(lines)
