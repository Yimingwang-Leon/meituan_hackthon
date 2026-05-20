from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("缺少 OPENAI_API_KEY")

from src.rule_parser import parse_rules
from src.rule_validation import print_validation_report, validate_rules

instructions_dir = Path(__file__).parent / "instructions"

for path in sorted(instructions_dir.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n{'='*60}")
    print(f"场景：{data['scenario']}（{data['id']}）")
    print(f"{'='*60}")

    rules = parse_rules(data["instruction"])
    for rule in rules:
        print(f"\n[{rule.rule_id}] ({rule.rule_type}/{rule.severity}) {rule.description}")
        if rule.trigger_condition:
            print(f"  触发：{rule.trigger_condition}")
        print(f"  期望：{rule.expected_behavior}")
        if rule.failure_criteria:
            for fc in rule.failure_criteria:
                print(f"  失败：{fc}")
        if rule.evidence_requirement:
            print(f"  证据：{rule.evidence_requirement}")
        if rule.checks:
            print(f"  代码 checks：{len(rule.checks)} 个")

    print_validation_report(rules, validate_rules(rules))
