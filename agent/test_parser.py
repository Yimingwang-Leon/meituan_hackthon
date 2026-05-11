from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("缺少 OPENAI_API_KEY")

from src.rule_parser import parse_rules

instructions_dir = Path(__file__).parent / "instructions"

for path in sorted(instructions_dir.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n{'='*60}")
    print(f"场景：{data['scenario']}（{data['id']}）")
    print(f"{'='*60}")

    rules = parse_rules(data["instruction"])
    for rule in rules:
        print(f"\n[{rule.rule_id}] ({rule.rule_type}/{rule.severity}) {rule.description}")
        print(f"  → {rule.evaluation_hint}")
