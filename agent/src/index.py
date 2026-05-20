from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import OUTBOUND_INSTRUCTIONS, make_outbound_agent
from .session import OutboundSession
from .types import SessionMeta


def _load_instruction(source: str | None, instructions_dir: Path) -> tuple[str, str]:
    if not source:
        return OUTBOUND_INSTRUCTIONS, "manual-default"

    candidate = Path(source)
    if candidate.exists():
        return _read_instruction_file(candidate), candidate.stem

    preset_path = instructions_dir / f"{source}.json"
    if preset_path.exists():
        return _read_instruction_file(preset_path), source

    return source, "manual-inline"


def _read_instruction_file(path: Path) -> str:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        instruction = data.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"{path} 缺少非空 instruction 字段")
        return instruction.strip()
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "缺少 OPENAI_API_KEY，请先在 agent/.env 中填写，或复制 agent/.env.example 为 agent/.env"
        )

    instructions_dir = project_root / "instructions"
    sessions_dir = project_root / "sessions"
    instruction_source = sys.argv[1] if len(sys.argv) > 1 else None
    instruction, source_label = _load_instruction(instruction_source, instructions_dir)

    print()
    print("=== 手动对话模式 ===")
    print(f"指令来源: {source_label}")
    print("输入 /quit 结束并归档。")
    print()

    session_meta = SessionMeta(
        session_id=source_label,
        source_label=source_label,
        instruction_snapshot=instruction,
    )
    session = OutboundSession(
        session_meta,
        agent=make_outbound_agent(instruction),
    )
    opening_turn = session.start()
    print(f"Agent: {opening_turn.reply_text}")

    if opening_turn.should_end:
        archive_path = session.save_archive(sessions_dir, "agent_end")
        print(f"会话已结束并归档: {archive_path}")
        return

    while True:
        try:
            answer = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            archive_path = session.save_archive(sessions_dir, "quit")
            print(f"会话已归档: {archive_path}")
            return

        if not answer:
            continue

        if answer == "/quit":
            archive_path = session.save_archive(sessions_dir, "quit")
            print(f"会话已归档: {archive_path}")
            return

        turn = session.reply(answer)
        print(f"Agent: {turn.reply_text}")

        if turn.should_end:
            archive_path = session.save_archive(sessions_dir, "agent_end")
            print(f"会话已结束并归档: {archive_path}")
            return
