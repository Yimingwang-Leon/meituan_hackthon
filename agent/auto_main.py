from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import OUTBOUND_INSTRUCTIONS, make_outbound_agent
from src.agent_spec import AgentSpec
from src.session import OutboundSession
from src.simulation_plan import SubPlan, build_simulation_plan
from src.simulator import UserSimulator
from src.test_case_generator import SimulationCase
from src.types import SessionMeta

MAX_TURNS = 15


def _has_llm_api_key() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _load_instruction(source: str | None, instructions_dir: Path) -> tuple[str, str]:
    if not source:
        return OUTBOUND_INSTRUCTIONS, "default"

    candidate = Path(source)
    if candidate.exists():
        return _read_instruction_file(candidate), candidate.stem

    preset_path = instructions_dir / f"{source}.json"
    if preset_path.exists():
        return _read_instruction_file(preset_path), source

    return source, "inline"


def _read_instruction_file(path: Path) -> str:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        instruction = data.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"{path} 缺少非空 instruction 字段")
        return instruction.strip()
    return path.read_text(encoding="utf-8").strip()


def run_session(
    source_label: str,
    simulation_case: SimulationCase,
    sub_plan: SubPlan,
    agent_spec: AgentSpec,
    sessions_dir: Path,
) -> None:
    session_id = f"{source_label}:{sub_plan.set_id}:{simulation_case.test_id}"
    session_label = (
        f"{sub_plan.set_id} · {simulation_case.target_rule_id} · "
        f"{simulation_case.case_type_label} · {simulation_case.profile_type}"
    )

    print(f"\n[{session_label}] {source_label}")

    agent = make_outbound_agent(sub_plan.filled_instruction)
    session_meta = SessionMeta(
        session_id=session_id,
        source_label=source_label,
        instruction_snapshot=sub_plan.filled_instruction,
        scenario_context=sub_plan.placeholder_values,
    )
    outbound = OutboundSession(session_meta, agent=agent)
    simulator = UserSimulator(
        simulation_case,
        agent_spec,
        sub_plan.placeholder_values,
        session_id=session_id,
    )

    def _archive() -> None:
        outbound.save_archive(
            sessions_dir,
            "agent_end",
            persona_type=simulation_case.profile_type,
            case_type=simulation_case.case_type,
            simulator_label=session_label,
            test_case_id=simulation_case.test_id,
            target_rule_id=simulation_case.target_rule_id,
            target_rule_type=simulation_case.test_goal.target_rule_type,
            target_rule_description=simulation_case.test_goal.rule_description,
            target_rule_evaluation_hint=simulation_case.test_goal.evaluation_hint,
            target_rule_severity=simulation_case.test_goal.severity,
            set_id=sub_plan.set_id,
            set_label=sub_plan.label,
        )

    agent_turn = outbound.start()
    print(f"  Agent: {agent_turn.reply_text}")

    for _ in range(MAX_TURNS):
        if agent_turn.should_end:
            _archive()
            return

        user_turn = simulator.reply(agent_turn.reply_text)
        print(f"  User: {user_turn.reply_text}")

        if user_turn.should_end:
            outbound.record_user(user_turn.reply_text)
            _archive()
            return

        agent_turn = outbound.reply(user_turn.reply_text)
        print(f"  Agent: {agent_turn.reply_text}")

    _archive()
    print(f"  [警告] 达到最大轮次 {MAX_TURNS}，强制结束")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    if not _has_llm_api_key():
        raise RuntimeError("缺少 DEEPSEEK_API_KEY 或 OPENAI_API_KEY，请在 agent/.env 中填写")

    sessions_dir = project_root / "sessions"
    instructions_dir = project_root / "instructions"

    # 用法：
    # python auto_main.py
    # python auto_main.py instructions/confirm.json 3
    # python auto_main.py confirm 2
    # python auto_main.py "你是一名..."
    instruction_source = sys.argv[1] if len(sys.argv) > 1 else None
    num_sets = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    instruction, source_label = _load_instruction(instruction_source, instructions_dir)

    print(f"\n构建 simulation plan（{num_sets} 个 placeholder set）...")
    plan = build_simulation_plan(instruction, num_sets=num_sets)
    for sub_plan in plan.sub_plans:
        for simulation_case in sub_plan.test_cases:
            run_session(
                source_label,
                simulation_case,
                sub_plan,
                plan.agent_spec,
                sessions_dir,
            )

    print("\n全部会话生成完成。")


if __name__ == "__main__":
    main()
