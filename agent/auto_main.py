from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import OUTBOUND_INSTRUCTIONS, make_outbound_agent
from src.agent_spec import AgentSpec
from src.orders import load_pending_orders
from src.session import OutboundSession
from src.simulation_plan import SimulationPlan, SubPlan, build_simulation_plan
from src.simulator import UserSimulator
from src.test_case_generator import SimulationCase
from src.types import LoadedOrder

MAX_TURNS = 15


def _load_scenario_instructions(instructions_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in instructions_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        mapping[data["id"]] = data["instruction"]
    return mapping


def run_session(
    loaded_order: LoadedOrder,
    simulation_case: SimulationCase,
    sub_plan: SubPlan,
    agent_spec: AgentSpec,
    sessions_dir: Path,
) -> None:
    print(
        f"\n[{sub_plan.set_id} · {simulation_case.target_rule_id}/{simulation_case.profile_type}] "
        f"{loaded_order.order.order_id}"
    )

    agent = make_outbound_agent(sub_plan.filled_instruction)
    outbound = OutboundSession(loaded_order, agent=agent)
    simulator = UserSimulator(loaded_order, simulation_case, agent_spec)
    session_label = (
        f"{sub_plan.set_id} · {simulation_case.target_rule_id} · "
        f"{simulation_case.profile_type}"
    )

    def _archive():
        outbound.save_archive(
            sessions_dir,
            "agent_end",
            persona_type=simulation_case.profile_type,
            simulator_label=session_label,
            test_case_id=simulation_case.test_id,
            target_rule_id=simulation_case.target_rule_id,
        )

    agent_turn = outbound.start()
    print(f"  数字人: {agent_turn.reply_text}")

    for _ in range(MAX_TURNS):
        if agent_turn.should_end:
            _archive()
            return

        user_turn = simulator.reply(agent_turn.reply_text)
        print(f"  用户: {user_turn.reply_text}")

        if user_turn.should_end:
            outbound.record_user(user_turn.reply_text)
            _archive()
            return

        agent_turn = outbound.reply(user_turn.reply_text)
        print(f"  数字人: {agent_turn.reply_text}")

    _archive()
    print(f"  [警告] 达到最大轮次 {MAX_TURNS}，强制结束")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，请在 agent/.env 中填写")

    orders_dir = project_root / "orders"
    sessions_dir = project_root / "sessions"
    instructions_dir = project_root / "instructions"

    # 可选参数：python auto_main.py confirm_001 3
    order_id_filter = sys.argv[1] if len(sys.argv) > 1 else None
    num_sets = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    scenario_instructions = _load_scenario_instructions(instructions_dir)

    orders = load_pending_orders(orders_dir)
    if order_id_filter:
        orders = [o for o in orders if o.order.order_id == order_id_filter]
    if not orders:
        print("未找到匹配的订单。")
        return

    plan_cache: dict[str, SimulationPlan] = {}
    for order in orders:
        instruction = scenario_instructions.get(
            order.order.scenario or "", OUTBOUND_INSTRUCTIONS
        )
        if instruction not in plan_cache:
            print(f"\n构建 simulation plan（{num_sets} 个 placeholder set）...")
            plan_cache[instruction] = build_simulation_plan(instruction, num_sets=num_sets)

        plan = plan_cache[instruction]
        for sub_plan in plan.sub_plans:
            for simulation_case in sub_plan.test_cases:
                run_session(order, simulation_case, sub_plan, plan.agent_spec, sessions_dir)

    print("\n所有订单处理完成。")


if __name__ == "__main__":
    main()
