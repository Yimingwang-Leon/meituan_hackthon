from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.orders import load_pending_orders
from src.persona import ALL_PERSONAS, UserPersona
from src.session import OutboundSession
from src.simulator import UserSimulator
from src.types import LoadedOrder

MAX_TURNS = 15


def run_session(loaded_order: LoadedOrder, persona: UserPersona, sessions_dir: Path) -> None:
    print(f"\n[{persona.persona_type}] {loaded_order.order.order_id}")

    outbound = OutboundSession(loaded_order)
    simulator = UserSimulator(loaded_order, persona)

    agent_turn = outbound.start()
    print(f"  数字人: {agent_turn.reply_text}")

    for _ in range(MAX_TURNS):
        if agent_turn.should_end:
            outbound.save_archive(sessions_dir, "agent_end")
            return

        user_turn = simulator.reply(agent_turn.reply_text)
        print(f"  用户: {user_turn.reply_text}")

        if user_turn.should_end:
            outbound.record_user(user_turn.reply_text)
            outbound.save_archive(sessions_dir, "agent_end")
            return

        agent_turn = outbound.reply(user_turn.reply_text)
        print(f"  数字人: {agent_turn.reply_text}")

    outbound.save_archive(sessions_dir, "agent_end")
    print(f"  [警告] 达到最大轮次 {MAX_TURNS}，强制结束")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，请在 agent/.env 中填写")

    orders_dir = project_root / "orders"
    sessions_dir = project_root / "sessions"

    orders = load_pending_orders(orders_dir)
    if not orders:
        print(f"未在 {orders_dir} 中找到可用订单文件。")
        return

    for order in orders:
        for persona in ALL_PERSONAS:
            run_session(order, persona, sessions_dir)

    print("\n所有订单处理完成。")


if __name__ == "__main__":
    main()
