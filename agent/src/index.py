from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .orders import load_pending_orders
from .session import OutboundSession


def _print_order_header(index: int, total: int, order) -> None:
    print()
    print(f"=== 订单 {index + 1}/{total} ===")
    print(f"订单编号: {order.order.order_id}")
    print(f"用户称呼: {order.order.user_name}")
    print(f"预计送达: {order.order.eta}")
    print(f"配送地址: {order.order.address}")
    if order.order.notes:
        print(f"备注: {order.order.notes}")
    print("输入 /next 进入下一单，输入 /quit 结束程序。")
    print()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "缺少 OPENAI_API_KEY，请先在 agent/.env 中填写，或复制 agent/.env.example 为 agent/.env"
        )

    orders_dir = project_root / "orders"
    sessions_dir = project_root / "sessions"

    orders = load_pending_orders(orders_dir)
    if not orders:
        print(f"未在 {orders_dir} 中找到可用订单文件。")
        return

    for index, order in enumerate(orders):
        _print_order_header(index, len(orders), order)

        session = OutboundSession(order)
        opening_turn = session.start()
        print(f"数字人: {opening_turn.reply_text}")

        if opening_turn.should_end:
            archive_path = session.save_archive(sessions_dir, "agent_end")
            print(f"会话已结束并归档: {archive_path}")
            continue

        while True:
            try:
                answer = input("用户: ").strip()
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

            if answer == "/next":
                archive_path = session.save_archive(sessions_dir, "next")
                print(f"会话已归档: {archive_path}")
                break

            turn = session.reply(answer)
            print(f"数字人: {turn.reply_text}")

            if turn.should_end:
                archive_path = session.save_archive(sessions_dir, "agent_end")
                print(f"会话已结束并归档: {archive_path}")
                break

    print()
    print("所有订单已处理完成。")
