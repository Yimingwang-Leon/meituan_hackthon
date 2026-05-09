from __future__ import annotations

import json
from pathlib import Path

from .types import LoadedOrder, OrderRecord


ORDER_EXTENSIONS = {".json"}


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"订单字段 {field_name} 缺失或为空")
    return value.strip()


def _parse_order_record(raw_value: object) -> OrderRecord:
    if not isinstance(raw_value, dict):
        raise ValueError("订单内容必须是 JSON 对象")

    return OrderRecord(
        user_name=_require_non_empty_string(raw_value.get("userName"), "userName"),
        order_id=_require_non_empty_string(raw_value.get("orderId"), "orderId"),
        eta=_require_non_empty_string(raw_value.get("eta"), "eta"),
        address=_require_non_empty_string(raw_value.get("address"), "address"),
        notes=raw_value.get("notes").strip() if isinstance(raw_value.get("notes"), str) and raw_value.get("notes").strip() else None,
    )


def load_pending_orders(orders_dir: str | Path) -> list[LoadedOrder]:
    orders_path = Path(orders_dir)
    files = sorted(
        [
            path
            for path in orders_path.iterdir()
            if path.is_file() and path.suffix.lower() in ORDER_EXTENSIONS
        ],
        key=lambda path: path.name,
    )

    loaded_orders: list[LoadedOrder] = []
    for file_path in files:
        raw_order = json.loads(file_path.read_text(encoding="utf-8"))
        loaded_orders.append(
            LoadedOrder(
                file_path=str(file_path),
                file_name=file_path.name,
                order=_parse_order_record(raw_order),
            )
        )

    return loaded_orders
