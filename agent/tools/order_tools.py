from __future__ import annotations

from typing import Any


MOCK_ORDERS: dict[str, dict[str, Any]] = {
    "order_001": {
        "order_id": "order_001",
        "status": "delivering",
        "masked_phone": "138****0001",
        "eta_minutes": 18,
        "refund_status": "not_requested",
    },
    "order_002": {
        "order_id": "order_002",
        "status": "cancel_requested",
        "masked_phone": "139****0002",
        "eta_minutes": None,
        "refund_status": "pending",
    },
}


def query_order(order_id: str) -> dict[str, Any]:
    order = MOCK_ORDERS.get(order_id)
    if order is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, **order}


def query_refund_status(order_id: str) -> dict[str, Any]:
    order = MOCK_ORDERS.get(order_id)
    if order is None:
        return {"found": False, "order_id": order_id, "refund_status": "unknown"}
    return {
        "found": True,
        "order_id": order_id,
        "refund_status": order.get("refund_status", "unknown"),
    }


def transfer_to_human(reason: str) -> dict[str, Any]:
    return {
        "transferred": True,
        "queue": "manual_outbound_support",
        "reason": reason,
    }


def update_callback_time(order_id: str, new_time: str) -> dict[str, Any]:
    return {
        "updated": True,
        "order_id": order_id,
        "new_time": new_time,
    }
