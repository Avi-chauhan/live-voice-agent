"""Tool: get_order_status

This tool looks up an order by order_id in the repository's `data/orders.json` file
and returns details. It exposes a TOOL dict with keys 'name' and 'call'.
"""
import os
import json
from typing import Dict, Any

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "orders.json")


def _load_orders() -> Dict[str, Any]:
    try:
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_order_status(args: Dict[str, Any]) -> Dict[str, Any]:
    """Arguments: {'order_id': '...'}

    Returns a dict with either order details or an error flag.
    """
    order_id = args.get("order_id")
    if not order_id:
        return {"error": "argument 'order_id' is required"}

    orders = _load_orders()
    order = orders.get(order_id)
    if not order:
        return {"found": False, "order_id": order_id}

    return {"found": True, "order": order}


TOOL = {
    "name": "get_order_status",
    "call": get_order_status,
}
