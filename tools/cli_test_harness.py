"""CLI test harness to simulate model -> tool -> model flow.

This script shows how the server would handle an assistant response that
requests a tool call, then injects the tool result back and gets a final reply.

Run:
    python3 tools/cli_test_harness.py

It's intentionally simple and doesn't require the Gemini API.
"""
import json
import os
import sys

# import the call_tool function from main.py
# adjust path so main.py can be imported
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import call_tool


def simulate_tool_call(tool_name, args):
    print(f"Simulating model tool_call -> {tool_name} with args {args}")
    result = call_tool(tool_name, args)
    print("Tool result:", result)
    # Simulate feeding the TOOL_RESULT back to the model and getting a final text
    # We'll just format a simple final message.
    if isinstance(result, dict) and result.get("error"):
        final = f"I tried to run {tool_name} but encountered an error: {result['error']}"
    else:
        final = f"Tool {tool_name} returned: {json.dumps(result)}"
    print("Final assistant response (simulated):\n", final)


if __name__ == '__main__':
    # examples
    simulate_tool_call("get_order_status", {"order_id": "ORD-1001"})
    print()
    simulate_tool_call("get_realtime_weather", {"city": "Ahmedabad, India"})