# Live Voice Agent (tool-calling demo)

This repository contains a demo voice agent server (FastAPI) configured to use Google Gemini (via `google-genai`) and demonstrates a simple tool-calling mechanism with pluggable tools.

What I added in this demo:

- `data/orders.json` — mock orders database
- `tools/get_order_status.py` — a pluggable tool that looks up `order_id` in `data/orders.json`
- Updated `main.py` — loads tools dynamically and instructs the model to emit structured JSON when invoking tools. The server will execute the tool and feed the result back to the model.

How the tool calling works (local/demo):

1. The assistant (Gemini) is instructed to respond with a JSON object when it wants to call a tool. Example:

```
{ "tool_call": { "name": "get_order_status", "arguments": { "order_id": "ORD-1001" } } }
```

2. The server detects this JSON, executes the corresponding tool in `tools/`, and returns the tool result back into the chat session so the assistant can produce the final spoken response.

Running the demo (text mode):

1. Ensure environment variables are set in `.env`: `NGROK_URL`, `GOOGLE_API_KEY`, optionally `PORT`.
2. Install requirements: `pip install -r requirements.txt` (prefer a virtualenv).
3. Run the server:

```bash
python main.py
```

Testing the order lookup flow:

- Add or use a phone/Twilio setup that posts messages to the WebSocket (existing Twilio ConversationRelay integration remains). For a quick local test you can modify `main.py` to call the chat flow directly (or send a `prompt` message to the websocket).

Extending with new tools:

1. Create a new Python file under `tools/` that exposes a `TOOL` dict:

```py
TOOL = {
    "name": "my_tool",
    "call": my_call_fn,
}
```

`my_call_fn` should accept a single `dict` argument and return a JSON-serializable object.

Notes and next steps:

- This demo expects the model to obey the tool-calling JSON contract. When integrating real Gemini/LLM models, you can prefer the model's built-in function-calling APIs if available — the scaffold here makes it easy to swap.
- If you want I can also add a small CLI test harness that simulates the LLM returning a tool call (useful offline) and demonstrates the full loop.
