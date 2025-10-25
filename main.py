import os
import json
import glob
import importlib.util
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PORT = int(os.getenv("PORT", "8000"))
DOMAIN = os.getenv("NGROK_URL") 
if not DOMAIN:
    raise ValueError("NGROK_URL environment variable not set.")
WS_URL = f"wss://{DOMAIN}/ws"

# Updated greeting to reflect the new model
WELCOME_GREETING = "Hi! I am a voice assistant powered by Outamation. Ask me anything!"

# System prompt for Gemini
# Gemini works well with a direct instruction like this.
SYSTEM_PROMPT = """You are a helpful and friendly voice assistant. This conversation is happening over a phone call, so your responses will be spoken aloud. 
Please adhere to the following rules:
1. Provide clear, concise, and direct answers.
2. Spell out all numbers (e.g., say 'one thousand two hundred' instead of 1200).
3. Do not use any special characters like asterisks, bullet points, or emojis.
4. Keep the conversation natural and engaging."""

# Tool-calling instructions for the model. If the model wants to invoke a tool, it MUST
# return a JSON object and ONLY that JSON object (no additional text). The JSON must
# have the shape:
# {
#   "tool_call": {
#       "name": "tool_name",
#       "arguments": { ... }
#   }
# }
# After the tool is executed, the assistant will receive the tool result and may then
# return a final text response.
SYSTEM_PROMPT += "\n\nIf you want to invoke an external tool, respond with only a JSON object like:\n{\n  \"tool_call\": {\n    \"name\": \"get_order_status\",\n    \"arguments\": {\n      \"order_id\": \"12345\"\n    }\n  }\n}\nDo not return any other text when making a tool call."

# --- Gemini API Initialization ---
# Get your Google API key from https://aistudio.google.com/app/apikey
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

genai.configure(api_key=GOOGLE_API_KEY)

# Configure the Gemini model. We pass the system prompt during initialization.
# gemini-2.5-flash-latest is a fast and capable model suitable for this use case.
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    system_instruction=SYSTEM_PROMPT
)

# Store active chat sessions
# We will now store Gemini's chat session objects
sessions = {}

# Tool registry loaded from the tools/ directory. Each tool module should expose a
# dict variable named TOOL with keys: 'name' and 'call' where 'call' is a function
# taking a dict of arguments and returning a serializable result (dict or str).
TOOLS: Dict[str, Any] = {}


def load_tools(tools_dir: str = "tools"):
    """Dynamically import tool modules from `tools_dir` and populate TOOLS."""
    pattern = os.path.join(tools_dir, "*.py")
    for path in glob.glob(pattern):
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                print(f"Failed to import tool {path}: {e}")
                continue
            tool = getattr(mod, "TOOL", None)
            if tool and "name" in tool and "call" in tool:
                TOOLS[tool["name"]] = tool["call"]
                print(f"Loaded tool: {tool['name']}")
            else:
                print(f"Tool file {path} does not expose TOOL dict with 'name' and 'call'.")


# Load tools at import time so they're available for incoming calls
load_tools()


def call_tool(tool_name: str, arguments: dict) -> Any:
    """Call a registered tool and return its result."""
    if tool_name not in TOOLS:
        return {"error": f"Tool '{tool_name}' not found"}
    try:
        result = TOOLS[tool_name](arguments)
        return result
    except Exception as e:
        return {"error": f"Tool '{tool_name}' raised exception: {e}"}

# Create FastAPI app
app = FastAPI()

async def gemini_response(chat_session, user_prompt):
    """Get a response from the Gemini API and stream it."""
    response = await chat_session.send_message_async(user_prompt)
    return response.text

@app.post("/twiml")
async def twiml_endpoint():
    """Endpoint that returns TwiML for Twilio to connect to the WebSocket"""
    # Note: Twilio ConversationRelay has built-in TTS. We specify a provider and voice.
    # You can change 'ElevenLabs' to 'Amazon' or 'Google' if you prefer their TTS.
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
    <Connect>
    <ConversationRelay url="{WS_URL}" welcomeGreeting="{WELCOME_GREETING}" ttsProvider="ElevenLabs" voice="FGY2WhTYpPnrIDTdsKH5" />
    </Connect>
    </Response>"""
    
    return Response(content=xml_response, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    call_sid = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "setup":
                call_sid = message["callSid"]
                print(f"Setup for call: {call_sid}")
                # Start a new chat session for this call
                sessions[call_sid] = model.start_chat(history=[])
                
            elif message["type"] == "prompt":
                if not call_sid or call_sid not in sessions:
                    print(f"Error: Received prompt for unknown call_sid {call_sid}")
                    continue

                user_prompt = message["voicePrompt"]
                print(f"Processing prompt: {user_prompt}")
                
                chat_session = sessions[call_sid]
                response_text = await gemini_response(chat_session, user_prompt)
                # The chat_session object automatically maintains history.

                # Check if assistant responded with a JSON tool_call request
                tool_called = False
                try:
                    parsed = json.loads(response_text)
                except Exception:
                    parsed = None

                if isinstance(parsed, dict) and "tool_call" in parsed:
                    tool_called = True
                    tool_spec = parsed["tool_call"]
                    tool_name = tool_spec.get("name")
                    tool_args = tool_spec.get("arguments", {})
                    print(f"Assistant requested tool: {tool_name} with args {tool_args}")

                    # Execute the tool
                    tool_result = call_tool(tool_name, tool_args)
                    print(f"Tool result: {tool_result}")

                    # Feed the tool result back into the chat session and ask for final response
                    # We send the tool result as plain text prefixed so the model can parse it.
                    follow_up_prompt = f"TOOL_RESULT: {json.dumps(tool_result)}\nPlease now produce the final spoken response to the user."
                    followup = await chat_session.send_message_async(follow_up_prompt)
                    final_text = followup.text

                    await websocket.send_text(
                        json.dumps({
                            "type": "text",
                            "token": final_text,
                            "last": True
                        })
                    )
                    print(f"Sent final response after tool call: {final_text}")

                if not tool_called:
                    # Send the complete response back to Twilio.
                    # Twilio's ConversationRelay will handle the text-to-speech conversion.
                    await websocket.send_text(
                        json.dumps({
                            "type": "text",
                            "token": response_text,
                            "last": True  # Indicate this is the full and final message
                        })
                    )
                    print(f"Sent response: {response_text}")
                
            elif message["type"] == "interrupt":
                print(f"Handling interruption for call {call_sid}.")
                
            else:
                print(f"Unknown message type received: {message['type']}")
                
    except WebSocketDisconnect:
        print(f"WebSocket connection closed for call {call_sid}")
        if call_sid in sessions:
            sessions.pop(call_sid)
            print(f"Cleared session for call {call_sid}")

if __name__ == "__main__":
    print(f"Starting server on port {PORT}")
    print(f"WebSocket URL for Twilio: {WS_URL}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)