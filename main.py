import asyncio
import base64
import binascii
import glob
import importlib.util
import io
import json
import os
import uuid
import wave
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PORT = int(os.getenv("PORT", "8000"))
DOMAIN = os.getenv("NGROK_URL")
if not DOMAIN:
    raise ValueError("NGROK_URL environment variable not set.")
WS_URL = f"wss://{DOMAIN}/ws"
HTTP_BASE_URL = f"https://{DOMAIN}"

LIVE_MODEL_NAME = os.getenv("GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-preview")
VOICE_NAME = os.getenv("GEMINI_VOICE_NAME", "charon") or "charon"

# Updated greeting to reflect the new model
WELCOME_GREETING = "Hi! I am a voice assistant powered by Outamation. Ask me anything!"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

GENAI_CLIENT = genai.Client(api_key=GOOGLE_API_KEY)

TOOLS: Dict[str, Any] = {}
TOOLS_META: Dict[str, Dict[str, Any]] = {}


def load_tools(tools_dir: str = "tools") -> None:
    """Dynamically import tool modules from `tools_dir` and populate registries."""
    pattern = os.path.join(tools_dir, "*.py")
    for path in glob.glob(pattern):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.startswith("_") or name in {"cli_test_harness"}:
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        if not spec or not spec.loader:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"Failed to import tool {path}: {exc}")
            continue

        tool = getattr(module, "TOOL", None)
        if not tool or "name" not in tool or "call" not in tool:
            print(f"Tool file {path} does not expose TOOL dict with 'name' and 'call'.")
            continue

        tool_name = tool["name"]
        TOOLS[tool_name] = tool["call"]
        TOOLS_META[tool_name] = {
            "description": tool.get("description", ""),
            "usage": tool.get("usage", {}),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        }
        print(f"Loaded tool: {tool_name}")


def build_tools_manifest() -> str:
    """Create manifest text describing tools for the system instruction."""
    if not TOOLS_META:
        return "No tools are currently available."

    lines: List[str] = ["Available tools:"]
    for name, meta in sorted(TOOLS_META.items()):
        desc = meta.get("description", "")
        usage = meta.get("usage", {})
        lines.append(f"- {name}: {desc} Usage example: {json.dumps(usage)}")

    lines.append("")
    lines.append("When you need external data or actions, call the relevant tool. Do not invent data.")
    lines.append(
        "For order inquiries, call get_order_status first. For weather questions, call get_realtime_weather first."
    )
    lines.append(
        "If a tool response returns an error, gracefully explain the issue or request clarification from the caller."
    )
    lines.append("If the caller asks about your capabilities, you may call list_tools to summarise them.")
    return "\n".join(lines)


def build_function_declarations() -> List[types.FunctionDeclaration]:
    """Convert tool metadata into Gemini function declarations."""
    declarations: List[types.FunctionDeclaration] = []
    for name, meta in sorted(TOOLS_META.items()):
        declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=meta.get("description", ""),
                parameters_json_schema=meta.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return declarations


# Load tools at import time so they're available for incoming calls
load_tools()


def list_tools(_: Dict[str, Any]) -> Dict[str, Any]:
    """Return the available tools with their description and usage."""
    items = []
    for name, meta in sorted(TOOLS_META.items()):
        items.append(
            {
                "name": name,
                "description": meta.get("description", ""),
                "usage": meta.get("usage", {}),
            }
        )
    return {"tools": items}


TOOLS["list_tools"] = list_tools
TOOLS_META["list_tools"] = {
    "description": "List available tools and their typical usage.",
    "usage": {},
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

SYSTEM_PROMPT = """
You are a friendly and precise voice assistant speaking on a live phone call. Respond in clear
spoken English, spell out numbers, and avoid special characters. Always keep replies concise and
grounded in real data.

You have two tools:
- get_realtime_weather: use it whenever the caller asks about weather, temperature, or conditions in
  any place.
- get_order_status: use it whenever the caller asks about an order, shipment, delivery, or tracking.

Workflow requirements:
1. Call the appropriate tool before you answer. Never guess or rely on memory.
2. If the caller spells or spaces an order identifier (for example “O R D one zero zero three”),
    normalise it to the canonical form before calling get_order_status.
3. After receiving a tool response, summarise it for the caller in natural speech.
4. Always return Gemini-generated audio so the caller hears your response. Include text alongside the
    audio only for logging; the caller must still receive audio.
5. If a required tool call fails or returns an error, apologise and explain the issue aloud. Do not
    respond with “I don’t know”; either call the tool or ask for the missing information.

Capability questions:
- If the caller asks what you can do, explicitly call list_tools, summarise each tool, and answer out
  loud so the caller hears your capabilities.
- Mention that you can check order status and live weather; do not invent extra skills.

Few-shot guidance:
Caller: “What is the weather in Paris?”
Assistant: (call get_realtime_weather {'city': 'Paris'})
Tool result: {'city': 'Paris', 'weather': 'Partly cloudy plus twenty degrees Celsius'}
Assistant: “The current weather in Paris is partly cloudy plus twenty degrees Celsius.”

Caller: “What can you do for me?”
Assistant: (call list_tools {})
Tool result: {'tools': [{'name': 'get_order_status', 'description': 'Lookup an order by order_id and return its status and details.'}, {'name': 'get_realtime_weather', 'description': 'Fetch current weather for a given city (uses wttr.in).'}]}
Assistant: “I can check your order status and fetch real-time weather updates. Just let me know which one you need.”

Caller: “Order I D O R D one zero zero two.”
Assistant: (call get_order_status {'order_id': 'ORD-1002'})
Tool result: {'found': True, 'order': {'order_id': 'ORD-1002', 'status': 'processing'}}
Assistant: “Order O R D dash one zero zero two is processing. There is no tracking number yet.”

Caller: “O R D one zero zero five status?”
Assistant: (call get_order_status {'order_id': 'ORD-1005'})
Tool result: {'found': False}
Assistant: “I could not find an order with I D O R D dash one zero zero five. Could you double-check it?”

Caller: “Try London then.”
Assistant: (call get_realtime_weather {'city': 'London'})
Tool result: {'city': 'London', 'weather': 'Clear plus seven degrees Celsius'}
Assistant: “The weather in London is clear plus seven degrees Celsius.”

Caller: “What is the current weather in Surat?”
Assistant: (call get_realtime_weather {'city': 'Surat'})
Tool result: {'city': 'Surat', 'weather': 'Overcast plus thirty degrees Celsius'}
Assistant: “In Surat it is overcast plus thirty degrees Celsius right now.”

Always follow this pattern.
"""

SYSTEM_INSTRUCTION = f"{SYSTEM_PROMPT}\n\n{build_tools_manifest()}"


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call a registered tool and return a JSON-serializable result."""
    if tool_name not in TOOLS:
        return {"error": f"Tool '{tool_name}' is not available."}

    try:
        result = TOOLS[tool_name](arguments)
    except Exception as exc:  # pragma: no cover - defensive logging
        return {"error": f"Tool '{tool_name}' raised an exception: {exc}"}

    if isinstance(result, dict):
        return result
    return {"result": result}


def expected_tool_for_prompt(user_prompt: str) -> Optional[str]:
    """Infer which tool must be used based on the caller's wording."""
    text = (user_prompt or "").lower()
    if not text:
        return None

    weather_keywords = ("weather", "temperature", "forecast", "rain", "climate")
    if any(keyword in text for keyword in weather_keywords):
        return "get_realtime_weather"

    capability_phrases = (
        "what can you do",
        "what else can you do",
        "how can you help",
        "what are you capable",
        "what services",
    )
    if any(phrase in text for phrase in capability_phrases):
        return "list_tools"

    # Simple detection for order-related questions, including spoken spellings of ORD IDs.
    order_keywords = ("order", "tracking", "shipment", "delivery", "parcel", "package")
    if any(keyword in text for keyword in order_keywords):
        return "get_order_status"

    compact = text.replace(" ", "")
    if compact.startswith("ord") or "ord-" in text or "ord" in text.split():
        return "get_order_status"

    return None


def _merge_audio_chunks(audio_chunks: List[Dict[str, Any]]) -> tuple[bytes, int]:
    """Decode Gemini inline audio parts into raw PCM bytes and determine sample rate."""
    pcm_buffer = bytearray()
    sample_rate = 24000

    for chunk in audio_chunks:
        data_b64 = chunk.get("data")
        if not data_b64:
            continue
        try:
            pcm_buffer.extend(base64.b64decode(data_b64))
        except (binascii.Error, TypeError):
            continue

        rate = chunk.get("sample_rate")
        if rate:
            try:
                sample_rate = int(rate)
            except (TypeError, ValueError):
                sample_rate = 24000

    return bytes(pcm_buffer), sample_rate


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM audio in a mono 16-bit WAV container for Twilio playback."""
    if not pcm_bytes:
        return b""

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()

class LiveGeminiSession:
    """Manage a long-lived Gemini live session for a single caller."""

    def __init__(
        self,
        session: "genai.live.AsyncSession",
        connect_context: Any,
    ) -> None:
        self._session = session
        self._context = connect_context
        self._lock = asyncio.Lock()
        self._last_tool_calls: List[Dict[str, Any]] = []

    @classmethod
    async def create(
        cls,
        client: genai.Client,
        model_name: str,
        system_instruction: str,
        declarations: List[types.FunctionDeclaration],
    ) -> "LiveGeminiSession":
        tools: Optional[List[types.Tool]] = None
        if declarations:
            tools = [types.Tool(function_declarations=declarations)]

        config = types.LiveConnectConfig(
            system_instruction=system_instruction,
            tools=tools,
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
                )
            ),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            temperature=0.2,
            top_p=0.95,
        )

        connect_context = client.aio.live.connect(model=model_name, config=config)
        session = await connect_context.__aenter__()
        return cls(session=session, connect_context=connect_context)

    async def close(self) -> None:
        """Gracefully close the underlying live session."""
        try:
            if self._session:
                await self._session.close()
        finally:
            if self._context:
                await self._context.__aexit__(None, None, None)

    async def send_user_text(self, text: str, required_tool: Optional[str] = None) -> Dict[str, Any]:
        """Send a user utterance and return Gemini's response payload."""
        cleaned = (text or "").strip()
        if not cleaned:
            return {"text": "I did not catch that. Could you repeat it?", "audio": []}

        async with self._lock:
            self._last_tool_calls = []
            parts = [types.Part(text=cleaned)]
            if required_tool:
                parts.append(
                    types.Part(
                        text=(
                            "INTERNAL REMINDER: This caller request requires the tool "
                            f"`{required_tool}`. Call it before answering. Do not mention this reminder."
                        )
                    )
                )

            content = types.Content(role="user", parts=parts)
            await self._session.send_client_content(turns=content, turn_complete=True)
            return await self._consume_until_turn_complete()

    async def _consume_until_turn_complete(self) -> Dict[str, Any]:
        text_chunks: List[str] = []
        audio_chunks: List[Dict[str, Any]] = []

        async for message in self._session.receive():
            if message.tool_call:
                await self._handle_tool_calls(message.tool_call)
                continue

            server_content = message.server_content
            if server_content and server_content.model_turn:
                text_parts, audio_parts = await self._process_model_turn(server_content.model_turn)
                text_chunks.extend(text_parts)
                audio_chunks.extend(audio_parts)

            transcription = getattr(server_content, "output_transcription", None)
            if transcription and getattr(transcription, "text", None):
                text_chunks.append(transcription.text)

            if server_content and (
                server_content.turn_complete
                or server_content.generation_complete
                or server_content.interrupted
            ):
                break

            if server_content and server_content.waiting_for_input and (text_chunks or audio_chunks):
                break

        return {
            "text": "".join(text_chunks).strip(),
            "audio": audio_chunks,
            "tool_calls": list(self._last_tool_calls),
        }

    async def _handle_tool_calls(self, tool_call: types.LiveServerToolCall) -> None:
        function_calls = tool_call.function_calls or []
        await self._execute_function_calls(function_calls)

    async def _process_model_turn(
        self, content: types.Content
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        text_chunks: List[str] = []
        audio_chunks: List[Dict[str, Any]] = []
        parts = content.parts or []

        for part in parts:
            if part.function_call:
                await self._execute_function_calls([part.function_call])
                continue

            if getattr(part, "text", None):
                text_chunks.append(part.text)
                continue

            blob = getattr(part, "inline_data", None)
            if blob and getattr(blob, "data", None):
                audio_chunks.append(self._serialise_audio_blob(blob))

        return text_chunks, audio_chunks

    @staticmethod
    def _serialise_audio_blob(blob: types.Blob) -> Dict[str, Any]:
        data_b64 = base64.b64encode(blob.data).decode("ascii") if blob.data else ""
        sample_rate = None
        mime_type = blob.mime_type or "audio/pcm;rate=24000"
        if ";" in mime_type:
            parts = mime_type.split(";")
            for piece in parts:
                piece = piece.strip()
                if piece.startswith("rate="):
                    try:
                        sample_rate = int(piece.split("=", 1)[1])
                    except ValueError:
                        sample_rate = None
        return {
            "data": data_b64,
            "mime_type": mime_type,
            "sample_rate": sample_rate or 24000,
        }

    async def _execute_function_calls(
        self, function_calls: List[types.FunctionCall]
    ) -> None:
        if not function_calls:
            return

        responses: List[types.FunctionResponse] = []
        for function_call in function_calls:
            name = function_call.name or ""
            payload = call_tool(name, function_call.args or {})
            print(f"Tool call -> {name}({function_call.args}) => {payload}")
            self._last_tool_calls.append(
                {
                    "name": name,
                    "arguments": function_call.args or {},
                    "response": payload,
                }
            )
            responses.append(
                types.FunctionResponse(
                    id=function_call.id,
                    name=name,
                    response=payload,
                )
            )

        if responses:
            await self._session.send_tool_response(function_responses=responses)

    async def enforce_tool_retry(self, required_tool: str, user_prompt: str) -> Dict[str, Any]:
        """Nudge Gemini to call a specific tool and retry the response."""
        reminder = (
            "INTERNAL REMINDER: The caller's last question was "
            f"\"{user_prompt}\". You must call the tool `{required_tool}` before answering. "
            "Call the tool now, then speak the answer aloud. Do not mention this reminder to the caller."
        )

        async with self._lock:
            self._last_tool_calls = []
            content = types.Content(role="user", parts=[types.Part(text=reminder)])
            await self._session.send_client_content(turns=content, turn_complete=True)
            return await self._consume_until_turn_complete()

# Create FastAPI app
app = FastAPI()

sessions: Dict[str, LiveGeminiSession] = {}
audio_cache: Dict[str, Dict[str, bytes]] = {}


async def start_call_session(call_sid: str) -> LiveGeminiSession:
    """Initialise a live Gemini session for the given call id."""
    existing = sessions.pop(call_sid, None)
    if existing:
        await existing.close()

    session = await LiveGeminiSession.create(
        client=GENAI_CLIENT,
        model_name=LIVE_MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION,
        declarations=build_function_declarations(),
    )
    sessions[call_sid] = session
    return session


async def end_call_session(call_sid: Optional[str]) -> None:
    if not call_sid:
        return
    session = sessions.pop(call_sid, None)
    if session:
        await session.close()
    audio_cache.pop(call_sid, None)

@app.post("/twiml")
async def twiml_endpoint():
    """Endpoint that returns TwiML for Twilio to connect to the WebSocket"""
    # Twilio ConversationRelay bridges phone audio to this WebSocket. We deliver Gemini-generated audio,
    # so no external TTS provider is declared here.
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
    <Connect>
    <ConversationRelay url="{WS_URL}" welcomeGreeting="{WELCOME_GREETING}" />
    </Connect>
    </Response>"""
    
    return Response(content=xml_response, media_type="text/xml")


@app.get("/audio/{call_sid}/{turn_id}.wav")
async def get_audio_asset(call_sid: str, turn_id: str):
    call_bucket = audio_cache.get(call_sid)
    if not call_bucket:
        raise HTTPException(status_code=404, detail="Call audio not found.")

    audio_bytes = call_bucket.get(turn_id)
    if not audio_bytes:
        raise HTTPException(status_code=404, detail="Audio clip not found.")

    return Response(content=audio_bytes, media_type="audio/wav")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint bridging Twilio ConversationRelay and Gemini Live."""
    await websocket.accept()
    call_sid: Optional[str] = None
    live_session: Optional[LiveGeminiSession] = None

    try:
        while True:
            payload = await websocket.receive_text()
            message = json.loads(payload)
            message_type = message.get("type")

            if message_type == "setup":
                call_sid = message.get("callSid")
                if not call_sid:
                    print("Setup message missing callSid; ignoring")
                    continue

                live_session = await start_call_session(call_sid)
                print(f"Setup complete for call: {call_sid}")

            elif message_type == "prompt":
                if not call_sid:
                    print("Prompt received before setup; ignoring")
                    continue

                live_session = live_session or sessions.get(call_sid)
                if not live_session:
                    live_session = await start_call_session(call_sid)

                user_prompt = message.get("voicePrompt", "")
                print(f"Processing prompt for {call_sid}: {user_prompt}")

                expected_tool = expected_tool_for_prompt(user_prompt)

                try:
                    response_payload = await live_session.send_user_text(user_prompt, expected_tool)
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"Gemini session error for {call_sid}: {exc}")
                    response_payload = {"text": "I'm sorry, I ran into an error handling that request.", "audio": []}

                tool_names = {
                    call.get("name")
                    for call in response_payload.get("tool_calls", []) or []
                    if call.get("name")
                }
                if expected_tool and expected_tool not in tool_names:
                    print(
                        f"Tool compliance retry for {call_sid}: expected {expected_tool}, received {tool_names or 'none'}"
                    )
                    try:
                        retry_payload = await live_session.enforce_tool_retry(expected_tool, user_prompt)
                        if retry_payload:
                            response_payload = retry_payload
                            tool_names = {
                                call.get("name")
                                for call in response_payload.get("tool_calls", []) or []
                                if call.get("name")
                            }
                    except Exception as exc:
                        print(f"Tool retry failed for {call_sid}: {exc}")

                text_response = (response_payload.get("text") or "").strip()
                audio_chunks = response_payload.get("audio") or []

                if not text_response and not audio_chunks:
                    text_response = "I'm not sure how to respond to that yet."

                audio_sent = False
                if audio_chunks:
                    pcm_bytes, sample_rate = _merge_audio_chunks(audio_chunks)
                    wav_bytes = _pcm_to_wav_bytes(pcm_bytes, sample_rate)
                    if wav_bytes:
                        turn_id = uuid.uuid4().hex
                        bucket = audio_cache.setdefault(call_sid, {})
                        bucket[turn_id] = wav_bytes
                        audio_url = f"{HTTP_BASE_URL}/audio/{call_sid}/{turn_id}.wav"
                        await websocket.send_text(json.dumps({"type": "play", "source": audio_url}))
                        audio_sent = True
                        print(
                            f"Queued audio for {call_sid}: url={audio_url} bytes={len(wav_bytes)} rate={sample_rate}"
                        )

                if text_response and not audio_sent:
                    await websocket.send_text(
                        json.dumps({"type": "text", "token": text_response, "last": True})
                    )

                print(
                    f"Sent response for {call_sid}: text_len={len(text_response)} audio_sent={audio_sent}"
                )

            elif message_type == "interrupt":
                print(f"Handling interruption for call {call_sid}.")

            else:
                print(f"Unknown message type received: {message_type}")

    except WebSocketDisconnect:
        print(f"WebSocket connection closed for call {call_sid}")
    finally:
        await end_call_session(call_sid)

if __name__ == "__main__":
    print(f"Starting server on port {PORT}")
    print(f"WebSocket URL for Twilio: {WS_URL}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)