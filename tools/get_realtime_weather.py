"""Simple realtime weather tool using wttr.in.

This module exposes a TOOL dict so the main app can auto-register it.
Expected arguments: {"city": "City Name"}
Returns a JSON-serializable dict with `city` and `weather` or an `error` key.
"""
import requests
from typing import Dict, Any


def _fetch_weather_for_city(city: str) -> str:
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    try:
        response = requests.get(url, timeout=5)
    except Exception as e:
        return f"error: {e}"
    if response.status_code == 200:
        return response.text.strip()
    return f"error: status {response.status_code}"


def get_realtime_weather(args: Dict[str, Any]) -> Dict[str, Any]:
    city = args.get("city") if isinstance(args, dict) else None
    if not city:
        return {"error": "argument 'city' is required"}

    weather = _fetch_weather_for_city(city)
    if weather.startswith("error:"):
        return {"error": weather}

    return {"city": city, "weather": weather}


TOOL = {
    "name": "get_realtime_weather",
    "call": get_realtime_weather,
    "description": "Fetch current weather for a given city (uses wttr.in).",
    "usage": {"city": "London"},
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name to query, for example 'Ahmedabad, India'."
            }
        },
        "required": ["city"],
        "additionalProperties": False
    }
}
