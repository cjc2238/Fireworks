"""One complete tool-calling round-trip, printed step by step."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

from fwlearn import MODEL, banner, client

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city. Use for any question about current weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]


def get_weather(city: str, unit: str = "celsius") -> dict:
    fake = {"paris": 18, "tokyo": 24, "new york": 21}
    c = fake.get(city.lower(), 20)
    return {"city": city, "temp": c if unit == "celsius" else round(c * 9 / 5 + 32), "unit": unit}


fw = client()
messages = [{"role": "user", "content": "What's the weather in Tokyo and in Paris, in fahrenheit?"}]

banner("Step 1: model decides to call tools")
r1 = fw.chat.completions.create(model=MODEL, messages=messages, tools=tools, temperature=0)
msg = r1.choices[0].message
print("finish_reason:", r1.choices[0].finish_reason)
for tc in msg.tool_calls or []:
    print(f"  tool_call id={tc.id} {tc.function.name}({tc.function.arguments})")

banner("Step 2: we execute them and send results back")
messages.append(msg)  # the assistant turn with tool_calls
for tc in msg.tool_calls or []:
    result = get_weather(**json.loads(tc.function.arguments))
    print("  result:", result)
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

banner("Step 3: model writes the final answer")
r2 = fw.chat.completions.create(model=MODEL, messages=messages, tools=tools, temperature=0)
print(r2.choices[0].message.content)
