"""Stream tool calls: show each call the moment it starts, assemble argument fragments,
run the tools, then stream the final answer.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import time

from fwlearn import MODEL, client

TOOLS = [{"type": "function", "function": {
    "name": "get_stock", "description": "Get current inventory for a product SKU.",
    "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}}}]
INVENTORY = {"FW-100": 42, "FW-200": 0, "FW-300": 7}

fw = client()
messages = [{"role": "user", "content": "How many FW-100, FW-200 and FW-300 do we have in stock?"}]
t0 = time.perf_counter()


def stream_turn():
    """Stream one assistant turn. Returns (assistant_message_dict, finish_reason)."""
    calls, text = {}, []
    for chunk in fw.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS,
                                            stream=True, temperature=0, reasoning_effort="none"):
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        if ch.delta.content:
            text.append(ch.delta.content)
            print(ch.delta.content, end="", flush=True)
        for d in ch.delta.tool_calls or []:           # deltas are keyed by index
            c = calls.setdefault(d.index, {"id": "", "name": "", "arguments": ""})
            if d.id:
                c["id"] = d.id
            if d.function and d.function.name:
                c["name"] = d.function.name
                print(f"\n[{time.perf_counter() - t0:5.2f}s] 🔧 {c['name']}(", end="", flush=True)
            if d.function and d.function.arguments:
                c["arguments"] += d.function.arguments   # JSON fragments; don't parse yet
                print(d.function.arguments, end="", flush=True)
        if ch.finish_reason:
            finish = ch.finish_reason
    msg = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        print(")")
        msg["tool_calls"] = [{"id": c["id"], "type": "function",
                              "function": {"name": c["name"], "arguments": c["arguments"]}}
                             for _, c in sorted(calls.items())]
    return msg, finish


msg, finish = stream_turn()
messages.append(msg)
if finish == "tool_calls":
    for tc in msg["tool_calls"]:
        sku = json.loads(tc["function"]["arguments"])["sku"]  # safe now: the call is complete
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps({"sku": sku, "in_stock": INVENTORY.get(sku, "unknown")})})
    print(f"[{time.perf_counter() - t0:5.2f}s] tools done, streaming answer:\n")
    stream_turn()
print(f"\n\n[{time.perf_counter() - t0:5.2f}s] total")
