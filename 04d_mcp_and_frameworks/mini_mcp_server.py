"""A minimal, dependency-free MCP server over stdio (newline-delimited JSON-RPC 2.0).

It implements the parts of MCP needed for tools: initialize, tools/list, tools/call, ping.
Real MCP hosts (Claude Code, Cursor, ...) can use it too:
    command: <path to python>   args: [<path to this file>]

To add a tool: write a function and add it to TOOLS with a JSON Schema. Clients discover it
automatically through tools/list.
"""

import json
import sys

PROTOCOL_VERSION = "2025-06-18"

GPU_PRICES = {"H100": 2.90, "H200": 4.00, "B200": 5.80}  # illustrative only
KB = {
    "cold-start": "Scaled-to-zero deployments return 503 DEPLOYMENT_SCALING_UP; requests are not queued. "
                  "Fix: min replicas >= 1 or client retries with backoff.",
    "prompt-caching": "Put static content first; use x-session-affinity; watch fireworks-cached-prompt-tokens.",
    "lora": "Serve one LoRA with live merge, or many with multi-LoRA (--enable-addons + load-lora).",
}


def gpu_price(gpu: str, hours: float = 1.0) -> str:
    g = gpu.upper()
    if g not in GPU_PRICES:
        raise ValueError(f"unknown GPU {gpu!r}; known: {sorted(GPU_PRICES)}")
    return json.dumps({"gpu": g, "hours": hours, "usd": round(GPU_PRICES[g] * hours, 2)})


def kb_search(topic: str) -> str:
    hits = {k: v for k, v in KB.items() if topic.lower() in k or k in topic.lower()}
    return json.dumps(hits or {"note": f"no article; topics are {sorted(KB)}"})


TOOLS = {
    "gpu_price": (gpu_price, "Price to rent a GPU type for some hours.", {
        "type": "object", "required": ["gpu"],
        "properties": {"gpu": {"type": "string", "enum": sorted(GPU_PRICES)},
                       "hours": {"type": "number", "default": 1}}}),
    "kb_search": (kb_search, "Search the support knowledge base by topic (e.g. 'cold-start', 'lora').", {
        "type": "object", "required": ["topic"], "properties": {"topic": {"type": "string"}}}),
}


def handle(req: dict) -> dict | None:
    method, rid, params = req.get("method"), req.get("id"), req.get("params") or {}
    if rid is None:                      # notifications (e.g. notifications/initialized) get no reply
        return None
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fireworks-course-mini-mcp", "version": "0.1.0"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [{"name": n, "description": d, "inputSchema": s} for n, (_, d, s) in TOOLS.items()]}
    elif method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in TOOLS:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
        try:  # tool failures are results with isError=True, so the model can see them
            result = {"content": [{"type": "text", "text": TOOLS[name][0](**args)}], "isError": False}
        except Exception as e:
            result = {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:               # one JSON-RPC message per line
        if not line.strip():
            continue
        resp = handle(json.loads(line))
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()           # stdio transport: flush every message
        # logs must go to stderr; stdout is reserved for protocol messages
        print(f"[mini-mcp] {json.loads(line).get('method')}", file=sys.stderr)


if __name__ == "__main__":
    main()
