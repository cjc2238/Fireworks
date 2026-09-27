"""A small but production-shaped agent: tool registry, argument validation, error recovery,
parallel execution, and an iteration cap.

Usage:  python 04_tool_calling/agent_loop.py "your question"
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ast
import json
import operator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from fwlearn import MODEL, client

# ---------------------------------------------------------------- tools
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def calculator(expression: str) -> float:
    """Safely evaluate arithmetic (no eval())."""
    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.BinOp):
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"unsupported expression: {expression}")
    return ev(ast.parse(expression, mode="eval").body)


def current_time(tz_offset_hours: int = 0) -> str:
    tz = timezone(timedelta(hours=tz_offset_hours))
    return datetime.now(tz).isoformat(timespec="minutes")


GPU_PRICES = {"H100": 2.90, "H200": 4.00, "B200": 5.80}  # illustrative $/hr, NOT real pricing


def gpu_price(gpu: str) -> dict:
    if gpu.upper() not in GPU_PRICES:
        raise KeyError(f"unknown gpu {gpu!r}; known: {list(GPU_PRICES)}")
    return {"gpu": gpu.upper(), "usd_per_hour": GPU_PRICES[gpu.upper()]}


REGISTRY = {
    "calculator": (calculator, "Evaluate an arithmetic expression like '(3+4)*2'. Use for ANY math.",
                   {"expression": {"type": "string"}}, ["expression"]),
    "current_time": (current_time, "Get the current time at a UTC offset in hours.",
                     {"tz_offset_hours": {"type": "integer"}}, []),
    "gpu_price": (gpu_price, "Look up the hourly price of a GPU type (H100, H200, B200).",
                  {"gpu": {"type": "string"}}, ["gpu"]),
}
TOOLS = [{"type": "function", "function": {
    "name": name, "description": desc,
    "parameters": {"type": "object", "properties": props, "required": req}}}
    for name, (_, desc, props, req) in REGISTRY.items()]


def run_tool(tc) -> dict:
    fn = REGISTRY.get(tc.function.name, (None,))[0]
    try:
        if fn is None:
            raise KeyError(f"no such tool {tc.function.name}")
        args = json.loads(tc.function.arguments or "{}")
        out = {"ok": True, "result": fn(**args)}
    except Exception as e:  # send errors back so the model can recover
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(f"   [tool] {tc.function.name}({tc.function.arguments}) -> {out}")
    return {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)}


# ---------------------------------------------------------------- loop
def run_agent(question: str, max_steps: int = 8) -> str:
    fw = client()
    messages = [
        {"role": "system", "content": "You are a precise assistant. Use tools for facts and math; never guess numbers."},
        {"role": "user", "content": question},
    ]
    for step in range(1, max_steps + 1):
        resp = fw.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, temperature=0.1)
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        print(f"step {step}: {len(msg.tool_calls)} tool call(s)")
        with ThreadPoolExecutor() as pool:  # parallel tool execution
            messages.extend(pool.map(run_tool, msg.tool_calls))
    return "Stopped: hit max_steps without a final answer."


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or (
        "How much does it cost to run 8 H100s and 4 B200s for 36 hours? "
        "Also what time is it in Tokyo (UTC+9)? And what's the price of an A100?"
    )
    print("Q:", q, "\n")
    print("\nA:", run_agent(q))
