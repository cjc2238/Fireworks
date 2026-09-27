"""Tool-selection eval: does the model pick the right tool (or no tool) with the right key argument?

Usage:  python 04b_advanced_tool_calling/tool_selection_eval.py [model-id ...]
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import asyncio
import json
import time

from fwlearn import BIG_MODEL, MODEL, async_client


def fn(name, desc, **props):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
        "required": list(props)}}}


# Deliberately confusable tools
TOOLS = [
    fn("search_orders", "Find a customer's orders by email.", email="customer email"),
    fn("get_order_status", "Get shipping status for ONE order by order id.", order_id="e.g. ORD-123"),
    fn("refund_order", "Issue a refund for an order.", order_id="e.g. ORD-123"),
    fn("get_invoice", "Get the invoice PDF link for an order.", order_id="e.g. ORD-123"),
    fn("search_docs", "Search product documentation for how-to questions.", query="search text"),
    fn("create_ticket", "Escalate to a human by creating a support ticket.", summary="issue summary"),
    fn("get_weather", "Current weather for a city.", city="city"),
    fn("convert_currency", "Convert an amount between currencies.", expression="e.g. '100 USD to EUR'"),
]

# (prompt, expected tool or None, (arg, substring expected in value) or None)
CASES = [
    ("Where is my order ORD-881?", "get_order_status", ("order_id", "ORD-881")),
    ("Show me all orders for jo@x.io", "search_orders", ("email", "jo@x.io")),
    ("I want my money back for ORD-12, it arrived broken", "refund_order", ("order_id", "ORD-12")),
    ("Can I get the receipt for ORD-77 for my expense report?", "get_invoice", ("order_id", "ORD-77")),
    ("How do I rotate my API key?", "search_docs", None),
    ("Nothing works and I've tried everything, I need a person", "create_ticket", None),
    ("What's 2+2?", None, None),
    ("Thanks, that's all!", None, None),
    ("How much is 50 euros in yen?", "convert_currency", None),
    ("Is it raining in Lima right now?", "get_weather", ("city", "Lima")),
]


async def run_case(fw, model, sem, prompt, exp_tool, exp_arg):
    async with sem:
        kw = {"reasoning_effort": "none"} if model == MODEL else {"max_tokens": 4000}
        t0 = time.perf_counter()
        r = await fw.chat.completions.create(
            model=model, temperature=0, tools=TOOLS,
            messages=[{"role": "system", "content": "You are a customer support agent. Use a tool only when needed."},
                      {"role": "user", "content": prompt}], **kw)
        dt = time.perf_counter() - t0
    calls = r.choices[0].message.tool_calls or []
    got = calls[0].function.name if calls else None
    ok = got == exp_tool
    if ok and exp_arg and calls:
        try:
            ok = exp_arg[1].lower() in str(json.loads(calls[0].function.arguments).get(exp_arg[0], "")).lower()
        except json.JSONDecodeError:
            ok = False
    return ok, got, dt


async def main(models):
    fw = async_client(max_retries=4, timeout=300)
    sem = asyncio.Semaphore(6)
    for model in models:
        res = await asyncio.gather(*(run_case(fw, model, sem, *c) for c in CASES))
        print(f"\n{model}")
        for (prompt, exp, _), (ok, got, dt) in zip(CASES, res):
            print(f"  {'PASS' if ok else 'FAIL'}  {dt:4.1f}s  expected={str(exp):<17} got={str(got):<17} {prompt[:45]}")
        acc = sum(r[0] for r in res) / len(res)
        print(f"  accuracy={acc:.0%}  avg latency={sum(r[2] for r in res) / len(res):.2f}s")


asyncio.run(main(sys.argv[1:] or [MODEL, BIG_MODEL]))
