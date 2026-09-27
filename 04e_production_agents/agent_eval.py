"""Outcome-based agent eval: each task runs against a fresh simulated inventory system, and success
is checked on the FINAL STATE, not on how the answer sounds. Passing trajectories are saved
for fine-tuning (see trajectories_to_sft.py).

Usage:  python 04e_production_agents/agent_eval.py [model ...] [--repeats N]
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import statistics
from concurrent.futures import ThreadPoolExecutor

from fwlearn import MODEL, ROOT
from fwlearn.agent import Agent, tool
from fwlearn.tracing import Tracer


class Inventory:
    """The environment. Tools act on it; checkers inspect it afterwards."""

    def __init__(self):
        self.stock = {"FW-100": 42, "FW-200": 0, "FW-300": 7}
        self.orders: list[dict] = []
        self.transfers: list[dict] = []

    def tools(self):
        @tool
        def get_stock(sku: str) -> int:
            """Units in stock for a SKU."""
            if sku not in self.stock:
                raise KeyError(f"unknown SKU {sku}; known: {sorted(self.stock)}")
            return self.stock[sku]

        @tool
        def list_skus() -> list[str]:
            """All known SKUs."""
            return sorted(self.stock)

        @tool
        def create_purchase_order(sku: str, quantity: int) -> str:
            """Order more units of a SKU from the supplier (quantity >= 1)."""
            if sku not in self.stock or quantity < 1:
                raise ValueError("invalid sku or quantity")
            self.orders.append({"sku": sku, "quantity": quantity})
            return f"PO-{len(self.orders)} created"

        @tool
        def transfer_stock(from_sku: str, to_sku: str, quantity: int) -> str:
            """Relabel units from one SKU to another (used for repackaging)."""
            if self.stock.get(from_sku, 0) < quantity:
                raise ValueError(f"only {self.stock.get(from_sku, 0)} units of {from_sku}")
            self.stock[from_sku] -= quantity
            self.stock[to_sku] = self.stock.get(to_sku, 0) + quantity
            self.transfers.append({"from": from_sku, "to": to_sku, "quantity": quantity})
            return "ok"

        return [get_stock, list_skus, create_purchase_order, transfer_stock]


# (task prompt, checker(env, answer_text) -> bool)
TASKS = [
    ("How many FW-100 units do we have?",
     lambda e, a: "42" in a and not e.orders),
    ("Order 25 more FW-200.",
     lambda e, a: e.orders == [{"sku": "FW-200", "quantity": 25}]),
    ("Any SKU with zero stock should get a purchase order for 10 units.",
     lambda e, a: e.orders == [{"sku": "FW-200", "quantity": 10}]),
    ("Move 5 units from FW-100 to FW-300, then tell me FW-300's new count.",
     lambda e, a: e.stock["FW-300"] == 12 and e.stock["FW-100"] == 37 and "12" in a),
    ("Order enough FW-300 to bring it up to exactly 20 units.",
     lambda e, a: e.orders == [{"sku": "FW-300", "quantity": 13}]),
    ("Transfer 50 units from FW-300 to FW-200.",           # impossible: only 7 units. Must NOT half-do it
     lambda e, a: e.stock == {"FW-100": 42, "FW-200": 0, "FW-300": 7} and not e.transfers),
]


def run_task(model: str, idx: int, prompt: str, check) -> dict:
    env = Inventory()
    tracer = Tracer(f"eval-t{idx}")
    agent = Agent(tools=env.tools(), model=model, on_event=tracer, max_steps=8,
                  reasoning_effort="none" if model == MODEL else None, extra={"max_tokens": 4000},
                  system="You operate an inventory system. Use tools; don't guess numbers. "
                         "If a request is impossible, don't partially do it: explain instead.")
    try:
        r = agent.run(prompt)
        ok = bool(check(env, r.text))
    except Exception as e:  # an agent crash counts as a failure, not a script crash
        return {"task": idx, "ok": False, "error": str(e)}
    s = tracer.summary()
    return {"task": idx, "ok": ok, "steps": r.steps, "tool_calls": r.tool_calls, "tool_errors": s["tool_errors"],
            "tokens": r.prompt_tokens + r.completion_tokens, "seconds": r.seconds, "stopped": r.stopped,
            "messages": r.messages, "tools": [t.schema for t in env.tools()]}


def main():
    args = sys.argv[1:]
    repeats = int(args[args.index("--repeats") + 1]) if "--repeats" in args else 2
    models = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--repeats")] or [MODEL]
    out = ROOT / "outputs" / "agent_trajectories.jsonl"

    for model in models:
        jobs = [(model, i, p, c) for i, (p, c) in enumerate(TASKS) for _ in range(repeats)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda j: run_task(*j), jobs))
        print(f"\n{model}  ({repeats} runs per task)")
        for i, (prompt, _) in enumerate(TASKS):
            rs = [r for r in results if r["task"] == i]
            print(f"  task {i}: {sum(r['ok'] for r in rs)}/{len(rs)} pass  {prompt[:60]}")
        done = [r for r in results if "steps" in r]
        print(f"  SUCCESS {sum(r['ok'] for r in results)}/{len(results)} = {statistics.mean(r['ok'] for r in results):.0%}"
              f" | avg steps {statistics.mean(r['steps'] for r in done):.1f}"
              f" | tool errors {sum(r['tool_errors'] for r in done)}"
              f" | avg tokens {statistics.mean(r['tokens'] for r in done):.0f}"
              f" | p50 {statistics.median(r['seconds'] for r in done):.1f}s")
        with out.open("a", encoding="utf-8") as f:
            for r in results:
                if r["ok"]:
                    f.write(json.dumps({"model": model, "task": r["task"], "tools": r["tools"],
                                        "messages": r["messages"]}, default=str) + "\n")
    print(f"\npassing trajectories appended to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
