"""Router + specialist agents. A grammar-constrained router picks a specialist; each specialist has
only its own tools (least privilege). Handoffs pass a clean task summary, not the raw transcript.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import MODEL, client
from fwlearn.agent import Agent, tool

# ---- billing specialist ----------------------------------------------------------------
INVOICES = {"acme": [{"id": "INV-9", "amount": 1260.0, "status": "paid"},
                     {"id": "INV-10", "amount": 1310.5, "status": "overdue"}]}


@tool
def list_invoices(account: str) -> list:
    """List invoices for an account with amount and status."""
    return INVOICES.get(account.lower(), [])


# ---- infra specialist ------------------------------------------------------------------
DEPLOYMENTS = {"acme": [{"id": "prod-llm", "state": "READY", "min_replicas": 0, "errors_24h": 212}]}


@tool
def list_deployments(account: str) -> list:
    """List an account's dedicated deployments with state, min replicas, and recent 503 error count."""
    return DEPLOYMENTS.get(account.lower(), [])


SPECIALISTS = {
    "billing": Agent(tools=[list_invoices], system="You are the billing specialist. Be precise about amounts."),
    "infra": Agent(tools=[list_deployments], system=(
        "You are the infrastructure specialist for Fireworks deployments. Diagnose from data. "
        "Know: min_replicas=0 means scale-to-zero, which causes 503 DEPLOYMENT_SCALING_UP on cold start.")),
    "general": Agent(tools=[], system="You are a friendly general support agent."),
}
ROUTER_GRAMMAR = 'root ::= "billing" | "infra" | "general"'


def route(message: str) -> str:
    r = client().chat.completions.create(
        model=MODEL, temperature=0, max_tokens=5, reasoning_effort="none",
        response_format={"type": "grammar", "grammar": ROUTER_GRAMMAR},  # can ONLY output a valid label
        messages=[{"role": "system", "content": (
                      "Route the support message to exactly one team:\n"
                      "billing = invoices, payments, refunds, pricing on a bill\n"
                      "infra = deployments, API errors (4xx/5xx), latency, scaling, outages\n"
                      "general = product features, dashboard/UI, how-to, anything else\n"
                      "Examples: 'charged twice' -> billing; '503s in prod' -> infra; "
                      "'is there a Rust SDK?' -> general")},
                  {"role": "user", "content": message}])
    return r.choices[0].message.content.strip()


def handle(account: str, message: str) -> str:
    team = route(message)
    # Clean handoff: the specialist gets a short task brief, not the router's transcript or tool calls.
    brief = f"Customer account: {account}\nCustomer message: {message}"
    result = SPECIALISTS[team].run(brief)
    return f"[routed to {team} | {result.tool_calls} tool calls]\n{result.text}"


if __name__ == "__main__":
    for msg in ["Why do I have an overdue invoice? How much is it?",
                "Our prod endpoint throws 503s every morning, what's going on?",
                "Do you have a dark mode in the dashboard?"]:
        print(f"\n>>> {msg}\n{handle('acme', msg)}")
