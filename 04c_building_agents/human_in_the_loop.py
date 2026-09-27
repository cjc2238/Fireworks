"""Human-in-the-loop: destructive tools require approval. Read-only tools run freely.
A denial goes back to the model as data, so it can explain and adapt instead of crashing.

Usage:  python 04c_building_agents/human_in_the_loop.py          # you approve/deny interactively
        python 04c_building_agents/human_in_the_loop.py --auto-deny
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn.agent import Agent, tool

DEPLOYMENTS = {"prod-llm": {"replicas": 4, "traffic_rps": 35}, "old-experiment": {"replicas": 1, "traffic_rps": 0}}


@tool
def list_deployments() -> dict:
    """List deployments with replica count and current traffic (requests/sec)."""
    return DEPLOYMENTS


@tool(requires_approval=True)
def delete_deployment(deployment_id: str) -> str:
    """Permanently delete a deployment."""
    # Don't write "requires approval" in the description: the model then asks in chat instead of
    # calling the tool, and the approval gate never runs. Enforce approval in the runtime.
    DEPLOYMENTS.pop(deployment_id)
    return f"deleted {deployment_id}"


def approve(name: str, args: dict) -> bool:
    if "--auto-deny" in sys.argv:
        print(f"\n  ⛔ auto-denied {name}({args})")
        return False
    return input(f"\n  ⚠ Agent wants to run {name}({args}). Approve? [y/N] ").strip().lower() == "y"


agent = Agent(tools=[list_deployments, delete_deployment], approve=approve,
              system="You manage Fireworks deployments. Check data before acting. Never delete anything serving "
                     "traffic. Just call tools: the system asks a human to approve risky actions automatically.")
r = agent.run("Clean up: delete any deployments with zero traffic to save money.")
print("\n" + r.text)
print("\nremaining deployments:", list(DEPLOYMENTS))
