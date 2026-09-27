"""Memory: keep a long conversation in budget.
- short-term: the recent turns, verbatim
- summary: older turns compressed by the model
- long-term: durable facts the agent chooses to save with a `remember` tool

Runs a scripted 8-turn conversation and prints how the prompt size stays bounded.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import MODEL, client
from fwlearn.agent import Agent, tool

FACTS: dict[str, str] = {}
KEEP_TURNS = 4          # recent user/assistant messages kept verbatim
SUMMARIZE_OVER = 8      # compress when history grows past this many messages


@tool
def remember(key: str, value: str) -> str:
    """Save a durable fact about the user (name, goals, preferences) for future turns."""
    FACTS[key] = value
    return "saved"


fw = client()
summary = ""
history: list[dict] = []


def system_prompt() -> str:
    facts = "\n".join(f"- {k}: {v}" for k, v in FACTS.items()) or "(none)"
    return (f"You are a study coach for a Fireworks AI course. Be brief.\n"
            f"Known facts about the user:\n{facts}\nSummary of earlier conversation:\n{summary or '(none)'}")


def compress():
    global summary, history
    old, history = history[:-KEEP_TURNS], history[-KEEP_TURNS:]
    text = "\n".join(f"{m['role']}: {m.get('content') or ''}" for m in old if m["role"] in ("user", "assistant"))
    r = fw.chat.completions.create(model=MODEL, reasoning_effort="none", max_tokens=200, messages=[
        {"role": "user", "content": f"Update this running summary with the new turns. Max 80 words.\n"
                                    f"Summary so far: {summary or '(none)'}\nNew turns:\n{text}"}])
    summary = r.choices[0].message.content.strip()
    print(f"   [compressed {len(old)} messages -> summary of {len(summary)} chars]")


SCRIPT = ["Hi! I'm Sam and I want a Fireworks solutions engineer job by December.",
          "I'm weak on deployments. What should I study first?",
          "What's a deployment shape?",
          "And autoscaling? Keep it short.",
          "What about cold starts?",
          "I prefer learning by doing, remember that.",
          "OK, plan my next 3 study sessions.",
          "What's my name, my goal, and how do I like to learn?"]

for turn in SCRIPT:
    agent = Agent(tools=[remember], system=system_prompt())
    r = agent.run(turn, history=[{"role": "system", "content": system_prompt()}] + history)
    # keep only user/assistant text turns in long-lived history (tool chatter stays out)
    history += [{"role": "user", "content": turn}, {"role": "assistant", "content": r.text}]
    print(f"\nyou> {turn}\nbot> {r.text.strip()[:300]}\n   [prompt tokens this turn: {r.prompt_tokens}]")
    if len(history) > SUMMARIZE_OVER:
        compress()

print("\nlong-term facts:", FACTS)
