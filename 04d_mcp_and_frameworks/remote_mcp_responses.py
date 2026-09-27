"""Remote MCP through the Responses API: Fireworks' servers call the MCP tools for you.
We use DeepWiki's public MCP server, which answers questions about GitHub repos.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import time

from fwlearn import MODEL, banner, openai_client

oai = openai_client(timeout=300)
MCP = [{"type": "mcp", "server_url": "https://mcp.deepwiki.com/mcp", "server_label": "deepwiki"}]


def show(resp, t0):
    print(f"({time.perf_counter() - t0:.1f}s) output items:", [o.type for o in resp.output])
    for o in resp.output:
        if o.type == "mcp_call":
            call = getattr(o, "mcp", None) or {"name": o.name, "arguments": o.arguments}
            print("  🔧 server-side MCP call:", call)
        elif o.type == "tool_output":
            print("  📦 tool output:", str(getattr(o, "output", ""))[:150].replace("\n", " "), "...")
    print("\n" + resp.output_text)


banner("Turn 1: the model decides to call DeepWiki; Fireworks executes it")
t0 = time.perf_counter()
r1 = oai.responses.create(model=MODEL, tools=MCP,
                          input="Using the deepwiki tools: what is the fw-ai/cookbook GitHub repo for? 3 sentences.")
show(r1, t0)

banner("Turn 2: stateful follow-up, no history resent (previous_response_id)")
t0 = time.perf_counter()
r2 = oai.responses.create(model=MODEL, tools=MCP, previous_response_id=r1.id,
                          input="Which training methods does it have recipes for? Bullet list.")
show(r2, t0)
print(f"\nusage turn 2: {r2.usage}")
