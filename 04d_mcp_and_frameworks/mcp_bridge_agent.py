"""A minimal MCP client that bridges an MCP server's tools into a Fireworks agent.

1. spawn the server as a subprocess (stdio transport)
2. initialize handshake -> tools/list
3. turn each MCP tool into a Fireworks function tool (inputSchema == parameters)
4. run the agent; each tool call becomes an MCP tools/call
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import itertools
import json
import subprocess

from fwlearn.agent import Agent, Tool

SERVER = pathlib.Path(__file__).with_name("mini_mcp_server.py")


class StdioMCPClient:
    def __init__(self, command: list[str]):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        self._ids = itertools.count(1)

    def _send(self, msg: dict):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        rid = next(self._ids)
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        while True:  # read until the response with our id arrives
            msg = json.loads(self.proc.stdout.readline())
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"MCP error {msg['error']}")
                return msg["result"]

    def initialize(self):
        info = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "fireworks-course-bridge", "version": "0.1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return info

    def as_fireworks_tools(self) -> list[Tool]:
        tools = []
        for t in self.request("tools/list")["tools"]:
            def call(_name=t["name"], **arguments):
                res = self.request("tools/call", {"name": _name, "arguments": arguments})
                text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
                if res.get("isError"):
                    raise RuntimeError(text)  # the agent runtime reports this to the model as data
                return text
            tools.append(Tool.from_schema(t["name"], t.get("description", ""), t["inputSchema"], call))
        return tools

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=5)


if __name__ == "__main__":
    mcp = StdioMCPClient([sys.executable, str(SERVER)])
    info = mcp.initialize()
    print("connected to", info["serverInfo"], "protocol", info["protocolVersion"])
    tools = mcp.as_fireworks_tools()
    print("discovered tools:", [t.name for t in tools], "\n")

    def log(kind, d):
        if kind == "tool":
            print(f"  🔧 MCP tools/call {d['name']}({d['arguments']}) ok={d['ok']}")

    agent = Agent(tools=tools, on_event=log,
                  system="You are a Fireworks support engineer. Use the knowledge base and pricing tools.")
    r = agent.run("Our deployment 503s every morning. Why, and what would keeping one H100 warm "
                  "cost for a 30-day month? Also price an A100 month for comparison.")
    print("\n" + r.text)
    mcp.close()
