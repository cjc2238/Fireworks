"""A small, readable agent runtime used by lessons 04b-04e.

It's deliberately about 200 lines so you can read all of it. Frameworks (lesson 04d) do the same
things with more features. Understanding this file means you can debug any of them.

    from fwlearn.agent import Agent, tool

    @tool
    def gpu_price(gpu: str) -> dict:
        '''Hourly price of a GPU type.'''
        return {...}

    result = Agent(tools=[gpu_price]).run("How much is an H100?")
    print(result.text)
"""

from __future__ import annotations

import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

from . import MODEL, client

MAX_TOOL_RESULT_CHARS = 6000  # keep tool output from flooding the context window


# ----------------------------------------------------------------------------- tools
@dataclass
class Tool:
    """A callable the model may invoke. Build one with @tool, or Tool.from_schema() for
    tools whose schema comes from elsewhere (e.g. an MCP server)."""

    fn: Callable[..., Any]
    name: str
    description: str
    parameters: dict
    args_model: type[BaseModel] | None = None
    requires_approval: bool = False

    @property
    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def run(self, raw_args: str | None) -> Any:
        if self.args_model is not None:  # validate + coerce with pydantic
            args = self.args_model.model_validate_json(raw_args or "{}").model_dump()
        else:
            args = json.loads(raw_args or "{}")
        return self.fn(**args)

    @classmethod
    def from_schema(cls, name: str, description: str, parameters: dict, fn: Callable[..., Any],
                    requires_approval: bool = False) -> "Tool":
        return cls(fn, name, description, parameters, None, requires_approval)


def tool(fn: Callable | None = None, *, name: str | None = None, requires_approval: bool = False):
    """Decorator: a typed Python function becomes a Tool.
    - the docstring is the description (the model reads it!)
    - type hints become the JSON Schema; use Annotated[str, Field(description=...)] for per-arg docs
    """
    def wrap(f: Callable) -> Tool:
        hints = get_type_hints(f, include_extras=True)
        fields = {
            p.name: (hints.get(p.name, str), ... if p.default is inspect.Parameter.empty else p.default)
            for p in inspect.signature(f).parameters.values()
        }
        model = create_model(f"{f.__name__}_args", **fields)
        params = model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        return Tool(f, name or f.__name__, inspect.getdoc(f) or "", params, model, requires_approval)

    return wrap(fn) if fn else wrap


# ----------------------------------------------------------------------------- results
@dataclass
class AgentResult:
    text: str
    messages: list[dict]
    steps: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    stopped: str  # "final" | "max_steps" | "budget" | "loop"
    seconds: float


def _to_dict(msg) -> dict:
    """SDK message object -> plain dict. Keeps reasoning_content for interleaved thinking."""
    d = msg.model_dump(exclude_none=True)
    d.pop("function_call", None)
    return d


# ----------------------------------------------------------------------------- agent
@dataclass
class Agent:
    tools: list[Tool] = field(default_factory=list)
    system: str = "You are a precise assistant. Use tools for facts and actions; never invent tool results."
    model: str = MODEL
    max_steps: int = 10
    temperature: float = 0.1
    reasoning_effort: str | None = "none"   # None = model default (may think before each step)
    token_budget: int | None = None         # stop if prompt+completion tokens exceed this
    approve: Callable[[str, dict], bool] | None = None   # human-in-the-loop hook
    on_event: Callable[[str, dict], None] | None = None  # tracing hook: (kind, data)
    extra: dict = field(default_factory=dict)            # extra create() kwargs

    def __post_init__(self):
        self._by_name = {t.name: t for t in self.tools}
        self._fw = client(max_retries=3, timeout=300)

    # -- helpers -------------------------------------------------------------
    def _emit(self, kind: str, **data):
        if self.on_event:
            self.on_event(kind, data)

    def _call_tool(self, tc) -> dict:
        name, raw = tc.function.name, tc.function.arguments
        t0 = time.perf_counter()
        t = self._by_name.get(name)
        try:
            if t is None:
                raise KeyError(f"unknown tool {name!r}; available: {sorted(self._by_name)}")
            if t.requires_approval:
                args = json.loads(raw or "{}")
                if not (self.approve and self.approve(name, args)):
                    raise PermissionError("a human reviewer DENIED this action; do not retry it")
            out = {"ok": True, "result": t.run(raw)}
        except ValidationError as e:  # tell the model exactly what was wrong so it can fix it
            out = {"ok": False, "error": "invalid arguments", "details": e.errors(include_url=False)}
        except Exception as e:
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        content = json.dumps(out, default=str)
        if len(content) > MAX_TOOL_RESULT_CHARS:
            content = content[:MAX_TOOL_RESULT_CHARS] + " ...[truncated]"
        self._emit("tool", name=name, arguments=raw, ok=out["ok"], output=content[:500],
                   seconds=round(time.perf_counter() - t0, 3))
        return {"role": "tool", "tool_call_id": tc.id, "content": content}

    # -- main loop -------------------------------------------------------------
    def run(self, user_input: str, history: list[dict] | None = None) -> AgentResult:
        t_start = time.perf_counter()
        messages = history[:] if history else [{"role": "system", "content": self.system}]
        messages.append({"role": "user", "content": user_input})
        schemas = [t.schema for t in self.tools]
        kwargs = dict(self.extra)
        if schemas:  # omit the field entirely when there are no tools; tools=None is a 400
            kwargs["tools"] = schemas
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        p_tok = c_tok = n_calls = 0
        seen: dict[str, int] = {}

        def done(text, stopped, step):
            self._emit("end", stopped=stopped, steps=step, tool_calls=n_calls,
                       prompt_tokens=p_tok, completion_tokens=c_tok)
            return AgentResult(text, messages, step, n_calls, p_tok, c_tok, stopped,
                               round(time.perf_counter() - t_start, 2))

        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            resp = self._fw.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature, **kwargs)
            msg = resp.choices[0].message
            p_tok += resp.usage.prompt_tokens
            c_tok += resp.usage.completion_tokens
            messages.append(_to_dict(msg))
            self._emit("llm", step=step, seconds=round(time.perf_counter() - t0, 3),
                       prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens,
                       finish_reason=resp.choices[0].finish_reason,
                       tool_calls=[(tc.function.name, tc.function.arguments) for tc in msg.tool_calls or []],
                       content=(msg.content or "")[:300])

            if not msg.tool_calls:
                return done(msg.content or "", "final", step)

            # loop detection: the exact same call 3 times means the agent is stuck
            for tc in msg.tool_calls:
                key = f"{tc.function.name}:{tc.function.arguments}"
                seen[key] = seen.get(key, 0) + 1
                if seen[key] >= 3:
                    return done(f"Stopped: repeated identical call {key}", "loop", step)

            n_calls += len(msg.tool_calls)
            with ThreadPoolExecutor(max_workers=8) as pool:  # parallel tool calls
                messages.extend(pool.map(self._call_tool, msg.tool_calls))

            if self.token_budget and p_tok + c_tok > self.token_budget:
                return done(f"Stopped: token budget {self.token_budget} exceeded", "budget", step)

        return done("Stopped: reached max_steps without a final answer.", "max_steps", self.max_steps)
