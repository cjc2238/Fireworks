"""A ReAct research agent over this course's lesson files: it searches, reads sections, and
answers with citations. This is agentic RAG: the model decides what to retrieve, and when to stop.

Usage:  python 04c_building_agents/research_agent.py "your question"
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import re
from typing import Annotated

from pydantic import Field

from fwlearn import ROOT
from fwlearn.agent import Agent, tool

LESSONS = {p.parent.name: p for p in sorted(ROOT.glob("*/lesson.md"))}


def _sections(text: str) -> dict[str, str]:
    parts = re.split(r"^(#{1,3} .+)$", text, flags=re.M)
    out, title = {}, "intro"
    for chunk in parts:
        if re.match(r"^#{1,3} ", chunk):
            title = chunk.lstrip("#").strip()
        elif chunk.strip():
            out[title] = out.get(title, "") + chunk
    return out


@tool
def list_lessons() -> list[str]:
    """List all lesson ids in the course (e.g. '09_deployments'). Use to orient yourself."""
    return list(LESSONS)


@tool
def search_course(query: Annotated[str, Field(description="2-5 keywords, e.g. 'lora deploy multi'")]) -> list[dict]:
    """Keyword-search every lesson section. Returns the best matching (lesson, section) pairs with a snippet."""
    terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    hits = []
    for lid, path in LESSONS.items():
        for title, body in _sections(path.read_text(encoding="utf-8")).items():
            low = (title + " " + body).lower()
            score = sum(low.count(t) for t in terms)
            if score:
                hits.append({"lesson": lid, "section": title, "score": score,
                             "snippet": " ".join(body.split())[:160]})
    return sorted(hits, key=lambda h: -h["score"])[:6]


@tool
def read_section(lesson: str, section: str) -> str:
    """Read one section of a lesson in full. Use the exact lesson id and section title from search results."""
    secs = _sections(LESSONS[lesson].read_text(encoding="utf-8"))
    if section not in secs:
        raise KeyError(f"no section {section!r}; sections are: {list(secs)}")
    return secs[section][:4000]


SYSTEM = """You are a research assistant for a Fireworks AI course.
Process: search -> read the 1-3 most relevant sections -> answer.
Answer ONLY from what you read. Cite every claim as [lesson > section].
If the course doesn't cover it, say so."""


def log(kind, d):
    if kind == "tool":
        print(f"  🔧 {d['name']}({d['arguments']})  ok={d['ok']}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How do I deploy a LoRA, and how does multi-LoRA differ from live merge?"
    print("Q:", q, "\n")
    r = Agent(tools=[list_lessons, search_course, read_section], system=SYSTEM, on_event=log).run(q)
    print("\n" + r.text)
    print(f"\n[{r.steps} steps, {r.tool_calls} tool calls, {r.prompt_tokens} prompt tokens, {r.seconds}s]")
