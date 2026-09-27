"""Every tool_choice mode against a question that NEEDS the tool and one that DOESN'T.
Watch for invented arguments when a tool is forced on an unrelated question.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import MODEL, client

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

MODES = {
    "auto": "auto",
    "none": "none",
    "required": "required",
    "forced": {"type": "function", "function": {"name": "get_weather"}},
}
QUESTIONS = {"relevant": "What's the weather in Oslo?", "irrelevant": "Tell me a one-line joke."}

fw = client()
print(f"{'mode':<10} {'question':<11} {'result'}")
for mode, choice in MODES.items():
    for qname, q in QUESTIONS.items():
        r = fw.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": q}],
                                       tools=TOOLS, tool_choice=choice, temperature=0,
                                       reasoning_effort="none", max_tokens=200)
        m = r.choices[0].message
        if m.tool_calls:
            res = "TOOL " + ", ".join(f"{t.function.name}({t.function.arguments})" for t in m.tool_calls)
            if qname == "irrelevant":
                res += "   <-- invented arguments!"
        else:
            res = "TEXT " + (m.content or "").strip().replace("\n", " ")[:60]
        print(f"{mode:<10} {qname:<11} {res}")
