"""See how temperature changes diversity: 3 samples at each setting."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import MODEL, banner, client

fw = client()
prompt = "Invent a name for a coffee shop run by robots. Reply with the name only."

for temp in [0.0, 0.7, 1.3]:
    banner(f"temperature={temp}")
    for _ in range(3):  # n=3 would do this in one request, but not every model supports n > 1
        resp = fw.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=20,
            reasoning_effort="none",
        )
        print(" -", resp.choices[0].message.content.strip())
