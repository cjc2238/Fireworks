"""Pure reward function for the triage task. No API calls, so it's fully unit-testable.

Score = weighted field accuracy, with penalties for format violations (anti-reward-hacking).
"""

import re

LINE_RE = re.compile(r"^team=([a-z]+); sev=(S[1-4]); tag=([a-z0-9]+(?:-[a-z0-9]+)*)$")
VALID_TEAMS = {"inference", "training", "billing", "accounts", "docs"}
WEIGHTS = {"team": 0.4, "sev": 0.4, "tag": 0.2}


def parse_strict(text: str):
    m = LINE_RE.match((text or "").strip())
    if not m:
        return None
    team, sev, tag = m.groups()
    return {"team": team, "sev": sev, "tag": tag} if team in VALID_TEAMS else None


def score_triage(prediction: str, ground_truth: str) -> tuple[float, str]:
    gold = parse_strict(ground_truth)
    if gold is None:
        return 0.0, "invalid ground truth"
    if prediction and len(prediction) > 200:
        return 0.0, "too long: likely rambling or hacking"
    pred = parse_strict(prediction)
    if pred is None:
        return 0.0, f"format violation: {prediction[:80]!r}"
    score = sum(w for f, w in WEIGHTS.items() if pred[f] == gold[f])
    wrong = [f for f in WEIGHTS if pred[f] != gold[f]]
    return round(score, 3), "all correct" if not wrong else f"wrong: {', '.join(wrong)}"
