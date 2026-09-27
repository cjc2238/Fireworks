"""Track cost per request/feature from usage data.

PRICES below are PLACEHOLDERS. Fill them in from https://fireworks.ai/pricing or the model page.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from collections import defaultdict
from dataclasses import dataclass, field

from fwlearn import MODEL, client

# $ per 1M tokens: (input, cached_input, output). PLACEHOLDER VALUES, update them!
PRICES = {MODEL: (0.50, 0.25, 2.00)}


@dataclass
class CostTracker:
    totals: dict = field(default_factory=lambda: defaultdict(float))

    def record(self, feature: str, model: str, resp, cached_tokens: int = 0):
        p_in, p_cached, p_out = PRICES.get(model, (0, 0, 0))
        u = resp.usage
        uncached = u.prompt_tokens - cached_tokens
        cost = (uncached * p_in + cached_tokens * p_cached + u.completion_tokens * p_out) / 1e6
        self.totals[feature] += cost
        return cost

    def report(self):
        for feat, c in sorted(self.totals.items(), key=lambda x: -x[1]):
            print(f"  {feat:<20} ${c:.6f}")
        print(f"  {'TOTAL':<20} ${sum(self.totals.values()):.6f}")


if __name__ == "__main__":
    fw, tracker = client(), CostTracker()
    for feature, prompt in [
        ("summarize", "Summarize the plot of Hamlet in 3 sentences."),
        ("classify", "Is 'I love this!' positive or negative? One word."),
        ("summarize", "Summarize the plot of Macbeth in 3 sentences."),
    ]:
        raw = fw.chat.completions.with_raw_response.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=150)
        cached = int(raw.headers.get("fireworks-cached-prompt-tokens", 0) or 0)
        cost = tracker.record(feature, MODEL, raw.parse(), cached)
        print(f"{feature:<10} ${cost:.6f}")
    print("\nBy feature:")
    tracker.report()
