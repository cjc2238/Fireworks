"""Validate an SFT JSONL file before uploading. Catches the mistakes that make jobs fail or learn nothing.

Usage:  python 10_fine_tuning_sft/validate_dataset.py outputs/sft_train.jsonl
"""

import json
import sys
from collections import Counter

VALID_ROLES = {"system", "user", "assistant", "tool"}

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/sft_train.jsonl"
errors, warnings, seen = [], [], Counter()
lengths, n = [], 0

for i, line in enumerate(open(path, encoding="utf-8"), 1):
    if not line.strip():
        continue
    n += 1
    try:
        row = json.loads(line)
    except json.JSONDecodeError as e:
        errors.append(f"line {i}: invalid JSON ({e})")
        continue
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        errors.append(f"line {i}: missing/empty 'messages'")
        continue
    for m in msgs:
        if m.get("role") not in VALID_ROLES:
            errors.append(f"line {i}: bad role {m.get('role')!r}")
        if "content" not in m and "tool_calls" not in m:
            errors.append(f"line {i}: message without content")
    if not any(m.get("role") == "assistant" and m.get("weight", 1) != 0 for m in msgs):
        errors.append(f"line {i}: no trainable assistant message (nothing to learn)")
    if msgs[-1].get("role") != "assistant":
        warnings.append(f"line {i}: last message isn't assistant")
    key = json.dumps(msgs, sort_keys=True)
    seen[key] += 1
    lengths.append(sum(len(str(m.get("content", ""))) for m in msgs))

dupes = sum(c - 1 for c in seen.values() if c > 1)
print(f"rows={n}  duplicates={dupes}  chars/row: min={min(lengths, default=0)} "
      f"avg={sum(lengths) // max(len(lengths), 1)} max={max(lengths, default=0)} (~4 chars/token)")
if n < 3:
    errors.append("fewer than 3 examples (Fireworks minimum)")
for w in warnings[:10]:
    print("WARN ", w)
for e in errors[:20]:
    print("ERROR", e)
print("\nOK to upload." if not errors else f"\n{len(errors)} error(s): fix before uploading.")
sys.exit(1 if errors else 0)
