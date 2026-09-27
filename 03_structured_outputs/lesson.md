# 03 · Structured Outputs

**Goal:** make the model return machine-readable data that is *guaranteed* to parse. Almost every
production integration (extraction, classification, agents, UI generation) depends on this.

## Three levels of control

| `response_format` | Guarantee | Use when |
|---|---|---|
| *(none, just "reply in JSON")* | None. The model *usually* complies. | Never in production |
| `{"type": "json_object"}` | Valid JSON, any shape | Quick prototypes |
| `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}` | Valid JSON **matching your schema** | Production (recommended) |
| `{"type": "grammar", "grammar": "<GBNF>"}` | Output matches an arbitrary grammar | Non-JSON formats: SQL subsets, enums, DSLs |

**How it works: constrained decoding.** At every decode step, Fireworks masks out any token that
would make the output invalid under the schema or grammar. The model *cannot* produce bad JSON. This
is enforced at the sampler, not by retrying.

## Gotchas (from the docs; customers hit all of these)

1. **The model doesn't "see" the schema** just because it's in `response_format`. Enforcement
   happens during generation, so *also* describe the format in the prompt. If you don't, the model
   may emit whitespace until it hits `max_tokens`.
2. **Truncation:** if `finish_reason == "length"`, the JSON is cut off and invalid. Raise `max_tokens`.
3. **Reasoning models:** `json_schema` mode *disables reasoning output*. To keep the thinking, put
   the schema in the prompt and validate afterwards (see `reasoning_plus_json.py`).
4. `pattern` regexes are best-effort (lookarounds ignored). External `$ref`s (http/file) aren't
   supported. In-document `#/$defs/...` refs work.

## Scripts

```powershell
python 03_structured_outputs\pydantic_extraction.py   # json_schema from a Pydantic model
python 03_structured_outputs\json_object_mode.py
python 03_structured_outputs\grammar_mode.py          # GBNF grammar: force one of N labels
python 03_structured_outputs\reasoning_plus_json.py   # keep reasoning, validate JSON yourself
```

## Exercises

1. Build an **invoice extractor**: given messy invoice text, return `vendor`, `date` (ISO),
   `line_items[]` (description, qty, unit_price) and `total`. Add a Pydantic validator checking
   that the line items sum to the total. How often does it fail?
2. Write a GBNF grammar that only allows a SQL `SELECT ... FROM ... WHERE ...` over a fixed list
   of table names.
3. Measure the latency overhead of `json_schema` vs no constraint for the same prompt. (It should
   be small. Knowing that is a selling point.)
