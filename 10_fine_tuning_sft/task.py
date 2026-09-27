"""The task we fine-tune for: support-ticket triage into a company-specific schema.

A small model with a short prompt tends to get the house rules wrong. After SFT it should
follow them without the long prompt.
"""

SYSTEM_SHORT = "Triage the ticket. Reply as: team=<team>; sev=<S1-S4>; tag=<tag>"

# The "house rules" the teacher sees; the fine-tuned student must learn them implicitly.
SYSTEM_TEACHER = SYSTEM_SHORT + """
House rules:
- team is one of: inference, training, billing, accounts, docs
- sev: S1 = production down / data loss; S2 = major degradation; S3 = question or minor bug; S4 = feature request
- tag is ONE lowercase kebab-case word describing the specific issue, e.g. rate-limit, cold-start,
  lora-deploy, invoice, sso, schema-error, timeout, dataset-upload
- Anything mentioning 503 or 'scaling up' is team=inference, tag=cold-start
- Refunds and invoices are ALWAYS S3 unless the account is suspended (then S2)
Output exactly one line, no extra words."""

TOPICS = [
    "429 rate limit errors", "503 on a deployment after idle", "fine-tuning job failed",
    "LoRA won't load onto deployment", "double charged on invoice", "account suspended with credits left",
    "SSO login broken", "JSON schema mode returns invalid output", "request timeouts on long prompts",
    "dataset upload rejected", "docs page for batch API is outdated", "wants a Rust SDK",
    "latency spiked in EU region", "RFT evaluator crashing", "API key leaked, need rotation",
]


def parse(line: str) -> dict:
    out = {}
    for part in line.strip().split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out
