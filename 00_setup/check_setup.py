"""Verify your API key works, discover your account ID, and make one inference call."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import requests

from fwlearn import CONTROL_URL, MODEL, auth_headers, banner, client

banner("1. Control plane: which accounts can this key see?")
r = requests.get(f"{CONTROL_URL}/accounts", headers=auth_headers(), timeout=30)
if r.status_code == 401:
    sys.exit("401 Unauthorized: the API key is wrong or revoked.")
r.raise_for_status()
for acct in r.json().get("accounts", []):
    account_id = acct["name"].split("/")[-1]
    print(f"  account_id = {account_id}   ({acct.get('displayName', '')})")
print("  -> put this in .env as FIREWORKS_ACCOUNT_ID")

banner(f"2. Data plane: one chat completion with {MODEL}")
resp = client().chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Reply with exactly: Fireworks is ready."}],
    max_tokens=20,
    reasoning_effort="none",  # skip hidden thinking: fast, short answers
)
print(" ", resp.choices[0].message.content)
print("\nAll good. On to lesson 01.")
