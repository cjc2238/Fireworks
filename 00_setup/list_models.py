"""List every model in the public Fireworks library that is currently live on serverless.

Uses the control-plane REST API: GET /v1/accounts/fireworks/models
Usage:  python 00_setup/list_models.py [search-term]
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import requests

from fwlearn import CONTROL_URL, auth_headers

search = sys.argv[1].lower() if len(sys.argv) > 1 else ""
url = f"{CONTROL_URL}/accounts/fireworks/models"


def fetch(params):
    models, token = [], None
    while True:
        if token:
            params["pageToken"] = token
        r = requests.get(url, headers=auth_headers(), params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        models += data.get("models", [])
        token = data.get("nextPageToken")
        if not token:
            return models


# Server-side filter (AIP-160 syntax). If the server rejects it, filter client-side instead.
try:
    models = fetch({"pageSize": 200, "filter": "supports_serverless=true"})
except requests.HTTPError:
    models = [m for m in fetch({"pageSize": 200}) if m.get("supportsServerless")]

rows = sorted(
    (m["name"], m.get("contextLength") or 0, m.get("kind", ""))
    for m in models
    if search in m["name"].lower()
)
print(f"{'MODEL ID':<70} {'CONTEXT':>8}  KIND")
for name, ctx, kind in rows:
    print(f"{name:<70} {ctx:>8}  {kind}")
print(f"\n{len(rows)} serverless models" + (f" matching '{search}'" if search else ""))
