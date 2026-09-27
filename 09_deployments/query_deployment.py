"""Query your dedicated deployment and handle cold starts (503 DEPLOYMENT_SCALING_UP).

Usage:  python 09_deployments/query_deployment.py <deployment-id>
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import time

import fireworks

from fwlearn import client, require_account_id

if len(sys.argv) < 2:
    sys.exit(__doc__)
model = f"accounts/{require_account_id()}/deployments/{sys.argv[1]}"
fw = client(max_retries=0, timeout=120)

t0 = time.perf_counter()
for attempt in range(40):  # cold starts can take a few minutes
    try:
        r = fw.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Say hi from a dedicated GPU."}], max_tokens=30)
        print(f"[{time.perf_counter() - t0:.0f}s] {r.choices[0].message.content}")
        break
    except fireworks.APIStatusError as e:
        if e.status_code == 503:
            print(f"[{time.perf_counter() - t0:.0f}s] 503, deployment scaling up... retrying in 15s")
            time.sleep(15)
        else:
            raise
