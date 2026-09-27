"""Thin REST helpers for the Fireworks control plane (datasets, jobs, deployments).

These use plain `requests` so you can see every endpoint. The Fireworks SDK also exposes
client.datasets, client.batch_inference_jobs, client.deployments, client.models, and
`firectl` wraps the same API. Knowing the raw REST shape helps you debug all three.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from . import CONTROL_URL, auth_headers, require_account_id


def _url(path: str) -> str:
    return f"{CONTROL_URL}/accounts/{require_account_id()}/{path}"


def _check(r: requests.Response) -> dict:
    if not r.ok:
        raise RuntimeError(f"{r.request.method} {r.url} -> {r.status_code}: {r.text[:500]}")
    return r.json() if r.text else {}


def get(path: str, **params) -> dict:
    return _check(requests.get(_url(path), headers=auth_headers(), params=params, timeout=60))


def post(path: str, body: dict | None = None, **params) -> dict:
    return _check(requests.post(_url(path), headers=auth_headers(), json=body or {}, params=params, timeout=120))


def delete(path: str) -> dict:
    return _check(requests.delete(_url(path), headers=auth_headers(), timeout=60))


# ------------------------------------------------------------------ datasets
def create_dataset(dataset_id: str, jsonl_path: str | Path) -> str:
    """Create a dataset entry and upload a JSONL file (<150MB; use firectl for bigger files)."""
    jsonl_path = Path(jsonl_path)
    n = sum(1 for line in jsonl_path.open(encoding="utf-8") if line.strip())
    post("datasets", {"datasetId": dataset_id, "dataset": {"userUploaded": {}, "exampleCount": str(n)}})
    with jsonl_path.open("rb") as f:
        headers = {k: v for k, v in auth_headers().items() if k != "Content-Type"}
        _check(requests.post(_url(f"datasets/{dataset_id}:upload"), headers=headers,
                             files={"file": (jsonl_path.name, f)}, timeout=600))
    return wait_for(f"datasets/{dataset_id}", ready_states={"READY", "STATE_READY"}, state_key="state", poll=3)["name"]


def download_dataset(dataset_id: str, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = get(f"datasets/{dataset_id}:getDownloadEndpoint").get("filenameToSignedUrls", {})
    paths = []
    for fname, signed in urls.items():
        p = out_dir / Path(fname).name
        p.write_bytes(requests.get(signed, timeout=600).content)
        paths.append(p)
    return paths


# ------------------------------------------------------------------ polling
TERMINAL_BAD = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "FAILED"}


def wait_for(path: str, ready_states: set[str], state_key: str = "state", poll: int = 20,
             timeout_s: int = 6 * 3600) -> dict:
    """Poll a resource until its state is in ready_states. Prints each state change."""
    t0, last = time.time(), None
    while True:
        res = get(path)
        state = res.get(state_key)
        if state != last:
            print(f"  [{time.strftime('%H:%M:%S')}] {path}: {state}")
            last = state
        if state in ready_states:
            return res
        if state in TERMINAL_BAD:
            raise RuntimeError(f"{path} ended in {state}: {json.dumps(res.get('status', res))[:500]}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(path)
        time.sleep(poll)
