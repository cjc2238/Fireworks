"""Shared helpers used by every lesson script.

Every script in this repo starts with:

    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from fwlearn import client, MODEL

so you can run any file directly with `python <folder>/<script>.py` from the repo root.

Model IDs change as Fireworks adds new models. Instead of editing every script,
override the defaults in your .env file (see .env.example) and run
`python 00_setup/list_models.py` to see what is currently available on serverless.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Windows consoles default to cp1252; model output is full Unicode.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("FIREWORKS_API_KEY")
if not API_KEY:
    sys.exit(
        "FIREWORKS_API_KEY is not set.\n"
        "1. Create a key at https://app.fireworks.ai/settings/users/api-keys\n"
        "2. Copy .env.example to .env and paste the key in."
    )

# --- Endpoints -------------------------------------------------------------
# Data plane: inference (chat, completions, embeddings, rerank). OpenAI-compatible.
INFERENCE_URL = "https://api.fireworks.ai/inference/v1"
# Control plane: accounts, models, deployments, datasets, jobs.
CONTROL_URL = "https://api.fireworks.ai/v1"

# --- Default models (override any of these in .env) ------------------------
MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/deepseek-v4p1-flash")
BIG_MODEL = os.getenv("FIREWORKS_BIG_MODEL", "accounts/fireworks/models/kimi-k3")
VISION_MODEL = os.getenv("FIREWORKS_VISION_MODEL", "accounts/fireworks/models/deepseek-v4p1-flash")
EMBED_MODEL = os.getenv("FIREWORKS_EMBED_MODEL", "fireworks/qwen3-embedding-8b")
RERANK_MODEL = os.getenv("FIREWORKS_RERANK_MODEL", "fireworks/qwen3-reranker-8b")
ACCOUNT_ID = os.getenv("FIREWORKS_ACCOUNT_ID", "")


def client(**kwargs):
    """Synchronous Fireworks SDK client (reads FIREWORKS_API_KEY from env)."""
    from fireworks import Fireworks

    return Fireworks(**kwargs)


def async_client(**kwargs):
    """Async Fireworks SDK client, for concurrent requests."""
    from fireworks import AsyncFireworks

    return AsyncFireworks(**kwargs)


def openai_client(**kwargs):
    """The official OpenAI SDK pointed at Fireworks. Proves the API is drop-in compatible."""
    from openai import OpenAI

    return OpenAI(api_key=API_KEY, base_url=INFERENCE_URL, **kwargs)


def auth_headers() -> dict[str, str]:
    """Headers for raw HTTP calls with `requests`."""
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def require_account_id() -> str:
    """Control-plane calls need your account ID. Run 00_setup/check_setup.py to find it."""
    if not ACCOUNT_ID:
        sys.exit(
            "FIREWORKS_ACCOUNT_ID is not set. Run `python 00_setup/check_setup.py` "
            "to discover it, then add it to .env."
        )
    return ACCOUNT_ID


def print_usage(response) -> None:
    """Pretty-print token usage from a chat completion."""
    u = getattr(response, "usage", None)
    if u:
        print(f"\n[usage] prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
