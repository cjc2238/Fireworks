"""A production-style wrapper: timeouts, retry with exponential backoff + jitter, and model fallback."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import logging
import random
import time

import fireworks

from fwlearn import BIG_MODEL, MODEL, client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("resilient")

RETRYABLE = {429, 500, 502, 503, 504}

# Disable SDK retries so we control the policy ourselves.
fw = client(max_retries=0, timeout=60.0)


def complete(messages, models=(MODEL, BIG_MODEL), max_attempts=4, base_delay=1.0, **kw):
    """Try each model in order; retry transient errors with backoff before falling back."""
    last_err = None
    for model in models:
        for attempt in range(max_attempts):
            try:
                t0 = time.perf_counter()
                r = fw.chat.completions.create(model=model, messages=messages, **kw)
                log.info("ok model=%s attempt=%d latency=%.2fs", model, attempt + 1, time.perf_counter() - t0)
                return r
            except fireworks.APIStatusError as e:
                last_err = e
                if e.status_code not in RETRYABLE:
                    log.error("non-retryable %s on %s: %s", e.status_code, model, e.message)
                    raise
                # honor Retry-After when the server sends one
                retry_after = e.response.headers.get("retry-after")
                delay = float(retry_after) if retry_after else base_delay * 2 ** attempt + random.uniform(0, 1)
                log.warning("status=%s model=%s retry in %.1fs", e.status_code, model, delay)
                time.sleep(delay)
            except (fireworks.APIConnectionError, fireworks.APITimeoutError) as e:
                last_err = e
                delay = base_delay * 2 ** attempt + random.uniform(0, 1)
                log.warning("connection/timeout on %s, retry in %.1fs", model, delay)
                time.sleep(delay)
        log.warning("giving up on %s, falling back", model)
    raise RuntimeError(f"all models failed; last error: {last_err}")


if __name__ == "__main__":
    r = complete([{"role": "user", "content": "Give one tip for handling 429 errors."}], max_tokens=100)
    print(r.choices[0].message.content)

    print("\nNow a non-retryable error (bad model id):")
    try:
        complete([{"role": "user", "content": "hi"}], models=("accounts/fireworks/models/does-not-exist",))
    except fireworks.APIStatusError as e:
        print(f"  -> {e.status_code}: raised immediately, no retries wasted")
