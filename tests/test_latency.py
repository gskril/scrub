"""Latency budget test (PLAN.md Sec 5 "Performance plan" / Sec 9
"Verification"): a warm cache-hit round trip must stay under 50ms and a cold
small-text-file redact under 500ms, against a REAL daemon subprocess (same
spawn pattern as tests/test_daemon.py: SCRUB_SOCKET/SCRUB_CACHE_DIR pointed
at tmp dirs, the Rampart model already in the local HF cache so warmup hits
disk, not the network).

Marked `slow` for anyone who wants to filter it out later, but nothing in
this repo's pytest config excludes that marker today, so it still runs in
the default `pytest tests/` suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

import scrub.client as client

_START_TIMEOUT = 60.0
_WARM_BUDGET_MS = 50.0
_COLD_BUDGET_MS = 500.0


def _wait_up(timeout: float = _START_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            resp = client.ping()
            if resp.get("ok"):
                return
        except OSError as e:
            last = e
        time.sleep(0.1)
    raise AssertionError(f"daemon did not come up in {timeout}s (last: {last})")


@contextmanager
def _daemon(base: Path):
    """Spawn a daemon with env pointed at `base`, restoring env afterwards
    (mirrors tests/test_daemon.py's `_daemon` helper)."""
    sock = base / "scrubd.sock"
    cache = base / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    saved = {k: os.environ.get(k) for k in ("SCRUB_SOCKET", "SCRUB_CACHE_DIR")}
    os.environ["SCRUB_SOCKET"] = str(sock)
    os.environ["SCRUB_CACHE_DIR"] = str(cache)
    logf = open(cache / "daemon.out", "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "scrub.daemon"],
        stdin=subprocess.DEVNULL,
        stdout=logf,
        stderr=logf,
    )
    try:
        _wait_up()
        yield {"proc": proc, "sock": sock, "cache": cache}
    finally:
        try:
            client.shutdown()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        logf.close()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    base = tmp_path_factory.mktemp("latency-live")
    with _daemon(base) as d:
        yield d


@pytest.mark.slow
def test_warm_cache_hit_round_trip_under_budget(live, tmp_path):
    """Second (and later) redacts of an unchanged file must be a pure
    content-hash cache hit: no extraction, no detection, just a socket
    round trip (PLAN.md Sec 5: "second read is a cache hit -> instant path
    return")."""
    src = tmp_path / "warm.txt"
    src.write_text(
        "Call Ravi Patel at (206) 555-0142 about the invoice.\n"
        "His SSN is 734-22-1156 and email ravi.patel@example.com.\n"
    )

    cold = client.redact(None, src)
    assert cold["ok"] is True
    assert cold["cache_hit"] is False  # sanity: this really was the first hit
    assert cold["found"] > 0

    # A couple of untimed warm round trips first, so the measurement isn't
    # skewed by first-call-on-this-connection jitter.
    for _ in range(3):
        warm_up = client.redact(None, src)
        assert warm_up["cache_hit"] is True

    samples_ms = []
    for _ in range(10):
        t0 = time.perf_counter()
        resp = client.redact(None, src)
        samples_ms.append((time.perf_counter() - t0) * 1000)
        assert resp["ok"] is True
        assert resp["cache_hit"] is True

    samples_ms.sort()
    median_ms = samples_ms[len(samples_ms) // 2]
    print(
        f"\n[latency] warm cache-hit round trip: median={median_ms:.2f}ms "
        f"min={samples_ms[0]:.2f}ms max={samples_ms[-1]:.2f}ms "
        f"(budget {_WARM_BUDGET_MS:.0f}ms)"
    )
    assert median_ms < _WARM_BUDGET_MS, (
        f"warm cache-hit round trip median {median_ms:.2f}ms exceeds the "
        f"{_WARM_BUDGET_MS:.0f}ms budget (PLAN.md Sec 5: 'sub-100ms added "
        "latency on a warm cache')"
    )


@pytest.mark.slow
def test_cold_small_text_file_under_budget(live, tmp_path):
    """A brand-new small text file (never seen by this daemon before) must
    extract + detect + redact + write within budget. The daemon holds the
    ONNX model resident (PLAN.md Sec 5), so this pays no warmup cost -- that
    already happened once, at daemon startup."""
    samples_ms = []
    for i in range(5):
        src = tmp_path / f"cold_{i}.txt"
        src.write_text(
            f"Contact Sofia Novak at (480) 555-0163 or sofia.novak.{i}@example.com "
            f"about invoice #{1000 + i}. Her SSN is 219-08-5567.\n"
        )
        t0 = time.perf_counter()
        resp = client.redact(None, src)
        samples_ms.append((time.perf_counter() - t0) * 1000)
        assert resp["ok"] is True
        assert resp["cache_hit"] is False
        assert resp["found"] > 0

    samples_ms.sort()
    median_ms = samples_ms[len(samples_ms) // 2]
    print(
        f"\n[latency] cold small-text-file redact: median={median_ms:.2f}ms "
        f"min={samples_ms[0]:.2f}ms max={samples_ms[-1]:.2f}ms "
        f"(budget {_COLD_BUDGET_MS:.0f}ms)"
    )
    assert median_ms < _COLD_BUDGET_MS, (
        f"cold small-text-file redact median {median_ms:.2f}ms exceeds the "
        f"{_COLD_BUDGET_MS:.0f}ms budget (PLAN.md Sec 5: 'low hundreds of ms "
        "on a cold text file')"
    )
