"""Integration tests for the scrub daemon (Phase 4).

Each test runs a real daemon in a subprocess with SCRUB_SOCKET + SCRUB_CACHE_DIR
pointed at tmp dirs, and drives it through the thin client. The Rampart model is
already cached locally, so warmup hits disk, not the network.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

import scrub.client as client

PII_TEXT = (
    "Call Maria Garcia at (312) 555-0148 about the loan.\n"
    "Her SSN is 458-02-6841 and email maria.garcia@example.com.\n"
)

_START_TIMEOUT = 60.0


def _wait_up(timeout: float = _START_TIMEOUT) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            resp = client.ping()
            if resp.get("ok"):
                return resp
        except OSError as e:
            last = e
        time.sleep(0.1)
    raise AssertionError(f"daemon did not come up in {timeout}s (last: {last})")


@contextmanager
def _daemon(base: Path, *, wait: bool = True):
    """Spawn a daemon with env pointed at `base`, restoring env afterwards."""
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
        if wait:
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
    base = tmp_path_factory.mktemp("daemon-live")
    with _daemon(base) as d:
        yield d


def test_ping(live):
    resp = client.ping()
    assert resp["ok"] is True
    assert resp["pong"] is True
    assert resp["pid"] == live["proc"].pid


def test_redact_cold_then_warm(live, tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text(PII_TEXT)

    t0 = time.perf_counter()
    cold = client.redact(None, src)
    cold_ms = (time.perf_counter() - t0) * 1000

    assert cold["ok"] is True
    assert cold["cache_hit"] is False
    assert cold["found"] > 0
    redacted = Path(cold["redacted_path"])
    assert redacted.is_file()
    body = redacted.read_text()
    assert "458-02-6841" not in body
    assert "Maria" not in body
    assert "[" in body and "]" in body  # placeholders present

    t0 = time.perf_counter()
    warm = client.redact(None, src)
    warm_ms = (time.perf_counter() - t0) * 1000

    assert warm["ok"] is True
    assert warm["cache_hit"] is True
    assert warm["found"] == cold["found"]
    assert warm["redacted_path"] == cold["redacted_path"]

    print(f"\n[timing] cold redact = {cold_ms:.1f} ms, warm (cache hit) = {warm_ms:.1f} ms")
    # Warm path skips extraction + inference entirely.
    assert warm_ms < cold_ms


def test_unknown_op_errors(live):
    # Talk to the socket directly to send an unsupported op.
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(os.environ["SCRUB_SOCKET"])
    s.sendall(b'{"op":"frobnicate"}\n')
    line = b""
    while b"\n" not in line:
        line += s.recv(4096)
    s.close()
    import json

    resp = json.loads(line)
    assert resp["ok"] is False
    assert "error" in resp


def test_shutdown(tmp_path):
    with _daemon(tmp_path / "shutdown") as d:
        resp = client.shutdown()
        assert resp["ok"] is True
        d["proc"].wait(timeout=10)
        assert d["proc"].poll() is not None
        # Socket + pidfile cleaned up.
        assert not d["sock"].exists()


def test_stale_socket_recovery(tmp_path):
    base = tmp_path / "stale"
    base.mkdir()
    sock = base / "scrubd.sock"
    # Leave a stale socket file behind (bound then closed, no listener).
    leftover = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    leftover.bind(str(sock))
    leftover.close()
    assert sock.exists()  # file lingers with no daemon behind it

    with _daemon(base) as d:  # must detect stale, unlink, bind, come up
        resp = client.ping()
        assert resp["ok"] is True
        assert resp["pid"] == d["proc"].pid
