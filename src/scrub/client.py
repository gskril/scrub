"""Thin Unix-socket client for the scrub daemon, used by the CLI and hook.

One JSON line out, one JSON line back, then the connection closes. Connect
timeouts are short (the daemon is local); the redact response timeout is
generous because a big PDF can take a while. `ensure_daemon` spawns a detached
daemon if none is answering and retries ping with exponential backoff, allowing
for the ~1s Rampart warmup at startup.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .config import Config, cache_dir, socket_path

_CONNECT_TIMEOUT = 0.5
_REDACT_TIMEOUT = 30.0
# Total time to wait for a freshly-spawned daemon to answer ping. Cold start
# imports onnxruntime and warms the model (~1-2s), so this is deliberately
# generous — a too-short deadline turns a slow start into a fail-closed deny.
_SPAWN_DEADLINE = float(os.environ.get("SCRUB_SPAWN_TIMEOUT", "10.0"))


class DaemonError(Exception):
    """The daemon could not be reached or returned no usable response."""


def _request(obj: dict, read_timeout: float) -> dict:
    sock_path = socket_path()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(_CONNECT_TIMEOUT)
    try:
        s.connect(str(sock_path))
        s.settimeout(read_timeout)
        s.sendall(json.dumps(obj).encode("utf-8") + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    line = buf.partition(b"\n")[0]
    if not line:
        raise DaemonError("empty response from daemon")
    return json.loads(line.decode("utf-8"))


def ping(config: Config | None = None) -> dict:
    return _request({"op": "ping"}, read_timeout=_CONNECT_TIMEOUT)


def redact(config: Config | None, path: Path) -> dict:
    return _request({"op": "redact", "path": str(path)}, read_timeout=_REDACT_TIMEOUT)


def redact_text_request(config: Config | None, text: str) -> dict:
    return _request({"op": "redact_text", "text": text}, read_timeout=_REDACT_TIMEOUT)


def shutdown(config: Config | None = None) -> dict:
    return _request({"op": "shutdown"}, read_timeout=_CONNECT_TIMEOUT)


def _spawn_daemon() -> None:
    cdir = cache_dir()
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        log = open(cdir / "scrubd.log", "ab")  # noqa: SIM115 (handed to child)
    except OSError:
        log = subprocess.DEVNULL
    subprocess.Popen(
        [sys.executable, "-m", "scrub.daemon"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
    )


def ensure_daemon(config: Config | None = None) -> dict:
    """Ensure a daemon is answering. Returns the ping response. Raises
    DaemonError if none comes up within the spawn deadline."""
    try:
        resp = ping(config)
        if resp.get("ok"):
            return resp
    except OSError:
        pass

    _spawn_daemon()

    deadline = time.monotonic() + _SPAWN_DEADLINE
    delay = 0.05
    while time.monotonic() < deadline:
        time.sleep(delay)
        try:
            resp = ping(config)
            if resp.get("ok"):
                return resp
        except OSError:
            pass
        delay = min(delay * 1.6, 0.5)
    raise DaemonError("scrub daemon did not come up in time")
