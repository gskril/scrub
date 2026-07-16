"""Long-lived scrub daemon.

Holds the Rampart model resident and serves redaction requests over a Unix
domain socket so the hook stays fast. Protocol is newline-delimited JSON
(see ARCHITECTURE.md):

  {"op":"redact","path":"/abs/file"}
      -> {"ok":true,"redacted_path":str|null,"found":N,"cache_hit":bool}
  {"op":"ping"}     -> {"ok":true,"pong":true,"pid":N}
  {"op":"shutdown"} -> {"ok":true} then the daemon exits.
  errors            -> {"ok":false,"error":"..."}

Run as ``python -m scrub.daemon``. Startup: build ``Pipeline.default(config)``,
warm up the Rampart detector (~1s), bind the socket (recovering a stale socket
left by a dead daemon), write a pidfile next to it. A single worker lock
serialises redaction (ONNX ``run()`` is thread-safe, but keeping one lock is
simpler and the model is fast). The daemon exits after
``config.daemon_idle_exit_s`` with no requests, and cleans up socket + pidfile
on SIGTERM/SIGINT.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cache import ResultCache
from .config import Config, cache_dir, socket_path
from .pipeline import Pipeline

_ACCEPT_POLL_S = 1.0  # how often the accept loop wakes to check idle/stop
_MAX_WORKERS = 8


class DaemonAlreadyRunning(Exception):
    """Another daemon already answers on this socket."""


def _pidfile_for(sock_path: Path) -> Path:
    return sock_path.parent / (sock_path.name + ".pid")


def _log_path() -> Path:
    return cache_dir() / "scrubd.log"


class Daemon:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.pipeline = Pipeline.default(config)
        self.cache = ResultCache(config)
        self.sock_path = socket_path()
        self.pid_path = _pidfile_for(self.sock_path)
        self._listener: socket.socket | None = None
        self._lock = threading.Lock()  # serialises redaction
        self._stop = threading.Event()
        self._last_activity = time.monotonic()

    # ------------------------------------------------------------- startup

    def warmup(self) -> None:
        """Load the Rampart ONNX session up front so the first real request is
        fast. Finds the detector by name; no-op if Rampart is disabled."""
        for detector in self.pipeline.detectors:
            if getattr(detector, "name", None) == "rampart":
                warmup = getattr(detector, "warmup", None)
                if callable(warmup):
                    warmup()

    def socket_alive(self) -> bool:
        """True if a live daemon already answers on our socket path."""
        if not self.sock_path.exists():
            return False
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect(str(self.sock_path))
            return True
        except OSError:
            return False
        finally:
            probe.close()

    def bind(self) -> None:
        """Bind the listening socket, recovering a stale socket file left by a
        dead daemon. Raises DaemonAlreadyRunning if a live daemon holds it."""
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.sock_path.exists():
            if self.socket_alive():
                raise DaemonAlreadyRunning(str(self.sock_path))
            # Stale socket from a crashed daemon — safe to remove.
            self.sock_path.unlink()

        old_umask = os.umask(0o177)  # -> socket created 0600
        try:
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(self.sock_path))
            except OSError as e:
                listener.close()
                # Lost a bind race with a simultaneously-spawned daemon.
                raise DaemonAlreadyRunning(str(self.sock_path)) from e
        finally:
            os.umask(old_umask)
        try:
            os.chmod(self.sock_path, 0o600)
        except OSError:
            pass
        listener.listen(64)
        listener.settimeout(_ACCEPT_POLL_S)
        self._listener = listener
        self._write_pidfile()

    def _write_pidfile(self) -> None:
        tmp = self.pid_path.parent / (self.pid_path.name + ".tmp")
        tmp.write_text(str(os.getpid()))
        os.replace(tmp, self.pid_path)

    # --------------------------------------------------------------- serve

    def serve(self) -> None:
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        assert self._listener is not None
        idle_limit = self.config.daemon_idle_exit_s
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            try:
                while not self._stop.is_set():
                    try:
                        conn, _ = self._listener.accept()
                    except socket.timeout:
                        if idle_limit and (
                            time.monotonic() - self._last_activity
                        ) > idle_limit:
                            break
                        continue
                    except OSError:
                        break
                    pool.submit(self._handle, conn)
            finally:
                self._cleanup()

    def _on_signal(self, signum, frame) -> None:  # noqa: ANN001
        self._stop.set()

    def _cleanup(self) -> None:
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        for p in (self.sock_path, self.pid_path):
            try:
                p.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------ handling

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(60.0)
            buf = b""
            while not self._stop.is_set():
                line, buf = self._recv_line(conn, buf)
                if line is None:
                    return
                self._last_activity = time.monotonic()
                stop_after = False
                try:
                    req = json.loads(line.decode("utf-8"))
                    resp = self._dispatch(req)
                    if req.get("op") == "shutdown":
                        stop_after = True
                except Exception as e:  # noqa: BLE001
                    resp = {"ok": False, "error": str(e)}
                try:
                    conn.sendall(json.dumps(resp).encode("utf-8") + b"\n")
                except OSError:
                    return
                if stop_after:
                    self._stop.set()
                    return

    @staticmethod
    def _recv_line(conn: socket.socket, buf: bytes) -> tuple[bytes | None, bytes]:
        while b"\n" not in buf:
            try:
                chunk = conn.recv(65536)
            except OSError:
                return None, buf
            if not chunk:
                return None, buf
            buf += chunk
        line, _, rest = buf.partition(b"\n")
        return line, rest

    def _dispatch(self, req: dict) -> dict:
        op = req.get("op")
        if op == "ping":
            return {"ok": True, "pong": True, "pid": os.getpid()}
        if op == "shutdown":
            return {"ok": True}
        if op == "redact":
            path = req.get("path")
            if not path:
                return {"ok": False, "error": "missing path"}
            return self._redact(Path(path))
        return {"ok": False, "error": f"unknown op {op!r}"}

    def _redact(self, path: Path) -> dict:
        with self._lock:
            entry = self.cache.get(path)
            if entry is not None:
                return {
                    "ok": True,
                    "redacted_path": str(entry.redacted_path) if entry.redacted_path else None,
                    "found": entry.found,
                    "cache_hit": True,
                }
            result = self.pipeline.redact_file(path)
            self.cache.put(path, result)
            return {
                "ok": True,
                "redacted_path": str(result.redacted_path) if result.redacted_path else None,
                "found": result.found,
                "cache_hit": False,
            }


def run(config: Config | None = None) -> None:
    """Start the daemon. If a live daemon already holds the socket (or wins a
    concurrent bind race), return quietly — the caller just needs *some* daemon
    answering."""
    config = config or Config.load()
    daemon = Daemon(config)
    if daemon.socket_alive():
        return
    daemon.warmup()
    try:
        daemon.bind()
    except DaemonAlreadyRunning:
        return
    daemon.serve()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
