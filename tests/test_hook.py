"""Tests for the scrub-hook PreToolUse entry point and the settings installer
(Phase 5)."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import scrub.client as client
from scrub import install as install_mod

PII_TEXT = (
    "Call Maria Garcia at (312) 555-0148.\n"
    "Her SSN is 458-02-6841 and email maria.garcia@example.com.\n"
)

_HOOK_CMD = [sys.executable, "-m", "scrub.hook"]
_START_TIMEOUT = 60.0


# --------------------------------------------------------------- daemon setup

def _wait_up(timeout: float = _START_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if client.ping().get("ok"):
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise AssertionError("daemon did not come up")


@pytest.fixture(scope="module")
def live_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("hook-daemon")
    sock = base / "scrubd.sock"
    cache = base / "cache"
    cache.mkdir()
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
        yield {"sock": sock, "cache": cache, "proc": proc}
    finally:
        try:
            client.shutdown()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_hook(payload: dict, env: dict | None = None) -> dict:
    proc = subprocess.run(
        _HOOK_CMD,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, f"hook exited {proc.returncode}: {proc.stderr}"
    return json.loads(proc.stdout)


# ------------------------------------------------------------------- hook cases

def test_pii_file_rewrites_input(live_env, tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text(PII_TEXT)
    payload = {
        "tool_name": "Read",
        "tool_input": {"file_path": str(src), "offset": 10, "limit": 5},
    }
    out = _run_hook(payload)["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    updated = out["updatedInput"]
    # file_path swapped to a redacted copy under the cache dir...
    assert updated["file_path"] != str(src)
    assert str(live_env["cache"]) in updated["file_path"]
    assert Path(updated["file_path"]).is_file()
    assert "458-02-6841" not in Path(updated["file_path"]).read_text()
    # ...and every other tool_input field echoed unchanged.
    assert updated["offset"] == 10
    assert updated["limit"] == 5


def test_clean_file_passthrough(live_env, tmp_path):
    src = tmp_path / "clean.py"
    src.write_text("def add(a, b):\n    return a + b\n")
    payload = {"tool_name": "Read", "tool_input": {"file_path": str(src)}}
    out = _run_hook(payload)["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    assert "updatedInput" not in out


def test_file_inside_cache_dir_passthrough(live_env):
    inside = live_env["cache"] / "redacted" / "already.redacted.txt"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_text("SSN 458-02-6841 for Maria Garcia\n")  # PII, but it's our output
    payload = {"tool_name": "Read", "tool_input": {"file_path": str(inside)}}
    out = _run_hook(payload)["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    assert "updatedInput" not in out  # never re-scrub our own output


def test_non_read_tool_passthrough(live_env, tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text(PII_TEXT)
    payload = {"tool_name": "Write", "tool_input": {"file_path": str(src), "content": "x"}}
    out = _run_hook(payload)["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    assert "updatedInput" not in out


def test_daemon_unreachable_fail_closed_denies(tmp_path):
    # A socket path whose parent is a FILE — mkdir/bind can never succeed (even
    # as root), so the spawned daemon dies and the client exhausts its retries.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    cache = tmp_path / "cache"
    cache.mkdir()
    env = os.environ.copy()
    env["SCRUB_SOCKET"] = str(blocker / "scrubd.sock")
    env["SCRUB_CACHE_DIR"] = str(cache)
    env["SCRUB_SPAWN_TIMEOUT"] = "1"  # keep the test fast
    env.pop("SCRUB_CONFIG_DIR", None)  # default fail_mode = closed

    src = tmp_path / "notes.txt"
    src.write_text(PII_TEXT)
    payload = {"tool_name": "Read", "tool_input": {"file_path": str(src)}}
    out = _run_hook(payload, env=env)["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "scrub:" in out["permissionDecisionReason"]
    assert "NOT shown" in out["permissionDecisionReason"]


def test_daemon_unreachable_fail_open_passes(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    cache = tmp_path / "cache"
    cache.mkdir()
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "config.toml").write_text('fail_mode = "open"\n')
    env = os.environ.copy()
    env["SCRUB_SOCKET"] = str(blocker / "scrubd.sock")
    env["SCRUB_CACHE_DIR"] = str(cache)
    env["SCRUB_CONFIG_DIR"] = str(cfgdir)
    env["SCRUB_SPAWN_TIMEOUT"] = "1"

    src = tmp_path / "notes.txt"
    src.write_text(PII_TEXT)
    payload = {"tool_name": "Read", "tool_input": {"file_path": str(src)}}
    out = _run_hook(payload, env=env)["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"  # fail-open passes original through
    assert "updatedInput" not in out


def test_relative_path_passthrough(live_env):
    payload = {"tool_name": "Read", "tool_input": {"file_path": "relative/notes.txt"}}
    out = _run_hook(payload)["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    assert "updatedInput" not in out


# ------------------------------------------------------------ install: pure merge

CMD = "/usr/local/bin/scrub-hook"


def test_merge_install_empty():
    settings = install_mod.merge_install({}, CMD)
    entries = settings["hooks"]["PreToolUse"]
    assert len(entries) == 1
    assert entries[0]["matcher"] == "Read"
    assert entries[0]["hooks"][0]["command"] == CMD
    assert entries[0]["hooks"][0]["timeout"] == 20


def test_merge_install_idempotent():
    s = install_mod.merge_install({}, CMD)
    s = install_mod.merge_install(copy.deepcopy(s), CMD)
    assert len(s["hooks"]["PreToolUse"]) == 1


def test_merge_install_preserves_unrelated():
    original = {
        "model": "sonnet",
        "hooks": {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "other"}]}
            ],
            "PostToolUse": [{"matcher": "Edit", "hooks": []}],
        },
    }
    merged = install_mod.merge_install(copy.deepcopy(original), CMD)
    assert merged["model"] == "sonnet"
    assert merged["hooks"]["PostToolUse"] == original["hooks"]["PostToolUse"]
    matchers = [e["matcher"] for e in merged["hooks"]["PreToolUse"]]
    assert "Bash" in matchers and "Read" in matchers


def test_merge_uninstall_restores():
    original = {
        "hooks": {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "other"}]}
            ]
        }
    }
    installed = install_mod.merge_install(copy.deepcopy(original), CMD)
    restored = install_mod.merge_uninstall(installed, CMD)
    assert restored == original


def test_merge_uninstall_prunes_to_empty():
    installed = install_mod.merge_install({}, CMD)
    restored = install_mod.merge_uninstall(installed, CMD)
    assert restored == {}


# ------------------------------------------------------ install: filesystem io

@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_install_creates_settings(fake_home):
    path, cmd = install_mod.install("user", command=CMD)
    assert path == fake_home / ".claude" / "settings.json"
    data = json.loads(path.read_text())
    assert install_mod.is_installed(data, CMD)


def test_install_preserves_existing_and_idempotent(fake_home):
    path = fake_home / ".claude" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model": "opus", "env": {"FOO": "1"}}))

    install_mod.install("user", command=CMD)
    install_mod.install("user", command=CMD)  # double install

    data = json.loads(path.read_text())
    assert data["model"] == "opus"
    assert data["env"] == {"FOO": "1"}
    assert len(data["hooks"]["PreToolUse"]) == 1  # not duplicated


def test_uninstall_restores(fake_home):
    path = fake_home / ".claude" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model": "opus"}))

    install_mod.install("user", command=CMD)
    _, removed = install_mod.uninstall("user", command=CMD)
    assert removed is True
    data = json.loads(path.read_text())
    assert data == {"model": "opus"}
    assert not install_mod.is_installed(data, CMD)
