"""`scrub-hook` — the Claude Code PreToolUse hook for `Read`.

Reads a hook payload on stdin, and on stdout emits a PreToolUse decision:

  - passthrough (allow, no change) for anything we shouldn't or can't scrub;
  - allow with `updatedInput` pointing `file_path` at a redacted copy when PII
    is found (the ENTIRE tool_input is echoed, only `file_path` swapped);
  - on any failure, honour `config.fail_mode`: "closed" (default) denies the
    read with a readable reason; "open" passes the original through and logs.

The hook is a thin client — no model loading here; it talks to the daemon,
spawning one if needed. It ALWAYS exits 0 with a valid JSON decision on stdout,
even on unexpected errors (last-resort fail-mode-aware emit).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .client import ensure_daemon, redact
from .config import Config, cache_dir
from .router import should_skip

_EVENT = "PreToolUse"


def _allow(updated_input: dict | None = None) -> dict:
    out: dict = {"hookSpecificOutput": {"hookEventName": _EVENT, "permissionDecision": "allow"}}
    if updated_input is not None:
        out["hookSpecificOutput"]["updatedInput"] = updated_input
    return out


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": _EVENT,
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"scrub: {reason}; original file NOT shown. Fix the scrub daemon "
                "or set fail_mode='open' in ~/.config/scrub/config.toml"
            ),
        }
    }


def _fail(fail_mode: str, reason: str) -> dict:
    if fail_mode == "open":
        print(f"scrub: {reason} (fail-open, passing original through)", file=sys.stderr)
        return _allow()
    return _deny(reason)


def _inside(path: Path, root: Path) -> bool:
    try:
        rpath = path.resolve()
        rroot = root.resolve()
    except OSError:
        return False
    return rroot == rpath or rroot in rpath.parents


def _decide(data: dict, config: Config) -> dict:
    if data.get("tool_name") != "Read":
        return _allow()

    tool_input = data.get("tool_input") or {}
    file_path = tool_input.get("file_path")
    if not file_path or not os.path.isabs(file_path):
        return _allow()

    path = Path(file_path)
    if not path.is_file():
        return _allow()

    # Never re-scrub our own output — reading a redacted copy back through the
    # hook would loop forever.
    if _inside(path, cache_dir()):
        return _allow()

    # allow_globs / skip_globs / size cap (should_skip covers allow_globs).
    skip, _reason = should_skip(path, config)
    if skip:
        return _allow()

    ensure_daemon(config)
    resp = redact(config, path)
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "daemon error"))

    if not resp.get("found"):
        return _allow()

    redacted_path = resp.get("redacted_path")
    if not redacted_path or not Path(redacted_path).is_file():
        # found > 0 but no usable redacted copy: this is a daemon bug, and
        # allowing the original through here would be a silent fail-open.
        raise RuntimeError("daemon reported findings but no redacted copy")

    updated_input = {**tool_input, "file_path": redacted_path}
    return _allow(updated_input)


def main() -> None:
    try:
        raw = sys.stdin.read()
    except Exception:  # noqa: BLE001
        raw = ""
    try:
        data = json.loads(raw) if raw.strip() else {}
    except ValueError:
        data = {}

    try:
        config = Config.load()
    except Exception:  # noqa: BLE001 — bad config must not crash the hook
        config = None
    fail_mode = config.fail_mode if config is not None else "closed"

    try:
        decision = _decide(data, config or Config())
    except Exception as e:  # noqa: BLE001
        decision = _fail(fail_mode, str(e) or e.__class__.__name__)

    sys.stdout.write(json.dumps(decision))
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
