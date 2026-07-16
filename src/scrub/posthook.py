"""`scrub-posthook` — the Claude Code PostToolUse hook for `Bash` and `Read`.

This is the source-agnostic companion to the PreToolUse `Read` hook. Where the
PreToolUse hook can only redirect a `Read` at a redacted *file* copy, this hook
scrubs the *output a tool already produced* on its way to the model — so PII in
`Bash` output (`cat`, `pdftotext`, `python`, arbitrary shell) is redacted too,
and nothing on disk is ever touched.

Mechanism (schema confirmed empirically against Claude Code, not the docs):

  stdin  : {"tool_name","tool_input","tool_response", ...}
             Bash content -> tool_response.stdout / .stderr
             Read content -> tool_response.file.content
  stdout : {"hookSpecificOutput":{"hookEventName":"PostToolUse",
            "updatedToolOutput": <a full copy of tool_response with the
                                  content field(s) replaced>}}

`updatedToolOutput` REPLACES the entire tool_response, so we copy the original
and swap only the text fields — preserving numLines/interrupted/etc. When
nothing is redacted we emit nothing (a bare passthrough). On failure we honour
`config.fail_mode`: "closed" (default) withholds the original output behind a
notice; "open" passes it through untouched.

The hook ALWAYS exits 0. Emitting no stdout is a valid passthrough.
"""

from __future__ import annotations

import copy
import json
import sys

from .client import ensure_daemon, redact_text_request
from .config import Config

_EVENT = "PostToolUse"
# Outputs larger than this are withheld (fail-closed) rather than scrubbed: a
# very large redaction could exceed the hook timeout, and a timed-out hook
# emits nothing — which Claude Code treats as passthrough, i.e. a fail-OPEN
# leak. Capping keeps us comfortably under the timeout so we never leak that
# way. Normal tool output is far below this.
_MAX_SCRUB_CHARS = 200_000
_WITHHELD = (
    "[scrub: could not redact this output, so it was withheld. Fix the scrub "
    "daemon or set fail_mode='open' in ~/.config/scrub/config.toml]"
)
_TOO_LARGE = (
    "[scrub: output too large to redact within the hook's time budget, so it "
    "was withheld to avoid leaking un-redacted content. Narrow the command "
    "(head/grep/limit) and try again.]"
)


def _emit(updated_tool_output: dict | None) -> None:
    """Print a PostToolUse decision, or nothing at all for a passthrough."""
    if updated_tool_output is None:
        return
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": _EVENT,
                "updatedToolOutput": updated_tool_output,
            }
        },
        sys.stdout,
    )
    sys.stdout.flush()


def _scrub(config: Config, text: str) -> tuple[str, int]:
    """Redact one string via the daemon. Returns (redacted, found)."""
    if not text:
        return text, 0
    ensure_daemon(config)
    resp = redact_text_request(config, text)
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "daemon error"))
    return resp.get("redacted", text), int(resp.get("found", 0))


def _withhold(data: dict, message: str) -> dict | None:
    """Build an updatedToolOutput whose text fields are replaced by `message`,
    hiding the original. Returns None if the shape isn't one we handle."""
    resp = data.get("tool_response")
    if not isinstance(resp, dict):
        return None
    updated = copy.deepcopy(resp)
    if data.get("tool_name") == "Bash":
        updated["stdout"] = message
        updated["stderr"] = ""
    elif data.get("tool_name") == "Read" and isinstance(updated.get("file"), dict):
        updated["file"]["content"] = message
    else:
        return None
    return updated


def _decide(data: dict, config: Config) -> dict | None:
    tool_name = data.get("tool_name")
    resp = data.get("tool_response")
    if not isinstance(resp, dict):
        return None

    if tool_name == "Bash":
        stdout, stderr = resp.get("stdout") or "", resp.get("stderr") or ""
        # Oversize -> withhold, ALWAYS (never honour fail-open here): a huge
        # redaction risks a hook timeout, which would silently fail open.
        if len(stdout) + len(stderr) > _MAX_SCRUB_CHARS:
            return _withhold(data, _TOO_LARGE)
        out, found_out = _scrub(config, stdout)
        err, found_err = _scrub(config, stderr)
        if found_out == 0 and found_err == 0:
            return None
        updated = copy.deepcopy(resp)
        updated["stdout"] = out
        updated["stderr"] = err
        return updated

    if tool_name == "Read":
        file_obj = resp.get("file")
        if not isinstance(file_obj, dict) or resp.get("type") != "text":
            return None  # non-text Read (image/notebook) — not our shape
        content = file_obj.get("content") or ""
        if len(content) > _MAX_SCRUB_CHARS:
            return _withhold(data, _TOO_LARGE)
        redacted, found = _scrub(config, content)
        if found == 0:
            return None
        updated = copy.deepcopy(resp)
        updated["file"]["content"] = redacted
        return updated

    return None


def _fail(fail_mode: str, data: dict, reason: str) -> dict | None:
    """Fail-open passes the original through (emit nothing). Fail-closed
    replaces the tool's text fields with a notice so the original is hidden."""
    if fail_mode == "open":
        print(f"scrub: {reason} (fail-open, passing original through)", file=sys.stderr)
        return None
    return _withhold(data, _WITHHELD)


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
        updated = _decide(data, config or Config())
    except Exception as e:  # noqa: BLE001
        updated = _fail(fail_mode, data, str(e) or e.__class__.__name__)

    _emit(updated)
    sys.exit(0)


if __name__ == "__main__":
    main()
