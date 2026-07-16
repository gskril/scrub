"""Install / uninstall the scrub PreToolUse hook into Claude Code settings.

`scrub install-hook [--user|--project]` writes a PreToolUse `Read` matcher into
`~/.claude/settings.json` (user) or `.claude/settings.json` (project). The merge
is non-destructive (other hooks and keys are preserved) and idempotent (a second
install is a no-op). `uninstall-hook` removes exactly what install added.

The merge helpers are pure functions over a settings dict so they're trivially
unit-testable; only `install`/`uninstall` touch the filesystem.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_MATCHER = "Read"
_TIMEOUT = 20


def settings_path(scope: str = "user") -> Path:
    if scope == "project":
        return Path.cwd() / ".claude" / "settings.json"
    if scope == "user":
        return Path.home() / ".claude" / "settings.json"
    raise ValueError(f"scope must be 'user' or 'project', got {scope!r}")


def resolve_hook_command() -> str:
    """Absolute command that runs the hook. Prefers the installed `scrub-hook`
    console script; falls back to `<python> -m scrub.hook`."""
    found = shutil.which("scrub-hook")
    if found:
        return str(Path(found).resolve())
    # Fall back to invoking the module with the current interpreter.
    return f"{sys.executable} -m scrub.hook"


def _hook_command_entry(command: str) -> dict:
    return {"type": "command", "command": command, "timeout": _TIMEOUT}


def _is_ours(hook: dict, command: str) -> bool:
    return hook.get("type") == "command" and hook.get("command") == command


def merge_install(settings: dict, command: str) -> dict:
    """Return settings with our PreToolUse `Read` hook present. Idempotent:
    if our exact command already sits under a `Read` matcher, nothing changes."""
    hooks = settings.setdefault("hooks", {})
    pretooluse = hooks.setdefault("PreToolUse", [])

    for entry in pretooluse:
        if entry.get("matcher") == _MATCHER:
            for h in entry.get("hooks", []):
                if _is_ours(h, command):
                    return settings  # already installed

    pretooluse.append({"matcher": _MATCHER, "hooks": [_hook_command_entry(command)]})
    return settings


def merge_uninstall(settings: dict, command: str) -> dict:
    """Remove exactly the hook `merge_install` added, pruning any structures
    left empty so the settings return to their prior shape."""
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return settings
    pretooluse = hooks.get("PreToolUse")
    if not isinstance(pretooluse, list):
        return settings

    new_pretooluse = []
    for entry in pretooluse:
        if entry.get("matcher") == _MATCHER and isinstance(entry.get("hooks"), list):
            kept = [h for h in entry["hooks"] if not _is_ours(h, command)]
            if not kept:
                continue  # drop a matcher entry left with no hooks
            entry = {**entry, "hooks": kept}
        new_pretooluse.append(entry)

    if new_pretooluse:
        hooks["PreToolUse"] = new_pretooluse
    else:
        hooks.pop("PreToolUse", None)
    if not hooks:
        settings.pop("hooks", None)
    return settings


def is_installed(settings: dict, command: str) -> bool:
    for entry in settings.get("hooks", {}).get("PreToolUse", []):
        if entry.get("matcher") == _MATCHER:
            if any(_is_ours(h, command) for h in entry.get("hooks", [])):
                return True
    return False


# ------------------------------------------------------------------ file io

def _read_settings(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = path.read_text().strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at top level")
    return data


def _write_settings(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def install(scope: str = "user", command: str | None = None) -> tuple[Path, str]:
    command = command or resolve_hook_command()
    path = settings_path(scope)
    settings = _read_settings(path)
    already = is_installed(settings, command)
    settings = merge_install(settings, command)
    if not already:
        _write_settings(path, settings)
    return path, command


def uninstall(scope: str = "user", command: str | None = None) -> tuple[Path, bool]:
    command = command or resolve_hook_command()
    path = settings_path(scope)
    if not path.is_file():
        return path, False
    settings = _read_settings(path)
    had = is_installed(settings, command)
    settings = merge_uninstall(settings, command)
    _write_settings(path, settings)
    return path, had
