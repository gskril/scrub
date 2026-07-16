"""`scrub-bashguard` — a PreToolUse hook for `Bash` that fails CLOSED when a
command would read the *bytes* of a PDF or image file.

Why this exists: the PostToolUse text-scrub is a reliable backstop for plain
text, but scrubbing PDF/image bytes after the fact is lossy — a PDF's content
streams fragment each value across drawing operators, which destroys the
context the ML detector needs, so fields like a driver's-license or medical
record number can slip through. The robust path for those media is the
file-level pipeline, which the `Read` tool already routes through (PreToolUse
redirects `Read` at a redacted copy). So rather than best-effort scrubbing of
decoded bytes, we deny the raw byte-read and point the agent back at `Read`.

Scope (per project decision): file *paths/names* are assumed PII-free, so
inspecting them is fine; only PDFs and images are guarded. A command is denied
when it references an existing PDF/image file AND either invokes a
content-reading utility (cat/python/strings/pdftotext/…) or isn't a recognised
metadata-only operation (ls/mv/cp/rm/git/…). When in doubt, deny.

Always exits 0 with a valid decision; emitting nothing is an allow.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

_EVENT = "PreToolUse"

# Utilities that read/emit file *contents* (or metadata that can carry PII).
_READERS = frozenset(
    {
        "cat", "head", "tail", "less", "more", "tac", "nl", "fold", "cut", "tr",
        "fmt", "col", "expand", "unexpand", "strings", "xxd", "od", "hexdump",
        "hd", "base64", "base32", "dd", "python", "python3", "perl", "ruby",
        "node", "deno", "php", "awk", "gawk", "sed", "grep", "egrep", "fgrep",
        "rg", "ag", "pdftotext", "pdfinfo", "pdfimages", "pdftoppm", "mutool",
        "qpdf", "pdffonts", "tesseract", "convert", "magick", "identify",
        "vipsheader", "exiftool", "exif", "gs",
    }
)

# Commands that only touch a file's name/metadata, never its contents.
_SAFE_META = frozenset(
    {
        "ls", "mv", "cp", "rm", "stat", "file", "chmod", "chown", "mkdir",
        "touch", "git", "find", "basename", "dirname", "realpath", "readlink",
        "du", "ln", "rsync", "scp", "echo", "printf", "test", "[", "which",
        "wc", "cd", "pwd", "mktemp", "cmp",
    }
)

_MEDIA_EXTS = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp",
     ".heic", ".heif", ".avif"}
)


def _allow() -> None:
    """An allow is an empty stdout (passthrough)."""
    return None


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": _EVENT,
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _is_pdf_or_image(path: Path) -> bool:
    """True if `path` is a PDF or image, by magic bytes (preferred) or, failing
    a read, by extension."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return path.suffix.lower() in _MEDIA_EXTS
    if head[:4] == b"%PDF":
        return True
    sigs = (
        b"\x89PNG\r\n\x1a\n",       # png
        b"\xff\xd8\xff",             # jpeg
        b"GIF87a", b"GIF89a",        # gif
        b"BM",                       # bmp
        b"II*\x00", b"MM\x00*",      # tiff
    )
    if any(head.startswith(s) for s in sigs):
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":  # webp
        return True
    if head[4:8] == b"ftyp":  # heic/heif/avif family
        return True
    # Unreadable-as-magic but clearly-named media (e.g. a path inside a -c
    # string we resolved): fall back to extension.
    return path.suffix.lower() in _MEDIA_EXTS


def _candidate_paths(command: str, cwd: Path) -> list[Path]:
    """Every existing file the command plausibly refers to: shlex tokens plus
    media-extension substrings dug out of quoted/embedded strings."""
    raw: set[str] = set()
    try:
        raw.update(shlex.split(command))
    except ValueError:
        raw.update(command.split())
    # Paths embedded in quotes/code, e.g. python -c "open('/x/y.pdf')".
    raw.update(re.findall(r"[\w./~\-]+\.(?:pdf|png|jpe?g|gif|bmp|tiff?|webp|heic|heif|avif)", command, re.IGNORECASE))

    out: list[Path] = []
    for tok in raw:
        tok = tok.strip().strip("'\"")
        if not tok or tok.startswith("-"):
            continue
        p = Path(os.path.expanduser(tok))
        if not p.is_absolute():
            p = cwd / p
        try:
            if p.is_file():
                out.append(p)
        except OSError:
            continue
    return out


def _command_words(command: str) -> list[str]:
    try:
        toks = shlex.split(command)
    except ValueError:
        toks = command.split()
    return [Path(t).name for t in toks]


def _decide(data: dict) -> dict | None:
    if data.get("tool_name") != "Bash":
        return _allow()
    command = (data.get("tool_input") or {}).get("command") or ""
    if not command.strip():
        return _allow()
    cwd = Path(data.get("cwd") or os.getcwd())

    media = [p for p in _candidate_paths(command, cwd) if _is_pdf_or_image(p)]
    if not media:
        return _allow()

    words = set(_command_words(command))
    reads = words & _READERS
    leading = next(iter(_command_words(command)), "")

    # Deny if a content reader is involved, or if the command isn't a
    # recognised metadata-only op (fail-closed on the unknown).
    if reads or leading not in _SAFE_META:
        names = ", ".join(sorted({p.name for p in media}))
        return _deny(
            f"scrub: reading the bytes of a PDF/image ({names}) via Bash is "
            "blocked — decoded PDF/image bytes can't be reliably redacted. "
            "Use the Read tool instead; it serves a redacted copy. (If you only "
            "need to move/list/delete the file, a metadata-only command like "
            "ls/mv/cp/rm is allowed.)"
        )
    return _allow()


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
        decision = _decide(data)
    except Exception:  # noqa: BLE001 — a guard bug must fail closed, not crash open
        cmd = (data.get("tool_input") or {}).get("command", "")
        # Only escalate to deny if media is even plausibly involved; otherwise
        # a guard crash shouldn't block every unrelated command.
        if re.search(r"\.(pdf|png|jpe?g|gif|bmp|tiff?|webp|heic|heif|avif)\b", cmd, re.IGNORECASE):
            decision = _deny("scrub: bashguard error while checking a PDF/image reference; blocked to fail closed.")
        else:
            decision = None

    if decision is not None:
        json.dump(decision, sys.stdout)
        sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
