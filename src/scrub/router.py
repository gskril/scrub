"""Route a file to a coarse kind by magic bytes, with an extension-agnostic
text fallback.

Never trust the extension alone for binary formats (PDF/image) — magic bytes
decide those via the `filetype` library. For everything else (source code,
markdown, JSON, CSV, `.env`, logs, and anything else that happens to be valid
text regardless of extension) we confirm by attempting a UTF-8 decode of a
sample.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import filetype

from .config import Config

# Extraction.kind / router classification values.
TEXT = "text"
PDF = "pdf"
IMAGE = "image"
BINARY_UNKNOWN = "binary_unknown"

_SNIFF_BYTES = 8192
_MAX_REPLACEMENT_RATIO = 0.05  # >5% U+FFFD replacement chars => treat as binary


class SkipFile(Exception):
    """Raised by the router or an extractor when a file cannot or should not
    be processed further.

    `reason` is a short machine-friendly tag (e.g. "binary_content",
    "max_file_bytes"); `detail` is a human-readable message. Callers
    (pipeline/cli/hook) treat this as "pass the original through untouched"
    rather than a hard error — the hook's fail-closed policy is layered on
    top of this by Phase 4+5, not here.
    """

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail or reason
        super().__init__(self.detail)


def classify(path: Path) -> str:
    """Classify a file's true type by magic bytes, with a text fallback.

    Returns one of TEXT, PDF, IMAGE, BINARY_UNKNOWN. Raises SkipFile if the
    path cannot be read at all (missing, permission denied, etc.).
    """
    path = Path(path)
    try:
        kind = filetype.guess(str(path))
    except FileNotFoundError as e:
        raise SkipFile("not_found", f"{path}: {e}") from e
    except OSError as e:
        raise SkipFile("read_error", f"{path}: {e}") from e

    if kind is not None:
        mime = kind.mime
        if mime == "application/pdf":
            return PDF
        if mime.startswith("image/"):
            return IMAGE
        # filetype matched some other known binary signature (zip, exe,
        # archive, office doc, audio, video, ...) — not text, not pdf/image.
        return BINARY_UNKNOWN

    # No recognized binary signature — the common case for source code,
    # markdown, JSON, CSV, .env, logs, plain text. Confirm via UTF-8 sniff.
    if _looks_like_text(path):
        return TEXT
    return BINARY_UNKNOWN


def _looks_like_text(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            sample = f.read(_SNIFF_BYTES)
    except OSError:
        return False
    if not sample:
        return True  # empty file: nothing to redact, nothing to skip either
    if b"\x00" in sample:
        return False  # NUL byte: definitely not text
    decoded = sample.decode("utf-8", errors="replace")
    if not decoded:
        return False
    replacement_count = decoded.count("�")
    return (replacement_count / len(decoded)) <= _MAX_REPLACEMENT_RATIO


def should_skip(path: Path, config: Config) -> tuple[bool, str]:
    """Decide whether a file should be skipped before extraction/detection.

    Returns (skip, reason). Precedence:
      1. `deny_globs` ("always scrub") overrides everything below — it is the
         most specific/explicit directive a user can give.
      2. `allow_globs` ("never scrub", e.g. the tool's own fixtures) -> skip.
      3. `skip_globs` (generated dirs, lockfiles, vcs internals) -> skip.
      4. `max_file_bytes` -> skip if the file exceeds the configured cap.
    """
    path = Path(path)
    path_str = path.as_posix()

    def matches_any(globs: list[str]) -> bool:
        return any(
            fnmatch.fnmatch(path_str, pat) or fnmatch.fnmatch(path.name, pat) for pat in globs
        )

    if not matches_any(config.deny_globs):
        if matches_any(config.allow_globs):
            return True, "allow_globs"
        if matches_any(config.skip_globs):
            return True, "skip_globs"

    try:
        size = path.stat().st_size
    except OSError:
        return True, "stat_failed"
    if size > config.max_file_bytes:
        return True, "max_file_bytes"

    return False, ""
