"""TextExtractor: turn a text-like file into an Extraction.

Handles source code, markdown, JSON, CSV, .env, logs — anything the router
classified as TEXT. Decodes as UTF-8 with a guard: if a meaningful fraction
of the decode came out as replacement characters, this wasn't really text
(the router's sniff can be fooled by a text-looking prefix on an otherwise
binary file) and we bail with SkipFile rather than redact garbage.
"""

from __future__ import annotations

from pathlib import Path

from ..router import SkipFile
from ..types import Extraction

_MAX_REPLACEMENT_RATIO = 0.05


class TextExtractor:
    """Extractor for Extraction.kind == "text"."""

    def extract(self, path: Path) -> Extraction:
        path = Path(path)
        try:
            raw = path.read_bytes()
        except OSError as e:
            raise SkipFile("read_error", f"could not read {path}: {e}") from e

        if not raw:
            return Extraction(text="", kind="text")

        text = raw.decode("utf-8", errors="replace")
        replacement_count = text.count("�")
        if len(text) and (replacement_count / len(text)) > _MAX_REPLACEMENT_RATIO:
            raise SkipFile(
                "binary_content",
                f"{path}: {replacement_count}/{len(text)} decoded chars were "
                "invalid UTF-8 (looks like binary, not text)",
            )
        return Extraction(text=text, kind="text")
