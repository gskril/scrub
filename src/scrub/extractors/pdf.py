"""PDF extractor: word-level text + bounding boxes via PyMuPDF.

Implements the `Extractor` protocol from `scrub.types`. Digital PDFs only —
scanned pages with no extractable text are recorded (internally, see
`PdfExtractor.pages_without_text`) and skipped; OCR is a post-v1 phase.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from ..types import Extraction, WordBox

logger = logging.getLogger(__name__)


class EncryptedPdfError(Exception):
    """Raised when a PDF is encrypted/password-protected and cannot be read."""


class PdfExtractor:
    """Extractor for digital PDFs.

    `extract()` builds `Extraction.text` by joining `page.get_text("words")`
    output with single spaces within a page and "\\n\\n" between pages. Every
    `WordBox` carries the exact (start, end) char range of its word in that
    text, so `text[wb.start:wb.end] == word` always holds (round-trip
    invariant), which the redactor relies on to map spans back to page rects.

    `pages_without_text` is NOT part of the `Extraction` contract (that
    dataclass uses `__slots__` and is owned by `scrub.types`) — it is tracked
    as instance state on the extractor for the most recently extracted
    document, and a warning is logged for each such page at extract time.
    """

    def __init__(self) -> None:
        self.pages_without_text: list[int] = []

    def extract(self, path: Path) -> Extraction:
        path = Path(path)
        doc = pymupdf.open(path)
        try:
            if doc.is_encrypted or doc.needs_pass:
                raise EncryptedPdfError(
                    f"{path}: PDF is encrypted/password-protected; cannot extract text"
                )

            pages_without_text: list[int] = []
            pieces: list[str] = []
            words: list[WordBox] = []
            cursor = 0

            for page_index in range(doc.page_count):
                page = doc[page_index]
                # get_text("words") returns tuples sorted in reading order:
                # (x0, y0, x1, y1, word, block_no, line_no, word_no)
                raw_words = [w for w in page.get_text("words") if w[4]]

                if page_index > 0:
                    sep = "\n\n"
                    pieces.append(sep)
                    cursor += len(sep)

                if not raw_words:
                    pages_without_text.append(page_index)
                    logger.warning(
                        "%s: page %d has no extractable text (likely scanned); "
                        "OCR extraction is post-v1, skipping",
                        path,
                        page_index,
                    )

                for i, w in enumerate(raw_words):
                    x0, y0, x1, y1, word_text = w[0], w[1], w[2], w[3], w[4]
                    if i > 0:
                        pieces.append(" ")
                        cursor += 1
                    start = cursor
                    pieces.append(word_text)
                    cursor += len(word_text)
                    end = cursor
                    words.append(
                        WordBox(
                            start=start,
                            end=end,
                            page=page_index,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                        )
                    )

            self.pages_without_text = pages_without_text
            text = "".join(pieces)
            return Extraction(text=text, kind="pdf", words=words)
        finally:
            doc.close()
