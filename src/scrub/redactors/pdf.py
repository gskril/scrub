"""PDF redactor: black-out redaction annotations + metadata scrub via PyMuPDF.

Implements the PDF side of the redaction contract in ARCHITECTURE.md: map
winning spans -> WordBoxes -> PyMuPDF redaction annotations, apply them (which
removes the underlying text, not just draws a box over it), then strip
document metadata and XMP so nothing PII-bearing survives a re-extraction or
metadata dump.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from ..config import Config
from ..types import Extraction, ReportEntity, Span
from .placeholders import assign_placeholders

# A redaction rectangle only needs to intersect a glyph to remove that glyph.
# Insetting is safer than padding: PDF word boxes on tightly-led lines can
# overlap vertically, and an expanded rectangle can therefore delete text on
# an adjacent line. The fill is applied after the intersecting glyphs are
# removed, so there are no antialiased remnants to cover.
REDACT_INSET_X = 0.5
REDACT_INSET_Y = 2.0

_BLACK = (0, 0, 0)


def _scrub_metadata(doc: pymupdf.Document) -> None:
    """Strip metadata + XMP using the strongest combination this PyMuPDF
    version offers. `Document.scrub()` (when present) is the broadest tool —
    it also clears metadata/xml_metadata/javascript/hidden text/embedded
    files/thumbnails/links — but we additionally call the narrower
    `set_metadata({})` / `del_xml_metadata()` explicitly so the behavior does
    not silently regress if `scrub()` is ever unavailable or its defaults
    change.
    """
    if hasattr(doc, "scrub"):
        doc.scrub(metadata=True, xml_metadata=True)
    doc.set_metadata({})
    if hasattr(doc, "del_xml_metadata"):
        doc.del_xml_metadata()


def redact_pdf(
    src_path: Path,
    dst_path: Path,
    spans: list[Span],
    extraction: Extraction,
    config: Config | None = None,
) -> list[ReportEntity]:
    """Redact `src_path` per `spans`, write the result to `dst_path`.

    `spans` are offsets into `extraction.text` (the same `Extraction`
    produced by `PdfExtractor.extract(src_path)`). Spans whose entity_type is
    in `config.public_types` are recorded in the returned report but not
    blacked out (ARCHITECTURE.md: "detected and reported but NOT redacted").
    """
    if extraction.kind != "pdf":
        raise ValueError(f"redact_pdf requires a pdf Extraction, got kind={extraction.kind!r}")

    cfg = config or Config.load()
    public_types = cfg.public_types
    placeholders = assign_placeholders(spans)

    page_rects: dict[int, list[pymupdf.Rect]] = {}
    entities: list[ReportEntity] = []

    for span, placeholder in zip(spans, placeholders):
        overlapping = [wb for wb in extraction.words if wb.start < span.end and span.start < wb.end]
        page = overlapping[0].page if overlapping else None

        entities.append(
            ReportEntity(
                placeholder=placeholder,
                entity_type=span.entity_type,
                start=span.start,
                end=span.end,
                confidence=span.confidence,
                source=span.source,
                page=page,
            )
        )

        if span.entity_type in public_types:
            continue  # public type: report it, don't black it out

        if not overlapping:
            # Every detected span is offsets into text built from these very
            # WordBoxes, so zero overlap means the offset mapping is broken.
            # Writing output anyway would ship a "sanitized" PDF that still
            # contains this value — fail instead (the hook turns this into a
            # deny under fail-closed).
            raise RuntimeError(
                f"PDF redaction cannot locate span {placeholder} "
                f"({span.entity_type}, chars {span.start}-{span.end}) on any page; "
                "refusing to write partially-redacted output"
            )

        for wb in overlapping:
            rect = pymupdf.Rect(
                wb.x0 + min(REDACT_INSET_X, (wb.x1 - wb.x0) / 4),
                wb.y0 + min(REDACT_INSET_Y, (wb.y1 - wb.y0) / 4),
                wb.x1 - min(REDACT_INSET_X, (wb.x1 - wb.x0) / 4),
                wb.y1 - min(REDACT_INSET_Y, (wb.y1 - wb.y0) / 4),
            )
            page_rects.setdefault(wb.page, []).append(rect)

    doc = pymupdf.open(src_path)
    try:
        if doc.is_encrypted or doc.needs_pass:
            raise ValueError(f"{src_path}: cannot redact an encrypted PDF")

        for page_index, rects in page_rects.items():
            page = doc[page_index]
            for rect in rects:
                page.add_redact_annot(rect, fill=_BLACK)
            # images=2: blank out overlapping image parts too (belt-and-braces
            # for scanned-looking embedded images); text=0 (default) removes
            # the underlying text content stream, not just paints over it.
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)

        _scrub_metadata(doc)

        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(dst_path, garbage=4, deflate=True)
    finally:
        doc.close()

    return entities
