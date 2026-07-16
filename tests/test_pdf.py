"""Standalone tests for the PDF extractor + redactor (Phase 3).

Deliberately does NOT import scrub.detectors or scrub.pipeline (owned by
concurrent phases) — Spans are hand-built from known fixture offsets via
`text.find()` so this file runs independently.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from scrub.extractors.pdf import PdfExtractor
from scrub.redactors.pdf import redact_pdf
from scrub.types import EntityType, Span

from tests.fixtures.make_pdfs import make_all

SSN = "458-02-6841"
GIVEN_NAME = "Maria"
SURNAME = "Garcia"
FULL_NAME = "Maria Garcia"


def _span(text: str, needle: str, entity_type: EntityType, source: str = "regex",
          confidence: float = 1.0, start: int = 0) -> Span:
    idx = text.find(needle, start)
    assert idx != -1, f"{needle!r} not found in extracted text"
    return Span(
        start=idx,
        end=idx + len(needle),
        entity_type=entity_type,
        text=needle,
        confidence=confidence,
        source=source,
    )


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory) -> dict[str, Path]:
    out_dir = tmp_path_factory.mktemp("pdf_fixtures")
    paths = make_all(out_dir)
    return {p.stem: p for p in paths}


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def test_word_roundtrip_invariant_holds_for_every_word(fixtures):
    """text[wb.start:wb.end] == the word, for every WordBox, across all fixtures."""
    extractor = PdfExtractor()
    for name, path in fixtures.items():
        extraction = extractor.extract(path)
        for wb in extraction.words:
            got = extraction.text[wb.start:wb.end]
            assert got == extraction.text[wb.start:wb.end]  # sanity: slicing works
        # Re-derive expected words directly from PyMuPDF and compare.
        doc = pymupdf.open(path)
        try:
            for wb in extraction.words:
                page = doc[wb.page]
                page_words = [w[4] for w in page.get_text("words") if w[4]]
                assert extraction.text[wb.start:wb.end] in page_words, (
                    f"{name}: {extraction.text[wb.start:wb.end]!r} round-trip mismatch"
                )
        finally:
            doc.close()


def test_fake_w2_extraction_contains_ssn_and_name(fixtures):
    extractor = PdfExtractor()
    extraction = extractor.extract(fixtures["fake_w2"])
    assert extraction.kind == "pdf"
    assert SSN in extraction.text
    assert "Maria" in extraction.text
    # Every WordBox exact round-trip on this fixture in particular.
    for wb in extraction.words:
        word = extraction.text[wb.start:wb.end]
        assert word.strip() == word
        assert len(word) > 0


def test_contract_is_multipage_with_boxes_on_every_page(fixtures):
    extractor = PdfExtractor()
    extraction = extractor.extract(fixtures["contract"])
    doc = pymupdf.open(fixtures["contract"])
    try:
        assert doc.page_count >= 2
    finally:
        doc.close()

    pages_with_boxes = {wb.page for wb in extraction.words}
    # Boxes present on page 0 and at least one later page (1+).
    assert 0 in pages_with_boxes
    assert any(p >= 1 for p in pages_with_boxes)

    # Party names repeat across pages.
    assert extraction.text.count("Maria Garcia") >= 2
    assert extraction.text.count("Acme Corp") >= 2

    # "\n\n" page separator is present between pages.
    assert "\n\n" in extraction.text


def test_extraction_records_pages_without_text_internally():
    # A page with no inserted text should be flagged, without touching the
    # Extraction dataclass contract (which is slots-only, owned by types.py).
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        blank_path = Path(td) / "blank.pdf"
        doc = pymupdf.open()
        doc.new_page()  # no text at all
        doc.save(blank_path)
        doc.close()

        extractor = PdfExtractor()
        extraction = extractor.extract(blank_path)
        assert extraction.kind == "pdf"
        assert extractor.pages_without_text == [0]


# --------------------------------------------------------------------------
# Redaction + recovery
# --------------------------------------------------------------------------


def test_redact_fake_w2_removes_ssn_and_name(fixtures, tmp_path):
    extractor = PdfExtractor()
    src = fixtures["fake_w2"]
    extraction = extractor.extract(src)

    ssn_span = _span(extraction.text, SSN, EntityType.SSN)
    name_span = _span(extraction.text, FULL_NAME, EntityType.GIVEN_NAME)

    dst = tmp_path / "fake_w2.redacted.pdf"
    entities = redact_pdf(src, dst, [ssn_span, name_span], extraction)

    assert len(entities) == 2
    assert dst.is_file()

    # (a) Recovery: re-extract text from the redacted PDF, none of the
    # redacted strings appear anywhere.
    redacted_doc = pymupdf.open(dst)
    try:
        full_text = "\n".join(redacted_doc[i].get_text() for i in range(redacted_doc.page_count))
    finally:
        redacted_doc.close()

    assert SSN not in full_text
    assert "Maria Garcia" not in full_text
    assert "Maria" not in full_text
    assert "Garcia" not in full_text

    # (d) Visual sanity: redacted page renders without error.
    redacted_doc = pymupdf.open(dst)
    try:
        pix = redacted_doc[0].get_pixmap()
        assert pix.width > 0 and pix.height > 0
    finally:
        redacted_doc.close()


def test_redaction_does_not_delete_text_on_adjacent_tightly_led_line(tmp_path):
    src = tmp_path / "tight-lines.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Authorization note", fontsize=8, fontname="helv")
    page.insert_text((72, 110), "Jordan authorize access", fontsize=8, fontname="helv")
    doc.save(src)
    doc.close()

    extraction = PdfExtractor().extract(src)
    span = _span(extraction.text, "Jordan", EntityType.GIVEN_NAME)
    dst = tmp_path / "tight-lines.redacted.pdf"
    redact_pdf(src, dst, [span], extraction)

    out = pymupdf.open(dst)
    try:
        text = out[0].get_text()
    finally:
        out.close()
    assert "Jordan" not in text
    assert "Authorization note" in text
    assert "authorize access" in text


def test_redact_letter_scrubs_metadata_and_xmp(fixtures, tmp_path):
    extractor = PdfExtractor()
    src = fixtures["letter"]
    extraction = extractor.extract(src)

    # Sanity: the source fixture really does carry PII metadata.
    src_doc = pymupdf.open(src)
    try:
        assert "Maria Garcia" in (src_doc.metadata.get("author") or "")
        assert "Maria Garcia" in (src_doc.metadata.get("title") or "")
    finally:
        src_doc.close()

    name_spans = []
    idx = 0
    while True:
        pos = extraction.text.find(FULL_NAME, idx)
        if pos == -1:
            break
        name_spans.append(
            Span(
                start=pos,
                end=pos + len(FULL_NAME),
                entity_type=EntityType.GIVEN_NAME,
                text=FULL_NAME,
                confidence=1.0,
                source="regex",
            )
        )
        idx = pos + len(FULL_NAME)
    assert len(name_spans) >= 2  # letterhead address block + salutation

    ref_spans = []
    idx = 0
    while True:
        pos = extraction.text.find("SCB-0042-7719", idx)
        if pos == -1:
            break
        ref_spans.append(
            Span(
                start=pos,
                end=pos + len("SCB-0042-7719"),
                entity_type=EntityType.CUSTOM,
                text="SCB-0042-7719",
                confidence=1.0,
                source="custom",
            )
        )
        idx = pos + len("SCB-0042-7719")
    assert len(ref_spans) >= 2  # "Re:" line + body reference

    dst = tmp_path / "letter.redacted.pdf"
    redact_pdf(src, dst, [*name_spans, *ref_spans], extraction)

    redacted_doc = pymupdf.open(dst)
    try:
        # (b) metadata has no author/title PII.
        meta = redacted_doc.metadata
        for key in ("author", "title", "subject", "keywords"):
            value = meta.get(key) or ""
            assert "Maria" not in value
            assert "Garcia" not in value

        # (c) xref/XMP metadata gone.
        assert redacted_doc.xref_xml_metadata() == 0
        xml_meta = redacted_doc.get_xml_metadata()
        assert not xml_meta

        # Recovery: name + reference number no longer extractable as text.
        full_text = redacted_doc[0].get_text()
        assert "Maria Garcia" not in full_text
        assert "SCB-0042-7719" not in full_text

        # (d) visual sanity.
        pix = redacted_doc[0].get_pixmap()
        assert pix.width > 0 and pix.height > 0
    finally:
        redacted_doc.close()


def test_redact_contract_removes_pii_from_every_page(fixtures, tmp_path):
    extractor = PdfExtractor()
    src = fixtures["contract"]
    extraction = extractor.extract(src)

    def _all_spans(needle: str, entity_type: EntityType) -> list[Span]:
        found = []
        idx = 0
        while True:
            pos = extraction.text.find(needle, idx)
            if pos == -1:
                break
            found.append(
                Span(
                    start=pos,
                    end=pos + len(needle),
                    entity_type=entity_type,
                    text=needle,
                    confidence=1.0,
                    source="regex",
                )
            )
            idx = pos + len(needle)
        assert found, f"{needle!r} not found in extracted text"
        return found

    spans: list[Span] = []
    spans += _all_spans("maria.garcia@example.com", EntityType.EMAIL)
    spans += _all_spans("(312) 555-0148", EntityType.PHONE)
    spans += _all_spans("021000021", EntityType.ROUTING_NUMBER)
    spans += _all_spans("000123456789", EntityType.BANK_ACCOUNT)
    spans += _all_spans(FULL_NAME, EntityType.GIVEN_NAME)

    dst = tmp_path / "contract.redacted.pdf"
    entities = redact_pdf(src, dst, spans, extraction)
    assert len(entities) == len(spans)

    redacted_doc = pymupdf.open(dst)
    try:
        assert redacted_doc.page_count == 3
        full_text = "\n".join(redacted_doc[i].get_text() for i in range(3))
    finally:
        redacted_doc.close()

    assert "maria.garcia@example.com" not in full_text
    assert "(312) 555-0148" not in full_text
    assert "021000021" not in full_text
    assert "000123456789" not in full_text
    assert "Maria Garcia" not in full_text


# --------------------------------------------------------------------------
# Passthrough (public types are reported, not redacted)
# --------------------------------------------------------------------------


def test_public_type_city_is_not_blacked_out(fixtures, tmp_path):
    extractor = PdfExtractor()
    src = fixtures["fake_w2"]
    extraction = extractor.extract(src)

    # "Springfield" is CITY, a DEFAULT_PUBLIC_TYPE — must survive redaction.
    city_span = _span(extraction.text, "Springfield", EntityType.CITY)

    dst = tmp_path / "fake_w2.city_passthrough.pdf"
    entities = redact_pdf(src, dst, [city_span], extraction)

    assert len(entities) == 1
    assert entities[0].entity_type == EntityType.CITY

    redacted_doc = pymupdf.open(dst)
    try:
        full_text = redacted_doc[0].get_text()
    finally:
        redacted_doc.close()

    # CITY is public: still present, not blacked out.
    assert "Springfield" in full_text


def test_redactable_span_alongside_public_span_only_blacks_out_redactable(fixtures, tmp_path):
    extractor = PdfExtractor()
    src = fixtures["fake_w2"]
    extraction = extractor.extract(src)

    ssn_span = _span(extraction.text, SSN, EntityType.SSN)
    city_span = _span(extraction.text, "Springfield", EntityType.CITY)

    dst = tmp_path / "fake_w2.mixed.pdf"
    redact_pdf(src, dst, [ssn_span, city_span], extraction)

    redacted_doc = pymupdf.open(dst)
    try:
        full_text = redacted_doc[0].get_text()
    finally:
        redacted_doc.close()

    assert SSN not in full_text
    assert "Springfield" in full_text
