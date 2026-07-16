"""End-to-end integration: full Pipeline.default() (regex + Rampart) over a
real text file and a generated sample PDF. Owned by the CTO/integration pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from make_pdfs import make_all  # noqa: E402

from scrub.config import Config
from scrub.pipeline import Pipeline
from scrub.types import EntityType


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> Pipeline:
    cache = tmp_path_factory.mktemp("cache")
    import scrub.config as config_mod

    # Route pipeline output into a temp cache dir for the whole module.
    orig = config_mod.cache_dir
    config_mod.cache_dir = lambda: cache
    import scrub.pipeline as pipeline_mod

    pipeline_mod.cache_dir = lambda: cache
    p = Pipeline.default(Config())
    yield p
    config_mod.cache_dir = orig
    pipeline_mod.cache_dir = orig


def test_text_file_end_to_end(pipeline, tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text(
        "Call Maria Garcia at (312) 555-0148 about the loan.\n"
        "Her SSN is 458-02-6841 and her email is maria.garcia@example.com.\n"
        "She lives at 431 Elmwood Avenue, Springfield, IL 62704.\n"
    )
    result = pipeline.redact_file(src)

    assert result.found > 0
    assert result.redacted_path is not None
    redacted = result.redacted_path.read_text()

    # Raw PII must be gone from the output text.
    for secret in ("Maria", "Garcia", "458-02-6841", "(312) 555-0148", "maria.garcia@example.com", "Elmwood"):
        assert secret not in redacted, f"{secret!r} leaked into redacted text"

    # SSN must come from the regex layer (conf 1.0), not Rampart's PHONE guess.
    ssn_entities = [e for e in result.entities if e.entity_type == EntityType.SSN]
    assert ssn_entities and all(e.source == "regex" for e in ssn_entities)

    # Name must come from Rampart.
    assert any(
        e.entity_type in (EntityType.GIVEN_NAME, EntityType.SURNAME) and e.source == "rampart"
        for e in result.entities
    )

    # Public types (city/state/zip) survive in the text but are reported.
    assert "Springfield" in redacted and "IL" in redacted

    # Report exists and contains no raw PII.
    report = result.report_path.read_text()
    for secret in ("Maria", "458-02-6841", "maria.garcia@example.com"):
        assert secret not in report


def test_placeholder_stability(pipeline, tmp_path):
    src = tmp_path / "memo.txt"
    src.write_text(
        "Maria Garcia signed. Later, Maria Garcia countersigned. Then Ravi Patel signed.\n"
    )
    result = pipeline.redact_file(src)
    redacted = result.redacted_path.read_text()
    # Same name -> same placeholder; different name -> different number.
    assert redacted.count("[GIVEN_NAME_1]") == 2
    assert "[GIVEN_NAME_2]" in redacted


def test_pdf_end_to_end(pipeline, tmp_path):
    pdfs = {p.name: p for p in make_all(tmp_path / "pdfs")}
    result = pipeline.redact_file(pdfs["fake_w2.pdf"])

    assert result.found > 0
    assert result.redacted_path is not None
    assert result.redacted_path.suffix == ".pdf"

    # Recovery: re-extract every page's text; the SSN and name must be gone.
    doc = pymupdf.open(result.redacted_path)
    try:
        full_text = "\n".join(page.get_text() for page in doc)
        meta = str(doc.metadata)
    finally:
        doc.close()
    for secret in ("458-02-6841", "Maria", "Garcia"):
        assert secret not in full_text, f"{secret!r} recoverable from redacted PDF"
        assert secret not in meta

    # Entities carry page numbers for the PDF medium.
    assert any(e.page is not None for e in result.entities)


def test_clean_file_passthrough(pipeline, tmp_path):
    src = tmp_path / "clean.py"
    src.write_text("def add(a, b):\n    return a + b\n")
    result = pipeline.redact_file(src)
    assert result.found == 0
    assert result.redacted_path is None


def test_value_propagation(pipeline, tmp_path):
    # Rampart often tags a name in prose but misses the identical string in
    # an address-block/tabular context. The propagation post-pass must redact
    # every literal occurrence of any detected value.
    src = tmp_path / "letter.txt"
    src.write_text(
        "Garcia\n88 Lighthouse Way\n\n"  # address-block context Rampart tends to miss
        "Dear Ms. Maria Garcia,\n\n"
        "Thank you for banking with us. We will contact you shortly.\n"
    )
    result = pipeline.redact_file(src)
    redacted = result.redacted_path.read_text()
    assert "Garcia" not in redacted, "detected value leaked at another occurrence"


def test_pdf_metadata_only_gets_scrubbed(pipeline, tmp_path):
    # Body is clean, but metadata carries a name: the PDF must NOT pass
    # through untouched — a scrubbed copy with empty metadata is required.
    src = tmp_path / "meta_only.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Quarterly maintenance checklist, all items nominal.")
    doc.set_metadata({"author": "Maria Garcia", "title": "Maria Garcia private notes"})
    doc.save(src)
    doc.close()

    result = pipeline.redact_file(src)
    assert result.redacted_path is not None, "metadata-bearing PDF passed through"

    out = pymupdf.open(result.redacted_path)
    try:
        leftover = {
            k: v
            for k, v in (out.metadata or {}).items()
            if v and k not in ("format", "encryption")  # intrinsic, never PII
        }
        assert not leftover, f"metadata survived the scrub: {leftover}"
    finally:
        out.close()


def test_pdf_span_without_wordbox_raises(tmp_path):
    from scrub.extractors.pdf import PdfExtractor
    from scrub.redactors.pdf import redact_pdf
    from scrub.types import Span

    pdfs = {p.name: p for p in make_all(tmp_path / "pdfs")}
    extraction = PdfExtractor().extract(pdfs["letter.pdf"])
    # A span pointing past the extracted text can't map to any WordBox;
    # writing output anyway would ship an unredacted "sanitized" PDF.
    bogus = Span(
        start=len(extraction.text) + 10,
        end=len(extraction.text) + 20,
        entity_type=EntityType.SSN,
        text="458-02-6841",
        confidence=1.0,
        source="regex",
    )
    with pytest.raises(RuntimeError, match="cannot locate span"):
        redact_pdf(pdfs["letter.pdf"], tmp_path / "out.pdf", [bogus], extraction, Config())


def test_custom_keywords(tmp_path):
    cfg = Config()
    cfg.custom_keywords = ["Project Nightjar"]
    cfg.enable_rampart = False  # keep it fast; custom tier is what's under test
    p = Pipeline.default(cfg)
    src = tmp_path / "codename.txt"
    src.write_text("Status update for Project Nightjar: on track.\n")
    spans = p.scan(src)
    assert any(s.entity_type == EntityType.CUSTOM for s in spans)
