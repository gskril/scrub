"""Real-model tests for RampartDetector.

The Rampart model is already cached locally at the pinned revision, so these
run actual ONNX inference (no network download at test time — snapshot_download
hits the cache). All PII is synthetic.
"""

from __future__ import annotations

import pytest

from scrub.config import RAMPART_REPO, RAMPART_REVISION
from scrub.detectors.rampart import RampartDetector
from scrub.types import EntityType


@pytest.fixture(scope="module")
def detector() -> RampartDetector:
    d = RampartDetector()
    d.warmup()
    return d


def _by_type(spans, etype):
    return [s for s in spans if s.entity_type == etype]


def _find(spans, text, etype):
    """Return the span of the given type whose source text matches, or None."""
    for s in spans:
        if s.entity_type == etype and s.text == text:
            return s
    return None


def test_smoke_sentence(detector: RampartDetector) -> None:
    text = "Maria Garcia lives at 123 Main St, Springfield, IL 62704."
    spans = detector.detect(text)

    # Names with correct char spans in the ORIGINAL text.
    maria = _find(spans, "Maria", EntityType.GIVEN_NAME)
    assert maria is not None
    assert (maria.start, maria.end) == (0, 5)
    assert text[maria.start : maria.end] == "Maria"

    garcia = _find(spans, "Garcia", EntityType.SURNAME)
    assert garcia is not None
    assert (garcia.start, garcia.end) == (6, 12)
    assert text[garcia.start : garcia.end] == "Garcia"

    # Address components present.
    assert _by_type(spans, EntityType.STREET_NAME)
    assert _by_type(spans, EntityType.CITY)
    assert _by_type(spans, EntityType.STATE)
    assert _by_type(spans, EntityType.ZIP_CODE)

    city = _by_type(spans, EntityType.CITY)[0]
    assert text[city.start : city.end] == "Springfield"
    zip_ = _by_type(spans, EntityType.ZIP_CODE)[0]
    assert text[zip_.start : zip_.end] == "62704"

    # Every span's stored text matches its offsets and carries the right source.
    for s in spans:
        assert s.text == text[s.start : s.end]
        assert s.source == "rampart"


def test_long_document_chunking(detector: RampartDetector) -> None:
    """A document longer than the 512-position window: entities near the end
    must still be found with correct ABSOLUTE offsets (chunking regression)."""
    filler = "John Smith works in Boston. " * 120
    tail = "Contact Fernanda Villalobos at 987 Sunset Boulevard, Portland, OR 97201."
    doc = filler + tail

    # Sanity: the document really is longer than one model window.
    enc = detector._tokenizer.encode(doc)  # type: ignore[union-attr]
    assert len(enc.ids) > 512

    spans = detector.detect(doc)

    fernanda = _find(spans, "Fernanda", EntityType.GIVEN_NAME)
    assert fernanda is not None
    assert fernanda.start == doc.index("Fernanda")
    assert doc[fernanda.start : fernanda.end] == "Fernanda"

    villalobos = _find(spans, "Villalobos", EntityType.SURNAME)
    assert villalobos is not None
    assert villalobos.start == doc.index("Villalobos")

    zip_ = _by_type(spans, EntityType.ZIP_CODE)
    assert any(doc[s.start : s.end] == "97201" for s in zip_)


def test_case_preservation(detector: RampartDetector) -> None:
    """Model is uncased, but offsets come from the tokenizer, so spans map to
    the ORIGINAL cased text."""
    text = "MARIA GARCIA lives at 123 Main St, Springfield, IL 62704."
    spans = detector.detect(text)

    maria = _find(spans, "MARIA", EntityType.GIVEN_NAME)
    assert maria is not None
    assert (maria.start, maria.end) == (0, 5)
    assert text[maria.start : maria.end] == "MARIA"

    garcia = _find(spans, "GARCIA", EntityType.SURNAME)
    assert garcia is not None
    assert text[garcia.start : garcia.end] == "GARCIA"


def test_no_pii_text(detector: RampartDetector) -> None:
    text = "The quick brown fox jumps over the lazy dog. It was a fine day."
    spans = detector.detect(text)
    assert spans == []


def test_empty_and_whitespace(detector: RampartDetector) -> None:
    assert detector.detect("") == []
    assert detector.detect("   \n\t ") == []


def test_pinned_revision_constants() -> None:
    # Guard against silent model drift.
    assert RAMPART_REPO == "nationaldesignstudio/rampart"
    assert RAMPART_REVISION == "b1993e4e68b082835b80ffc65acc03325ea2e501"
