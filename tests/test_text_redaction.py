"""Tests for scrub.redactors.text (unit level) and scrub.pipeline.Pipeline
(end-to-end on synthetic .txt / .env fixtures).

All PII values are synthetic; SSN/card/routing reuse the plan's designated
known-fake fixtures where convenient.
"""

from __future__ import annotations

import json

import orjson
import pytest

from scrub.config import Config
from scrub.detectors.regex_rules import RegexDetector
from scrub.pipeline import Pipeline, resolve_overlaps
from scrub.redactors.text import redact_text
from scrub.types import EntityType, Span


def make_span(entity_type, start, end, text, confidence=1.0, source="regex"):
    return Span(
        start=start, end=end, entity_type=entity_type, text=text, confidence=confidence, source=source
    )


# --------------------------------------------------------------------------
# redact_text: placeholder application + numbering + public passthrough
# --------------------------------------------------------------------------


def test_single_span_replaced_with_placeholder():
    text = "call 415-555-0100 now"
    span = make_span(EntityType.PHONE, 5, 17, "415-555-0100")
    redacted, entities = redact_text(text, [span], Config())
    assert redacted == "call [PHONE_1] now"
    assert len(entities) == 1
    assert entities[0].placeholder == "[PHONE_1]"
    assert entities[0].entity_type == EntityType.PHONE
    assert entities[0].start == 5 and entities[0].end == 17


def test_numbering_increments_per_first_appearance():
    text = "a@example.com then b@example.com then a@example.com again"
    d = RegexDetector()
    spans = d.detect(text)
    redacted, entities = redact_text(text, spans, Config())
    # a@example.com appears 1st and 3rd -> both [EMAIL_1]; b@example.com -> [EMAIL_2]
    assert redacted == "[EMAIL_1] then [EMAIL_2] then [EMAIL_1] again"


def test_same_text_case_insensitive_reuses_placeholder():
    text = "AKIAABCDEFGHIJKLMNOP then akiaabcdefghijklmnop"
    span1 = make_span(EntityType.API_KEY, 0, 20, "AKIAABCDEFGHIJKLMNOP")
    span2 = make_span(EntityType.API_KEY, 26, 46, "akiaabcdefghijklmnop")
    redacted, entities = redact_text(text, [span1, span2], Config())
    assert redacted == "[API_KEY_1] then [API_KEY_1]"
    assert entities[0].placeholder == entities[1].placeholder == "[API_KEY_1]"


def test_different_entity_types_get_independent_counters():
    text = "email a@example.com and ssn 458-02-6841"
    d = RegexDetector()
    spans = d.detect(text)
    redacted, entities = redact_text(text, spans, Config())
    assert redacted == "email [EMAIL_1] and ssn [SSN_1]"


def test_public_types_are_reported_but_not_redacted():
    text = "see https://example.com/path for details"
    span = make_span(EntityType.URL, 4, 30, "https://example.com/path")
    config = Config()  # URL is in DEFAULT_PUBLIC_TYPES
    redacted, entities = redact_text(text, [span], config)
    assert redacted == text  # unchanged — URL is public
    assert len(entities) == 1
    assert entities[0].entity_type == EntityType.URL
    assert entities[0].placeholder == "[URL_1]"  # still recorded, just not applied


def test_mixed_public_and_redactable_spans():
    text = "contact a@example.com or visit https://example.com"
    d = RegexDetector()
    spans = d.detect(text)
    config = Config()
    redacted, entities = redact_text(text, spans, config)
    assert redacted == "contact [EMAIL_1] or visit https://example.com"
    assert {e.entity_type for e in entities} == {EntityType.EMAIL, EntityType.URL}


def test_no_spans_returns_original_text_unchanged():
    text = "nothing sensitive here at all"
    redacted, entities = redact_text(text, [], Config())
    assert redacted == text
    assert entities == []


def test_report_entities_never_contain_raw_text():
    text = "ssn 458-02-6841 email a@example.com"
    d = RegexDetector()
    spans = d.detect(text)
    _, entities = redact_text(text, spans, Config())
    for e in entities:
        assert not hasattr(e, "text")
        # placeholder is a synthetic tag, never the raw matched substring
        assert "458-02-6841" not in e.placeholder
        assert "a@example.com" not in e.placeholder


# --------------------------------------------------------------------------
# resolve_overlaps (pipeline-level merge placeholder for Phase 0+1)
# --------------------------------------------------------------------------


def test_resolve_overlaps_prefers_longest():
    a = make_span(EntityType.SSN, 0, 11, "458-02-6841")
    b = make_span(EntityType.PHONE, 0, 3, "458")
    result = resolve_overlaps([a, b])
    assert result == [a]


def test_resolve_overlaps_keeps_disjoint_spans_sorted():
    a = make_span(EntityType.EMAIL, 10, 20, "x")
    b = make_span(EntityType.PHONE, 0, 5, "y")
    result = resolve_overlaps([a, b])
    assert [s.start for s in result] == [0, 10]


# --------------------------------------------------------------------------
# Pipeline end-to-end: synthetic .txt and .env with mixed PII
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRUB_CACHE_DIR", str(tmp_path / "cache"))


def test_pipeline_redacts_txt_file_end_to_end(tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text(
        "Customer: Maria Garcia\n"
        "SSN: 458-02-6841\n"
        "Email: maria.garcia@example.com\n"
        "Phone: (415) 555-2671\n"
        "Card: 4111 1111 1111 1111\n"
    )
    pipeline = Pipeline(Config())
    result = pipeline.redact_file(src)

    assert result.found > 0
    assert result.redacted_path is not None
    assert result.redacted_path.exists()
    assert result.report_path is not None
    assert result.report_path.exists()

    redacted_text = result.redacted_path.read_text()
    assert "458-02-6841" not in redacted_text
    assert "maria.garcia@example.com" not in redacted_text
    assert "4111 1111 1111 1111" not in redacted_text
    assert "[SSN_1]" in redacted_text
    assert "[EMAIL_1]" in redacted_text
    assert "[CREDIT_CARD_1]" in redacted_text

    report = orjson.loads(result.report_path.read_bytes())
    assert report["found"] == result.found
    raw_blob = json.dumps(report)
    assert "458-02-6841" not in raw_blob
    assert "maria.garcia@example.com" not in raw_blob
    assert "4111 1111 1111 1111" not in raw_blob


def test_pipeline_redacts_env_file_end_to_end(tmp_path):
    src = tmp_path / "secrets.env"
    src.write_text(
        "AWS_ACCESS_KEY=AKIAABCDEFGHIJKLMNOP\n"
        "STRIPE_KEY=sk_live_bbbbbbbbbbbbbbbbbbbbbbbb\n"
        "SUPPORT_EMAIL=support@example.com\n"
        "DEBUG=true\n"
    )
    pipeline = Pipeline(Config())
    result = pipeline.redact_file(src)

    assert result.found > 0
    redacted_text = result.redacted_path.read_text()
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted_text
    assert "sk_live_bbbbbbbbbbbbbbbbbbbbbbbb" not in redacted_text
    assert "support@example.com" not in redacted_text
    assert "DEBUG=true" in redacted_text  # non-PII lines pass through untouched
    assert "[API_KEY_1]" in redacted_text
    assert "[API_KEY_2]" in redacted_text
    assert "[EMAIL_1]" in redacted_text


def test_pipeline_passthrough_when_nothing_found(tmp_path):
    src = tmp_path / "clean.txt"
    src.write_text("Just a friendly note with no PII whatsoever.\n")
    pipeline = Pipeline(Config())
    result = pipeline.redact_file(src)
    assert result.found == 0
    assert result.redacted_path is None
    assert result.report_path is None


def test_pipeline_skips_files_over_size_cap(tmp_path):
    src = tmp_path / "big.txt"
    src.write_text("SSN 458-02-6841 " * 100)
    config = Config(max_file_bytes=10)
    pipeline = Pipeline(config)
    result = pipeline.redact_file(src)
    assert result.found == 0
    assert result.redacted_path is None


def test_pipeline_skips_binary_file(tmp_path):
    src = tmp_path / "blob.bin"
    src.write_bytes(bytes(range(256)) * 4)
    pipeline = Pipeline(Config())
    result = pipeline.redact_file(src)
    assert result.found == 0
    assert result.redacted_path is None


def test_pipeline_scan_detects_without_writing(tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("SSN: 458-02-6841\nEmail: a@example.com\n")
    pipeline = Pipeline(Config())
    spans = pipeline.scan(src)
    assert {s.entity_type for s in spans} == {EntityType.SSN, EntityType.EMAIL}
    # scan must not create any output files
    cache_redacted_dir = tmp_path / "cache" / "redacted"
    assert not cache_redacted_dir.exists()


def test_pipeline_public_type_only_file_is_passthrough(tmp_path):
    # Only a URL (a DEFAULT_PUBLIC_TYPES member) — nothing redactable, so no
    # file should be written even though something was detected.
    src = tmp_path / "link.txt"
    src.write_text("See https://example.com/docs for more info.\n")
    pipeline = Pipeline(Config())
    result = pipeline.redact_file(src)
    assert result.found == 0
    assert result.redacted_path is None


def test_pipeline_content_hash_deterministic_filename(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("ssn 458-02-6841\n")
    pipeline = Pipeline(Config())
    result1 = pipeline.redact_file(src)
    result2 = pipeline.redact_file(src)
    assert result1.redacted_path == result2.redacted_path
