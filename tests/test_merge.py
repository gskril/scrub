"""Pure-unit tests for the two-detector merge glue (no model needed)."""

from __future__ import annotations

from scrub.detectors.merge import mask_for_ml, merge_spans
from scrub.types import EntityType, Span


def _rx(start, end, etype=EntityType.SSN, text="x", conf=1.0) -> Span:
    return Span(start, end, etype, text, conf, "regex")


def _ml(start, end, etype=EntityType.GIVEN_NAME, text="x", conf=0.9) -> Span:
    return Span(start, end, etype, text, conf, "rampart")


# --------------------------------------------------------------- mask_for_ml


def test_mask_preserves_length() -> None:
    text = "SSN 458-02-6841 belongs to Maria."
    spans = [_rx(4, 15, text=text[4:15])]
    masked = mask_for_ml(text, spans)
    assert len(masked) == len(text)


def test_mask_replaces_only_span_and_keeps_offsets() -> None:
    text = "SSN 458-02-6841 belongs to Maria."
    spans = [_rx(4, 15, text=text[4:15])]
    masked = mask_for_ml(text, spans)
    # The digits are gone.
    assert "458-02-6841" not in masked
    # Everything outside the span is byte-for-byte identical.
    assert masked[:4] == text[:4]
    assert masked[15:] == text[15:]
    # "Maria" is still where it was — offsets stable for the ML pass.
    assert masked.index("Maria") == text.index("Maria")


def test_mask_no_spans_is_identity() -> None:
    text = "nothing to mask here"
    assert mask_for_ml(text, []) == text


def test_mask_multiple_spans() -> None:
    text = "a 111 b 222 c"
    spans = [_rx(2, 5, EntityType.PHONE), _rx(8, 11, EntityType.PHONE)]
    masked = mask_for_ml(text, spans)
    assert len(masked) == len(text)
    assert "111" not in masked and "222" not in masked
    assert masked[0:2] == "a " and masked[5:8] == " b " and masked[11:] == " c"


def test_mask_clamps_out_of_range() -> None:
    text = "short"
    # Defensive: a span past the end should not crash or change length.
    masked = mask_for_ml(text, [_rx(2, 999)])
    assert len(masked) == len(text)


# --------------------------------------------------------------- merge_spans


def test_regex_beats_ml_on_overlap() -> None:
    # Regex SSN overlaps an ML span that misclassified the same chars.
    regex = [_rx(0, 11, EntityType.SSN, conf=1.0)]
    ml = [_ml(0, 11, EntityType.PHONE, conf=0.95)]
    out = merge_spans(regex, ml)
    assert len(out) == 1
    assert out[0].source == "regex"
    assert out[0].entity_type == EntityType.SSN


def test_disjoint_spans_all_survive() -> None:
    regex = [_rx(0, 5, EntityType.SSN)]
    ml = [_ml(10, 15, EntityType.GIVEN_NAME)]
    out = merge_spans(regex, ml)
    assert len(out) == 2
    assert [s.start for s in out] == [0, 10]


def test_longest_ml_wins_on_overlap() -> None:
    short = _ml(2, 6, EntityType.GIVEN_NAME, conf=0.99)
    long = _ml(0, 10, EntityType.STREET_NAME, conf=0.60)
    out = merge_spans([], [short, long])
    assert len(out) == 1
    assert out[0] is long  # longest wins even at lower confidence


def test_highest_confidence_breaks_length_tie() -> None:
    a = _ml(0, 5, EntityType.GIVEN_NAME, conf=0.70)
    b = _ml(0, 5, EntityType.SURNAME, conf=0.95)
    out = merge_spans([], [a, b])
    assert len(out) == 1
    assert out[0] is b


def test_result_is_sorted_and_non_overlapping() -> None:
    regex = [_rx(20, 25, EntityType.EMAIL), _rx(0, 5, EntityType.SSN)]
    ml = [
        _ml(3, 8, EntityType.GIVEN_NAME),  # overlaps regex 0-5 -> dropped
        _ml(10, 15, EntityType.SURNAME),  # survives
        _ml(22, 30, EntityType.URL),  # overlaps regex 20-25 -> dropped
    ]
    out = merge_spans(regex, ml)
    starts = [s.start for s in out]
    assert starts == sorted(starts)
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start  # non-overlapping
    # regex hits both survived; the surviving ML gap span is present.
    assert (0, 5) in [(s.start, s.end) for s in out]
    assert (20, 25) in [(s.start, s.end) for s in out]
    assert (10, 15) in [(s.start, s.end) for s in out]


def test_empty_inputs() -> None:
    assert merge_spans([], []) == []


def test_regex_pre_empts_overlapping_longer_ml() -> None:
    # Even a longer ML span loses to a regex span it overlaps.
    regex = [_rx(5, 10, EntityType.SSN)]
    ml = [_ml(0, 20, EntityType.STREET_NAME, conf=0.99)]
    out = merge_spans(regex, ml)
    assert len(out) == 1
    assert out[0].source == "regex"
