"""Two-detector merge glue: mask regex hits before the ML pass, then union
the regex and ML spans into one non-overlapping, sorted result.

`merge_spans` supersedes `pipeline.resolve_overlaps` for the regex+Rampart
case; it is written to be a drop-in (pure function over Span lists).
"""

from __future__ import annotations

from ..types import Span

# Sentinel used to blank out regex hits before the model sees the text. A
# private-use / block character the tokenizer won't split into meaningful
# subwords, so Rampart can't re-derive or mangle an already-caught entity.
_SENTINEL = "█"  # █


def mask_for_ml(text: str, regex_spans: list[Span]) -> str:
    """Replace each regex-detected span with a same-length run of sentinel
    characters. Text length and every character offset outside the masked
    ranges are preserved exactly, so ML spans map back onto the original text.
    """
    if not regex_spans:
        return text
    chars = list(text)
    n = len(chars)
    for span in regex_spans:
        start = max(0, span.start)
        end = min(n, span.end)
        for i in range(start, end):
            chars[i] = _SENTINEL
    return "".join(chars)


def merge_spans(regex_spans: list[Span], ml_spans: list[Span]) -> list[Span]:
    """Union of regex and ML spans, resolved to a sorted, non-overlapping list.

    Precedence on overlap:
      1. a validated regex span (conf 1.0) always beats an ML span;
      2. among competing spans within the same tier, the LONGEST wins, then
         the highest confidence.

    Implemented as a greedy accept over candidates ordered by
    (regex-first, longest, highest-confidence): a candidate is accepted only if
    it does not overlap anything already accepted, so a regex span — considered
    before any ML span — pre-empts every ML span it touches.
    """

    def sort_key(s: Span) -> tuple[int, int, float, int]:
        is_regex = 0 if s.source == "regex" else 1  # regex tier first
        return (is_regex, -len(s), -s.confidence, s.start)

    candidates = sorted([*regex_spans, *ml_spans], key=sort_key)
    accepted: list[Span] = []
    for cand in candidates:
        if any(cand.overlaps(a) for a in accepted):
            continue
        accepted.append(cand)
    accepted.sort(key=lambda s: (s.start, s.end))
    return accepted
