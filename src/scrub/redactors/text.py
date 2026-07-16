"""Text redactor: apply a resolved span list to text, producing placeholders
and a leak-free report.

Assumes `spans` is already sorted by start and overlap-free (that's the
pipeline's/merge's job, not this module's). Every span is reported; only
spans whose entity_type is NOT in `config.public_types` are actually
replaced in the returned text.
"""

from __future__ import annotations

from ..config import Config
from ..types import ReportEntity, Span
from .placeholders import assign_placeholders


def redact_text(
    text: str, spans: list[Span], config: Config
) -> tuple[str, list[ReportEntity]]:
    """Replace redactable spans with stable `[{TYPE}_{n}]` placeholders.

    Numbering comes from the shared `assign_placeholders` helper (same rules
    for text and PDF redaction) — this holds for both redacted and
    public/passthrough entity types, so the report stays consistent even
    though public types aren't substituted into the text.

    Returns (redacted_text, entities) where `entities` covers every span
    (public and redacted alike) as ReportEntity — never the raw matched
    text, so report.json built from this is safe to write to disk.
    """
    placeholders = assign_placeholders(spans)
    entities: list[ReportEntity] = []
    pieces: list[str] = []
    cursor = 0

    for span, placeholder in zip(spans, placeholders):
        if span.entity_type not in config.public_types:
            pieces.append(text[cursor : span.start])
            pieces.append(placeholder)
            cursor = span.end
        # public types: leave the source text in place, cursor untouched —
        # it gets carried through by the next span's (or the trailing) slice.

        entities.append(
            ReportEntity(
                placeholder=placeholder,
                entity_type=span.entity_type,
                start=span.start,
                end=span.end,
                confidence=span.confidence,
                source=span.source,
            )
        )

    pieces.append(text[cursor:])
    return "".join(pieces), entities
