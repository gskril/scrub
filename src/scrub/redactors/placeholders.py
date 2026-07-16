"""Shared placeholder assignment used by every redactor.

Placeholder format: `[{ENTITY_TYPE}_{n}]`, n starting at 1 per entity type in
order of first appearance in the file. The same source text (case-insensitive)
reuses its placeholder within a file so the model can follow references.
"""

from __future__ import annotations

from ..types import Span


def assign_placeholders(spans: list[Span]) -> list[str]:
    """Return one placeholder per input span, aligned by index."""
    counters: dict[str, int] = {}
    seen: dict[tuple[str, str], str] = {}
    order = sorted(range(len(spans)), key=lambda i: spans[i].start)
    result: list[str] = [""] * len(spans)
    for i in order:
        span = spans[i]
        key = (span.entity_type.value, span.text.casefold())
        placeholder = seen.get(key)
        if placeholder is None:
            counters[span.entity_type.value] = counters.get(span.entity_type.value, 0) + 1
            placeholder = f"[{span.entity_type.value}_{counters[span.entity_type.value]}]"
            seen[key] = placeholder
        result[i] = placeholder
    return result
