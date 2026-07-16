"""Custom keyword detector: case-insensitive literal matches of user-supplied
terms (case names, project codenames) from config.custom_keywords.

Runs in the deterministic tier: its hits are masked before the ML pass and
win overlaps like regex hits (confidence 1.0).
"""

from __future__ import annotations

import re

from ..types import EntityType, Span


class CustomKeywordDetector:
    name = "custom"

    def __init__(self, keywords: list[str]) -> None:
        self._patterns = [
            re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)", re.IGNORECASE)
            for kw in keywords
            if kw.strip()
        ]

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for pattern in self._patterns:
            for m in pattern.finditer(text):
                spans.append(
                    Span(
                        start=m.start(),
                        end=m.end(),
                        entity_type=EntityType.CUSTOM,
                        text=m.group(),
                        confidence=1.0,
                        source="custom",
                    )
                )
        return spans
