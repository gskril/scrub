"""Core data types shared by every scrub component.

These are the contracts between extractors, detectors, redactors, the daemon,
and the hook. Keep them dependency-light: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable


class EntityType(StrEnum):
    """Union of the deterministic layer's types and Rampart's 17 categories.

    Detector implementations must map their internal labels onto these.
    """

    # Deterministic (regex + validator) layer
    SSN = "SSN"
    ITIN = "ITIN"
    EIN = "EIN"
    CREDIT_CARD = "CREDIT_CARD"
    ROUTING_NUMBER = "ROUTING_NUMBER"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    IP_ADDRESS = "IP_ADDRESS"
    MAC_ADDRESS = "MAC_ADDRESS"
    API_KEY = "API_KEY"
    PRIVATE_KEY = "PRIVATE_KEY"
    JWT = "JWT"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    DEVICE_ID = "DEVICE_ID"
    CREDENTIAL = "CREDENTIAL"
    MEDICAL_ID = "MEDICAL_ID"
    HEALTH_INFORMATION = "HEALTH_INFORMATION"
    FINANCIAL_INFORMATION = "FINANCIAL_INFORMATION"
    # Shared between layers (regex catches structured forms, Rampart contextual ones)
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    URL = "URL"
    # Rampart ML layer
    GIVEN_NAME = "GIVEN_NAME"
    SURNAME = "SURNAME"
    TAX_ID = "TAX_ID"
    GOVERNMENT_ID = "GOVERNMENT_ID"
    PASSPORT = "PASSPORT"
    DRIVERS_LICENSE = "DRIVERS_LICENSE"
    BUILDING_NUMBER = "BUILDING_NUMBER"
    STREET_NAME = "STREET_NAME"
    SECONDARY_ADDRESS = "SECONDARY_ADDRESS"
    CITY = "CITY"
    STATE = "STATE"
    ZIP_CODE = "ZIP_CODE"
    # Config-driven custom keyword list
    CUSTOM = "CUSTOM"


# Entity types that are kept (not redacted) unless config says otherwise.
# Mirrors Rampart's default public set; everything else is redacted by default.
DEFAULT_PUBLIC_TYPES: frozenset[EntityType] = frozenset(
    {EntityType.CITY, EntityType.STATE, EntityType.ZIP_CODE, EntityType.URL}
)


@dataclass(slots=True)
class Span:
    """A detected entity as character offsets into an Extraction's text."""

    start: int
    end: int  # exclusive
    entity_type: EntityType
    text: str  # the matched source text (never serialized into reports)
    confidence: float  # 1.0 for validated regex hits
    source: str  # "regex" | "rampart" | "custom"

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class WordBox:
    """A word's location in a paginated/visual medium (PDF, image)."""

    start: int  # char offset range this word occupies in Extraction.text
    end: int
    page: int  # 0-based
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(slots=True)
class Extraction:
    """Normalized output of any extractor: text plus optional coordinates.

    For plain text media, `words` is empty and character offsets ARE the
    coordinates. For visual media, `words` maps offset ranges to boxes so the
    redactor can black out pixels / apply PDF redactions.
    """

    text: str
    kind: str  # "text" | "pdf" | "image"
    words: list[WordBox] = field(default_factory=list)


@dataclass(slots=True)
class ReportEntity:
    """One redaction, as recorded in report.json. Contains NO raw PII."""

    placeholder: str  # e.g. "[GIVEN_NAME_1]"
    entity_type: EntityType
    start: int
    end: int
    confidence: float
    source: str
    page: int | None = None


@dataclass(slots=True)
class RedactionResult:
    """What the pipeline returns for one input file."""

    original_path: Path
    redacted_path: Path | None  # None when nothing was found (passthrough)
    found: int
    entities: list[ReportEntity]
    report_path: Path | None = None
    cache_hit: bool = False


@runtime_checkable
class Detector(Protocol):
    """detect() returns spans over the given text. Implementations must be
    safe to call repeatedly from a long-lived daemon (no per-call model load).
    """

    name: str

    def detect(self, text: str) -> list[Span]: ...


@runtime_checkable
class Extractor(Protocol):
    """Turn a file into normalized text (+ coordinates for visual media)."""

    def extract(self, path: Path) -> Extraction: ...
