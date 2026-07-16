"""Pipeline: router -> extractor -> detectors -> resolve_overlaps -> redactor
-> emitter.

Phase 0+1 ships this with regex-only detection and a text-only
extractor/redactor registry. Later phases register more entries (pdf/image
extractors and redactors, keyed by `Extraction.kind`) — the registries below
exist so that's additive, not a rewrite.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import orjson

from .config import Config, cache_dir
from .detectors.custom import CustomKeywordDetector
from .detectors.merge import DETERMINISTIC_SOURCES, mask_for_ml, merge_spans
from .detectors.regex_rules import RegexDetector
from .extractors.text import TextExtractor
from .redactors.placeholders import assign_placeholders
from .redactors.text import redact_text
from .router import SkipFile, classify, should_skip
from .types import Detector, Extraction, Extractor, RedactionResult, ReportEntity, Span

RedactorFn = Callable[[str, list[Span], Config], tuple[str, list[ReportEntity]]]

# Detectors in this set run first on the raw text; their hits are masked to
# sentinels before ML detectors run, and they win overlaps in the merge
# (merge.DETERMINISTIC_SOURCES must agree with this set).
_DETERMINISTIC_DETECTORS = DETERMINISTIC_SOURCES


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Merge spans into a non-overlapping, start-sorted list.

    Phase 0+1 placeholder rule (all spans here come from regex detectors, so
    "regex wins" is moot today): longest match wins, then highest
    confidence, ties broken by input order. Phase 2's `detectors/merge.py`
    supersedes this once Rampart spans enter the mix — that module documents
    itself as the drop-in replacement.
    """

    def sort_key(item: tuple[int, Span]) -> tuple[int, int, float, int]:
        idx, s = item
        is_regex = 0 if s.source == "regex" else 1
        return (is_regex, -len(s), -s.confidence, idx)

    ordered = sorted(enumerate(spans), key=sort_key)
    chosen: list[Span] = []
    for _, span in ordered:
        if any(span.overlaps(c) for c in chosen):
            continue
        chosen.append(span)
    chosen.sort(key=lambda s: s.start)
    return chosen


def _propagate_values(text: str, spans: list[Span]) -> list[Span]:
    """Value propagation: if a value was detected anywhere in the document,
    redact ALL its other literal occurrences too.

    Rampart's per-occurrence recall is context-dependent — it can tag
    "Garcia" in a salutation but miss the identical string in an address
    block two lines up, leaving `[GIVEN_NAME_1] Garcia` in the output. A
    detected value is sensitive everywhere it appears, so this deterministic
    post-pass closes that leak class. Word-boundary anchored,
    case-insensitive, values shorter than 3 chars skipped (too collision-
    prone, e.g. state codes).
    """
    occupied = sorted((s.start, s.end) for s in spans)

    def overlaps_existing(start: int, end: int) -> bool:
        return any(start < e and s < end for s, e in occupied)

    extra: list[Span] = []
    seen: set[str] = set()
    for span in spans:
        value = span.text
        key = value.casefold()
        # Only propagate reasonably confident detections — propagation
        # amplifies its source, so a borderline ML false positive must not
        # spread. (Real names in awkward contexts score ~0.6; code-token
        # false positives are filtered upstream by the capitalization rule.)
        if span.confidence < 0.6 or len(value.strip()) < 3 or key in seen:
            continue
        seen.add(key)
        pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.IGNORECASE)
        for m in pattern.finditer(text):
            if m.start() == span.start or overlaps_existing(m.start(), m.end()):
                continue
            extra.append(
                Span(
                    start=m.start(),
                    end=m.end(),
                    entity_type=span.entity_type,
                    text=m.group(),
                    confidence=span.confidence,
                    source="propagated",
                )
            )
            occupied.append((m.start(), m.end()))
    return extra


def _pdf_has_metadata(path: Path) -> bool:
    """True if the PDF carries any document metadata or XMP. Metadata can
    hold PII (author, title, creator paths) even when the body is clean, so
    a metadata-bearing PDF must never pass through unscrubbed."""
    import pymupdf

    try:
        doc = pymupdf.open(path)
    except Exception:  # noqa: BLE001 — unreadable PDFs are handled upstream
        return False
    try:
        # "format" and "encryption" are intrinsic document properties PyMuPDF
        # always reports, not author-supplied metadata — never PII.
        if any(
            (v or "").strip()
            for k, v in (doc.metadata or {}).items()
            if k not in ("format", "encryption")
        ):
            return True
        try:
            return bool((doc.get_xml_metadata() or "").strip())
        except Exception:  # noqa: BLE001
            return False
    finally:
        doc.close()


def _entity_to_dict(e: ReportEntity) -> dict:
    return {
        "placeholder": e.placeholder,
        "entity_type": e.entity_type.value,
        "start": e.start,
        "end": e.end,
        "confidence": e.confidence,
        "source": e.source,
        "page": e.page,
    }


class Pipeline:
    """Composes route -> extract -> detect -> resolve -> redact -> emit."""

    def __init__(
        self,
        config: Config,
        detectors: list[Detector] | None = None,
        extractors: dict[str, Extractor] | None = None,
        redactors: dict[str, RedactorFn] | None = None,
    ) -> None:
        self.config = config
        self.detectors: list[Detector] = detectors if detectors is not None else [RegexDetector()]
        # Extraction.kind -> Extractor. Phase 3 adds "pdf"; a future phase "image".
        self.extractors: dict[str, Extractor] = extractors or {"text": TextExtractor()}
        # Extraction.kind -> redact function. Phase 3 adds "pdf".
        self.redactors: dict[str, RedactorFn] = redactors or {"text": redact_text}

    def register_extractor(self, kind: str, extractor: Extractor) -> None:
        self.extractors[kind] = extractor

    def register_redactor(self, kind: str, redactor: RedactorFn) -> None:
        self.redactors[kind] = redactor

    def register_detector(self, detector: Detector) -> None:
        self.detectors.append(detector)

    @classmethod
    def default(cls, config: Config | None = None) -> "Pipeline":
        """Fully-wired pipeline per config: regex + custom keywords + Rampart
        detectors, text + pdf extractors/redactors. This is what the CLI,
        daemon, and hook use; the bare constructor stays regex/text-only for
        unit-level composition.
        """
        from .detectors.rampart import RampartDetector
        from .extractors.pdf import PdfExtractor
        from .redactors.pdf import redact_pdf  # noqa: F401  (used via kind dispatch)

        config = config or Config.load()
        detectors: list[Detector] = []
        if config.enable_regex:
            detectors.append(RegexDetector())
        if config.custom_keywords:
            detectors.append(CustomKeywordDetector(config.custom_keywords))
        if config.enable_rampart:
            detectors.append(RampartDetector(config))
        pipeline = cls(config, detectors=detectors)
        pipeline.register_extractor("pdf", PdfExtractor())
        return pipeline

    # -- internal: shared by redact_file() and scan() -----------------------

    def _route_and_extract(self, path: Path) -> Extraction | None:
        """Returns None if the file is skipped (unreadable/skip-glob/too big/
        no extractor registered for its kind/undecodable)."""
        skip, _reason = should_skip(path, self.config)
        if skip:
            return None
        try:
            kind = classify(path)
        except SkipFile:
            return None
        extractor = self.extractors.get(kind)
        if extractor is None:
            return None
        try:
            return extractor.extract(path)
        except SkipFile:
            return None

    def _detect(self, extraction: Extraction) -> list[Span]:
        """Two-stage detection: deterministic detectors (regex, custom
        keywords) run on the raw text first; their hits are masked to
        sentinels so ML detectors can't re-derive or mangle them (Rampart
        misclassifies e.g. SSNs as PHONE at low confidence — the regex layer
        must own those). The tiers are then merged, deterministic wins.
        """
        det_spans: list[Span] = []
        ml_detectors: list[Detector] = []
        for detector in self.detectors:
            if detector.name in _DETERMINISTIC_DETECTORS:
                det_spans.extend(detector.detect(extraction.text))
            else:
                ml_detectors.append(detector)
        det_spans = resolve_overlaps(det_spans)

        if not ml_detectors:
            return det_spans

        masked = mask_for_ml(extraction.text, det_spans)
        ml_spans: list[Span] = []
        for detector in ml_detectors:
            ml_spans.extend(detector.detect(masked))
        # ML spans carry offsets into the masked text, whose length and
        # non-masked offsets are identical to the original — but their .text
        # must be re-sliced from the original so reports/placeholders don't
        # contain sentinel characters.
        for s in ml_spans:
            s.text = extraction.text[s.start : s.end]
        merged = merge_spans(det_spans, ml_spans)
        return merge_spans(merged, _propagate_values(extraction.text, merged))

    def _passthrough_entities(self, spans: list[Span]) -> list[ReportEntity]:
        """ReportEntities for a passthrough result (only public-type spans
        were found, nothing written) — keeps scan/report output consistent."""
        placeholders = assign_placeholders(spans)
        return [
            ReportEntity(
                placeholder=ph,
                entity_type=s.entity_type,
                start=s.start,
                end=s.end,
                confidence=s.confidence,
                source=s.source,
            )
            for s, ph in zip(spans, placeholders)
        ]

    # -- public API -----------------------------------------------------

    def scan(self, path: Path) -> list[Span]:
        """Detect-only: route, extract, detect, resolve. No file is written."""
        path = Path(path)
        extraction = self._route_and_extract(path)
        if extraction is None:
            return []
        return self._detect(extraction)

    def redact_file(self, path: Path) -> RedactionResult:
        path = Path(path)
        extraction = self._route_and_extract(path)
        if extraction is None:
            return RedactionResult(original_path=path, redacted_path=None, found=0, entities=[])

        spans = self._detect(extraction)
        found = sum(1 for s in spans if s.entity_type not in self.config.public_types)

        # PDFs with document metadata/XMP get a scrubbed copy even when the
        # body is clean — metadata (author, title, creator) can carry PII
        # that body detection never sees.
        needs_output = found > 0 or (
            extraction.kind == "pdf" and _pdf_has_metadata(path)
        )
        if not needs_output:
            entities = self._passthrough_entities(spans)
            return RedactionResult(
                original_path=path, redacted_path=None, found=0, entities=entities
            )

        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        out_dir = cache_dir() / "redacted"
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lstrip(".") or "txt"
        redacted_path = out_dir / f"{content_hash}.redacted.{ext}"

        if extraction.kind == "pdf":
            from .redactors.pdf import redact_pdf

            entities = redact_pdf(path, redacted_path, spans, extraction, self.config)
        else:
            redactor = self.redactors.get(extraction.kind)
            if redactor is None:
                return RedactionResult(
                    original_path=path, redacted_path=None, found=0, entities=[]
                )
            redacted_text, entities = redactor(extraction.text, spans, self.config)
            redacted_path.write_text(redacted_text, encoding="utf-8")

        report_path = out_dir / f"{content_hash}.report.json"
        report = {
            "original_path": str(path),
            "content_hash": content_hash,
            "found": found,
            "entities": [_entity_to_dict(e) for e in entities],
        }
        report_path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))

        return RedactionResult(
            original_path=path,
            redacted_path=redacted_path,
            found=found,
            entities=entities,
            report_path=report_path,
        )
