"""Round-trip / no-leak assertion (PLAN.md Sec 9): redact every golden-corpus
document (`eval/corpus.py`), then re-scan the output and assert no
ground-truth PII survives -- not in the redacted text / re-extracted PDF
text, not in report.json, not in the redacted PDF's metadata/XMP.

Two tiers, mirroring the release-blocking split ARCHITECTURE.md/PLAN.md draw
between the deterministic and ML detector layers (see also
`eval/run_eval.py`, which runs the same underlying check as a reporting
harness rather than a pytest gate):

  - Structured/regex+validator PII (SSN, EIN, CREDIT_CARD, ROUTING_NUMBER,
    BANK_ACCOUNT, API_KEY, PRIVATE_KEY, JWT) is release-blocking per PLAN.md
    Sec 9: ANY leak fails this test, hard, no tolerance.
  - Names/addresses/ITIN come from the Rampart ML layer, which PLAN.md Secs
    9-10 are explicit is alpha and NOT claimed to be 100% -- and
    `eval/run_eval.py`'s own measured recall on this corpus is ~84-100%, not
    100%. Hard-failing this file on every such miss would make it fail
    permanently on expected, known model behavior rather than catching real
    regressions. So this file still performs the exact same round-trip
    re-scan for the ML layer and prints every miss (a real drop is visible
    immediately in CI output), but only fails if recall drops below a floor
    well under the measured baseline -- i.e. it's a regression guard (model
    fails to load, a masking bug blanks the document, ...), not a recall
    benchmark. `eval/run_eval.py` is the place that tracks the real number.

Runs the whole corpus through ONE shared Pipeline (module-scoped fixture) and
redacts each document exactly once (module-scoped `redacted_results`), so
this stays fast despite ~115 ground-truth values across 15 documents.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from corpus import CorpusDoc, RELEASE_BLOCKING_TYPES, build_corpus  # noqa: E402

from scrub.config import Config
from scrub.pipeline import Pipeline

# Regression floor for the ML layer (names/addresses/ITIN). Measured recall
# on this corpus (see eval/run_eval.py output) is ~84-100%; 0.60 leaves real
# breakage (a broken model load, a masking bug that blanks the document,
# thresholds misconfigured to reject everything) plenty of room to be caught
# while not being sensitive to Rampart's normal per-sentence variance.
_ML_RECALL_FLOOR = 0.60


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> list[CorpusDoc]:
    out = tmp_path_factory.mktemp("no_leak_corpus")
    return build_corpus(out)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    # cache_dir() (scrub.config) re-reads this env var on every call, so
    # pointing it at a throwaway dir is enough to keep this test off the
    # real ~/.cache/scrub -- no monkeypatching needed.
    cache = tmp_path_factory.mktemp("no_leak_cache")
    saved = os.environ.get("SCRUB_CACHE_DIR")
    os.environ["SCRUB_CACHE_DIR"] = str(cache)
    try:
        yield Pipeline.default(Config())
    finally:
        if saved is None:
            os.environ.pop("SCRUB_CACHE_DIR", None)
        else:
            os.environ["SCRUB_CACHE_DIR"] = saved


@pytest.fixture(scope="module")
def redacted_results(pipeline, corpus) -> dict[str, object]:
    """Redact every ground-truth-bearing corpus doc exactly once; every
    assertion below shares this instead of re-running the pipeline per value."""
    results = {}
    for doc in corpus:
        if doc.is_control:
            continue
        results[doc.name] = pipeline.redact_file(doc.path)
    return results


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _pdf_sources(path: Path) -> list[str]:
    doc = pymupdf.open(path)
    try:
        text = "\n".join(page.get_text() for page in doc)
        meta = json.dumps(doc.metadata)
        xml = doc.get_xml_metadata() or ""
    finally:
        doc.close()
    return [text, meta, xml]


def _sources_for(doc: CorpusDoc, result) -> list[str]:
    """Every place downstream PII could still be hiding after redact_file().
    `result.redacted_path` is None only when nothing was found -- in that
    case a Read hook shows the ORIGINAL file, so that's what gets scanned."""
    target = result.redacted_path or doc.path
    sources: list[str] = (
        _pdf_sources(target)
        if doc.kind == "pdf"
        else [target.read_text(encoding="utf-8", errors="replace")]
    )
    if result.report_path is not None:
        sources.append(result.report_path.read_text(encoding="utf-8", errors="replace"))
    return sources


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_release_blocking_types_never_leak(corpus, redacted_results):
    """Hard gate (PLAN.md Sec 9): SSN/EIN/CREDIT_CARD/ROUTING_NUMBER/
    BANK_ACCOUNT/API_KEY/PRIVATE_KEY/JWT must never survive redaction in the
    redacted text/PDF text, report.json, or redacted PDF metadata/XMP."""
    leaks: list[str] = []
    checked = 0
    for doc in corpus:
        if doc.is_control:
            continue
        result = redacted_results[doc.name]
        sources = _sources_for(doc, result)
        for item in doc.ground_truth:
            if item.entity_type not in RELEASE_BLOCKING_TYPES:
                continue
            checked += 1
            if any(item.value in s for s in sources):
                leaks.append(f"{doc.name}: {item.entity_type.value} {item.value!r}")

    assert checked > 0, "no release-blocking ground truth in the corpus -- test would be vacuous"
    assert not leaks, "release-blocking PII leaked past redaction:\n" + "\n".join(leaks)


def test_ml_layer_recall_regression_guard(corpus, redacted_results, capsys):
    """Soft gate: see module docstring. Every miss is printed either way
    (visible with `pytest -s` or on failure) so a real drop is easy to spot
    even though this only fails below the regression floor."""
    leaks: list[str] = []
    total = 0
    for doc in corpus:
        if doc.is_control:
            continue
        result = redacted_results[doc.name]
        sources = _sources_for(doc, result)
        for item in doc.ground_truth:
            if item.entity_type in RELEASE_BLOCKING_TYPES:
                continue
            total += 1
            if any(item.value in s for s in sources):
                leaks.append(f"{doc.name}: {item.entity_type.value} {item.value!r}")

    assert total > 0, "no ML-layer ground truth in the corpus -- test would be vacuous"
    recall = 1 - len(leaks) / total
    if leaks:
        print(f"\n[test_no_leak] ML-layer misses ({len(leaks)}/{total}, recall={recall:.1%}):")
        for line in leaks:
            print(f"  {line}")

    assert recall >= _ML_RECALL_FLOOR, (
        f"ML-layer (name/address) recall dropped to {recall:.1%} "
        f"(floor {_ML_RECALL_FLOOR:.0%}) -- this looks like a real regression, "
        "not normal Rampart variance. Misses:\n" + "\n".join(leaks)
    )


def test_redacted_output_has_no_residual_release_blocking_pii(corpus, pipeline, redacted_results):
    """Idempotency check: re-scanning the ALREADY-redacted text/PDF with the
    full pipeline must not find any more release-blocking (regex+validator)
    PII -- if it did, either the redactor silently skipped a resolved span or
    a placeholder collided with a real pattern."""
    for doc in corpus:
        if doc.is_control:
            continue
        result = redacted_results[doc.name]
        if result.redacted_path is None:
            continue  # nothing was redacted; covered by the leak test above
        rescanned = pipeline.scan(result.redacted_path)
        residual = [
            s
            for s in rescanned
            if s.entity_type in RELEASE_BLOCKING_TYPES and s.source in ("regex", "custom")
        ]
        assert not residual, (
            f"{doc.name}: re-scanning the redacted output still finds release-blocking "
            f"PII: {[(s.entity_type.value, s.text) for s in residual]}"
        )


def test_control_docs_have_no_ground_truth(corpus):
    """Sanity check on the corpus itself: every doc marked as a clean
    false-positive control really does carry zero ground truth."""
    controls = [d for d in corpus if d.is_control]
    assert controls, "corpus has no clean control documents"
    for doc in controls:
        assert doc.ground_truth == [], f"{doc.name}: control doc has ground truth {doc.ground_truth}"
