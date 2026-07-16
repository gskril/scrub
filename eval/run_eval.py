"""Golden-corpus eval harness (Phase 6) -- the release-blocking recall gate
from PLAN.md Sec 9.

Runs `Pipeline.default()` over `eval/corpus.py`'s golden corpus and computes:

  - Per-entity-type, VALUE-LEVEL recall: a ground-truth value counts as
    "caught" only if it survives nowhere downstream -- not in the redacted
    text / re-extracted PDF text, not in report.json, not in the redacted
    PDF's metadata/XMP. This mirrors PLAN.md's round-trip/no-leak assertion,
    scored per value instead of asserted per document (that's what
    tests/test_no_leak.py does).
  - A false-positive proxy: total redactions made in the corpus's clean
    control documents (ground truth == []). Any redaction there is a false
    positive by construction.

Release-blocking gate (PLAN.md Sec 9 / ARCHITECTURE.md): recall on
SSN, EIN, CREDIT_CARD, ROUTING_NUMBER, BANK_ACCOUNT, API_KEY, PRIVATE_KEY,
JWT must be 100%. This script exits 1 if any of those falls short. Name/
address recall (GIVEN_NAME, SURNAME, STREET_NAME, ITIN, ...) is reported but
never gates the exit code -- Rampart is alpha and PLAN.md is explicit that
cross-domain ML recall is not a 100% target.

Runnable directly:

    python3 eval/run_eval.py

Writes eval/results.json alongside the printed table.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))
from corpus import CorpusDoc, RELEASE_BLOCKING_TYPES, build_corpus  # noqa: E402

# Route the pipeline's cache (redacted copies + report.json) to a throwaway
# dir. cache_dir() re-reads this env var on every call (see scrub.config), so
# setting it before Pipeline.default() runs is enough -- no monkeypatching,
# and nothing from this eval run lands in the real ~/.cache/scrub.
_CACHE_DIR = Path(tempfile.mkdtemp(prefix="scrub-eval-cache-"))
os.environ.setdefault("SCRUB_CACHE_DIR", str(_CACHE_DIR))

import pymupdf  # noqa: E402

from scrub.config import Config  # noqa: E402
from scrub.pipeline import Pipeline  # noqa: E402

CORPUS_DIR = _EVAL_DIR / "generated"
RESULTS_PATH = _EVAL_DIR / "results.json"


def _pdf_sources(path: Path) -> list[str]:
    doc = pymupdf.open(path)
    try:
        text = "\n".join(page.get_text() for page in doc)
        meta = json.dumps(doc.metadata)
        xml = doc.get_xml_metadata() or ""
    finally:
        doc.close()
    return [text, meta, xml]


def _output_sources(doc: CorpusDoc, result) -> list[str]:
    """Every place downstream PII could still be hiding after redact_file()."""
    # None means "nothing found, nothing written" -- what a Read hook would
    # show downstream is the ORIGINAL file, unredacted. That's a leak for
    # every ground-truth value in that doc, and the substring check below
    # against the original correctly reports it as one.
    target = result.redacted_path or doc.path
    sources: list[str] = list(_pdf_sources(target)) if doc.kind == "pdf" else [
        target.read_text(encoding="utf-8", errors="replace")
    ]
    if result.report_path is not None:
        sources.append(result.report_path.read_text(encoding="utf-8", errors="replace"))
    return sources


def run() -> dict:
    config = Config()
    pipeline = Pipeline.default(config)
    docs = build_corpus(CORPUS_DIR)

    per_type: dict[str, dict[str, int]] = defaultdict(lambda: {"caught": 0, "total": 0})
    misses: list[dict] = []
    fp_docs: list[dict] = []
    fp_total = 0

    t0 = time.perf_counter()
    for doc in docs:
        result = pipeline.redact_file(doc.path)

        if doc.is_control:
            redacted_types = sorted(
                {
                    e.entity_type.value
                    for e in result.entities
                    if e.entity_type not in config.public_types
                }
            )
            if result.found > 0:
                fp_total += result.found
                fp_docs.append({"doc": doc.name, "found": result.found, "types": redacted_types})
            continue

        sources = _output_sources(doc, result)
        for item in doc.ground_truth:
            key = item.entity_type.value
            per_type[key]["total"] += 1
            leaked = any(item.value in s for s in sources)
            if leaked:
                misses.append({"doc": doc.name, "entity_type": key, "value": item.value})
            else:
                per_type[key]["caught"] += 1
    elapsed_s = time.perf_counter() - t0

    blocking_report = {}
    blocking_ok = True
    for etype in sorted(RELEASE_BLOCKING_TYPES, key=lambda e: e.value):
        stats = per_type.get(etype.value, {"caught": 0, "total": 0})
        total = stats["total"]
        recall = (stats["caught"] / total) if total else None
        ok = (recall == 1.0) if total else False  # no ground truth at all is also a failure
        blocking_ok = blocking_ok and ok
        blocking_report[etype.value] = {"caught": stats["caught"], "total": total, "recall": recall, "ok": ok}

    other_report = {}
    for key, stats in sorted(per_type.items()):
        if key in {e.value for e in RELEASE_BLOCKING_TYPES}:
            continue
        total = stats["total"]
        other_report[key] = {
            "caught": stats["caught"],
            "total": total,
            "recall": (stats["caught"] / total) if total else None,
        }

    n_control = sum(1 for d in docs if d.is_control)
    n_control_leaky = len(fp_docs)

    results = {
        "corpus_docs": len(docs),
        "corpus_dir": str(CORPUS_DIR),
        "elapsed_s": round(elapsed_s, 3),
        "release_blocking": blocking_report,
        "release_blocking_pass": blocking_ok,
        "other_recall": other_report,
        "misses": misses,
        "false_positive_proxy": {
            "control_docs": n_control,
            "control_docs_with_redactions": n_control_leaky,
            "total_redactions_in_controls": fp_total,
            "details": fp_docs,
        },
    }
    return results


def _fmt_pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def print_report(results: dict) -> None:
    print(f"\nscrub eval -- {results['corpus_docs']} corpus docs, "
          f"{results['elapsed_s']}s\n")

    print("RELEASE-BLOCKING (must be 100%):")
    print(f"  {'type':16} {'caught/total':>14} {'recall':>8}  gate")
    for etype, stats in results["release_blocking"].items():
        mark = "PASS" if stats["ok"] else "FAIL"
        print(f"  {etype:16} {stats['caught']:>6}/{stats['total']:<7} "
              f"{_fmt_pct(stats['recall']):>8}  {mark}")

    print("\nREPORTED (not gated -- ML layer, alpha):")
    print(f"  {'type':16} {'caught/total':>14} {'recall':>8}")
    for etype, stats in results["other_recall"].items():
        print(f"  {etype:16} {stats['caught']:>6}/{stats['total']:<7} {_fmt_pct(stats['recall']):>8}")

    fp = results["false_positive_proxy"]
    print(f"\nFALSE-POSITIVE PROXY (clean controls, ground truth == []):")
    print(f"  {fp['control_docs_with_redactions']}/{fp['control_docs']} control docs had a redaction; "
          f"{fp['total_redactions_in_controls']} total redaction(s)")
    for d in fp["details"]:
        print(f"    {d['doc']}: {d['found']} redaction(s), types={d['types']}")

    if results["misses"]:
        print(f"\nMISSES ({len(results['misses'])} ground-truth values that leaked):")
        for m in results["misses"]:
            print(f"  [{m['entity_type']:12}] {m['doc']:36} value={m['value']!r}")

    gate = "PASS" if results["release_blocking_pass"] else "FAIL"
    print(f"\nRELEASE GATE: {gate}\n")


def main() -> int:
    results = run()
    print_report(results)
    RESULTS_PATH.write_bytes(json.dumps(results, indent=2).encode("utf-8"))
    print(f"Wrote {RESULTS_PATH}")
    return 0 if results["release_blocking_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
