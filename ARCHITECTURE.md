# scrub — internal engineering brief

Read PLAN.md first for the product design. This file is the build contract:
validated facts, interfaces, and file ownership. Do not change files another
task owns; do not change `src/scrub/types.py` or `src/scrub/config.py` without
flagging it in your final report.

## Validated facts (do not re-research these)

- Python 3.11.15; deps already installed: onnxruntime, tokenizers,
  huggingface_hub, pymupdf, filetype, typer, pydantic, orjson, pytest.
  Package installed editable (`pip install -e .`), console scripts `scrub`
  and `scrub-hook` are wired in pyproject.toml.
- Rampart model is downloaded and verified working from pure Python:
  - HF repo `nationaldesignstudio/rampart`, revision pinned in
    `scrub/config.py` (`RAMPART_REVISION`).
  - Files: `onnx/model_q4.onnx` (14.7 MB), `tokenizer.json`, `config.json`
    (id2label with 35 BIO labels / 17 entity types).
  - ONNX inputs: `input_ids`, `attention_mask`, `token_type_ids`
    (all int64, [batch, seq]); output `logits` [batch, seq, 35].
  - `tokenizers.Tokenizer.from_file("tokenizer.json")` gives char offsets
    via `encoding.offsets`. Model is uncased. max_position_embeddings=512.
  - Smoke test confirmed: names/street/city/state/zip/phone detected with
    correct char offsets. An SSN was misclassified as B-PHONE at ~0.7 conf —
    this is WHY the regex layer runs first and masks its hits with sentinels
    before the model pass (Rampart upstream does the same).
- `scrub download` uses
  `huggingface_hub.snapshot_download(RAMPART_REPO, revision=RAMPART_REVISION)`
  to install the pinned model. Runtime commands pass `local_files_only=True`
  and fail with an actionable error if the snapshot is missing or incomplete.

## Contracts (already written — build against, don't redefine)

- `scrub.types`: `EntityType`, `Span`, `WordBox`, `Extraction`,
  `ReportEntity`, `RedactionResult`, `Detector` / `Extractor` protocols,
  `DEFAULT_PUBLIC_TYPES`.
- `scrub.config`: `Config.load()`, `config_dir()`, `cache_dir()`,
  `socket_path()`, `RAMPART_REPO`, `RAMPART_REVISION`.
- Placeholder format: `[{ENTITY_TYPE}_{n}]`, n starts at 1 per file in order
  of first appearance; the same source text (case-insensitive for names)
  reuses its placeholder within a file.
- Entities whose type is in `config.public_types` are detected and reported
  but NOT redacted.

## File ownership

| Task | Owns |
|---|---|
| Phase 0+1 | `router.py`, `extractors/text.py`, `detectors/regex_rules.py`, `redactors/text.py`, `pipeline.py`, `cli.py`, `tests/test_router.py`, `tests/test_regex_rules.py`, `tests/test_text_redaction.py` |
| Phase 2 | `detectors/rampart.py`, `detectors/merge.py`, `tests/test_rampart.py`, `tests/test_merge.py` |
| Phase 3 | `extractors/pdf.py`, `redactors/pdf.py`, `tests/fixtures/make_pdfs.py`, `tests/test_pdf.py` |
| Phase 4+5 | `daemon.py`, `client.py`, `cache.py`, `hook.py`, `install.py`, `tests/test_daemon.py`, `tests/test_hook.py` |
| Phase 6 | `eval/` harness, `tests/test_no_leak.py`, `tests/test_latency.py`, `README.md` |

## Key integration points

- `pipeline.py` (Phase 0+1 owns the skeleton) exposes
  `Pipeline.redact_file(path: Path) -> RedactionResult` and composes:
  router → extractor → detectors → merge → redactor → emitter. Phase 0+1
  ships it with regex-only detection and text-only redaction; later phases
  register their detector/extractor/redactor — design for that (registries
  or constructor injection, keep it simple).
- Detector merge rule (Phase 2 owns `merge.py`): union spans; on overlap the
  validated regex span wins over ML; otherwise longest, then highest
  confidence. Regex hits are masked to same-length sentinel characters before
  the Rampart pass so the model can't re-derive them.
- PDF redaction (Phase 3): map winning spans → WordBoxes → PyMuPDF redact
  annotations, `apply_redactions()`, then `doc.set_metadata({})` and strip
  XMP (`doc.del_xml_metadata()`). Recovery test must prove re-extraction
  returns none of the redacted strings.
- Daemon protocol (Phase 4+5): newline-delimited JSON over unix socket.
  Request: `{"op": "redact", "path": "/abs/file"}`.
  Response: `{"ok": true, "redacted_path": "...|null", "found": N,
  "cache_hit": bool}` or `{"ok": false, "error": "..."}`.
  Hook must handle: daemon not running (spawn + retry with backoff),
  fail-closed = emit permissionDecision "deny" with a readable reason.
- Hook stdout on rewrite: full `tool_input` echoed with only `file_path`
  replaced (updatedInput replaces the ENTIRE input object). See PLAN.md §1/§4.

## Testing rules

- Every phase ships pytest tests that run offline. Rampart tests require the
  pinned model to have been installed by `scrub download`; runtime model
  loading is cache-only and never falls back to a network request.
- All PII in fixtures is synthetic. Use these known-fake values where
  possible: SSN 458-02-6841 (invalid area is fine to synthesize), names like
  "Maria Garcia", card 4111 1111 1111 1111, routing 021000021.
- Run `python3 -m pytest tests/ -x -q` before reporting done; report the
  actual output.
- Do NOT run `git commit` or `git push` — the CTO session handles VCS.
