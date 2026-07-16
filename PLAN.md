# PII Redactor CLI — Design & Build Plan

A performant local CLI that takes **any** media file (text, source, PDF, image, Office doc), detects personally identifiable information, and returns a **redacted copy**. Designed to run as a **Claude Code hook** so files are scrubbed *before the model ever sees them*.

**Decisions locked in for this plan**

- **Language:** Python
- **Detection engine:** Rampart (National Design Studio) — regex + MiniLM, ~14.7 MB, ~4 ms/text. This is the **only** detector in the core build; the tool is fully functional with just Rampart and requires **no large model download**.
- **Rampart integration:** **Path A (pure Python) from day 1** — ONNX weights via `onnxruntime` + `tokenizers`, validated against Rampart's published TypeScript eval harness as the golden reference. No Node sidecar (see §3).
- **v1 scope:** **text/code/config + digital PDF** (phases 0–3, 5, 6). OCR, images, and Office docs are follow-ups.
- **Fail mode default:** **fail-closed** — if the daemon or an extractor fails, deny the Read with a clear reason (§6).
- **OCR engine (future phase):** **PaddleOCR**, accepting the heavier install for better accuracy on messy scans.
- **Future idea (not now):** an optional OpenAI Privacy Filter "deep pass" (~2.8 GB) for context-heavy tax/legal PDFs. Explicitly out of scope for v1 — noted so the architecture leaves room, but nothing about the working tool depends on it.
- **Hook behavior:** Redact in-place for the agent — the original file on disk is never modified; the agent transparently reads a sanitized copy.

Working name used throughout: **`scrub`**. Rename freely.

---

## 1. The architectural choice that drives everything

*(Corrected 2026-07: an earlier draft claimed PostToolUse cannot rewrite tool output. That has since shipped — `PostToolUse` now supports `updatedToolOutput`, which replaces the tool's result. PreToolUse is no longer the only option, but it remains the better one here:)*

- **`updatedToolOutput` is capped by the 10,000-char hook output limit** — useless for large files.
- **Images and PDFs are rendered natively by Claude Code** — you can't return redacted pixels through a text output rewrite, but you *can* point `Read` at a redacted copy on disk.
- **A path rewrite composes with the content-hash cache** — repeated reads return the same path instantly.

So the primary mechanism is **PreToolUse**, which returns `updatedInput` to modify the tool's arguments before it runs (`updatedToolOutput` stays in the toolbox as a fallback, e.g. for a later Bash-hardening pass). The design is:

> Intercept `Read` in a **PreToolUse** hook → produce a redacted copy of the target file on disk → rewrite `file_path` to point at that copy → let `Read` proceed against the sanitized file.

This is clean: the model reads a real file, just a scrubbed one, and the original is untouched. It also sidesteps the 10,000-char stdout cap on hooks, because the hook returns a *path*, not the content.

```json
// PreToolUse stdout on a match:
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "updatedInput": { "file_path": "/Users/greg/.cache/scrub/ab12cd34.redacted.pdf" }
  }
}
```

> ⚠️ `updatedInput` **replaces the entire `tool_input` object**, not just the fields you name. The hook must echo back every other field from the incoming `tool_input` (e.g. `Read`'s `offset` and `limit`) with only `file_path` swapped.

---

## 2. High-level architecture

```
                        ┌───────────────────────────────────────────┐
   file path ──▶  CLI ──▶│  scrub daemon (persistent, model in RAM)  │
  (or stdin)             │                                           │
                        │  1. Router      detect true type (magic)   │
                        │  2. Extractor   media → text + coordinates  │
                        │  3. Detector    regex + Rampart (+deep)     │
                        │  4. Redactor    mask text / blackout pixels │
                        │  5. Emitter     write copy + JSON report    │
                        └───────────────────────────────────────────┘
                                          │
                    sanitized file path  ◀┘  +  report.json
```

The **daemon** is the key to being "performant." Loading an ONNX model and compiling regexes on every `Read` would add hundreds of ms. Instead a long-lived process holds the model in memory and the CLI is a thin client over a Unix domain socket. A content-hash cache means repeated reads of the same unchanged file are instant.

### Components

**1. Router** — Identify the file by magic bytes (`python-magic`/`filetype`), never trust the extension alone. Route to the right extractor. Unknown/binary types with no text extractor pass through untouched (and are logged).

**2. Extractors** — normalize any medium into `(text, spans-with-coordinates)`:

| Input | Library | Notes |
|---|---|---|
| Text, source code, `.md`, `.json`, `.csv`, `.env`, logs | native read | Character offsets are the "coordinates." |
| PDF (digital) | **PyMuPDF** (`pymupdf`) | Word-level bounding boxes via `page.get_text("words")`. |
| PDF (scanned) | PyMuPDF render → **PaddleOCR** *(post-v1)* | Rasterize page, OCR the image; boxes come from OCR. (PyMuPDF's built-in `get_textpage_ocr()` is Tesseract-only, so scanned pages route through the same PaddleOCR path as images.) |
| Images `.png/.jpg/.tiff/.webp` | **PaddleOCR** *(post-v1)* | Word boxes from detection results. Chosen over Tesseract for accuracy on messy scans, accepting the heavier install. |
| `.docx / .pptx / .xlsx` | `python-docx`, `python-pptx`, `openpyxl` | Text runs; redact by run replacement. |

**3. Detector** — Rampart's two-layer approach, reimplemented in Python:

- **Deterministic layer:** regexes *with validators* — SSN, ITIN, EIN, credit card (Luhn), ABA routing (checksum), bank account, phone, email, IP/MAC, URLs, and secrets (API keys, private keys, JWTs). Validation kills false positives that plain regex produces.
- **ML layer:** the Rampart MiniLM model for context-dependent entities (names, street addresses) that rules can't enumerate. Run the published weights via **ONNX Runtime** (see §3). ~14.7 MB — downloaded once, cached.
- **Merge:** union all spans, resolve overlaps (longest/highest-confidence wins), apply allow/deny lists and custom keywords.

> **Future, off by default:** a heavyweight **OpenAI Privacy Filter** (1.5B, ~2.8 GB) deep pass for high-stakes tax/legal PDFs. The interface below (`Detector` protocol) is designed so this drops in later as a second detector, gated by config, downloaded only if you ever opt in. **v1 ships without it and needs no such download.**

**4. Redactor** — transform by medium:

- **Text/code:** replace each span with a typed placeholder — `[GIVEN_NAME_1]`, `[SSN_1]`, `[ACCOUNT_NUMBER_1]`. Stable numbering per file so the model can still follow references. Optionally write a **reversible vault** (encrypted, local) mapping placeholder→original so agent edits can be re-expanded on write-back (advanced, §7).
- **PDF (digital):** PyMuPDF redaction annotations + `page.apply_redactions()` — this *removes* the underlying text, not just draws a box. Then **scrub metadata** (`doc.set_metadata({})`, `doc.xref_*`) and, for maximum safety, rasterize affected pages so nothing is recoverable via copy/paste or `pdftotext`.
- **Images / scanned PDF:** paint solid rectangles over the detected boxes at the pixel level and flatten. Irreversible by construction.

**5. Emitter** — write the sanitized file to `~/.cache/scrub/<contenthash>.redacted.<ext>`, plus a `report.json` (entities by type, page/offset, confidence). Return the path on the socket.

---

## 3. Running Rampart from Python (the honest part)

Rampart is officially shipped as an **NPM/TypeScript** library (`@nationaldesignstudio/rampart`, transformers.js). You picked Python, so there are two integration paths:

**Path A — Pure Python (recommended for a single-language tool).**
The model weights are on Hugging Face (`nationaldesignstudio/rampart`) in ONNX form. Load them with `onnxruntime` + the HF `tokenizers` library and reimplement the glue that the JS lib does: token-classification → span decode → placeholder substitution → (optional) reveal vault. The deterministic regex layer you write yourself anyway. **Risk:** you're reproducing Rampart's pre/post-processing; pin the model revision and validate against their published examples so your output matches theirs.

**Path B — Node sidecar.**
Ship the official Rampart NPM lib in a tiny Node service the Python daemon calls over the same socket/IPC. **Pro:** exact fidelity to Rampart's pipeline, upgrades for free. **Con:** a Node runtime dependency inside a "Python tool."

**Decision: Path A from day 1.** Building Path B first means building the integration twice and temporarily shipping a Node runtime inside a Python tool. Rampart publishes not just the weights but a **TypeScript eval harness** (CC BY 4.0) — run that harness once, offline, to generate a golden input→output corpus, and validate the Python port against it in CI. That gives the "known-good reference" Path B was meant to provide without ever shipping Node. Pin the HF model revision.

> Note: Rampart is **alpha**, **text-only**, and covers **7 Latin-script languages** (EN/ES/FR/DE/IT/PT/NL). Everything image/PDF is *your* extractor feeding text into it. Since you only need English, this is fine.

---

## 4. Claude Code hook integration

### 4.1 Settings

`~/.claude/settings.json` (or project `.claude/settings.json`):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read",
        "hooks": [
          { "type": "command", "command": "/usr/local/bin/scrub-hook", "timeout": 20 }
        ]
      }
    ]
  }
}
```

*(Corrected 2026-07: the real schema uses a singular `matcher` string and a `hooks` array with `"type": "command"` — an earlier draft used a `matchers`/`handlers` shape that doesn't exist and would silently never fire.)*

### 4.2 The hook script (`scrub-hook`)

A thin, fast client — **no model loading here**; it just talks to the daemon.

```
1. Read JSON from stdin → tool_input.file_path
2. If path is missing / not a regular file / in an allowlisted-safe dir → emit passthrough, exit 0
3. socket call: daemon.redact(path)  →  { redacted_path, found: N }  (uses content-hash cache)
4. If found == 0 → passthrough (allow, no updatedInput)
5. If found  > 0 → allow with updatedInput.file_path = redacted_path
6. exit 0
```

Passthrough / allow example (nothing found):

```json
{ "hookSpecificOutput": { "permissionDecision": "allow" } }
```

### 4.3 Coverage gaps to decide on explicitly

- **`Read` is covered; `Bash` is not.** An agent running `cat secrets.pdf` bypasses a Read-only hook. Options: also match `Bash` and inspect the command (fragile), or accept Read-only coverage and document it. Recommend Read-first, Bash as a later hardening pass.
- **Pasted content / `UserPromptSubmit`.** This hook can *add context* or *block*, but **cannot rewrite** the user's prompt text. So for PII the user pastes directly, the realistic behaviors are (a) block+warn, or (b) inject a note. It can't silently scrub the prompt. Worth a separate `UserPromptSubmit` handler if inbound paste is a concern.
- **Images via `Read`.** Claude Code renders images; rewriting `file_path` to a blacked-out copy works and the model sees only the redacted pixels.

---

## 5. Performance plan

Target: sub-100 ms added latency on a warm cache; low hundreds of ms on a cold text file; PDFs/OCR bounded by extraction, not detection.

- **Daemon holds the model resident** (ONNX session, compiled regexes) — no per-call startup.
- **Content-hash cache** keyed on `(sha256(bytes), mtime)`. Agents re-read the same files constantly; second read is a cache hit → instant path return.
- **No heavyweight model on the hot path** — Rampart is tiny and resident. (If the future OpenAI deep pass is ever enabled, it loads lazily only when a file trips its trigger; never for ordinary source files.)
- **Skip early.** Binary/unknown types, files over a size cap, and generated/`node_modules`-style paths short-circuit before extraction.
- **ONNX Runtime on CPU** is enough — Rampart is ~4 ms/text on WebGPU; short files stay in low ms on CPU. Batch long documents by chunk.
- Async hook (`async: true`) is *not* appropriate here — we must block `Read` until the copy exists. Keep it synchronous but fast.

Rust was on the table for raw speed, but with a resident Python daemon + ONNX + cache the CLI overhead isn't the bottleneck (extraction/OCR is), and Python keeps you in one ecosystem for Rampart, PyMuPDF, and OCR — the parts that actually matter. Revisit Rust only if you later want a single distributable static binary.

---

## 6. Configuration surface

`~/.config/scrub/config.toml`:

- **engines:** enable/disable regex, rampart; per-engine confidence thresholds. (A `deep(openai)` toggle exists but is **off by default and not installed** — see §7.)
- **per-type redaction style:** placeholders vs blackout vs pseudonyms (§7), per file class.
- **entity toggles:** which of the ~17 categories to act on; custom keyword list (case names, project codenames).
- **allow/deny paths:** never-scrub globs (e.g. your own fixtures) and always-scrub globs.
- **cache:** dir, max size, TTL.
- **fail mode:** on extractor/daemon error → `fail-open` (allow original, log) or `fail-closed` (deny read). **Default: fail-closed** — the tool's whole promise is that raw PII never reaches the model, so the safe behavior must be what runs when nobody is looking. Fail-open is an explicit opt-in. (The hook must therefore auto-spawn the daemon on first call and handle the startup race, or a dead daemon blocks every `Read`.)

---

## 7. Future ideas (explicitly not v1)

- **OpenAI Privacy Filter deep pass.** The heavyweight second detector (1.5B, ~2.8 GB) for context-heavy tax/legal PDFs where Rampart's OpenPII-tuned recall may generalize less well. Rationale it's deferred: it's a big download, needs a warm daemon/torch, and Rampart already covers the common case. Design hook: everything speaks a `Detector` protocol (`detect(text) -> spans`), so adding it later is a new class + a config toggle + a lazy model download — no rearchitecting. **Do not install torch or pull this model for v1.**
- **Consistent pseudonyms** instead of placeholders: `Maria Garcia → Alex Rivera`, stable within a file, so the agent reasons about references naturally. Needs a per-file deterministic fake-data generator (Faker seeded by entity hash).
- **Reversible vault + write-back.** Keep an encrypted placeholder→original map. When the agent later *writes* to a file (Edit/Write hook), re-expand placeholders so real values land back on disk. Powerful but adds a second hook and real security surface — the vault becomes a PII honeypot; encrypt at rest and scope by session.
- **`UserPromptSubmit` guard** for pasted PII (block+warn).
- **Bash hardening** to catch `cat`/`less` reads.

---

## 8. Build phases

| Phase | Deliverable | Proves |
|---|---|---|
| 0 | CLI skeleton + type router + text passthrough | Plumbing works end to end |
| 1 | Deterministic regex+validator layer, text placeholder redaction | Catches structured PII (SSN/EIN/cards) with low FP |
| 2 | Rampart integration (Path A, pure Python, validated against the TS eval harness) | Context PII (names/addresses) |
| 3 | PDF text extraction + `apply_redactions` + metadata scrub | Real IRS/contract PDFs, unrecoverable |
| 4 | Daemon + Unix socket + content-hash cache | Hook-grade latency |
| 5 | `scrub-hook` + settings.json wiring | Works inside Claude Code |
| 6 | Eval harness + hardening | Measured recall/precision, regression-proof |
| — | *Post-v1:* PaddleOCR path for images + scanned PDFs + pixel blackout; Office docs (`.docx/.pptx/.xlsx`) | "Any media" becomes true |
| — | *Future:* OpenAI deep pass, pseudonyms, vault, UserPromptSubmit (§7) | Depth & reversibility, when needed |

**v1 = phases 0–5** (text/code/config + digital PDF, wired into Claude Code). A useful daily-driver exists at the end of **Phase 5**; OCR and Office extractors slot in behind the same extractor interface afterward.

---

## 9. Verification (don't skip)

- **Golden corpus** of synthetic docs with known PII (tax forms, contracts, code with secrets, scanned images). Score **recall & precision per entity type**; treat recall on SSN/EIN/account numbers as release-blocking.
- **Recovery tests** on redacted PDFs: `pdftotext`, copy-paste, and metadata dump must return *nothing* sensitive. This is the exact failure mode of naive "black box" tools.
- **Round-trip / no-leak assertion:** the sanitized output is re-scanned; if any raw PII survives, fail the build.
- **Latency budget test** in daemon mode (cold vs warm cache) to keep the hook snappy.
- **Fidelity test** for the Rampart Python port (Path A) against the Node reference (Path B) on the same inputs.

---

## 10. Honest risks & limits

- **Rampart is alpha and officially JS-only.** Python use means running raw weights + your own glue; pin the model revision, validate against the reference. It may change under you.
- **OCR quality is the ceiling for scans.** Garbage OCR → missed PII. Budget for a good OCR engine and spot-checks on image-heavy inputs.
- **No detector is 100%,** and cross-domain recall drops sharply. This hook is **defense-in-depth, not a compliance guarantee** — keep human review for anything you'd stake liability on.
- **The vault (if built) concentrates PII.** Encrypt, scope, and consider whether you want it at all.
- **Coverage is Read-shaped.** Bash reads and direct pastes need separate handling.
- **Fail mode matters.** A crashed daemon must not silently pass raw files to the model unless you chose fail-open on purpose. (Default is fail-closed — §6.)
- **Unix domain sockets mean macOS/Linux only** for v1. Windows would need a named-pipe or TCP-on-localhost transport; explicitly out of scope until someone asks.

---

## 11. Dependency shortlist

**Core (v1):** `python 3.11+`, `onnxruntime`, `tokenizers`, `huggingface_hub`, `pymupdf`, `filetype`/`python-magic`, `typer` (CLI), `pydantic`/`tomllib` (config), `orjson`/`msgspec` (IPC). Only download is Rampart's ~14.7 MB model.

**Post-v1 (OCR + Office phases):** `paddleocr` (+ PaddlePaddle), `pillow`, `python-docx`, `python-pptx`, `openpyxl`.

**Not installed for v1 (future only):** `torch` + `transformers` and the ~2.8 GB `openai/privacy-filter` weights (deep pass); `cryptography` (vault). Node + `@nationaldesignstudio/rampart` is used only offline, once, to generate the golden validation corpus from Rampart's eval harness (§3) — it is never a runtime dependency.
