# scrub

`scrub` is a local PII redactor that runs as a Claude Code hook: before the
`Read` tool hands a file's contents to the model, `scrub` intercepts the
call, writes a sanitized copy to a local cache, and points `Read` at that
copy instead. The original file on disk is never touched. Detection is
two-layer — deterministic regex+validator rules for structured PII (SSNs,
credit cards, API keys, and strongly labeled identity, credential, financial,
and medical fields) plus a small local ML model ([Rampart][rampart])
for context-dependent PII (names, addresses) — and everything runs on your
machine; nothing is sent anywhere. **This is defense-in-depth, not a
compliance guarantee.** No detector is 100%, coverage is `Read`-shaped (a
`Bash cat` bypasses it today), and you should not treat this as a substitute
for not putting sensitive data in a repo in the first place. See
[PLAN.md](PLAN.md) for the full design rationale and honest list of risks
and limits (§10), and [ARCHITECTURE.md](ARCHITECTURE.md) for implementation
details.

[rampart]: https://huggingface.co/nationaldesignstudio/rampart

## Requirements

- Python 3.11+
- macOS or Linux only — the daemon communicates over a Unix domain socket
  (no Windows support without a named-pipe/TCP transport, which doesn't
  exist yet)
- ~15 MB Rampart model, auto-downloaded from Hugging Face
  (`nationaldesignstudio/rampart`) on first run and cached locally after that

## Development setup on macOS

Modern Homebrew Python installations are [externally managed][pep-668], so a
direct `pip install .` is intentionally blocked. Do not use
`--break-system-packages`. Because this repository is an active development
project, use a virtual environment and an editable install instead.

### If you already have Python

Check the version you already have:

```bash
python3 --version
```

If that reports Python 3.11 or newer, use it directly:

```bash
git clone <this-repo-url> scrub
cd scrub
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

If `python3` is not found or reports a version older than 3.11, follow the next
section instead.

### If you need Python 3.11+

Install Python 3.11 with [Homebrew][homebrew]. If `brew` is not installed,
install Homebrew first from [brew.sh][homebrew].

```bash
brew install python@3.11
git clone <this-repo-url> scrub
cd scrub
"$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The `brew --prefix` expression locates Homebrew correctly on both Apple
Silicon and Intel Macs.

### Working in the project

The editable install means source changes take effect immediately without
reinstalling the package. It provides two commands while the environment is
active: `scrub` (CLI) and `scrub-hook` (the Claude Code hook entry point).

Activate the environment whenever you open a new Terminal window to work on
the project:

```bash
cd /path/to/scrub
source .venv/bin/activate
```

The Claude Code hook stores the absolute path to `.venv/bin/scrub-hook`, so
keep this virtual environment in place while the hook is installed.

[pep-668]: https://peps.python.org/pep-0668/
[homebrew]: https://brew.sh/

### Linux

After installing Python 3.11 or newer and its `venv` support through your
distribution, use the same project setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Verify the installation

Check that the command is available:

```bash
scrub --help
scrub scan somefile.txt
python -m pytest
```

On first run this downloads the Rampart model (~15 MB) into the Hugging
Face cache; subsequent runs are instant. `scrub scan` only detects and
prints what it found — it writes nothing. To actually produce a redacted
copy:

```bash
scrub redact somefile.txt
```

## Set up the Claude Code hook

```bash
scrub install-hook --user        # writes to ~/.claude/settings.json
# or
scrub install-hook --project     # writes to ./.claude/settings.json
```

This merges a `PreToolUse` hook for the `Read` tool into your Claude Code
settings — non-destructively (other hooks and keys are preserved) and
idempotently (running it again is a no-op). It adds something equivalent to:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read",
        "hooks": [
          { "type": "command", "command": "/path/to/scrub-hook", "timeout": 20 }
        ]
      }
    ]
  }
}
```

**Verify it's working:**

1. Create a test file with fake PII:
   ```bash
   echo "Call Maria Garcia at (312) 555-0148, SSN 458-02-6841." > /tmp/pii_test.txt
   ```
2. In Claude Code, ask it to read that file (e.g. "read /tmp/pii_test.txt").
3. Claude should see placeholders like `[GIVEN_NAME_1] [SURNAME_1] at
   [PHONE_1], SSN [SSN_1].` instead of the raw text. If you want to confirm
   from the CLI first, without going through Claude Code at all:
   ```bash
   scrub redact /tmp/pii_test.txt
   ```

If nothing gets redacted and you expected it to, check `scrub daemon
status` (below) and see **How it works** for the fail-open/fail-closed
distinction.

## How it works

A long-lived daemon holds the Rampart ONNX model and compiled regexes in
memory so per-`Read` overhead stays low (median warm cache-hit round trip is
well under a millisecond on this dev machine; see `tests/test_latency.py`).
The `scrub-hook` PreToolUse hook is a thin client: it asks the daemon to
redact the target file (spawning the daemon if it isn't running yet), and if
anything was found, rewrites `Read`'s `file_path` argument to point at the
redacted copy — `Read` proceeds normally against the sanitized file, and the
original is never modified. Repeated reads of an unchanged file are served
from a content-hash cache, so the second `Read` of the same file is instant.

**Fail mode matters.** By default `scrub` is **fail-closed**: if the daemon
can't be reached or an extractor errors out, the hook **denies** the `Read`
rather than silently showing the model raw, unredacted content. If a broken
daemon is blocking every `Read`:

```bash
scrub daemon status     # is it running? what's wrong?
scrub daemon start       # try starting it manually, see the error
```

If you'd rather see original files whenever `scrub` itself is broken (fail
open instead of fail closed — an explicit, deliberate trade of safety for
availability), set this in `~/.config/scrub/config.toml`:

```toml
fail_mode = "open"
```

## Configuration

`~/.config/scrub/config.toml` (all fields optional; shown values are the
defaults except where noted):

```toml
# Deny closes a Read on daemon/extractor failure (safe default); "open"
# passes the original file through and logs instead.
fail_mode = "closed"

[engines]
regex = true              # deterministic regex+validator layer
rampart = true             # ML layer (names, addresses, ...)
rampart_confidence = 0.5   # global minimum confidence for Rampart spans

# Per-type confidence floors (applied as max(global, per-type)). Address
# component types misfire on source code more than plain prose, so they get
# a stricter floor by default.
[engines.rampart_type_thresholds]
SECONDARY_ADDRESS = 0.85
STREET_NAME = 0.85
BUILDING_NUMBER = 0.85

[entities]
# Types that are DETECTED and reported but NOT redacted (kept in the output).
public_types = ["CITY", "STATE", "ZIP_CODE", "URL"]
# Extra literal terms to always redact (case names, project codenames, ...)
# — matched case-insensitively as whole words, tagged CUSTOM.
custom_keywords = ["Project Nightjar"]

[paths]
allow = []          # glob patterns: NEVER scrub (e.g. your own test fixtures)
deny = []           # glob patterns: ALWAYS scrub, overrides allow/skip
skip = [            # glob patterns: skip without scrubbing (defaults shown)
    "**/node_modules/**",
    "**/.git/**",
    "**/__pycache__/**",
    "**/*.lock",
]

[limits]
max_file_bytes = 20971520   # 20 MiB; larger files are skipped untouched

[cache]
max_bytes = 524288000       # 500 MiB cap on ~/.cache/scrub
```

## What gets detected

| Layer | Entity types |
|---|---|
| Regex + validator (deterministic, confidence 1.0) | `SSN`, `ITIN`, `EIN`, `CREDIT_CARD` (Luhn), `ROUTING_NUMBER` (ABA checksum), `BANK_ACCOUNT`, `IP_ADDRESS`, `MAC_ADDRESS`, `API_KEY` (AWS/GitHub/Stripe key formats + generic high-entropy secret assignments), `PRIVATE_KEY` (PEM blocks), `JWT` |
| Shared (regex catches structured forms, Rampart catches contextual ones) | `EMAIL`, `PHONE`, `URL` |
| Rampart ML layer (context-dependent) | `GIVEN_NAME`, `SURNAME`, `TAX_ID`, `GOVERNMENT_ID`, `PASSPORT`, `DRIVERS_LICENSE`, `BUILDING_NUMBER`, `STREET_NAME`, `SECONDARY_ADDRESS`, `CITY`, `STATE`, `ZIP_CODE` |
| Config-driven | `CUSTOM` (your `custom_keywords`) |

`CITY`, `STATE`, `ZIP_CODE`, and `URL` are detected and listed in
`report.json` but left in place by default (`public_types`) — everything
else is replaced with a stable per-file placeholder like `[SSN_1]` or
`[GIVEN_NAME_2]`. See [`eval/`](eval/) for the measured recall/precision
harness: regex+validator entity types (the release-blocking set — SSN, EIN,
CREDIT_CARD, ROUTING_NUMBER, BANK_ACCOUNT, API_KEY, PRIVATE_KEY, JWT) are
held to 100% recall on the golden corpus; run `python3 eval/run_eval.py` for
current numbers, including the ML layer's name/address recall (which is
real but not 100% — see below).

## Known limitations

- **`Read`-tool coverage only.** An agent running `Bash cat secrets.pdf` (or
  `less`, `head`, etc.) bypasses this hook entirely. Bash hardening is a
  documented future idea (PLAN.md §7), not yet built.
- **Pasted PII in prompts is not covered.** There's no hook that can rewrite
  the user's own prompt text before the model sees it.
- **Digital text/PDF only.** Scanned PDFs, images, and Office docs
  (`.docx`/`.pptx`/`.xlsx`) are not yet supported — OCR and Office
  extractors are a planned follow-up (PLAN.md §8).
- **English / Latin-script only** (Rampart covers EN/ES/FR/DE/IT/PT/NL, but
  this build is validated against English fixtures).
- **No detector is 100%.** The regex+validator layer is held to 100% recall
  on its golden corpus (see `eval/`) because it's deterministic and
  checksummed. The ML layer (names/addresses) is alpha and measurably
  imperfect — expect occasional misses, especially in unusual sentence
  structures. Treat this as defense-in-depth, not a compliance guarantee,
  and keep human review for anything you'd stake liability on.
- **Rampart itself is alpha** and officially shipped as a TypeScript/ONNX
  library; this is a from-scratch Python port of its pre/post-processing
  against a pinned model revision. It may need revalidation if that
  revision is ever bumped.

## Uninstall

Run these commands from the repository while its virtual environment is
active. Uninstall the hook before deleting `.venv`, because the hook points to
an executable inside it.

```bash
scrub uninstall-hook --user      # or --project, matching how you installed it
deactivate
rm -rf .venv
rm -rf ~/.cache/scrub            # redacted-copy cache + daemon socket/pidfile
```
