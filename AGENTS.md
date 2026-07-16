# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
`scrub` is a **local, single-package Python CLI** (Python 3.11+) that detects and
redacts PII in files, and ships a Claude Code `PreToolUse` hook so files are
scrubbed before the model reads them. Three entry points, one package:
`scrub` (CLI), `scrub-hook` (Claude Code hook), and a long-lived `scrub daemon`.
See `README.md` and `ARCHITECTURE.md` for the full design; `PLAN.md` is the
original design doc.

> Note: on `main` the application code may live only in an open PR (the code is
> under `src/scrub/`, `tests/`, `eval/`, plus `pyproject.toml`). If those files
> and `pyproject.toml` are not present in your checkout, the code hasn't been
> merged yet — check out the implementation PR branch to run the app.

### Environment / dependencies
- A Python virtualenv is created at `/workspace/.venv` by the startup update
  script (`pip install -e ".[dev]"`), but only when `pyproject.toml` exists.
- Run tools via `/workspace/.venv/bin/<tool>` (e.g. `.venv/bin/scrub`,
  `.venv/bin/pytest`) or `source /workspace/.venv/bin/activate` first.
- Standard install/test/run commands are documented in `README.md` and
  `pyproject.toml` (`[project.scripts]`, `[tool.pytest.ini_options]`); don't
  duplicate them — use those.

### Non-obvious gotchas
- **Rampart ML model auto-downloads on first run** (~15 MB from Hugging Face,
  repo `nationaldesignstudio/rampart`) into the HF cache (`~/.cache/huggingface`).
  This is cached (and persisted in the VM snapshot), so later runs are offline.
  No `HF_TOKEN` is required (you'll see an unauthenticated-rate-limit warning —
  harmless). If the cache is ever cold, the first `scrub scan/redact` needs
  network access.
- **The hook is fail-closed by default.** If the daemon is unreachable or an
  extractor errors, `scrub-hook` returns a `deny` decision (it will NOT show the
  original file). So a broken/stopped daemon blocks reads. Start it explicitly
  with `.venv/bin/scrub daemon start` and check `.venv/bin/scrub daemon status`.
  Set `fail_mode = "open"` in `~/.config/scrub/config.toml` only if you want the
  opposite behavior.
- **Daemon uses a Unix domain socket** — Linux/macOS only. The CLI auto-spawns
  the daemon when needed; tests run their own daemons in temp dirs via
  `SCRUB_SOCKET` / `SCRUB_CACHE_DIR`.
- **`eval/run_eval.py` rewrites the tracked file `eval/results.json`.** Revert
  it (`git checkout -- eval/results.json`) if you didn't intend to update the
  golden results.

### Quick sanity check (hello-world)
```bash
printf 'Maria Garcia, SSN 458-02-6841, maria@example.com\n' > /tmp/pii.txt
.venv/bin/scrub scan /tmp/pii.txt        # detect only
.venv/bin/scrub redact /tmp/pii.txt --out /tmp/pii.redacted.txt && cat /tmp/pii.redacted.txt
.venv/bin/scrub daemon start
echo '{"tool_name":"Read","tool_input":{"file_path":"/tmp/pii.txt"}}' | .venv/bin/scrub-hook
```
