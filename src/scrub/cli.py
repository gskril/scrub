"""scrub CLI.

`scrub redact FILE [--out PATH] [--json]` — redact PII, write a sanitized
copy (to the cache, or --out), print a summary.
`scrub scan FILE` — detect only; lists entity types + counts, writes nothing.

Structured as a bare `typer.Typer()` so later phases can register more
subcommands with `app.add_typer(...)` (daemon control, hook install) without
touching this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import Config
from .pipeline import Pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True, help="scrub — local PII redactor")


@app.command()
def redact(
    file: Path = typer.Argument(..., exists=True, readable=True, help="File to redact"),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Also copy the redacted file to this path"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print a machine-readable JSON summary instead of text"
    ),
) -> None:
    """Redact PII in FILE, writing a sanitized copy (cache dir, or --out)."""
    config = Config.load()
    pipeline = Pipeline.default(config)
    result = pipeline.redact_file(file)

    display_path = result.redacted_path
    if out is not None and result.redacted_path is not None:
        out.write_bytes(result.redacted_path.read_bytes())
        display_path = out

    if json_output:
        payload = {
            "original_path": str(result.original_path),
            "redacted_path": str(display_path) if display_path else None,
            "found": result.found,
            "entities": [
                {
                    "entity_type": e.entity_type.value,
                    "start": e.start,
                    "end": e.end,
                    "confidence": e.confidence,
                    "source": e.source,
                }
                for e in result.entities
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        if result.found == 0:
            typer.echo(f"scrub: no redactable PII found in {file} (passthrough)")
        else:
            plural = "y" if result.found == 1 else "ies"
            typer.echo(f"scrub: redacted {result.found} entit{plural} -> {display_path}")
            counts: dict[str, int] = {}
            for e in result.entities:
                if e.entity_type not in config.public_types:
                    counts[e.entity_type.value] = counts.get(e.entity_type.value, 0) + 1
            for etype, n in sorted(counts.items()):
                typer.echo(f"  {etype}: {n}")
    raise typer.Exit(code=0)


@app.command()
def scan(
    file: Path = typer.Argument(..., exists=True, readable=True, help="File to scan"),
) -> None:
    """Detect PII in FILE without writing anything."""
    config = Config.load()
    pipeline = Pipeline.default(config)
    spans = pipeline.scan(file)

    if not spans:
        typer.echo(f"scrub: no PII detected in {file}")
        raise typer.Exit(code=0)

    counts: dict[str, int] = {}
    for s in spans:
        counts[s.entity_type.value] = counts.get(s.entity_type.value, 0) + 1

    typer.echo(f"scrub: {len(spans)} entities detected in {file}")
    for etype, n in sorted(counts.items()):
        typer.echo(f"  {etype}: {n}")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
