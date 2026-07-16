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


# --------------------------------------------------------------- hook install

@app.command("install-hook")
def install_hook_cmd(
    project: bool = typer.Option(False, "--project", help="Install into ./.claude/settings.json"),
    user: bool = typer.Option(False, "--user", help="Install into ~/.claude/settings.json (default)"),
) -> None:
    """Register the scrub PreToolUse hook in Claude Code settings."""
    from . import install as install_mod

    if project and user:
        typer.echo("scrub: pass at most one of --project / --user", err=True)
        raise typer.Exit(code=2)
    scope = "project" if project else "user"
    path, command = install_mod.install(scope)
    typer.echo(f"scrub: hook installed ({scope}) -> {path}")
    typer.echo(f"  command: {command}")
    raise typer.Exit(code=0)


@app.command("uninstall-hook")
def uninstall_hook_cmd(
    project: bool = typer.Option(False, "--project", help="Uninstall from ./.claude/settings.json"),
    user: bool = typer.Option(False, "--user", help="Uninstall from ~/.claude/settings.json (default)"),
) -> None:
    """Remove the scrub PreToolUse hook from Claude Code settings."""
    from . import install as install_mod

    if project and user:
        typer.echo("scrub: pass at most one of --project / --user", err=True)
        raise typer.Exit(code=2)
    scope = "project" if project else "user"
    path, removed = install_mod.uninstall(scope)
    if removed:
        typer.echo(f"scrub: hook removed ({scope}) -> {path}")
    else:
        typer.echo(f"scrub: no scrub hook found in {path} (nothing to do)")
    raise typer.Exit(code=0)


# --------------------------------------------------------------- daemon control

daemon_app = typer.Typer(add_completion=False, help="Manage the scrub daemon")
app.add_typer(daemon_app, name="daemon")


@daemon_app.command("start")
def daemon_start() -> None:
    """Start the scrub daemon (no-op if one is already running)."""
    from . import client

    config = Config.load()
    try:
        resp = client.ensure_daemon(config)
    except client.DaemonError as e:
        typer.echo(f"scrub: could not start daemon: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"scrub: daemon running (pid {resp.get('pid')})")
    raise typer.Exit(code=0)


@daemon_app.command("stop")
def daemon_stop() -> None:
    """Stop the running scrub daemon."""
    from . import client

    try:
        client.shutdown()
    except OSError:
        typer.echo("scrub: no daemon running")
        raise typer.Exit(code=0)
    typer.echo("scrub: daemon stopped")
    raise typer.Exit(code=0)


@daemon_app.command("status")
def daemon_status() -> None:
    """Report whether the scrub daemon is running."""
    from . import client

    try:
        resp = client.ping()
    except OSError:
        typer.echo("scrub: daemon not running")
        raise typer.Exit(code=1)
    typer.echo(f"scrub: daemon running (pid {resp.get('pid')})")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
