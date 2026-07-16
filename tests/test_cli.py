"""CLI behavior for explicit model installation and offline runtime errors."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from scrub.cli import app
from scrub.detectors.rampart import ModelNotDownloadedError

runner = CliRunner()


def test_bare_root_command_shows_help() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "scrub — local PII redactor" in result.output
    for command in ("download", "redact", "scan", "install-hook", "uninstall-hook", "daemon"):
        assert command in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["download", "--help"],
        ["redact", "--help"],
        ["scan", "--help"],
        ["install-hook", "--help"],
        ["uninstall-hook", "--help"],
        ["daemon", "--help"],
        ["daemon", "start", "--help"],
        ["daemon", "stop", "--help"],
        ["daemon", "status", "--help"],
    ],
)
def test_every_help_command_succeeds(args: list[str]) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.parametrize(
    ("command", "options"),
    [
        ("scan", []),
        ("redact", ["--out", "--json"]),
    ],
)
def test_bare_file_commands_show_help(command: str, options: list[str]) -> None:
    result = runner.invoke(app, [command])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert command in result.output
    assert "FILE" in result.output
    for option in options:
        assert option in result.output


def test_download_command_installs_model(monkeypatch) -> None:
    model_dir = Path("/cache/rampart")
    monkeypatch.setattr("scrub.detectors.rampart.download_model", lambda: model_dir)

    result = runner.invoke(app, ["download"])

    assert result.exit_code == 0
    assert "Rampart model ready" in result.stdout
    assert str(model_dir) in result.stdout


def test_download_failure_is_readable(monkeypatch) -> None:
    def fail():
        raise OSError("network unavailable")

    monkeypatch.setattr("scrub.detectors.rampart.download_model", fail)

    result = runner.invoke(app, ["download"])

    assert result.exit_code == 1
    assert "model download failed: network unavailable" in result.output


def test_scan_without_model_has_actionable_error(monkeypatch, tmp_path) -> None:
    src = tmp_path / "input.txt"
    src.write_text("Maria Garcia")

    def missing_model(self, path):
        raise ModelNotDownloadedError(
            "Rampart model is not installed; run `scrub download` while online first"
        )

    monkeypatch.setattr("scrub.pipeline.Pipeline.scan", missing_model)

    result = runner.invoke(app, ["scan", str(src)])

    assert result.exit_code == 1
    assert "run `scrub download`" in result.output


def test_bare_daemon_command_shows_help() -> None:
    result = runner.invoke(app, ["daemon"])

    assert result.exit_code == 0
    assert "Manage the scrub daemon" in result.output
    assert "start" in result.output
    assert "stop" in result.output
    assert "status" in result.output


def test_daemon_status_when_stopped(monkeypatch) -> None:
    def unavailable():
        raise FileNotFoundError

    monkeypatch.setattr("scrub.client.ping", unavailable)
    result = runner.invoke(app, ["daemon", "status"])

    assert result.exit_code == 1
    assert "daemon not running" in result.output


def test_daemon_start_reports_pid(monkeypatch) -> None:
    monkeypatch.setattr("scrub.client.ensure_daemon", lambda config: {"ok": True, "pid": 1234})
    result = runner.invoke(app, ["daemon", "start"])

    assert result.exit_code == 0
    assert "daemon running (pid 1234)" in result.output


def test_daemon_stop_when_stopped_is_success(monkeypatch) -> None:
    def unavailable():
        raise FileNotFoundError

    monkeypatch.setattr("scrub.client.shutdown", unavailable)
    result = runner.invoke(app, ["daemon", "stop"])

    assert result.exit_code == 0
    assert "no daemon running" in result.output


@pytest.mark.parametrize("command", ["install-hook", "uninstall-hook"])
def test_hook_commands_reject_conflicting_scopes(command: str) -> None:
    result = runner.invoke(app, [command, "--project", "--user"])

    assert result.exit_code == 2
    assert "pass at most one of --project / --user" in result.output
