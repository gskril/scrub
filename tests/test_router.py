"""Tests for scrub.router: magic-byte classification + should_skip policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from scrub.config import Config
from scrub.router import BINARY_UNKNOWN, IMAGE, PDF, TEXT, classify, should_skip


def write(tmp_path: Path, name: str, content: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


# --------------------------------------------------------------------------
# classify()
# --------------------------------------------------------------------------


def test_classify_plain_text_by_extension_agnostic_sniff(tmp_path):
    p = write(tmp_path, "notes", b"just some plain english text, no extension\n")
    assert classify(p) == TEXT


def test_classify_source_code(tmp_path):
    p = write(tmp_path, "main.py", b"def f(x):\n    return x + 1\n")
    assert classify(p) == TEXT


def test_classify_json(tmp_path):
    p = write(tmp_path, "data.json", b'{"a": 1, "b": [1, 2, 3]}')
    assert classify(p) == TEXT


def test_classify_env_file(tmp_path):
    p = write(tmp_path, ".env", b"API_KEY=abc123\nDEBUG=true\n")
    assert classify(p) == TEXT


def test_classify_csv(tmp_path):
    p = write(tmp_path, "data.csv", b"name,age\nAlice,30\nBob,40\n")
    assert classify(p) == TEXT


def test_classify_log_file(tmp_path):
    p = write(tmp_path, "app.log", b"2026-01-01 12:00:00 INFO started\n")
    assert classify(p) == TEXT


def test_classify_mislabeled_extension_still_text(tmp_path):
    # .bin extension but actually text content — magic-byte/sniff logic
    # should win over the extension.
    p = write(tmp_path, "data.bin", b"hello this is actually text data\n")
    assert classify(p) == TEXT


def test_classify_pdf_by_magic_bytes(tmp_path):
    # Minimal-but-valid PDF header is enough for filetype's magic-byte sniff.
    pdf_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
    p = write(tmp_path, "doc.pdf", pdf_bytes)
    assert classify(p) == PDF


def test_classify_pdf_extension_lie_is_ignored(tmp_path):
    # File claims to be a .txt but is actually a PDF by magic bytes.
    pdf_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
    p = write(tmp_path, "sneaky.txt", pdf_bytes)
    assert classify(p) == PDF


def test_classify_png_by_magic_bytes(tmp_path):
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    p = write(tmp_path, "image.png", png_header)
    assert classify(p) == IMAGE


def test_classify_jpeg_by_magic_bytes(tmp_path):
    jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 64
    p = write(tmp_path, "photo.jpg", jpeg_header)
    assert classify(p) == IMAGE


def test_classify_binary_unknown_for_random_bytes(tmp_path):
    p = write(tmp_path, "blob.dat", bytes(range(256)) * 4)
    assert classify(p) == BINARY_UNKNOWN


def test_classify_gzip_is_binary_unknown(tmp_path):
    # gzip magic bytes: filetype recognizes it, it's neither pdf nor image.
    p = write(tmp_path, "archive.tar.gz", b"\x1f\x8b\x08\x00" + b"\x00" * 32)
    assert classify(p) == BINARY_UNKNOWN


def test_classify_empty_file_is_text(tmp_path):
    p = write(tmp_path, "empty.txt", b"")
    assert classify(p) == TEXT


# --------------------------------------------------------------------------
# should_skip()
# --------------------------------------------------------------------------


def test_should_skip_respects_skip_globs(tmp_path):
    config = Config()
    p = tmp_path / "node_modules" / "pkg" / "index.js"
    p.parent.mkdir(parents=True)
    p.write_text("console.log('hi')")
    skip, reason = should_skip(p, config)
    assert skip is True
    assert reason == "skip_globs"


def test_should_skip_respects_lockfile_glob(tmp_path):
    config = Config()
    p = tmp_path / "package-lock.lock"
    p.write_text("{}")
    skip, reason = should_skip(p, config)
    assert skip is True
    assert reason == "skip_globs"


def test_should_skip_allow_globs_wins_over_normal_file(tmp_path):
    config = Config(allow_globs=["*.fixture.txt"])
    p = tmp_path / "sample.fixture.txt"
    p.write_text("some content")
    skip, reason = should_skip(p, config)
    assert skip is True
    assert reason == "allow_globs"


def test_should_skip_deny_globs_overrides_allow_globs(tmp_path):
    config = Config(allow_globs=["*.txt"], deny_globs=["*.txt"])
    p = tmp_path / "sample.txt"
    p.write_text("some content")
    skip, reason = should_skip(p, config)
    assert skip is False


def test_should_skip_deny_globs_overrides_skip_globs(tmp_path):
    config = Config(deny_globs=["**/node_modules/**"])
    p = tmp_path / "node_modules" / "pkg" / "index.js"
    p.parent.mkdir(parents=True)
    p.write_text("console.log('hi')")
    skip, reason = should_skip(p, config)
    assert skip is False


def test_should_skip_max_file_bytes(tmp_path):
    config = Config(max_file_bytes=10)
    p = tmp_path / "big.txt"
    p.write_text("this is definitely more than ten bytes")
    skip, reason = should_skip(p, config)
    assert skip is True
    assert reason == "max_file_bytes"


def test_should_skip_normal_small_file_is_not_skipped(tmp_path):
    config = Config()
    p = tmp_path / "normal.txt"
    p.write_text("hello world")
    skip, reason = should_skip(p, config)
    assert skip is False
    assert reason == ""


def test_should_skip_missing_file_is_skipped(tmp_path):
    config = Config()
    p = tmp_path / "does_not_exist.txt"
    skip, reason = should_skip(p, config)
    assert skip is True
