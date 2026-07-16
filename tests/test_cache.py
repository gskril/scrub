"""Unit tests for the content-hash result cache (Phase 4)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from scrub.cache import ResultCache, _config_fingerprint
from scrub.config import Config
from scrub.types import RedactionResult


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    cdir = tmp_path / "cache"
    cdir.mkdir()
    monkeypatch.setenv("SCRUB_CACHE_DIR", str(cdir))
    return cdir


def _make_result(cdir: Path, name: str, found: int, size: int) -> RedactionResult:
    """A RedactionResult whose redacted file really exists on disk (size bytes)."""
    if found == 0:
        return RedactionResult(
            original_path=Path("/x") / name, redacted_path=None, found=0, entities=[]
        )
    redacted = cdir / "redacted" / f"{name}.redacted.txt"
    redacted.parent.mkdir(parents=True, exist_ok=True)
    redacted.write_bytes(b"x" * size)
    report = cdir / "redacted" / f"{name}.report.json"
    report.write_text("{}")
    return RedactionResult(
        original_path=Path("/x") / name,
        redacted_path=redacted,
        found=found,
        entities=[],
        report_path=report,
    )


def _source(tmp_path: Path, name: str, content: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_hit_and_miss(cache_env, tmp_path):
    cache = ResultCache(Config())
    src = _source(tmp_path, "a.txt", b"hello world")

    assert cache.get(src) is None  # miss before put

    result = _make_result(cache_env, "a", found=3, size=50)
    cache.put(src, result)

    entry = cache.get(src)
    assert entry is not None
    assert entry.found == 3
    assert entry.redacted_path == result.redacted_path

    other = _source(tmp_path, "b.txt", b"different bytes")
    assert cache.get(other) is None


def test_found_zero_is_cached(cache_env, tmp_path):
    cache = ResultCache(Config())
    src = _source(tmp_path, "clean.txt", b"nothing sensitive")
    cache.put(src, _make_result(cache_env, "clean", found=0, size=0))
    entry = cache.get(src)
    assert entry is not None
    assert entry.found == 0
    assert entry.redacted_path is None


def test_persists_across_instances(cache_env, tmp_path):
    src = _source(tmp_path, "a.txt", b"hello")
    ResultCache(Config()).put(src, _make_result(cache_env, "a", found=2, size=10))
    # A fresh cache instance loads the on-disk index.
    entry = ResultCache(Config()).get(src)
    assert entry is not None and entry.found == 2


def test_config_fingerprint_invalidation(cache_env, tmp_path):
    src = _source(tmp_path, "a.txt", b"hello")
    cfg_a = Config()
    ResultCache(cfg_a).put(src, _make_result(cache_env, "a", found=1, size=10))
    assert ResultCache(cfg_a).get(src) is not None

    cfg_b = Config()
    cfg_b.custom_keywords = ["Project Nightjar"]
    assert _config_fingerprint(cfg_a) != _config_fingerprint(cfg_b)
    # Same file bytes, different config -> different key -> miss.
    assert ResultCache(cfg_b).get(src) is None


def test_missing_redacted_file_invalidates(cache_env, tmp_path):
    cache = ResultCache(Config())
    src = _source(tmp_path, "a.txt", b"hello")
    result = _make_result(cache_env, "a", found=1, size=10)
    cache.put(src, result)
    assert cache.get(src) is not None

    result.redacted_path.unlink()  # file evicted/deleted out from under us
    assert cache.get(src) is None
    # And the stale entry is gone from a freshly loaded index.
    assert ResultCache(Config()).get(src) is None


def test_eviction_respects_budget(cache_env, tmp_path):
    cfg = Config()
    cfg.cache_max_bytes = 250
    cache = ResultCache(cfg)

    sources = []
    for i in range(3):
        src = _source(tmp_path, f"f{i}.txt", f"content-{i}".encode())
        cache.put(src, _make_result(cache_env, f"f{i}", found=1, size=100))
        sources.append(src)
        time.sleep(0.01)  # keep access times strictly ordered

    total = sum(e.size for e in cache._entries.values())
    assert total <= 250

    # Oldest (f0) evicted; its file is gone. Newest (f2) survives.
    assert cache.get(sources[0]) is None
    assert not (cache_env / "redacted" / "f0.redacted.txt").exists()
    assert cache.get(sources[2]) is not None
    assert (cache_env / "redacted" / "f2.redacted.txt").exists()
