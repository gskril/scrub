"""Content-hash result cache for the scrub daemon.

Key: ``sha256(file bytes)`` combined with a *config fingerprint* so that a
config change (different keywords, thresholds, engine toggles, …) invalidates
prior results instead of returning stale redactions. Value: the redacted-file
path, the entity count, and the report path — exactly the paths the pipeline
already writes under ``cache_dir()/redacted``. We reuse those paths as the
cached value rather than copying anything.

The index is a single small JSON file under ``cache_dir()``. Writes are atomic
(temp file + ``os.replace``) so a crash mid-write never corrupts it, and a
corrupt/missing index is tolerated by starting empty. An LRU-ish eviction keeps
the total size of cached redacted files under ``config.cache_max_bytes``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config, cache_dir
from .types import RedactionResult

_INDEX_VERSION = 1


def _config_fingerprint(config: Config) -> str:
    """A short, stable hash of every config field that affects redaction
    output. Fields that only affect operational behaviour (cache size, fail
    mode, idle timeout) are intentionally excluded — they don't change what a
    file redacts to."""
    payload = {
        "enable_regex": config.enable_regex,
        "enable_rampart": config.enable_rampart,
        "rampart_confidence": config.rampart_confidence,
        "rampart_type_thresholds": sorted(
            (k.value, float(v)) for k, v in config.rampart_type_thresholds.items()
        ),
        "public_types": sorted(t.value for t in config.public_types),
        "custom_keywords": list(config.custom_keywords),
        "allow_globs": list(config.allow_globs),
        "deny_globs": list(config.deny_globs),
        "skip_globs": list(config.skip_globs),
        "max_file_bytes": config.max_file_bytes,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class CacheEntry:
    redacted_path: Path | None
    report_path: Path | None
    found: int
    size: int  # bytes of the redacted file (0 when nothing written)
    atime: float  # last access, for LRU-ish eviction


class ResultCache:
    """Persistent content-hash cache. Thread-safe for use from the daemon."""

    def __init__(self, config: Config, index_path: Path | None = None) -> None:
        self.config = config
        self.fingerprint = _config_fingerprint(config)
        self.index_path = index_path or (cache_dir() / "index.json")
        self._lock = threading.RLock()
        self._entries: dict[str, CacheEntry] = {}
        self._load()

    # ---------------------------------------------------------------- keys

    def _key(self, path: Path) -> str:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return f"{digest}:{self.fingerprint}"

    # ------------------------------------------------------------ index io

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            data = json.loads(self.index_path.read_text())
        except (OSError, ValueError):
            # Corrupt or unreadable index — start fresh (crash-safe).
            return
        for k, d in data.get("entries", {}).items():
            try:
                self._entries[k] = CacheEntry(
                    redacted_path=Path(d["redacted_path"]) if d.get("redacted_path") else None,
                    report_path=Path(d["report_path"]) if d.get("report_path") else None,
                    found=int(d.get("found", 0)),
                    size=int(d.get("size", 0)),
                    atime=float(d.get("atime", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                continue

    def _persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": _INDEX_VERSION,
            "entries": {
                k: {
                    "redacted_path": str(e.redacted_path) if e.redacted_path else None,
                    "report_path": str(e.report_path) if e.report_path else None,
                    "found": e.found,
                    "size": e.size,
                    "atime": e.atime,
                }
                for k, e in self._entries.items()
            },
        }
        tmp = self.index_path.parent / (self.index_path.name + ".tmp")
        tmp.write_bytes(json.dumps(data).encode("utf-8"))
        os.replace(tmp, self.index_path)  # atomic

    # -------------------------------------------------------------- public

    def get(self, path: Path) -> CacheEntry | None:
        """Return a cached entry for `path`, or None on miss/invalidation.

        Verifies that a cached redacted file still exists on disk; if it was
        evicted or deleted out from under us, the stale entry is dropped and
        this counts as a miss. Reading bumps the entry's access time (LRU).
        """
        key = self._key(path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.found > 0 and (
                entry.redacted_path is None or not entry.redacted_path.exists()
            ):
                del self._entries[key]
                self._persist()
                return None
            entry.atime = time.time()
            self._persist()
            return entry

    def put(self, path: Path, result: RedactionResult) -> CacheEntry:
        """Store the pipeline's result for `path` and evict if over budget."""
        key = self._key(path)
        rp = result.redacted_path
        size = 0
        if rp is not None:
            try:
                size = rp.stat().st_size
            except OSError:
                size = 0
        entry = CacheEntry(
            redacted_path=rp,
            report_path=result.report_path,
            found=result.found,
            size=size,
            atime=time.time(),
        )
        with self._lock:
            self._entries[key] = entry
            self._evict()
            self._persist()
        return entry

    # ------------------------------------------------------------ eviction

    def _evict(self) -> None:
        """Drop oldest entries (and delete their files) until the total size of
        cached redacted files is within `cache_max_bytes`. Caller holds lock."""
        budget = self.config.cache_max_bytes
        total = sum(e.size for e in self._entries.values())
        if total <= budget:
            return
        # Oldest access first.
        for key, entry in sorted(self._entries.items(), key=lambda kv: kv[1].atime):
            if total <= budget:
                break
            for p in (entry.redacted_path, entry.report_path):
                if p is not None:
                    try:
                        p.unlink()
                    except OSError:
                        pass
            total -= entry.size
            del self._entries[key]
