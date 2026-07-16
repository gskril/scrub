"""Configuration: ~/.config/scrub/config.toml with safe defaults.

Fail mode defaults to CLOSED: if the daemon or an extractor breaks, the hook
denies the Read rather than silently passing raw PII to the model.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .types import DEFAULT_PUBLIC_TYPES, EntityType

# Pinned Rampart revision — bump deliberately, never implicitly (model is alpha).
RAMPART_REPO = "nationaldesignstudio/rampart"
RAMPART_REVISION = "b1993e4e68b082835b80ffc65acc03325ea2e501"


def config_dir() -> Path:
    return Path(os.environ.get("SCRUB_CONFIG_DIR", Path.home() / ".config" / "scrub"))


def cache_dir() -> Path:
    return Path(os.environ.get("SCRUB_CACHE_DIR", Path.home() / ".cache" / "scrub"))


def socket_path() -> Path:
    return Path(os.environ.get("SCRUB_SOCKET", cache_dir() / "scrubd.sock"))


@dataclass(slots=True)
class Config:
    # engines
    enable_regex: bool = True
    enable_rampart: bool = True
    rampart_confidence: float = 0.5
    # Per-type minimum confidence overrides (applied as max(global, per-type)).
    # Address-component types misfire on source code (e.g. "return a + b"
    # scores ~0.74 as SECONDARY_ADDRESS) while real addresses score ~0.98+,
    # so they get a stricter floor by default.
    rampart_type_thresholds: dict[EntityType, float] = field(
        default_factory=lambda: {
            EntityType.SECONDARY_ADDRESS: 0.85,
            EntityType.STREET_NAME: 0.85,
            EntityType.BUILDING_NUMBER: 0.85,
        }
    )
    # entity policy: everything is redacted except these
    public_types: set[EntityType] = field(default_factory=lambda: set(DEFAULT_PUBLIC_TYPES))
    custom_keywords: list[str] = field(default_factory=list)
    # paths
    allow_globs: list[str] = field(default_factory=list)  # never scrub (fixtures etc.)
    deny_globs: list[str] = field(default_factory=list)  # always scrub
    skip_globs: list[str] = field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/.git/**",
            "**/__pycache__/**",
            "**/*.lock",
        ]
    )
    # limits
    max_file_bytes: int = 20 * 1024 * 1024
    # cache
    cache_max_bytes: int = 500 * 1024 * 1024
    # fail mode: "closed" (deny read on error) or "open" (allow original, log)
    fail_mode: str = "closed"
    # daemon
    daemon_idle_exit_s: int = 3600

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or config_dir() / "config.toml"
        cfg = cls()
        if not path.is_file():
            return cfg
        data = tomllib.loads(path.read_text())
        engines = data.get("engines", {})
        cfg.enable_regex = engines.get("regex", cfg.enable_regex)
        cfg.enable_rampart = engines.get("rampart", cfg.enable_rampart)
        cfg.rampart_confidence = engines.get("rampart_confidence", cfg.rampart_confidence)
        if "rampart_type_thresholds" in engines:
            cfg.rampart_type_thresholds = {
                EntityType(k): float(v)
                for k, v in engines["rampart_type_thresholds"].items()
            }
        entities = data.get("entities", {})
        if "public_types" in entities:
            cfg.public_types = {EntityType(t) for t in entities["public_types"]}
        cfg.custom_keywords = entities.get("custom_keywords", cfg.custom_keywords)
        paths = data.get("paths", {})
        cfg.allow_globs = paths.get("allow", cfg.allow_globs)
        cfg.deny_globs = paths.get("deny", cfg.deny_globs)
        cfg.skip_globs = paths.get("skip", cfg.skip_globs)
        limits = data.get("limits", {})
        cfg.max_file_bytes = limits.get("max_file_bytes", cfg.max_file_bytes)
        cache = data.get("cache", {})
        cfg.cache_max_bytes = cache.get("max_bytes", cfg.cache_max_bytes)
        fail = data.get("fail_mode", cfg.fail_mode)
        if fail not in ("open", "closed"):
            raise ValueError(f"fail_mode must be 'open' or 'closed', got {fail!r}")
        cfg.fail_mode = fail
        return cfg
