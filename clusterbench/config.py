"""ServerConfig — declarative launch configuration for the ClusterBench server.

A single YAML file holds everything needed to start the server: which path
(mock/real), where LiteLLM lives (base + metrics URLs), the API key, the default
model, and the HTTP bind. `run_server.py` loads it, then lets explicit CLI flags
override individual fields for ad-hoc runs.

Precedence (lowest → highest): dataclass defaults → YAML file → CLI flags.

The *model*, *streaming*, and *scrape_interval_s* fields are server-level
**defaults**: a `POST /api/run` body may still override them per run. Everything
else (URLs, key, bind, path) is fixed for the server's lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

# Log levels uvicorn accepts; validated at load so a typo fails fast at startup
# rather than deep inside uvicorn.
_LOG_LEVELS = {"critical", "error", "warning", "info", "debug"}


@dataclass
class ServerConfig:
    """Everything needed to launch the server. Field names match the YAML keys
    and the CLI flags (with `-` ↔ `_`)."""

    # --- HTTP bind --------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    # --- Paths ------------------------------------------------------------
    results_dir: str = "results"
    # None → <results_dir>/miniswe, resolved by the caller.
    runner_root: str | None = None

    # --- Wire / model endpoint -------------------------------------------
    # base_url: OpenAI-style endpoint the agent's LLM calls hit (LiteLLM, or
    # mock_litellm on the mock path). metrics_url: LiteLLM /metrics to scrape.
    base_url: str = "http://localhost:4000/v1"
    metrics_url: str = "http://localhost:4000/metrics"
    api_key: str = "sk-mock"

    # --- Path selector ----------------------------------------------------
    # real=False drives the in-process mock; real=True shells out to
    # mini-swe-agent (needs the `real` extra + Docker).
    real: bool = False

    # --- Per-run defaults (a POST /api/run body may override these) -------
    model: str = "gpt-4o-mini"
    streaming: bool = True
    scrape_interval_s: float = 1.0
    step_limit: int = 0

    def __post_init__(self) -> None:
        if self.log_level not in _LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {sorted(_LOG_LEVELS)}, "
                f"got {self.log_level!r}"
            )
        self.port = int(self.port)
        self.step_limit = int(self.step_limit)
        self.scrape_interval_s = float(self.scrape_interval_s)
        self.real = bool(self.real)
        self.streaming = bool(self.streaming)

    @classmethod
    def _field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ServerConfig":
        """Build from a mapping, rejecting unknown keys so typos in the YAML
        (e.g. `base_ur`) fail loudly instead of being silently ignored."""
        known = cls._field_names()
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"unknown config key(s): {sorted(unknown)}; "
                f"valid keys are {sorted(known)}"
            )
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "ServerConfig":
        """Load a ServerConfig from a YAML file. An empty file yields all
        defaults."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"config file {path} must contain a YAML mapping, "
                f"got {type(raw).__name__}"
            )
        return cls.from_dict(raw)

    def merge_overrides(self, overrides: dict[str, Any]) -> "ServerConfig":
        """Return a copy with the given non-None overrides applied. Used to
        layer CLI flags on top of the file (CLI wins)."""
        known = self._field_names()
        merged = {f.name: getattr(self, f.name) for f in fields(self)}
        for k, v in overrides.items():
            if v is None:
                continue
            if k not in known:
                raise ValueError(f"unknown override key: {k!r}")
            merged[k] = v
        return ServerConfig(**merged)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


__all__ = ["ServerConfig"]
