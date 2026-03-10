"""
services/config_service.py

Purpose:
    Persistent configuration storage and normalization.

What this file does:
    - Loads interface settings from disk.
    - Saves interface settings to disk.
    - Normalizes expected value types.
    - Provides the configured Ollama base URL.

How this file fits into the system:
    All user-editable settings flow through this service so the rest of the
    program does not have to manage JSON parsing, defaults, or validation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 11434,
    "model": "",
    "timeout_seconds": 300,
    "system_prompt": "You are a helpful local assistant.",
    "thinking_mode": True,
    "thinking_level": "medium",
    "enable_web_search": False,
    "searxng_base_url": "http://127.0.0.1:8080",
    "web_search_category": "general",
    "web_search_language": "all",
    "web_search_time_range": "none",
    "web_safe_search": 1,
    "web_max_results": 8,
    "web_max_pages": 2,
    "enable_search_reranker": False,
    "search_reranker_backend": "disabled",
    "search_reranker_model": "",
    "sentence_transformers_reranker_model": "BAAI/bge-reranker-base",
    "sentence_transformers_reranker_device": "auto",
    "web_fetch_enabled": True,
    "web_fetch_max_response_bytes": 1500000,
    "web_fetch_max_redirects": 4,
    "web_fetch_allowlist": "",
    "web_fetch_blocklist": "",
}


class ConfigService:
    """Load, validate, and persist application settings for the local LLM UI."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the configuration service and load settings from disk.

        Args:
            config_path: Optional override path for the JSON config file. This is
                useful for testing or running isolated copies of the app.
        """
        self.config_path = config_path or Path(__file__).resolve().parent.parent / "config.json"
        self._config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load settings from disk, falling back to defaults when needed."""
        if not self.config_path.exists():
            self.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()

        try:
            with self.config_path.open("r", encoding="utf-8") as file:
                raw_config = json.load(file)
        except (json.JSONDecodeError, OSError):
            raw_config = {}

        merged = self._merge_with_defaults(raw_config)
        if merged != raw_config:
            self.save_config(merged)
        return merged

    def save_config(self, new_config: Dict[str, Any]) -> None:
        """Persist configuration values to disk using an atomic replace.

        Stability note:
            Writing directly to the live config file risks leaving a truncated or
            half-written JSON file behind if the process is interrupted mid-write.
            This method now writes to a temporary sibling file first and then
            atomically replaces the real config file.
        """
        normalized_config = self._merge_with_defaults(new_config)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.config_path.with_suffix(f"{self.config_path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(normalized_config, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, self.config_path)
        self._config = normalized_config

    def get_config(self) -> Dict[str, Any]:
        """Return a copy of the currently loaded configuration."""
        return self._config.copy()

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Apply partial updates to the current configuration and save them."""
        merged = self._config.copy()
        merged.update(updates)
        self.save_config(merged)
        return self.get_config()

    def get_base_url(self) -> str:
        """Build the Ollama base URL from the saved host and port values."""
        host = str(self._config.get("host", DEFAULT_CONFIG["host"])).strip() or str(DEFAULT_CONFIG["host"])
        port = self._safe_int(self._config.get("port"), int(DEFAULT_CONFIG["port"]))
        return f"http://{host}:{port}"

    def _merge_with_defaults(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """Combine user config with defaults and normalize expected value types.

        Long-term stability note:
            User-edited JSON files can drift into invalid states such as
            non-numeric ports or timeouts. Earlier revisions allowed those bad
            values to raise exceptions during startup. This method now normalizes
            each field defensively so a damaged config file falls back to safe
            defaults instead of crashing the whole application.
        """
        merged = DEFAULT_CONFIG.copy()
        merged.update(raw_config or {})

        merged["host"] = str(merged.get("host", DEFAULT_CONFIG["host"])).strip() or DEFAULT_CONFIG["host"]
        merged["port"] = self._safe_int(merged.get("port"), int(DEFAULT_CONFIG["port"]))
        merged["model"] = str(merged.get("model", DEFAULT_CONFIG["model"])).strip()
        merged["timeout_seconds"] = max(5, self._safe_int(merged.get("timeout_seconds"), int(DEFAULT_CONFIG["timeout_seconds"])))
        merged["system_prompt"] = str(
            merged.get("system_prompt", DEFAULT_CONFIG["system_prompt"])
        ).strip() or DEFAULT_CONFIG["system_prompt"]
        merged["thinking_mode"] = self._safe_bool(merged.get("thinking_mode"), bool(DEFAULT_CONFIG["thinking_mode"]))

        thinking_level = str(merged.get("thinking_level", DEFAULT_CONFIG["thinking_level"])).strip().lower()
        if thinking_level not in {"low", "medium", "high"}:
            thinking_level = DEFAULT_CONFIG["thinking_level"]
        merged["thinking_level"] = thinking_level

        merged["enable_web_search"] = self._safe_bool(
            merged.get("enable_web_search"),
            bool(DEFAULT_CONFIG["enable_web_search"]),
        )
        merged["searxng_base_url"] = str(
            merged.get("searxng_base_url", DEFAULT_CONFIG["searxng_base_url"])
        ).strip() or DEFAULT_CONFIG["searxng_base_url"]

        search_category = str(
            merged.get("web_search_category", DEFAULT_CONFIG["web_search_category"])
        ).strip().lower()
        if search_category not in {"general", "news", "science", "it"}:
            search_category = DEFAULT_CONFIG["web_search_category"]
        merged["web_search_category"] = search_category

        merged["web_search_language"] = str(
            merged.get("web_search_language", DEFAULT_CONFIG["web_search_language"])
        ).strip() or DEFAULT_CONFIG["web_search_language"]

        time_range = str(
            merged.get("web_search_time_range", DEFAULT_CONFIG["web_search_time_range"])
        ).strip().lower()
        if time_range not in {"none", "day", "week", "month", "year"}:
            time_range = DEFAULT_CONFIG["web_search_time_range"]
        merged["web_search_time_range"] = time_range

        merged["web_safe_search"] = max(
            0,
            min(2, self._safe_int(merged.get("web_safe_search"), int(DEFAULT_CONFIG["web_safe_search"]))),
        )
        merged["web_max_results"] = max(
            1,
            min(20, self._safe_int(merged.get("web_max_results"), int(DEFAULT_CONFIG["web_max_results"]))),
        )
        merged["web_max_pages"] = max(
            1,
            min(5, self._safe_int(merged.get("web_max_pages"), int(DEFAULT_CONFIG["web_max_pages"]))),
        )
        merged["enable_search_reranker"] = self._safe_bool(
            merged.get("enable_search_reranker"),
            bool(DEFAULT_CONFIG["enable_search_reranker"]),
        )
        reranker_backend = str(
            merged.get("search_reranker_backend", DEFAULT_CONFIG["search_reranker_backend"])
        ).strip().lower()
        if reranker_backend not in {"disabled", "ollama", "sentence_transformers"}:
            reranker_backend = "disabled"
        merged["search_reranker_backend"] = reranker_backend
        merged["search_reranker_model"] = str(
            merged.get("search_reranker_model", DEFAULT_CONFIG["search_reranker_model"])
        ).strip()
        merged["sentence_transformers_reranker_model"] = str(
            merged.get(
                "sentence_transformers_reranker_model",
                DEFAULT_CONFIG["sentence_transformers_reranker_model"],
            )
        ).strip() or DEFAULT_CONFIG["sentence_transformers_reranker_model"]
        sentence_transformers_device = str(
            merged.get(
                "sentence_transformers_reranker_device",
                DEFAULT_CONFIG["sentence_transformers_reranker_device"],
            )
        ).strip().lower()
        if sentence_transformers_device not in {"auto", "cpu", "cuda", "mps"}:
            sentence_transformers_device = DEFAULT_CONFIG["sentence_transformers_reranker_device"]
        merged["sentence_transformers_reranker_device"] = sentence_transformers_device
        merged["web_fetch_enabled"] = self._safe_bool(
            merged.get("web_fetch_enabled"),
            bool(DEFAULT_CONFIG["web_fetch_enabled"]),
        )
        merged["web_fetch_max_response_bytes"] = max(
            1024,
            min(10000000, self._safe_int(
                merged.get("web_fetch_max_response_bytes"),
                int(DEFAULT_CONFIG["web_fetch_max_response_bytes"]),
            )),
        )
        merged["web_fetch_max_redirects"] = max(
            0,
            min(10, self._safe_int(
                merged.get("web_fetch_max_redirects"),
                int(DEFAULT_CONFIG["web_fetch_max_redirects"]),
            )),
        )
        merged["web_fetch_allowlist"] = str(
            merged.get("web_fetch_allowlist", DEFAULT_CONFIG["web_fetch_allowlist"])
        ).strip()
        merged["web_fetch_blocklist"] = str(
            merged.get("web_fetch_blocklist", DEFAULT_CONFIG["web_fetch_blocklist"])
        ).strip()
        return merged

    def _safe_int(self, value: Any, default: int) -> int:
        """Return an integer value or a safe default when coercion fails."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _safe_bool(self, value: Any, default: bool) -> bool:
        """Return a predictable boolean from JSON-ish config values."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        return default
