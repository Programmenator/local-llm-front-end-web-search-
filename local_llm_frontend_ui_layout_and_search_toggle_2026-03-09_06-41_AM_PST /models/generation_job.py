"""
models/generation_job.py

Purpose:
    Explicit in-flight generation state model for one Ollama request.

What this file does:
    - Defines the immutable request snapshot captured when the user presses Send.
    - Stores the request lifecycle state for one streaming generation.
    - Keeps streaming output and error details attached to the request that produced them.

How this file fits into the system:
    The controller uses this model to prevent request state from drifting when
    the UI changes while a response is still streaming. This is the core object
    that isolates in-flight request state from mutable UI/session/config state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from models.chat_message import ChatMessage


@dataclass
class GenerationJob:
    """Represent one in-flight LLM generation request.

    Why this object exists:
        Earlier versions relied on mutable controller fields such as the active
        session and current config after the request had already started. That
        made it possible for a background worker to finish against the wrong
        session or mismatched model metadata. This object fixes that by freezing
        the request inputs at send time and tracking the request lifecycle in
        one place.

    Attributes:
        request_id: Unique identifier for this generation attempt.
        session_id: Target conversation session ID captured at send time.
        session_title_at_start: Session title visible when the request began.
        model_name: Exact model used for this request.
        host: Ollama host captured at send time.
        port: Ollama port captured at send time.
        timeout_seconds: Timeout used for this request.
        system_prompt: Prompt snapshot captured at send time.
        thinking_mode: Whether thinking mode was enabled for this request.
        thinking_level: Requested thinking intensity for compatible models.
        enable_web_search: Whether web-search tools were enabled for this request.
        searxng_base_url: SearXNG instance root captured at send time.
        web_search_category: Default SearXNG category for this request.
        web_search_language: Requested search language for this request.
        web_search_time_range: Requested freshness window for this request.
        web_safe_search: Saved safe-search value for this request.
        web_max_results: Maximum search candidates exposed to the model.
        enable_search_reranker: Whether a dedicated reranker stage was enabled for this request.
        search_reranker_backend: Selected second-stage reranker backend.
        search_reranker_model: Ollama model name used when the Ollama reranker backend is selected.
        sentence_transformers_reranker_model: Hugging Face model name used by the sentence-transformers reranker backend.
        sentence_transformers_reranker_device: Device preference for the sentence-transformers reranker backend.
        web_max_pages: Maximum fetched pages allowed for this request.
        web_fetch_enabled: Whether page fetching was enabled for this request.
        web_fetch_max_response_bytes: Maximum response size allowed for one fetched page.
        web_fetch_max_redirects: Maximum redirect hops allowed for one fetched page.
        web_fetch_allowlist: Optional allowed-domain rules for fetched pages.
        web_fetch_blocklist: Optional denied-domain rules for fetched pages.
        message_history: Frozen copy of the message history sent to Ollama.
        user_message_text: The newly submitted user message that triggered the request.
        started_at: When the request started.
        status: Request lifecycle state such as pending, streaming, completed, or failed.
        response_text: Final accumulated assistant content.
        thinking_text: Final accumulated assistant thinking text.
        error_text: Error captured if the request fails.
        estimated_output_tokens: Approximate output token count derived from streamed text.
        actual_output_tokens: Final token count reported by Ollama when available.
        tokens_per_second: Final or estimated throughput for the assistant output stream.
        throughput_source: Whether throughput came from Ollama metrics or local estimation.
    """

    request_id: str
    session_id: str
    session_title_at_start: str
    model_name: str
    host: str
    port: int
    timeout_seconds: int
    system_prompt: str
    thinking_mode: bool
    thinking_level: str
    enable_web_search: bool = False
    searxng_base_url: str = ""
    web_search_category: str = "general"
    web_search_language: str = "all"
    web_search_time_range: str = "none"
    web_safe_search: int = 1
    web_max_results: int = 8
    enable_search_reranker: bool = False
    search_reranker_backend: str = "disabled"
    search_reranker_model: str = ""
    sentence_transformers_reranker_model: str = "BAAI/bge-reranker-base"
    sentence_transformers_reranker_device: str = "auto"
    web_max_pages: int = 2
    web_fetch_enabled: bool = True
    web_fetch_max_response_bytes: int = 1500000
    web_fetch_max_redirects: int = 4
    web_fetch_allowlist: str = ""
    web_fetch_blocklist: str = ""
    message_history: List[ChatMessage] = field(default_factory=list)
    user_message_text: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    response_text: str = ""
    thinking_text: str = ""
    error_text: str = ""
    estimated_output_tokens: int = 0
    actual_output_tokens: int = 0
    tokens_per_second: float = 0.0
    throughput_source: str = ""

    def to_request_options(self) -> Dict[str, Any]:
        """Return the frozen request settings needed by the Ollama client.

        System interaction summary:
            The controller passes this dictionary into the Ollama service so the
            HTTP request uses the exact model, prompt, host, port, timeout,
            and web-search settings that were active when the user hit Send,
            even if the live config is edited later.
        """
        return {
            "model": self.model_name,
            "host": self.host,
            "port": self.port,
            "timeout_seconds": self.timeout_seconds,
            "system_prompt": self.system_prompt,
            "thinking_mode": self.thinking_mode,
            "thinking_level": self.thinking_level,
            "enable_web_search": self.enable_web_search,
            "searxng_base_url": self.searxng_base_url,
            "web_search_category": self.web_search_category,
            "web_search_language": self.web_search_language,
            "web_search_time_range": self.web_search_time_range,
            "web_safe_search": self.web_safe_search,
            "web_max_results": self.web_max_results,
            "enable_search_reranker": self.enable_search_reranker,
            "search_reranker_backend": self.search_reranker_backend,
            "search_reranker_model": self.search_reranker_model,
            "sentence_transformers_reranker_model": self.sentence_transformers_reranker_model,
            "sentence_transformers_reranker_device": self.sentence_transformers_reranker_device,
            "web_max_pages": self.web_max_pages,
            "web_fetch_enabled": self.web_fetch_enabled,
            "web_fetch_max_response_bytes": self.web_fetch_max_response_bytes,
            "web_fetch_max_redirects": self.web_fetch_max_redirects,
            "web_fetch_allowlist": self.web_fetch_allowlist,
            "web_fetch_blocklist": self.web_fetch_blocklist,
        }
