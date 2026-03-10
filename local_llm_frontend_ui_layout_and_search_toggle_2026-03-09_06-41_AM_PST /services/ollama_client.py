"""
services/ollama_client.py

Purpose:
    HTTP client for Ollama chat, model discovery, model unloading, and optional
    native tool-calling integration for SearXNG-backed web search.

What this file does:
    - Sends streaming chat requests to Ollama.
    - Adapts think-mode settings per model family.
    - Lists installed models and tests server connectivity.
    - Requests explicit model unload to free VRAM.
    - Optionally exposes `search_web` and `fetch_url_content` tools to models
      when web search is enabled in the saved configuration.
    - Executes those tools locally and feeds the tool results back to Ollama so
      the final answer can be grounded in selected search results instead of the
      first raw search hits.

How this file fits into the system:
    This service is the boundary between the desktop app and the Ollama HTTP
    API. It centralizes model request formatting so controller and UI code do
    not need to understand Ollama payload structure or tool-call message flow.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional
import re
from urllib import error, request

from models.chat_message import ChatMessage
from services.config_service import ConfigService
from services.reranker_service import (
    NonLlmSearchRerankerService,
    SearchRerankerService,
    SentenceTransformersSearchRerankerService,
)
from services.search_service import SearchService
from services.web_fetch_service import WebFetchService

StreamCallback = Callable[[str, str], None]


class OllamaClient:
    """Communicate with a local Ollama server over HTTP."""

    def __init__(self, config_service: ConfigService) -> None:
        """Store configuration access for building Ollama API requests."""
        self.config_service = config_service

    def chat_stream(
        self,
        messages: List[ChatMessage],
        on_chunk: StreamCallback,
        request_options: Dict[str, Any] | None = None,
    ) -> Dict[str, str]:
        """Send the full conversation to Ollama and stream the assistant reply.

        When web search is enabled, this method first offers the model two tools:
        `search_web` and `fetch_url_content`. If the model calls either tool,
        the client executes the tool locally and performs a second streamed chat
        request using the tool results. If the model does not call a tool, the
        initial non-streaming reply is returned directly.
        """
        config = self._resolve_request_options(request_options)
        model_name = str(config.get("model", "")).strip()
        if not model_name:
            raise RuntimeError("No model is selected. Open Settings or use the model dropdown first.")

        chat_messages = self._build_chat_messages(
            str(config.get("system_prompt", "")),
            messages,
            web_tools_enabled=self._web_tools_enabled(config),
        )
        if self._web_tools_enabled(config):
            return self._chat_with_optional_tools(chat_messages, on_chunk, config)
        return self._stream_chat_once(chat_messages, on_chunk, config)

    def test_connection(self) -> Dict[str, Any]:
        """Verify that the configured Ollama server is reachable."""
        try:
            response = self._get_json("/api/tags")
            model_count = len(response.get("models", []))
            return {
                "ok": True,
                "message": f"Connected successfully. Found {model_count} model(s).",
            }
        except RuntimeError as exc:
            return {"ok": False, "message": str(exc)}

    def list_models(self) -> List[str]:
        """Return the names of models currently available in Ollama."""
        response = self._get_json("/api/tags")
        models = response.get("models", [])
        names = [str(item.get("name", "")) for item in models if item.get("name")]
        return sorted(names)

    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Ask Ollama to immediately unload one model from memory."""
        cleaned_model = model_name.strip()
        if not cleaned_model:
            raise RuntimeError("No model is selected, so there is nothing to unload.")

        payload = {
            "model": cleaned_model,
            "prompt": "",
            "keep_alive": 0,
            "stream": False,
        }
        return self._request_json(endpoint="/api/generate", method="POST", payload=payload)

    def _chat_with_optional_tools(
        self,
        chat_messages: List[Dict[str, Any]],
        on_chunk: StreamCallback,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Offer web tools to Ollama, execute any tool calls, then stream the final answer.

        Request-scoped runtime state is tracked here so UI-exposed limits such as
        ``web_max_pages`` are enforced during execution rather than merely saved
        in configuration. This prevents one model turn from fetching more pages
        than the user explicitly allowed in Settings.
        """
        runtime_state = self._build_tool_runtime_state(config)
        initial_payload = {
            "model": str(config["model"]),
            "messages": chat_messages,
            "stream": False,
            "think": self._build_think_value(str(config["model"]), config),
            "tools": self._build_web_tools_schema(config),
        }
        response = self._request_json_with_request_options("/api/chat", "POST", initial_payload, config)
        message = response.get("message", {}) or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            fallback_messages = self._maybe_force_web_search_fallback(
                chat_messages,
                message,
                config,
                runtime_state=runtime_state,
            )
            if fallback_messages is not None:
                return self._stream_chat_once(fallback_messages, on_chunk, config)

            content = str(message.get("content", "")).strip()
            thinking = str(message.get("thinking", "")).strip()
            if thinking:
                on_chunk("thinking", thinking)
            if content:
                on_chunk("content", content)
            return {
                "content": content or "[Model returned an empty response.]",
                "thinking": thinking,
                "output_tokens": int(response.get("eval_count", 0) or 0),
                "tokens_per_second": self._compute_tokens_per_second(response),
                "throughput_source": "ollama_eval" if response.get("eval_duration") else "",
            }

        augmented_messages = list(chat_messages)
        augmented_messages.append(self._build_post_tool_grounding_message())
        augmented_messages.append(message)
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("function", {}).get("name", "")).strip()
            arguments = tool_call.get("function", {}).get("arguments", {}) or {}
            tool_result = self._execute_tool_call(tool_name, arguments, config, runtime_state=runtime_state)
            augmented_messages.append(self._build_tool_result_message(tool_call, tool_result))

        return self._stream_chat_once(augmented_messages, on_chunk, config)

    def _maybe_force_web_search_fallback(
        self,
        chat_messages: List[Dict[str, Any]],
        assistant_message: Dict[str, Any],
        config: Dict[str, Any],
        runtime_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]] | None:

        """Run one defensive search when a tool-capable model claims no web access.

        Some local models support tools but still answer with generic text such as
        "I do not have internet access" instead of emitting a tool call. When that
        happens and web tools are enabled, this fallback runs one search against the
        latest user message and re-asks the model with the normalized results.

        The fallback now reuses the same request-scoped runtime state as the main
        tool path so fetched-page limits stay consistent across both code paths.
        It also preserves the original assistant draft in the message history so
        the follow-up pass can revise that answer instead of responding as if the
        first pass never happened.
        """
        content = str(assistant_message.get("content", "") or "").strip()
        latest_user_message = self._get_latest_user_message(chat_messages)
        if not latest_user_message:
            return None

        should_force_search = self._response_indicates_missing_web_access(content)
        if not should_force_search and self._prompt_likely_requires_web_search(latest_user_message):
            should_force_search = True
        if not should_force_search:
            return None

        search_arguments = {
            "query": latest_user_message,
            "category": str(config.get("web_search_category", "general")),
            "time_range": str(config.get("web_search_time_range", "none")),
            "max_results": int(config.get("web_max_results", 8) or 8),
        }
        tool_result = self._execute_tool_call(
            "search_web",
            search_arguments,
            config,
            runtime_state=runtime_state,
        )
        fallback_messages = list(chat_messages)
        if assistant_message:
            fallback_messages.append({
                "role": str(assistant_message.get("role", "assistant") or "assistant"),
                "content": str(assistant_message.get("content", "") or ""),
            })
        fallback_messages.append({
            "role": "system",
            "content": (
                "Web search is enabled in this application. The previous assistant draft failed to use it. "
                "Revise the prior answer using the retrieved search results below. If the results are insufficient, "
                "say what is still missing. Do not claim that web access is unavailable."
            ),
        })
        fallback_messages.append(self._build_post_tool_grounding_message())
        fallback_messages.append({
            "role": "system",
            "content": "SEARCH_WEB_RESULTS: " + json.dumps(tool_result, ensure_ascii=False),
        })
        return fallback_messages

    def _response_indicates_missing_web_access(self, content: str) -> bool:
        """Return True when the draft answer admits no internet or browsing access.

        This intentionally catches both direct statements ("I do not have internet
        access") and softer evasions used by some local models when current news
        is requested ("I don't have access to real-time news updates").
        """
        normalized = content.strip().lower()
        if not normalized:
            return False
        patterns = [
            r"\bi do not have (?:direct )?(?:internet|web|browsing) access\b",
            r"\bi don't have (?:direct )?(?:internet|web|browsing) access\b",
            r"\bi cannot access the internet\b",
            r"\bi can't access the internet\b",
            r"\bi am unable to browse\b",
            r"\bi can't browse the web\b",
            r"\bi cannot browse the web\b",
            r"\bno internet access\b",
            r"\b(?:do not|don't) have access to real[- ]time (?:news|updates|information)\b",
            r"\b(?:cannot|can't) provide (?:live|real[- ]time|current) (?:news|updates|information)\b",
        ]
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _prompt_likely_requires_web_search(self, user_message: str) -> bool:
        """Return True when the user's request strongly implies current/external lookup.

        This fallback exists because some Ollama-served models ignore tool schemas
        even when the runtime supports them. For obviously web-dependent prompts,
        the client now force-runs one search instead of trusting an ungrounded
        freeform answer.
        """
        normalized = user_message.strip().lower()
        if not normalized:
            return False
        explicit_lookup_patterns = [
            r"\bsearch for\b",
            r"\blook up\b",
            r"\bfind online\b",
            r"\bon the internet\b",
            r"\bcheck the web\b",
            r"\bbrowse for\b",
        ]
        if any(re.search(pattern, normalized) for pattern in explicit_lookup_patterns):
            return True

        recency_plus_external_patterns = [
            r"\blatest\s+(?:news|headlines|updates|price|prices|weather|forecast|scores?|standings|release|releases|version|versions|report|reports)\b",
            r"\bcurrent\s+(?:news|events|weather|forecast|price|prices|score|scores|standings|version|versions|conditions|status)\b",
            r"\btoday(?:'s)?\s+(?:news|weather|forecast|score|scores|headlines|price|prices)\b",
            r"\brecent\s+(?:news|headlines|updates|developments)\b",
            r"\bthis\s+(?:week|month|year)'?s\s+(?:news|weather|forecast|release|releases|updates)\b",
            r"\bwhat(?:'s| is) the latest on\b",
        ]
        return any(re.search(pattern, normalized) for pattern in recency_plus_external_patterns)

    def _get_latest_user_message(self, chat_messages: List[Dict[str, Any]]) -> str:
        """Return the newest user message content from the chat payload."""
        for message in reversed(chat_messages):
            if str(message.get("role", "")) == "user":
                return str(message.get("content", "") or "").strip()
        return ""


    def _build_tool_result_message(
        self,
        tool_call: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return one protocol-safe tool result message for a prior assistant tool call.

        Ollama's chat API expects tool results to directly follow the assistant
        message that emitted ``tool_calls``. When an ID is available, it is
        preserved as ``tool_call_id`` so multi-tool responses can be paired with
        the originating call deterministically.
        """
        message = {
            "role": "tool",
            "content": json.dumps(tool_result, ensure_ascii=False),
        }
        tool_call_id = str(tool_call.get("id", "") or "").strip()
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        tool_name = str(tool_call.get("function", {}).get("name", "") or "").strip()
        if tool_name:
            message["name"] = tool_name
        return message

    def _build_post_tool_grounding_message(self) -> Dict[str, str]:
        """Return a system message that tells the model how to use tool output.

        Some local models can successfully call a web-search tool, then still
        hesitate because the returned results are newer than their pretraining
        cutoff. This instruction makes the runtime contract explicit: tool
        output is fresh external context retrieved by the application at answer
        time, so the model should use it instead of refusing due to cutoff age.
        """
        return {
            "role": "system",
            "content": (
                "The application has already retrieved live external data for this turn. "
                "Treat tool results and injected SEARCH_WEB_RESULTS as current runtime evidence, even when they are newer "
                "than your training cutoff. Do not refuse or down-rank them just because they are post-cutoff. "
                "Use the retrieved data directly, attribute it as web-search results when helpful, and only note uncertainty "
                "when the retrieved content itself is incomplete or conflicting."
            ),
        }

    def _stream_chat_once(
        self,
        chat_messages: List[Dict[str, Any]],
        on_chunk: StreamCallback,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform one normal streamed chat request and accumulate the final output."""
        payload = {
            "model": str(config["model"]),
            "messages": chat_messages,
            "stream": True,
            "think": self._build_think_value(str(config["model"]), config),
        }

        content_parts: List[str] = []
        thinking_parts: List[str] = []

        try:
            stream_items = self._stream_json_lines("/api/chat", payload, config)
            result = self._consume_stream(stream_items, on_chunk, content_parts, thinking_parts)
        except RuntimeError as exc:
            if self._should_retry_with_boolean_thinking(exc, payload.get("think")):
                payload["think"] = True
                content_parts.clear()
                thinking_parts.clear()
                stream_items = self._stream_json_lines("/api/chat", payload, config)
                result = self._consume_stream(stream_items, on_chunk, content_parts, thinking_parts)
            else:
                raise

        return result

    def _build_web_tools_schema(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the tool schema list exposed to tool-capable models."""
        search_description = (
            "Search the web through the configured SearXNG instance and return normalized candidates. "
            "Use this when external information is needed instead of assuming the first result is best. "
            "A deterministic first-pass reranker always reorders candidates using query, title, snippet, and domain relevance signals before you see them. "
            "When the dedicated search reranker is enabled in settings, the optional second stage runs after the deterministic reranker using either Ollama or sentence-transformers, depending on the selected backend."
        )
        max_pages = max(1, int(config.get("web_max_pages", 2) or 2))
        fetch_description = (
            "Fetch readable page text for one selected search result URL after you have decided the page is relevant. "
            f"At most {max_pages} page fetch(es) may be performed in a single answer. "
            "Only public http/https targets are allowed. Localhost, private IP ranges, link-local targets, blocked "
            "domains, oversized responses, unsupported content types, and redirect chains beyond the configured limit "
            "will be rejected by the application."
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": search_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query to run."},
                            "category": {
                                "type": "string",
                                "description": "SearXNG result category.",
                                "enum": ["general", "news", "science", "it"],
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Optional result freshness window.",
                                "enum": ["none", "day", "week", "month", "year"],
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "How many normalized results to return.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
        if bool(config.get("web_fetch_enabled", True)):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_url_content",
                        "description": fetch_description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "The page URL to fetch."},
                            },
                            "required": ["url"],
                        },
                    },
                }
            )
        return tools

    def _execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        config: Dict[str, Any],
        runtime_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one local tool requested by the model and return structured output.

        ``runtime_state`` tracks request-scoped limits that must be enforced
        while a single answer is being built. The most important one is the page
        fetch counter behind ``web_max_pages`` so the backend behavior matches
        the limit exposed in the Settings UI.
        """
        if tool_name == "search_web":
            service = SearchService(
                base_url=str(config.get("searxng_base_url", "http://127.0.0.1:8080")),
                timeout_seconds=int(config.get("timeout_seconds", 30)),
            )
            query = str(arguments.get("query", ""))
            search_result = service.search(
                query,
                category=str(arguments.get("category", config.get("web_search_category", "general"))),
                language=str(config.get("web_search_language", "all")),
                time_range=str(arguments.get("time_range", config.get("web_search_time_range", "none"))),
                safe_search=int(config.get("web_safe_search", 1)),
                max_results=min(
                    int(arguments.get("max_results", config.get("web_max_results", 8)) or 8),
                    int(config.get("web_max_results", 8) or 8),
                ),
            )
            search_result = self._apply_non_llm_search_reranker(
                query=query,
                search_result=search_result,
            )
            if self._search_reranker_enabled(config):
                return self._apply_search_reranker(
                    query=query,
                    search_result=search_result,
                    config=config,
                )
            search_result["reranker"] = {
                "enabled": False,
                "applied": False,
                "backend": "disabled",
                "model": str(config.get("search_reranker_model", "") or "").strip(),
                "reason": "Search reranker is disabled in settings.",
            }
            return search_result
        if tool_name == "fetch_url_content":
            if not bool(config.get("web_fetch_enabled", True)):
                raise RuntimeError("Web page fetching is disabled in settings.")

            page_limit = max(1, int(config.get("web_max_pages", 2) or 2))
            state = runtime_state if runtime_state is not None else self._build_tool_runtime_state(config)
            fetched_pages = int(state.get("page_fetch_count", 0) or 0)
            if fetched_pages >= page_limit:
                return {
                    "ok": False,
                    "error": (
                        f"Page fetch limit reached for this answer. The current Settings value allows at most "
                        f"{page_limit} fetched page(s) per request."
                    ),
                    "page_fetch_limit": page_limit,
                    "page_fetch_count": fetched_pages,
                }

            service = WebFetchService(
                timeout_seconds=int(config.get("timeout_seconds", 30)),
                max_response_bytes=int(config.get("web_fetch_max_response_bytes", 1500000) or 1500000),
                max_redirects=int(config.get("web_fetch_max_redirects", 4) or 4),
                domain_allowlist=self._parse_domain_rules(config.get("web_fetch_allowlist", "")),
                domain_blocklist=self._parse_domain_rules(config.get("web_fetch_blocklist", "")),
            )
            result = service.fetch(str(arguments.get("url", "")))
            state["page_fetch_count"] = fetched_pages + 1
            state["page_fetch_limit"] = page_limit
            return result
        raise RuntimeError(f"Model requested an unknown tool: {tool_name}")


    def _apply_non_llm_search_reranker(
        self,
        *,
        query: str,
        search_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply the deterministic first-pass reranker to the raw SearXNG results."""
        try:
            service = NonLlmSearchRerankerService()
            return service.rerank_search_payload(query, search_result)
        except Exception as exc:
            fallback = dict(search_result)
            fallback["heuristic_reranker"] = {
                "enabled": True,
                "applied": False,
                "method": "query_title_snippet_domain_scoring",
                "error": str(exc),
            }
            return fallback

    def _apply_search_reranker(
        self,
        *,
        query: str,
        search_result: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return reranked search results or the original order when reranking fails.

        The reranker stage is intentionally non-fatal. Search remains usable even
        if the dedicated reranker backend is missing, fails to load, returns
        malformed output, or times out. In those cases the original order from
        the deterministic first-pass reranker is preserved and explanatory
        metadata is attached for traceability.
        """
        backend = self._get_search_reranker_backend(config)
        model_name = self._get_search_reranker_model_name(config, backend)
        try:
            if backend == "sentence_transformers":
                service = SentenceTransformersSearchRerankerService(
                    model_name=model_name,
                    device=str(config.get("sentence_transformers_reranker_device", "auto")),
                )
            else:
                service = SearchRerankerService(
                    host=str(config.get("host", "127.0.0.1")),
                    port=int(config.get("port", 11434)),
                    model_name=model_name,
                    timeout_seconds=int(config.get("timeout_seconds", 30)),
                )
            return service.rerank_search_payload(query, search_result)
        except Exception as exc:
            fallback = dict(search_result)
            fallback["reranker"] = {
                "enabled": True,
                "applied": False,
                "backend": backend,
                "model": model_name,
                "error": str(exc),
            }
            return fallback

    def _search_reranker_enabled(self, config: Dict[str, Any]) -> bool:
        """Return True when the dedicated pre-main-model reranker should run."""
        if not bool(config.get("enable_search_reranker", False)):
            return False
        backend = self._get_search_reranker_backend(config)
        if backend == "sentence_transformers":
            return bool(str(config.get("sentence_transformers_reranker_model", "")).strip())
        if backend == "ollama":
            return bool(str(config.get("search_reranker_model", "")).strip())
        return False

    def _get_search_reranker_backend(self, config: Dict[str, Any]) -> str:
        """Return the configured reranker backend, preserving legacy behavior."""
        backend = str(config.get("search_reranker_backend", "")).strip().lower()
        if backend in {"disabled", "ollama", "sentence_transformers"}:
            return backend
        if bool(str(config.get("search_reranker_model", "")).strip()):
            return "ollama"
        return "disabled"

    def _get_search_reranker_model_name(self, config: Dict[str, Any], backend: str) -> str:
        """Return the model name associated with the configured reranker backend."""
        if backend == "sentence_transformers":
            return str(config.get("sentence_transformers_reranker_model", "") or "").strip()
        return str(config.get("search_reranker_model", "") or "").strip()

    def _build_tool_runtime_state(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create per-request tool execution counters and limits.

        This object exists so one answer can enforce UI-configured limits across
        multiple model-emitted tool calls without mutating the saved application
        configuration itself.
        """
        return {
            "page_fetch_count": 0,
            "page_fetch_limit": max(1, int(config.get("web_max_pages", 2) or 2)),
        }


    def _parse_domain_rules(self, value: Any) -> List[str]:
        """Convert a comma/newline separated domain rules string into a clean list."""
        raw_text = str(value or "")
        if not raw_text.strip():
            return []
        normalized = raw_text.replace("\n", ",")
        return [entry.strip() for entry in normalized.split(",") if entry.strip()]

    def _web_tools_enabled(self, config: Dict[str, Any]) -> bool:
        """Return True when web search tooling should be offered to the model."""
        if not bool(config.get("enable_web_search", False)):
            return False
        base_url = str(config.get("searxng_base_url", "")).strip()
        return bool(base_url)

    def _build_think_value(self, model_name: str, config: Dict[str, Any]) -> bool | str:
        """Choose the correct Ollama think field value for the selected model."""
        if not config.get("thinking_mode"):
            return False

        normalized_name = model_name.strip().lower()
        if "gpt-oss" in normalized_name:
            level = str(config.get("thinking_level", "medium")).strip().lower()
            return level if level in {"low", "medium", "high"} else "medium"

        return True

    def _should_retry_with_boolean_thinking(self, exc: RuntimeError, think_value: object) -> bool:
        """Return True when a think-level string should be retried as boolean True."""
        if not isinstance(think_value, str):
            return False

        error_text = str(exc).lower()
        return "think value" in error_text and "not supported for this model" in error_text

    def _consume_stream(
        self,
        stream_items: Iterable[Dict[str, Any]],
        on_chunk: StreamCallback,
        content_parts: List[str],
        thinking_parts: List[str],
    ) -> Dict[str, Any]:
        """Read streamed Ollama events, accumulate text, and capture final metrics."""
        eval_count = 0
        tokens_per_second = 0.0
        throughput_source = ""

        for item in stream_items:
            message = item.get("message", {})
            content_chunk = str(message.get("content", ""))
            thinking_chunk = str(message.get("thinking", ""))

            if thinking_chunk:
                thinking_parts.append(thinking_chunk)
                on_chunk("thinking", thinking_chunk)
            if content_chunk:
                content_parts.append(content_chunk)
                on_chunk("content", content_chunk)

            if item.get("done"):
                try:
                    eval_count = int(item.get("eval_count", 0) or 0)
                except (TypeError, ValueError):
                    eval_count = 0
                tokens_per_second = self._compute_tokens_per_second(item)
                if tokens_per_second > 0:
                    throughput_source = "ollama_eval"

        return {
            "content": "".join(content_parts).strip(),
            "thinking": "".join(thinking_parts).strip(),
            "output_tokens": eval_count,
            "tokens_per_second": tokens_per_second,
            "throughput_source": throughput_source,
        }

    def _compute_tokens_per_second(self, payload: Dict[str, Any]) -> float:
        """Return eval_count / eval_duration when Ollama provided final metrics."""
        try:
            eval_count = int(payload.get("eval_count", 0) or 0)
            eval_duration = int(payload.get("eval_duration", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
        if eval_count <= 0 or eval_duration <= 0:
            return 0.0
        return eval_count / (eval_duration / 1_000_000_000)

    def _resolve_request_options(self, request_options: Dict[str, Any] | None) -> Dict[str, Any]:
        """Return the request settings used for one chat request."""
        if request_options is None:
            return self.config_service.get_config()
        return request_options.copy()

    def _build_base_url(self, request_options: Dict[str, Any]) -> str:
        """Build an Ollama base URL from frozen request settings."""
        host = str(request_options.get("host", "127.0.0.1")).strip() or "127.0.0.1"
        port = int(request_options.get("port", 11434))
        return f"http://{host}:{port}"

    def _build_chat_messages(
        self,
        system_prompt: str,
        messages: List[ChatMessage],
        *,
        web_tools_enabled: bool = False,
    ) -> List[Dict[str, str]]:
        """Convert local messages into Ollama chat messages and inject tool guidance.

        When web tools are enabled, the payload includes a short system-level rule
        reminding the model that internet access is available through tools. This
        reduces the chance that a tool-capable model answers from habit with
        "I cannot browse the web" instead of issuing a tool call.
        """
        chat_messages: List[Dict[str, str]] = []
        if system_prompt.strip():
            chat_messages.append({"role": "system", "content": system_prompt.strip()})
        if web_tools_enabled:
            chat_messages.append({
                "role": "system",
                "content": (
                    "Web search tools are available in this application. When a user asks for current, external, "
                    "or internet-sourced information, use the provided tools instead of saying that you lack internet access. "
                    "Use search_web first, then use fetch_url_content only for pages you judge relevant."
                ),
            })

        for message in messages:
            if message.role not in {"user", "assistant", "system"}:
                continue
            chat_messages.append({"role": message.role, "content": message.content})
        return chat_messages

    def _get_json(self, endpoint: str) -> Dict[str, Any]:
        """Issue a GET request to Ollama and parse the JSON response."""
        return self._request_json(endpoint=endpoint, method="GET")

    def _stream_json_lines(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        request_options: Dict[str, Any],
    ) -> Iterable[Dict[str, Any]]:
        """Issue a streaming POST request and yield newline-delimited JSON objects."""
        url = f"{self._build_base_url(request_options)}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with request.urlopen(req, timeout=int(request_options["timeout_seconds"])) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        payload_item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"Ollama returned invalid streaming JSON: {line}") from exc
                    yield payload_item
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach the Ollama server. Check the host, port, and whether Ollama is running."
            ) from exc

    def _request_json(
        self,
        endpoint: str,
        method: str,
        payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Send a standard HTTP request to Ollama and return parsed JSON data."""
        config = self.config_service.get_config()
        url = f"{self.config_service.get_base_url()}{endpoint}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=data, method=method)
        req.add_header("Content-Type", "application/json")

        try:
            with request.urlopen(req, timeout=int(config["timeout_seconds"])) as response:
                raw_body = response.read().decode("utf-8")
                return json.loads(raw_body)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach the Ollama server. Check the host, port, and whether Ollama is running."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc

    def _request_json_with_request_options(
        self,
        endpoint: str,
        method: str,
        payload: Dict[str, Any],
        request_options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send a non-streaming Ollama request using a frozen GenerationJob config snapshot."""
        url = f"{self._build_base_url(request_options)}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with request.urlopen(req, timeout=int(request_options["timeout_seconds"])) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach the Ollama server. Check the host, port, and whether Ollama is running."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc
