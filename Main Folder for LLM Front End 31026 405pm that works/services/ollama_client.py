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
    - Optionally reranks real SearXNG results through a second Ollama model before the main model sees them
      when web search is enabled in the saved configuration.
    - Executes those tools locally and feeds the tool results back to Ollama so
      the final answer can be grounded in selected search results instead of the
      first raw search hits.
    - Accumulates source catalog entries across multiple search_web calls within
      a single turn. Each search call offsets its source IDs by the current
      catalog length so IDs remain globally unique (first search: S1..S10, second
      search: S11..S20). This prevents ID collisions that previously required
      clearing the catalog between calls — which had the side effect of
      invalidating citations from earlier searches in the same turn.

How this file fits into the system:
    This service is the boundary between the desktop app and the Ollama HTTP
    API. It centralizes model request formatting so controller and UI code do
    not need to understand Ollama payload structure or tool-call message flow.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Callable, Dict, Iterable, List
import re
from urllib import error, parse, request

from models.chat_message import ChatMessage
from services.config_service import ConfigService
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
    ) -> Dict[str, Any]:
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
            web_max_results=int(config.get("web_max_results", 8) or 8),
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
        """Ask Ollama to immediately unload one model from memory.

        Unload strategy:
            1. Prefer the local ``ollama stop <model>`` CLI when available because
               the official Ollama FAQ presents that as the direct unload command.
            2. Fall back to ``/api/generate`` with ``keep_alive=0`` because the
               API docs state that this unloads the model immediately after the
               request completes.
            3. Fall back again to ``/api/chat`` with ``keep_alive=0`` for
               environments where chat-only model serving is more reliable.

        Why this changed:
            The earlier PyQt6 revision used only the ``/api/generate`` path. That
            can work, but it gave the user no second path if their environment
            responded better to the CLI stop command, and the settings dialog did
            not distinguish between a successful unload request and a stale UI
            readout that simply had not refreshed yet.
        """
        cleaned_model = model_name.strip()
        if not cleaned_model:
            raise RuntimeError("No model is selected, so there is nothing to unload.")

        cli_result = self._unload_model_via_cli(cleaned_model)
        if cli_result.get("ok"):
            return cli_result

        generate_payload = {
            "model": cleaned_model,
            "prompt": "",
            "keep_alive": 0,
            "stream": False,
        }
        try:
            response = self._request_json(endpoint="/api/generate", method="POST", payload=generate_payload)
            response.setdefault("ok", True)
            response.setdefault("unload_method", "api_generate_keep_alive_0")
            return response
        except Exception as generate_exc:
            chat_payload = {
                "model": cleaned_model,
                "messages": [],
                "keep_alive": 0,
                "stream": False,
            }
            try:
                response = self._request_json(endpoint="/api/chat", method="POST", payload=chat_payload)
                response.setdefault("ok", True)
                response.setdefault("unload_method", "api_chat_keep_alive_0")
                response.setdefault("fallback_from", str(generate_exc))
                return response
            except Exception as chat_exc:
                raise RuntimeError(
                    f"Failed to unload model '{cleaned_model}' via CLI, /api/generate, and /api/chat. "
                    f"Last error: {chat_exc}"
                ) from chat_exc

    def _unload_model_via_cli(self, model_name: str) -> Dict[str, Any]:
        """Attempt to unload a model with the local ``ollama stop`` CLI.

        Returns a small status dictionary instead of raising so callers can
        cleanly fall back to the API-based unload path when the CLI is missing
        or the stop command fails.
        """
        if not shutil.which("ollama"):
            return {"ok": False, "message": "ollama CLI is not installed or not on PATH."}

        completed = subprocess.run(
            ["ollama", "stop", model_name],
            capture_output=True,
            text=True,
            timeout=15,
        )
        stdout_text = (completed.stdout or "").strip()
        stderr_text = (completed.stderr or "").strip()
        if completed.returncode == 0:
            return {
                "ok": True,
                "done": True,
                "done_reason": "stopped",
                "model": model_name,
                "message": stdout_text or f"Stopped {model_name} via ollama stop.",
                "unload_method": "ollama_cli_stop",
            }

        return {
            "ok": False,
            "message": stderr_text or stdout_text or f"ollama stop returned exit code {completed.returncode}.",
            "unload_method": "ollama_cli_stop",
        }

    def _chat_with_optional_tools(
        self,
        chat_messages: List[Dict[str, Any]],
        on_chunk: StreamCallback,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Offer web tools to Ollama, execute any tool calls, then stream the final answer.

        Source-validation note:
            This path now keeps a deterministic source catalog built only from
            real SearXNG results (and optional fetches derived from those
            results). Final citations are validated against that catalog before
            they are surfaced to the UI.
        """
        source_catalog: list[dict[str, Any]] = []
        fetch_count: int = 0
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
            fallback_messages, source_catalog = self._maybe_force_web_search_fallback(chat_messages, message, config)
            if fallback_messages is not None:
                return self._stream_chat_once(fallback_messages, on_chunk, config, source_catalog=source_catalog)

            content = str(message.get("content", "")).strip()
            thinking = str(message.get("thinking", "")).strip()
            if thinking:
                on_chunk("thinking", thinking)
            if content:
                on_chunk("content", content)
            sources_used = self._build_validated_sources_from_answer(content, source_catalog)
            return {
                "content": content or "[Model returned an empty response.]",
                "thinking": thinking,
                "output_tokens": int(response.get("eval_count", 0) or 0),
                "tokens_per_second": self._compute_tokens_per_second(response),
                "throughput_source": "ollama_eval" if response.get("eval_duration") else "",
                "sources_used": sources_used,
                "invalid_citations": self._find_invalid_citation_ids(content, source_catalog),
            }

        augmented_messages = list(chat_messages)
        augmented_messages.append(message)
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("function", {}).get("name", "")).strip()
            arguments = tool_call.get("function", {}).get("arguments", {}) or {}
            if tool_name == "fetch_url_content":
                max_pages = int(config.get("web_max_pages", 2) or 2)
                if fetch_count >= max_pages:
                    augmented_messages.append({
                        "role": "tool",
                        "content": json.dumps({"error": f"Page fetch limit of {max_pages} reached for this request."}),
                    })
                    continue
                fetch_count += 1
            tool_result = self._execute_tool_call(tool_name, arguments, config, source_catalog=source_catalog)
            augmented_messages.append({
                "role": "tool",
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

        # Post-tool grounding goes AFTER all tool result messages so the
        # assistant→tool pairing is intact. Inserting a system message between
        # the assistant tool-call message and the tool result message causes
        # most Ollama-served models to ignore or mis-read the tool result,
        # which is why the model stopped producing [S1]-style citations.
        augmented_messages.append(self._build_post_tool_grounding_message())

        return self._stream_chat_once(augmented_messages, on_chunk, config, source_catalog=source_catalog)

    def _maybe_force_web_search_fallback(
        self,
        chat_messages: List[Dict[str, Any]],
        assistant_message: Dict[str, Any],
        config: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]] | None, list[dict[str, Any]]]:
        """Run one defensive search when a tool-capable model claims no web access.

        Returns both the fallback message list and the source catalog gathered
        from the forced SearXNG search. This path always starts with an empty
        catalog because it runs as a single forced search on a model that did
        not issue any tool calls. The catalog is separate so the final answer
        can later validate citations deterministically.
        """
        content = str(assistant_message.get("content", "") or "").strip()
        latest_user_message = self._get_latest_user_message(chat_messages)
        if not latest_user_message:
            return None, []

        should_force_search = self._response_indicates_missing_web_access(content)
        claimed_no_access = should_force_search
        if not should_force_search and self._prompt_likely_requires_web_search(latest_user_message):
            should_force_search = True
        if not should_force_search:
            return None, []

        search_arguments = {
            "query": self._extract_search_query(latest_user_message),
            "category": str(config.get("web_search_category", "general")),
            "time_range": str(config.get("web_search_time_range", "none")),
            "max_results": int(config.get("web_max_results", 8) or 8),
        }
        source_catalog: list[dict[str, Any]] = []
        tool_result = self._execute_tool_call("search_web", search_arguments, config, source_catalog=source_catalog)

        # Use different notice text depending on whether the model incorrectly claimed
        # no web access (claimed_no_access=True) or simply failed to call the tool for
        # a query that clearly requires current information (claimed_no_access=False).
        # The distinction matters for thinking-mode models: a generic "you claimed no
        # internet" correction confuses them when no such claim was actually made, and
        # can cause them to respond to the correction itself rather than the user query.
        if claimed_no_access:
            notice = (
                "You said you had no web access, but search results are now available below. "
                "The current date is March 10, 2026. "
                "Answer the user's question using the retrieved results. "
                "Cite every factual claim with the source ID, e.g. [S1]. "
                "Do not say web access is unavailable."
            )
        else:
            notice = (
                "Current web search results for the user's question are provided below. "
                "The current date is March 10, 2026. "
                "Answer the user's question directly using these results. "
                "Cite every factual claim with its source ID, e.g. [S1] or [S2]. "
                "Do not restate what you can or cannot do — answer the question."
            )

        fallback_messages = list(chat_messages)
        fallback_messages.append({
            "role": "system",
            "content": notice,
        })
        fallback_messages.append(self._build_post_tool_grounding_message())
        fallback_messages.append({
            "role": "system",
            "content": "SEARCH_WEB_RESULTS: " + json.dumps(tool_result, ensure_ascii=False),
        })
        return fallback_messages, source_catalog

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
        patterns = [
            r"\blatest\b",
            r"\bnews\b",
            r"\brecent\b",
            r"\bcurrent\b",
            r"\btoday\b",
            r"\bthis week\b",
            r"\bsearch for\b",
            r"\blook up\b",
            r"\bfind online\b",
            r"\bon the internet\b",
        ]
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _get_latest_user_message(self, chat_messages: List[Dict[str, Any]]) -> str:
        """Return the newest user message content from the chat payload."""
        for message in reversed(chat_messages):
            if str(message.get("role", "")) == "user":
                return str(message.get("content", "") or "").strip()
        return ""

    def _extract_search_query(self, user_message: str) -> str:
        """Strip conversational framing from a user message to produce a keyword query.

        Why this exists:
            The fallback search path uses the raw user message as the SearXNG query
            when the model failed to call the tool itself. User messages are natural
            language ("Give me the latest on Iran's conflict") which produces worse
            SearXNG results than keyword form ("Iran conflict latest 2026").

            This helper strips common conversational prefixes and punctuation so the
            fallback query lands closer to what a person would type into a search box.
            It is intentionally conservative: only well-known preamble patterns are
            removed. Anything ambiguous is left unchanged and the full message is used.
        """
        text = user_message.strip()
        if not text:
            return text

        # Remove common conversational request prefixes, case-insensitively.
        prefixes = [
            r"^(?:can you |could you |please |can you please |would you please )?",
            r"(?:give me |tell me |show me |find me |search for |look up |get me )?",
            r"(?:the latest (?:on |about |news on |news about )?|"
            r"recent (?:news (?:on |about )?)?|"
            r"current (?:news (?:on |about )?)?|"
            r"news (?:on |about )?|"
            r"what(?:'s| is) (?:the latest (?:on |about )?|happening with |going on with )?|"
            r"what(?:'s| are) (?:the )?(?:latest |recent |current )?(?:updates? (?:on |about )?)?)?",
        ]
        combined = "".join(prefixes)
        cleaned = re.sub(combined, "", text, count=1, flags=re.IGNORECASE).strip()

        # If stripping left something meaningful, use it; otherwise fall back to original.
        # "Meaningful" means at least 3 characters and not just punctuation.
        if len(cleaned) >= 3 and re.search(r"[a-zA-Z]", cleaned):
            return cleaned.rstrip("?.!,")
        return text.rstrip("?.!,")

    def _build_post_tool_grounding_message(self) -> Dict[str, str]:
        """Return a system message that defines temporal grounding precedence.

        The local model may carry older parametric knowledge that conflicts with
        fresh search results. This instruction tells the model that, for
        time-sensitive questions, retrieved web evidence outranks internal
        knowledge and 2026 is the live present context for this application.
        """
        return {
            "role": "system",
            "content": (
                "Today's date is March 10, 2026. "
                "The search results above are live data retrieved for this request and are newer than your training. "
                "Use them as your primary source for any time-sensitive or current-events claims. "
                "Cite every factual claim with its bracketed source ID, such as [S1] or [S2]. "
                "Do not present your training knowledge as a competing source against the retrieved results. "
                "Only flag uncertainty when the retrieved results themselves conflict or are incomplete."
            ),
        }

    def _stream_chat_once(
        self,
        chat_messages: List[Dict[str, Any]],
        on_chunk: StreamCallback,
        config: Dict[str, Any],
        *,
        source_catalog: list[dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Perform one normal streamed chat request and accumulate the final output.

        Args:
            source_catalog: Validated source metadata derived from real SearXNG
                results and optional fetches. When present, final citations are
                checked against this catalog before being returned to the UI.
        """
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

        validated_catalog = source_catalog or []
        result["sources_used"] = self._build_validated_sources_from_answer(result.get("content", ""), validated_catalog)
        result["invalid_citations"] = self._find_invalid_citation_ids(result.get("content", ""), validated_catalog)
        return result

    def _build_web_tools_schema(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the tool schema list exposed to tool-capable models.

        The configured web_max_results value is injected into the max_results
        property description and set as the JSON Schema default. Without this,
        the model has no signal about the user-configured result ceiling and
        will freely pick a small number (typically 3-5) regardless of what is
        set in config. By surfacing the configured value here, the model treats
        the user setting as the expected request size rather than a silent cap
        that the model never knew existed.
        """
        configured_max = int(config.get("web_max_results", 8) or 8)
        search_description = (
            "Search the web through the configured SearXNG instance and return normalized candidates. "
            "Use this when external information is needed instead of assuming the first result is best. Each result includes a stable source ID like S1 that must be used for citations."
        )
        fetch_description = (
            "Fetch readable page text for one selected search result URL after you have decided the page is relevant. Preserve and reuse the original source ID when citing that page."
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
                            "query": {
                                "type": "string",
                                "description": (
                                    "Keyword search query for SearXNG. Use concise keywords, not full sentences. "
                                    "Good: 'Iran conflict latest 2026'. Bad: 'Give me the latest on Iran conflict'."
                                ),
                            },
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
                                # Embed the user-configured value so the model
                                # uses it as the expected request size rather than
                                # defaulting to a small arbitrary number.
                                "description": f"How many normalized results to return. Use {configured_max} unless the user explicitly asked for a different amount.",
                                "default": configured_max,
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
        *,
        source_catalog: list[dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Execute one local tool requested by the model and return structured output.

        Args:
            source_catalog: Mutable catalog that accumulates deterministic source
                metadata across all search_web calls within one turn. Each new
                search_web call appends its results rather than replacing earlier
                ones. Source IDs are offset by the current catalog length before
                each search so IDs remain globally unique across multiple searches
                (e.g. first search yields S1..S10, second yields S11..S20). This
                means the model can cite results from any search call and the
                citation will still resolve correctly at validation time.
        """
        if tool_name == "search_web":
            service = SearchService(
                base_url=str(config.get("searxng_base_url", "http://127.0.0.1:8080")),
                timeout_seconds=int(config.get("timeout_seconds", 30)),
            )
            # Offset source IDs by current catalog length so a second (or third)
            # search_web call does not produce IDs that collide with earlier results.
            # Without the offset, both calls produce S1..SN and the catalog.clear()
            # was the only thing preventing the LLM from citing a mismatched result.
            # With accumulation + offset, all IDs are globally unique for the turn.
            id_offset = len(source_catalog) if source_catalog is not None else 0
            result = service.search(
                str(arguments.get("query", "")),
                category=str(arguments.get("category", config.get("web_search_category", "general"))),
                language=str(config.get("web_search_language", "all")),
                time_range=str(arguments.get("time_range", config.get("web_search_time_range", "none"))),
                safe_search=int(config.get("web_safe_search", 1)),
                max_results=min(
                    int(arguments.get("max_results", config.get("web_max_results", 8)) or 8),
                    int(config.get("web_max_results", 8) or 8),
                ),
                id_offset=id_offset,
            )
            prepared = self._prepare_search_tool_result(result)
            prepared = self._maybe_rerank_search_results(prepared, config)
            if source_catalog is not None:
                # Accumulate rather than replace. Earlier search results stay in
                # the catalog so citations from a first search_web call remain
                # valid even after a second search_web call runs.
                source_catalog.extend(prepared.get("results", []))
            return prepared
        if tool_name == "fetch_url_content":
            if not bool(config.get("web_fetch_enabled", True)):
                raise RuntimeError("Web page fetching is disabled in settings.")
            service = WebFetchService(timeout_seconds=int(config.get("timeout_seconds", 30)))
            fetched = service.fetch(str(arguments.get("url", "")))
            return self._prepare_fetch_tool_result(fetched, source_catalog or [])
        raise RuntimeError(f"Model requested an unknown tool: {tool_name}")



    def _maybe_rerank_search_results(self, result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Optionally reorder SearXNG results through a second Ollama reranker model.

        Why this exists:
            SearXNG returns broad candidates. This helper inserts an explicit
            retrieval-stage reranker before those candidates are shown to the
            main answer model, using another Ollama-served model selected by the
            user on the main screen. Only source IDs present in the real search
            results are accepted, so the reranker cannot invent candidates.
        """
        reranker_model = str(config.get("reranker_model", "") or "").strip()
        results = list(result.get("results", []))
        if not reranker_model or len(results) < 2:
            prepared = dict(result)
            prepared["reranked"] = False
            prepared["reranker_model"] = reranker_model
            return prepared

        source_lookup = {str(item.get("source_id", "")).strip(): item for item in results if str(item.get("source_id", "")).strip()}
        if len(source_lookup) < 2:
            prepared = dict(result)
            prepared["reranked"] = False
            prepared["reranker_model"] = reranker_model
            return prepared

        payload = {
            "model": reranker_model,
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a search result reranker. You will be given a query and a list of search result candidates. "
                        "Reorder the candidates from most relevant to least relevant for the query. "
                        "You MUST include every candidate ID in your output — do not drop any. "
                        "Return ONLY a JSON object in exactly this form, with no other text before or after it: "
                        "{\"ordered_source_ids\": [\"S3\", \"S1\", \"S2\"]}. "
                        "Use only IDs that appear in the candidate list. Do not invent new IDs."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_reranker_prompt(str(result.get("query", "")), results),
                },
            ],
        }
        rerank_request_options = dict(config)
        rerank_request_options["model"] = reranker_model

        try:
            response = self._request_json_with_request_options("/api/chat", "POST", payload, rerank_request_options)
            content = str((response.get("message") or {}).get("content", "")).strip()
            ordered_ids = self._extract_reranked_source_ids(content, source_lookup.keys())
        except Exception as exc:
            prepared = dict(result)
            prepared["reranked"] = False
            prepared["reranker_model"] = reranker_model
            prepared["rerank_error"] = str(exc)
            return prepared

        if not ordered_ids:
            prepared = dict(result)
            prepared["reranked"] = False
            prepared["reranker_model"] = reranker_model
            prepared["rerank_error"] = "Reranker returned no valid source IDs."
            return prepared

        ordered_results = [dict(source_lookup[source_id]) for source_id in ordered_ids if source_id in source_lookup]
        for item in results:
            source_id = str(item.get("source_id", "")).strip()
            if source_id and source_id not in ordered_ids:
                ordered_results.append(dict(item))

        for index, item in enumerate(ordered_results, start=1):
            item["rerank_position"] = index
        prepared = dict(result)
        prepared["results"] = ordered_results
        prepared["reranked"] = True
        prepared["reranker_model"] = reranker_model
        prepared["rerank_order"] = ordered_ids
        return prepared

    def _build_reranker_prompt(self, query: str, results: list[dict[str, Any]]) -> str:
        """Build the compact candidate list sent to the reranker model."""
        candidate_lines = []
        for item in results:
            candidate_lines.append(
                f"{item.get('source_id','')} | title={item.get('title','')} | domain={item.get('domain','')} | "
                f"snippet={item.get('snippet','')}"
            )
        return "Query:\n" + query.strip() + "\n\nCandidates:\n" + "\n".join(candidate_lines)

    def _extract_reranked_source_ids(self, content: str, valid_ids: Iterable[str]) -> list[str]:
        """Extract a validated ordered source-ID list from reranker JSON output."""
        valid_set = {str(item).strip() for item in valid_ids if str(item).strip()}
        if not valid_set:
            return []
        parsed_ids: list[str] = []
        try:
            payload = json.loads(content)
            raw_ids = payload.get("ordered_source_ids", []) if isinstance(payload, dict) else []
            if isinstance(raw_ids, list):
                for item in raw_ids:
                    source_id = str(item).strip()
                    if source_id in valid_set and source_id not in parsed_ids:
                        parsed_ids.append(source_id)
        except Exception:
            for item in re.findall(r"S\d{1,3}", str(content or "")):
                if item in valid_set and item not in parsed_ids:
                    parsed_ids.append(item)
        return parsed_ids

    def _prepare_search_tool_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add deterministic citation and link metadata to normalized search results."""
        prepared_results: list[dict[str, Any]] = []
        for item in result.get("results", []):
            source_id = str(item.get("source_id") or item.get("id") or "").strip()
            url = str(item.get("url", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            highlight_text = self._select_highlight_text(snippet)
            prepared_item = dict(item)
            prepared_item["source_id"] = source_id
            prepared_item["citation_text"] = f"[{source_id}]" if source_id else ""
            prepared_item["citation_url"] = self._build_text_fragment_url(url, highlight_text)
            prepared_item["highlight_text"] = highlight_text
            prepared_item["validated"] = bool(url and source_id)
            prepared_item["validation_reason"] = "Validated against returned SearXNG results."
            prepared_results.append(prepared_item)
        prepared = dict(result)
        prepared["results"] = prepared_results
        return prepared

    def _prepare_fetch_tool_result(self, fetched: Dict[str, Any], source_catalog: list[dict[str, Any]]) -> Dict[str, Any]:
        """Attach fetched-page excerpt metadata to the matching validated search result."""
        requested_url = str(fetched.get("url", "")).strip()
        final_url = str(fetched.get("final_url", requested_url)).strip()
        match = self._match_source_by_url(requested_url or final_url, source_catalog)
        excerpt_text = self._select_highlight_text(str(fetched.get("text", "")).strip())
        payload = dict(fetched)
        payload["source_id"] = match.get("source_id") if match else ""
        payload["highlight_text"] = excerpt_text
        payload["citation_url"] = self._build_text_fragment_url(final_url or requested_url, excerpt_text)
        if match:
            match["fetched_title"] = str(fetched.get("title", "")).strip()
            match["fetched_final_url"] = final_url
            if excerpt_text:
                match["highlight_text"] = excerpt_text
                match["citation_url"] = payload["citation_url"]
            match["fetched_text_excerpt"] = excerpt_text
        return payload

    def _extract_citation_ids(self, content: str) -> list[str]:
        """Return bracketed citation IDs such as S1 or S2 in appearance order."""
        seen: list[str] = []
        for item in re.findall(r"\[(S\d{1,3})\]", str(content or "")):
            if item not in seen:
                seen.append(item)
        return seen

    def _build_validated_sources_from_answer(self, content: str, source_catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only citation IDs that map to real SearXNG results from this request."""
        if not source_catalog:
            return []
        lookup = {str(item.get("source_id", "")).strip(): dict(item) for item in source_catalog if str(item.get("source_id", "")).strip()}
        validated_sources: list[dict[str, Any]] = []
        for source_id in self._extract_citation_ids(content):
            item = lookup.get(source_id)
            if not item:
                continue
            prepared = dict(item)
            prepared["citation_text"] = f"[{source_id}]"
            prepared["citation_url"] = str(prepared.get("citation_url") or self._build_text_fragment_url(str(prepared.get("url", "")), str(prepared.get("highlight_text", ""))))
            prepared["validated"] = True
            validated_sources.append(prepared)
        return validated_sources

    def _find_invalid_citation_ids(self, content: str, source_catalog: list[dict[str, Any]]) -> list[str]:
        """Return cited source IDs that were not present in the real search results."""
        valid_ids = {str(item.get("source_id", "")).strip() for item in source_catalog}
        return [source_id for source_id in self._extract_citation_ids(content) if source_id not in valid_ids]

    def _match_source_by_url(self, url: str, source_catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Return the matching source entry for a fetched URL when possible."""
        cleaned = str(url or "").strip()
        if not cleaned:
            return None
        for item in source_catalog:
            for candidate in (item.get("url"), item.get("final_url"), item.get("fetched_final_url")):
                if str(candidate or "").strip() == cleaned:
                    return item
        return None

    def _select_highlight_text(self, text: str, max_length: int = 140) -> str:
        """Choose a short text fragment suitable for browser text-fragment links."""
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return ""
        if len(cleaned) <= max_length:
            return cleaned
        truncated = cleaned[:max_length].rsplit(" ", 1)[0].strip()
        return truncated or cleaned[:max_length]

    def _build_text_fragment_url(self, url: str, highlight_text: str) -> str:
        """Build a best-effort browser text-fragment link for source validation."""
        cleaned_url = str(url or "").strip()
        fragment_text = self._select_highlight_text(highlight_text)
        if not cleaned_url or not fragment_text:
            return cleaned_url
        separator = '&' if '#' in cleaned_url else '#'
        return f"{cleaned_url}{separator}:~:text={parse.quote(fragment_text, safe='')}"

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
        web_max_results: int = 8,
    ) -> List[Dict[str, str]]:
        """Convert local messages into Ollama chat messages and inject tool guidance.

        When web tools are enabled, the payload includes a short system-level rule
        reminding the model that internet access is available through tools. This
        reduces the chance that a tool-capable model answers from habit with
        "I cannot browse the web" instead of issuing a tool call.

        The configured web_max_results value is embedded in the grounding message
        so the model receives an explicit instruction about the expected result
        count. Without this, models ignore the tool schema default and pick an
        arbitrary small number (typically 3-5) regardless of the user's setting.
        """
        chat_messages: List[Dict[str, str]] = []
        if system_prompt.strip():
            chat_messages.append({"role": "system", "content": system_prompt.strip()})
        if web_tools_enabled:
            chat_messages.append({
                "role": "system",
                "content": (
                    f"You have access to web search tools. Today's date is March 10, 2026. "
                    f"For any question about current events, recent news, or time-sensitive facts, "
                    f"call search_web with max_results={web_max_results} before answering — do not answer from memory alone. "
                    f"When calling search_web, write the query as concise keywords, not as a full sentence. "
                    f"For example, use 'Iran conflict latest 2026' not 'Give me the latest on Iran conflict'. "
                    f"After searching, cite every factual claim with the bracketed source ID from the result, such as [S1] or [S2]. "
                    f"Retrieved search results take precedence over your training knowledge. "
                    f"Only use fetch_url_content for a page you have already found via search_web and judge worth reading in full. "
                    f"Never invent a source ID or URL."
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
