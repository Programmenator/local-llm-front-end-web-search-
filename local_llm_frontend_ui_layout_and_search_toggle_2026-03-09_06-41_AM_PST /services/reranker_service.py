"""
services/reranker_service.py

Purpose:
    Reorder normalized SearXNG search results before the main chat model sees
    them.

What this file does:
    - Applies a deterministic non-LLM reranking pass using query/title/snippet/
      domain relevance signals.
    - Optionally applies a second Ollama-served reranker model after the
      deterministic pass.
    - Optionally applies a sentence-transformers cross-encoder reranker after
      the deterministic pass without routing through Ollama.
    - Rebuilds the search payload in reranked order while preserving every
      original candidate.
    - Falls back cleanly when the optional Ollama reranker output is incomplete
      by appending any missing IDs in original order.

How this file fits into the system:
    This service layer sits between raw SearXNG retrieval and the main answer
    model. The deterministic stage answers the first retrieval question,
    "Which candidates are most relevant to the query?", before any optional LLM
    judgment is used.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Sequence
from urllib import error, request


class NonLlmSearchRerankerService:
    """Deterministically rerank normalized search candidates by lexical relevance."""

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it",
        "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "who", "why",
        "with", "vs", "latest", "news", "today", "current",
    }

    def rerank_search_payload(
        self,
        query: str,
        search_payload: Dict[str, Any],
        *,
        top_n: int | None = None,
    ) -> Dict[str, Any]:
        """Return a copy of one normalized search payload in deterministic rank order."""
        normalized_query = str(query or "").strip()
        results = list(search_payload.get("results", []) or [])
        payload_copy = dict(search_payload)
        if not normalized_query or len(results) <= 1:
            payload_copy["heuristic_reranker"] = {
                "enabled": True,
                "applied": False,
                "reason": "Skipped because there were fewer than two results or the query was empty.",
            }
            return payload_copy

        scored_results = []
        for item in results:
            score, score_breakdown = self._score_candidate(normalized_query, item)
            enriched = dict(item)
            enriched["heuristic_score"] = round(score, 6)
            enriched["heuristic_score_breakdown"] = score_breakdown
            scored_results.append(enriched)

        ranked_results = sorted(
            scored_results,
            key=lambda item: (
                -float(item.get("heuristic_score", 0.0)),
                int(item.get("position", 0) or 0),
            ),
        )
        if top_n is not None:
            ranked_results = ranked_results[: max(1, int(top_n))]

        payload_copy["results"] = ranked_results
        payload_copy["result_count"] = len(ranked_results)
        payload_copy["heuristic_reranker"] = {
            "enabled": True,
            "applied": True,
            "method": "query_title_snippet_domain_scoring",
            "top_n": len(ranked_results),
            "ordered_result_ids": [str(item.get("id", "")).strip() for item in ranked_results],
            "signals": ["query", "title", "snippet", "domain"],
        }
        return payload_copy

    def _score_candidate(self, query: str, item: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
        """Return one composite relevance score plus the signal breakdown."""
        query_tokens = self._tokenize(query)
        title = str(item.get("title", "") or "")
        snippet = str(item.get("snippet", "") or "")
        domain = str(item.get("domain", "") or "")
        url = str(item.get("url", "") or "")

        title_tokens = self._tokenize(title)
        snippet_tokens = self._tokenize(snippet)
        domain_tokens = self._tokenize(domain.replace(".", " "))
        url_tokens = self._tokenize(url.replace("/", " ").replace("-", " "))

        if not query_tokens:
            return 0.0, {
                "title_overlap": 0.0,
                "snippet_overlap": 0.0,
                "domain_overlap": 0.0,
                "phrase_match": 0.0,
                "direct_answer_bonus": 0.0,
            }

        phrase_match = 1.0 if query.strip().lower() in f"{title} {snippet}".lower() else 0.0
        title_overlap = self._overlap_ratio(query_tokens, title_tokens)
        snippet_overlap = self._overlap_ratio(query_tokens, snippet_tokens)
        domain_overlap = self._overlap_ratio(query_tokens, domain_tokens | url_tokens)
        direct_answer_bonus = self._direct_answer_bonus(title, url)

        score_breakdown = {
            "title_overlap": round(title_overlap, 6),
            "snippet_overlap": round(snippet_overlap, 6),
            "domain_overlap": round(domain_overlap, 6),
            "phrase_match": phrase_match,
            "direct_answer_bonus": direct_answer_bonus,
        }
        weighted_score = (
            title_overlap * 4.0
            + snippet_overlap * 2.5
            + domain_overlap * 1.25
            + phrase_match * 2.0
            + direct_answer_bonus
        )
        return weighted_score, score_breakdown

    def _tokenize(self, text: str) -> set[str]:
        """Return a normalized token set for coarse lexical relevance scoring."""
        raw_tokens = re.findall(r"[a-z0-9]{2,}", str(text or "").lower())
        return {token for token in raw_tokens if token not in self._STOPWORDS}

    def _overlap_ratio(self, query_tokens: set[str], candidate_tokens: set[str]) -> float:
        """Return how much of the query token set appears in the candidate tokens."""
        if not query_tokens or not candidate_tokens:
            return 0.0
        overlap = len(query_tokens & candidate_tokens)
        return overlap / max(1, len(query_tokens))

    def _direct_answer_bonus(self, title: str, url: str) -> float:
        """Prefer documentation/article pages over generic homepages when other signals are close."""
        normalized = f"{title} {url}".lower()
        bonus = 0.0
        preferred_markers = ["docs", "documentation", "guide", "tutorial", "reference", "api", "article", "story"]
        homepage_markers = ["home", "homepage", "/$", "index"]
        if any(marker in normalized for marker in preferred_markers):
            bonus += 0.75
        if title.strip() and len(self._tokenize(title)) <= 2 and url.rstrip("/").count("/") <= 2:
            bonus -= 0.35
        if any(marker in normalized for marker in homepage_markers):
            bonus -= 0.15
        return bonus


_MODEL_CACHE: dict[tuple[str, str], Any] = {}


class SentenceTransformersSearchRerankerService:
    """Use a sentence-transformers cross-encoder to rerank already-screened results.

    This backend is intended for Hugging Face style reranker models such as
    ``BAAI/bge-reranker-base``. It loads the model lazily on first use and then
    reuses it from a small in-process cache so repeated searches do not keep
    paying full model initialization cost.
    """

    def __init__(self, *, model_name: str, device: str = "auto") -> None:
        cleaned_model = str(model_name).strip()
        if not cleaned_model:
            raise ValueError("Sentence-transformers reranker model name cannot be empty.")
        self.model_name = cleaned_model
        self.device = self._resolve_device(device)

    def rerank_search_payload(self, query: str, search_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of one normalized search payload in reranked order."""
        normalized_query = str(query or "").strip()
        results = list(search_payload.get("results", []) or [])
        if not normalized_query or len(results) <= 1:
            payload_copy = dict(search_payload)
            payload_copy["reranker"] = {
                "enabled": True,
                "applied": False,
                "backend": "sentence_transformers",
                "model": self.model_name,
                "reason": "Skipped because there were fewer than two results or the query was empty.",
            }
            return payload_copy

        document_texts = [self._build_candidate_text(item) for item in results]
        scores = self._score_pairs(query, document_texts)
        indexed_scores = list(enumerate(scores))
        ranked_indexes = [
            index
            for index, _score in sorted(
                indexed_scores,
                key=lambda item: (-float(item[1]), int(results[item[0]].get("position", 0) or 0)),
            )
        ]

        ranked_results: List[Dict[str, Any]] = []
        for index in ranked_indexes:
            enriched = dict(results[index])
            enriched["sentence_transformers_score"] = round(float(scores[index]), 6)
            ranked_results.append(enriched)

        payload_copy = dict(search_payload)
        payload_copy["results"] = ranked_results
        payload_copy["result_count"] = len(ranked_results)
        payload_copy["reranker"] = {
            "enabled": True,
            "applied": True,
            "backend": "sentence_transformers",
            "model": self.model_name,
            "device": self.device,
            "ordered_result_ids": [str(item.get("id", "")).strip() for item in ranked_results],
        }
        return payload_copy

    def _score_pairs(self, query: str, documents: Sequence[str]) -> List[float]:
        """Score each query-document pair with the configured cross-encoder."""
        model = self._get_model()
        pairs = [(query, document) for document in documents]
        raw_scores = model.predict(pairs, convert_to_numpy=False, show_progress_bar=False)
        return [float(score) for score in raw_scores]

    def _get_model(self) -> Any:
        """Return a cached sentence-transformers model instance for this backend."""
        cache_key = (self.model_name, self.device)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        model = self._load_model(self.model_name, self.device)
        _MODEL_CACHE[cache_key] = model
        return model

    def _load_model(self, model_name: str, device: str) -> Any:
        """Import and initialize the sentence-transformers cross-encoder lazily."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it in the program environment before using the "
                "sentence-transformers reranker backend."
            ) from exc

        try:
            return CrossEncoder(model_name, device=device)
        except Exception as exc:  # pragma: no cover - exact backend error varies by local environment.
            raise RuntimeError(
                f"Could not load sentence-transformers reranker model '{model_name}' on device '{device}': {exc}"
            ) from exc

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve the requested runtime device for sentence-transformers execution."""
        normalized = str(requested_device or "auto").strip().lower() or "auto"
        if normalized != "auto":
            return normalized
        try:
            import torch
        except ImportError:
            return "cpu"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _build_candidate_text(self, item: Dict[str, Any]) -> str:
        """Build one compact candidate text block for the reranker model."""
        title = str(item.get("title", "") or "").strip()
        snippet = str(item.get("snippet", "") or "").strip()
        domain = str(item.get("domain", "") or "").strip()
        url = str(item.get("url", "") or "").strip()
        return (
            f"Title: {title}\n"
            f"Snippet: {snippet}\n"
            f"Domain: {domain}\n"
            f"URL: {url}"
        ).strip()


class SearchRerankerService:
    """Use an Ollama model to reorder already-screened search candidates."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        model_name: str,
        timeout_seconds: int = 30,
    ) -> None:
        """Store the Ollama connection and reranker model details."""
        cleaned_model = str(model_name).strip()
        if not cleaned_model:
            raise ValueError("Reranker model name cannot be empty.")
        self.host = str(host).strip() or "127.0.0.1"
        self.port = int(port)
        self.model_name = cleaned_model
        self.timeout_seconds = max(5, int(timeout_seconds))

    def rerank_search_payload(self, query: str, search_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of one normalized search payload in reranked order."""
        normalized_query = str(query or "").strip()
        results = list(search_payload.get("results", []) or [])
        if not normalized_query or len(results) <= 1:
            payload_copy = dict(search_payload)
            payload_copy["reranker"] = {
                "enabled": True,
                "applied": False,
                "model": self.model_name,
                "reason": "Skipped because there were fewer than two results or the query was empty.",
            }
            return payload_copy

        ordered_ids, notes = self._request_reranked_ids(normalized_query, results)
        reordered_results = self._reorder_results(results, ordered_ids)

        payload_copy = dict(search_payload)
        payload_copy["results"] = reordered_results
        payload_copy["result_count"] = len(reordered_results)
        payload_copy["reranker"] = {
            "enabled": True,
            "applied": True,
            "model": self.model_name,
            "ordered_result_ids": [item.get("id") for item in reordered_results],
            "notes": notes,
        }
        return payload_copy

    def _request_reranked_ids(self, query: str, results: List[Dict[str, Any]]) -> tuple[List[str], str]:
        """Ask the reranker model for an ordered list of result IDs."""
        payload = {
            "model": self.model_name,
            "prompt": self._build_prompt(query, results),
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        response = self._request_json("/api/generate", payload)
        raw_text = str(response.get("response", "") or "").strip()
        if not raw_text:
            raise RuntimeError("The reranker model returned an empty response.")
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"The reranker model returned invalid JSON: {raw_text}") from exc

        raw_ids = parsed.get("ordered_result_ids", [])
        if not isinstance(raw_ids, list):
            raise RuntimeError("The reranker JSON must contain an 'ordered_result_ids' list.")
        ordered_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
        notes = str(parsed.get("notes", "") or "").strip()
        return ordered_ids, notes

    def _build_prompt(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Return the strict reranking prompt sent to the Ollama model."""
        result_lines = []
        for item in results:
            result_lines.append(
                {
                    "id": str(item.get("id", "")).strip(),
                    "title": str(item.get("title", "")).strip(),
                    "domain": str(item.get("domain", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "snippet": str(item.get("snippet", "")).strip(),
                    "published_date": item.get("published_date"),
                    "position": item.get("position"),
                    "heuristic_score": item.get("heuristic_score"),
                }
            )
        serialized_results = json.dumps(result_lines, ensure_ascii=False, indent=2)
        prompt = (
            "You are a search result reranker for a local LLM web-search pipeline. "
            "The candidate list has already passed through a deterministic first-pass relevance filter. "
            "Refine that order so the most relevant items to the user's query come first. "
            "Prefer direct answer pages over broad homepages. Prefer higher-authority or more official sources when relevance is similar. "
            "When the query asks for current information, prefer fresher results when the snippets indicate recency. "
            "Return only valid JSON with this exact shape: "
            '{"ordered_result_ids": ["r2", "r1"], "notes": "brief reason"}. '
            "Use only IDs that appear in the candidate list. Include each candidate at most once. Do not add commentary outside JSON.\n\n"
            f"USER_QUERY:\n{query}\n\n"
            f"CANDIDATE_RESULTS:\n{serialized_results}"
        )
        return prompt

    def _reorder_results(self, results: List[Dict[str, Any]], ordered_ids: List[str]) -> List[Dict[str, Any]]:
        """Apply the reranker ID order while preserving all original results."""
        by_id = {str(item.get("id", "")).strip(): item for item in results}
        consumed = set()
        reordered: List[Dict[str, Any]] = []

        for result_id in ordered_ids:
            if result_id in by_id and result_id not in consumed:
                reordered.append(by_id[result_id])
                consumed.add(result_id)

        for item in results:
            result_id = str(item.get("id", "")).strip()
            if result_id and result_id not in consumed:
                reordered.append(item)
                consumed.add(result_id)
        return reordered

    def _request_json(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Issue one non-streaming JSON request to Ollama and parse the response."""
        url = f"http://{self.host}:{self.port}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Reranker model request failed with HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama reranker model at {url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama reranker returned invalid JSON.") from exc
