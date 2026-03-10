"""
services/search_service.py

Purpose:
    Query a configured SearXNG instance and normalize the returned search
    results into a stable schema the rest of the application can trust.

What this file does:
    - Sends search requests to the configured SearXNG `/search` endpoint.
    - Requests JSON output so results are machine-readable.
    - Normalizes titles, snippets, domains, engines, and positions.
    - Returns a bounded result set so downstream model prompts stay predictable.

How this file fits into the system:
    This service is the retrieval entry point for web search. The Ollama client
    can expose its output to tool-calling models without coupling model logic to
    SearXNG's raw response shape.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict
from urllib import error, parse, request


class SearchService:
    """Perform normalized web searches against a SearXNG instance."""

    def __init__(self, base_url: str, timeout_seconds: int = 30) -> None:
        """Store connection details used for later SearXNG requests.

        Args:
            base_url: Root URL of the SearXNG instance, for example
                ``http://127.0.0.1:8080``.
            timeout_seconds: Request timeout applied to search calls.
        """
        self.base_url = self._normalize_base_url(base_url)
        self.timeout_seconds = max(5, int(timeout_seconds))

    def search(
        self,
        query: str,
        *,
        category: str = 'general',
        language: str = 'all',
        time_range: str = 'none',
        safe_search: int = 1,
        max_results: int = 8,
    ) -> Dict[str, Any]:
        """Run one SearXNG search and return normalized results.

        Args:
            query: User or model supplied search string.
            category: SearXNG category hint such as ``general`` or ``news``.
            language: Language code or ``all``.
            time_range: Optional freshness hint such as ``day`` or ``week``.
            safe_search: Integer SearXNG safe-search level.
            max_results: Upper bound on results returned to the caller.

        Returns:
            Normalized search payload containing ``query`` and ``results``.

        Raises:
            RuntimeError: If SearXNG is unreachable, returns invalid JSON, or the
                instance rejects JSON output.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise RuntimeError('Search query cannot be empty.')

        params = {
            'q': cleaned_query,
            'format': 'json',
            'categories': category or 'general',
            'language': language or 'all',
            'safesearch': str(max(0, min(2, int(safe_search)))),
        }
        if time_range and time_range != 'none':
            params['time_range'] = time_range

        url = f"{self.base_url}/search?{parse.urlencode(params)}"
        req = request.Request(url=url, method='GET')
        req.add_header('Accept', 'application/json')

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            if exc.code == 403 and 'format' in body.lower():
                raise RuntimeError(
                    f'SearXNG at {self.base_url} rejected JSON output. Enable format=json in the instance search settings.'
                ) from exc
            raise RuntimeError(f'SearXNG returned HTTP {exc.code} from {url}: {body}') from exc
        except error.URLError as exc:
            raise RuntimeError(self._build_connection_error_message(url, exc.reason)) from exc
        except ValueError as exc:
            raise RuntimeError(
                'The configured SearXNG URL is invalid. Include a host and use http:// or https://, '
                'for example http://127.0.0.1:8080.'
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f'SearXNG at {url} returned invalid JSON. Verify that /search?q=test&format=json works in a browser or curl.'
            ) from exc

        raw_results = payload.get('results', [])
        bounded_results = raw_results[: max(1, min(20, int(max_results)))]
        return {
            'query': cleaned_query,
            'result_count': len(bounded_results),
            'results': [self._normalize_result(item, index + 1) for index, item in enumerate(bounded_results)],
        }

    def test_connection(self) -> Dict[str, Any]:
        """Run a lightweight probe so the settings window can verify connectivity."""
        try:
            response = self.search('searxng', max_results=1)
        except RuntimeError as exc:
            return {'ok': False, 'message': str(exc)}
        return {
            'ok': True,
            'message': (
                f"Connected successfully to {self.base_url}. "
                f"Retrieved {response.get('result_count', 0)} search result(s)."
            ),
        }

    def _build_connection_error_message(self, request_url: str, reason: object) -> str:
        """Convert low-level network failures into user-facing setup guidance.

        This keeps the settings test actionable. Most failures during setup are
        wrong host, wrong port, a pasted `/search` URL instead of the instance
        root, or a server that is not running yet.
        """
        parsed_url = parse.urlparse(self.base_url)
        host = parsed_url.hostname or 'unknown-host'
        port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
        base_message = (
            f"Could not connect to SearXNG at {self.base_url}. "
            f"The app tried {request_url}."
        )

        if isinstance(reason, ConnectionRefusedError):
            return (
                f"{base_message} Connection was refused. This usually means the host/port is wrong or "
                f"the SearXNG service is not running. Check that the instance is listening on {host}:{port} "
                f"and test {self.base_url}/search?q=test&format=json in a browser or curl."
            )
        if isinstance(reason, socket.gaierror):
            return (
                f"{base_message} The hostname could not be resolved. Check for a typo in the host name and "
                f"make sure you used the correct machine address."
            )
        if isinstance(reason, TimeoutError):
            return (
                f"{base_message} The request timed out after {self.timeout_seconds} seconds. "
                f"Check that the instance is reachable and increase the timeout if needed."
            )

        detail = str(reason).strip() or 'Unknown connection error.'
        return (
            f"{base_message} {detail} Check that the base URL is correct and verify "
            f"{self.base_url}/search?q=test&format=json manually."
        )

    def _normalize_base_url(self, base_url: str) -> str:
        """Normalize the configured SearXNG root URL into a usable form.

        Users often enter a host like ``localhost:8080`` in the settings UI.
        ``urllib`` treats that as an invalid URL because it is missing a scheme.
        This helper repairs the common case by prepending ``http://`` and then
        validating that a hostname exists.
        """
        cleaned = str(base_url).strip()
        if not cleaned:
            raise ValueError('SearXNG base URL cannot be empty.')
        if '://' not in cleaned:
            cleaned = f'http://{cleaned}'
        parsed_url = parse.urlparse(cleaned)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError('SearXNG base URL must include a valid host.')

        normalized_path = parsed_url.path.rstrip('/')
        if normalized_path.endswith('/search'):
            normalized_path = normalized_path[:-len('/search')]

        rebuilt = parse.urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            normalized_path,
            '',
            '',
            '',
        ))
        return rebuilt.rstrip('/')

    def _normalize_result(self, item: Dict[str, Any], position: int) -> Dict[str, Any]:
        """Convert one raw SearXNG result into the internal schema."""
        url = str(item.get('url', '')).strip()
        parsed = parse.urlparse(url)
        engines = item.get('engines', []) or []
        return {
            'id': f'r{position}',
            'title': str(item.get('title', '')).strip(),
            'url': url,
            'domain': parsed.netloc,
            'snippet': str(item.get('content', '') or item.get('snippet', '')).strip(),
            'engine': ', '.join(str(engine) for engine in engines if engine),
            'category': str(item.get('category', '')).strip(),
            'published_date': str(item.get('publishedDate', '') or item.get('published_date', '')).strip() or None,
            'position': position,
        }
