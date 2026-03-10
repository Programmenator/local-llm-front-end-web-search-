"""
services/web_fetch_service.py

Purpose:
    Fetch a selected web page and extract readable text for the model.

What this file does:
    - Validates that a URL uses an allowed web scheme.
    - Downloads the page with a bounded timeout.
    - Extracts the HTML title and visible paragraph text.
    - Truncates oversized payloads so prompts stay within practical limits.

How this file fits into the system:
    Search snippets are often not enough for grounded answers. This service is
    the second retrieval stage that turns a chosen search result into readable
    page text the model can reason over.
"""

from __future__ import annotations

from html import unescape
from typing import Any, Dict, List
from urllib import error, parse, request

from bs4 import BeautifulSoup


class WebFetchService:
    """Download and normalize readable text from one web page."""

    def __init__(self, timeout_seconds: int = 30, max_characters: int = 12000) -> None:
        """Store fetch limits used for page retrieval.

        Args:
            timeout_seconds: Network timeout for page fetches.
            max_characters: Maximum extracted text characters returned to the caller.
        """
        self.timeout_seconds = max(5, int(timeout_seconds))
        self.max_characters = max(1000, int(max_characters))

    def fetch(self, url: str) -> Dict[str, Any]:
        """Download one page and return extracted readable text.

        Raises:
            RuntimeError: If the URL is invalid, unreachable, or returns an
                unsupported content type.
        """
        cleaned_url = url.strip()
        if not cleaned_url:
            raise RuntimeError('No URL was provided for page fetch.')

        parsed_url = parse.urlparse(cleaned_url)
        if parsed_url.scheme not in {'http', 'https'}:
            raise RuntimeError('Only http and https URLs are allowed for web fetch.')

        req = request.Request(
            url=cleaned_url,
            method='GET',
            headers={'User-Agent': 'LocalLLMFrontend/1.0 (+web-fetch)'},
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                content_type = str(response.headers.get('Content-Type', '')).lower()
                if 'text/html' not in content_type and 'application/xhtml+xml' not in content_type:
                    raise RuntimeError(f'Unsupported content type for fetch: {content_type or "unknown"}')
                html = response.read().decode('utf-8', errors='replace')
                final_url = response.geturl()
        except error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'Web fetch returned HTTP {exc.code}: {body}') from exc
        except error.URLError as exc:
            raise RuntimeError('Could not fetch the requested page URL.') from exc

        return self._extract_text(cleaned_url, final_url, html)

    def _extract_text(self, requested_url: str, final_url: str, html: str) -> Dict[str, Any]:
        """Extract a readable title and body text from downloaded HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        title_text = soup.title.get_text(' ', strip=True) if soup.title else final_url

        for tag in soup(['script', 'style', 'noscript', 'svg']):
            tag.decompose()

        paragraphs: List[str] = []
        for node in soup.find_all(['p', 'li']):
            text = unescape(node.get_text(' ', strip=True))
            if len(text) >= 40:
                paragraphs.append(text)

        if not paragraphs:
            fallback = unescape(soup.get_text(' ', strip=True))
            paragraphs = [fallback]

        combined_text = '\n\n'.join(paragraphs).strip()

        return {
            'url': requested_url,
            'final_url': final_url,
            'title': title_text,
            'content_type': 'text/html',
            'text': combined_text,
        }
