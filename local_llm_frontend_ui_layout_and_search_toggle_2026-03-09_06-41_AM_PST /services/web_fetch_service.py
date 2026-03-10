"""
services/web_fetch_service.py

Purpose:
    Fetch a selected web page and extract readable text for the model.

What this file does:
    - Validates that a URL uses an allowed web scheme.
    - Resolves hostnames before network access and blocks local/private targets.
    - Caps redirects and re-validates each redirect target before following it.
    - Enforces content-type and response-size limits before returning text.
    - Extracts the HTML title and visible paragraph text.
    - Truncates oversized extracted payloads so prompts stay within practical limits.

How this file fits into the system:
    Search snippets are often not enough for grounded answers. This service is
    the second retrieval stage that turns a chosen search result into readable
    page text the model can reason over. Because it is the network boundary for
    page retrieval, it also enforces the fetch safety rules shown to the user in
    the settings window and described in the documentation.
"""

from __future__ import annotations

import ipaddress
import socket
from html import unescape
from typing import Any, Callable, Dict, Iterable, List, Sequence
from urllib import error, parse, request

from bs4 import BeautifulSoup

_ALLOWED_CONTENT_TYPES = {
    'text/html',
    'application/xhtml+xml',
}
_DEFAULT_MAX_REDIRECTS = 4
_DEFAULT_MAX_RESPONSE_BYTES = 1_500_000
_DEFAULT_ALLOWED_SCHEMES = {'http', 'https'}


class _ValidatedRedirectHandler(request.HTTPRedirectHandler):
    """Redirect handler that re-validates every target before following it.

    Why this exists:
        urllib follows redirects automatically. For this application that is too
        permissive because a safe public URL could redirect to localhost or a
        private address. This handler gives WebFetchService one place to enforce
        redirect count limits and target validation for each hop.
    """

    def __init__(self, validator: Callable[[str], None], max_redirects: int) -> None:
        self._validator = validator
        self._max_redirects = max(0, int(max_redirects))

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        redirect_count = int(getattr(req, '_safe_redirect_count', 0)) + 1
        if redirect_count > self._max_redirects:
            raise RuntimeError(
                f'Web fetch exceeded the redirect limit of {self._max_redirects} hop(s).'
            )
        self._validator(newurl)
        redirected_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected_request is not None:
            setattr(redirected_request, '_safe_redirect_count', redirect_count)
        return redirected_request


class WebFetchService:
    """Download and normalize readable text from one web page.

    System interaction summary:
        - Called by OllamaClient when the model selects a page to read.
        - Applies network safety policy before any outbound request is sent.
        - Returns a structured payload that can be injected back into the model
          conversation as grounded runtime evidence.
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        max_characters: int = 12000,
        *,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
        max_redirects: int = _DEFAULT_MAX_REDIRECTS,
        allowed_content_types: Iterable[str] | None = None,
        domain_allowlist: Sequence[str] | None = None,
        domain_blocklist: Sequence[str] | None = None,
    ) -> None:
        """Store fetch limits and safety policy used for page retrieval.

        Args:
            timeout_seconds: Network timeout for page fetches.
            max_characters: Maximum extracted text characters returned to the caller.
            max_response_bytes: Maximum response body size accepted from the server.
            max_redirects: Maximum number of redirects allowed for one fetch.
            allowed_content_types: Exact MIME types accepted for fetched pages.
            domain_allowlist: Optional list of domains allowed for fetches.
            domain_blocklist: Optional list of domains denied for fetches.
        """
        self.timeout_seconds = max(5, int(timeout_seconds))
        self.max_characters = max(1000, int(max_characters))
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self.max_redirects = max(0, int(max_redirects))
        self.allowed_content_types = {
            self._normalize_content_type(value)
            for value in (allowed_content_types or _ALLOWED_CONTENT_TYPES)
            if self._normalize_content_type(value)
        } or set(_ALLOWED_CONTENT_TYPES)
        self.domain_allowlist = self._normalize_domain_rules(domain_allowlist)
        self.domain_blocklist = self._normalize_domain_rules(domain_blocklist)

    def fetch(self, url: str) -> Dict[str, Any]:
        """Download one page and return extracted readable text.

        Raises:
            RuntimeError: If the URL is invalid, blocked by policy, unreachable,
                too large, redirects unsafely, or returns an unsupported content
                type.
        """
        cleaned_url = url.strip()
        if not cleaned_url:
            raise RuntimeError('No URL was provided for page fetch.')

        self._validate_url_target(cleaned_url)

        req = request.Request(
            url=cleaned_url,
            method='GET',
            headers={'User-Agent': 'LocalLLMFrontend/1.0 (+web-fetch)'},
        )
        setattr(req, '_safe_redirect_count', 0)
        opener = request.build_opener(self._build_redirect_handler_for_test())

        try:
            with opener.open(req, timeout=self.timeout_seconds) as response:
                content_type_header = str(response.headers.get('Content-Type', '')).strip()
                content_type = self._normalize_content_type(content_type_header)
                if content_type not in self.allowed_content_types:
                    allowed_text = ', '.join(sorted(self.allowed_content_types))
                    raise RuntimeError(
                        'Unsupported content type for fetch: '
                        f'{content_type or "unknown"}. Allowed types: {allowed_text}.'
                    )

                content_length = self._safe_int(response.headers.get('Content-Length'))
                if content_length is not None and content_length > self.max_response_bytes:
                    raise RuntimeError(
                        'Web fetch blocked because the response body is too large '
                        f'({content_length} bytes > limit of {self.max_response_bytes} bytes).'
                    )

                html_bytes = self._read_response_with_limit(response)
                encoding = self._get_response_charset(response)
                html = html_bytes.decode(encoding or 'utf-8', errors='replace')
                final_url = response.geturl()
                self._validate_url_target(final_url)
        except error.HTTPError as exc:
            body = exc.read(min(2048, self.max_response_bytes)).decode('utf-8', errors='replace')
            raise RuntimeError(f'Web fetch returned HTTP {exc.code}: {body}') from exc
        except error.URLError as exc:
            raise RuntimeError('Could not fetch the requested page URL.') from exc

        return self._extract_text(cleaned_url, final_url, html, content_type)


    def _build_redirect_handler_for_test(self) -> _ValidatedRedirectHandler:
        """Return the redirect validator used for a fetch.

        Why this helper exists:
            The production fetch path and the automated tests should exercise the
            same redirect validation logic. Returning the configured handler from
            one helper keeps those paths aligned without duplicating setup code.
        """
        return _ValidatedRedirectHandler(self._validate_url_target, self.max_redirects)

    def _extract_text(
        self,
        requested_url: str,
        final_url: str,
        html: str,
        content_type: str,
    ) -> Dict[str, Any]:
        """Extract a readable title and body text from downloaded HTML.

        Extraction strategy:
            1. Remove obviously non-content tags.
            2. Prefer article-like containers when available.
            3. Fall back to paragraph/list extraction from the whole document.
            4. Truncate the final extracted text to the configured character cap.
        """
        soup = BeautifulSoup(html, 'html.parser')
        title_text = soup.title.get_text(' ', strip=True) if soup.title else final_url

        for tag in soup(['script', 'style', 'noscript', 'svg', 'iframe', 'canvas']):
            tag.decompose()

        candidate_roots = [
            soup.find('article'),
            soup.find('main'),
            soup.find(attrs={'role': 'main'}),
        ]
        candidate_root = next((node for node in candidate_roots if node is not None), soup)

        paragraphs: List[str] = []
        for node in candidate_root.find_all(['p', 'li']):
            text = unescape(node.get_text(' ', strip=True))
            if len(text) >= 40:
                paragraphs.append(text)

        if not paragraphs and candidate_root is not soup:
            for node in soup.find_all(['p', 'li']):
                text = unescape(node.get_text(' ', strip=True))
                if len(text) >= 40:
                    paragraphs.append(text)

        if not paragraphs:
            fallback = unescape(soup.get_text(' ', strip=True))
            paragraphs = [fallback]

        combined_text = '\n\n'.join(part for part in paragraphs if part).strip()
        if len(combined_text) > self.max_characters:
            combined_text = combined_text[: self.max_characters].rstrip() + '…'

        return {
            'url': requested_url,
            'final_url': final_url,
            'title': title_text,
            'content_type': content_type,
            'text': combined_text,
            'fetch_policy': {
                'max_response_bytes': self.max_response_bytes,
                'max_redirects': self.max_redirects,
                'allowed_content_types': sorted(self.allowed_content_types),
                'domain_allowlist': sorted(self.domain_allowlist),
                'domain_blocklist': sorted(self.domain_blocklist),
            },
        }

    def _validate_url_target(self, url: str) -> None:
        """Reject unsafe or disallowed targets before network access.

        This method is the main network-policy gate for the page-fetch tool.
        It validates scheme, hostname presence, allow/block rules, and resolved
        address safety before the program attempts a connection.
        """
        parsed_url = parse.urlparse(url.strip())
        if parsed_url.scheme.lower() not in _DEFAULT_ALLOWED_SCHEMES:
            raise RuntimeError('Only http and https URLs are allowed for web fetch.')

        hostname = (parsed_url.hostname or '').strip().lower().rstrip('.')
        if not hostname:
            raise RuntimeError('The requested page URL is missing a hostname.')

        if hostname in {'localhost'}:
            raise RuntimeError('Blocked web fetch target: localhost is not allowed.')

        self._enforce_domain_rules(hostname)
        self._validate_hostname_addresses(hostname)

    def _enforce_domain_rules(self, hostname: str) -> None:
        """Apply optional allowlist/blocklist rules to a hostname."""
        if self._matches_domain_rules(hostname, self.domain_blocklist):
            raise RuntimeError(f'Blocked web fetch target: {hostname} matches the domain blocklist.')
        if self.domain_allowlist and not self._matches_domain_rules(hostname, self.domain_allowlist):
            allowed_text = ', '.join(sorted(self.domain_allowlist))
            raise RuntimeError(
                f'Blocked web fetch target: {hostname} is not in the domain allowlist ({allowed_text}).'
            )

    def _validate_hostname_addresses(self, hostname: str) -> None:
        """Resolve a hostname and reject local, private, or otherwise internal IPs."""
        try:
            ip_literal = ipaddress.ip_address(hostname)
        except ValueError:
            ip_literal = None

        if ip_literal is not None:
            self._ensure_public_ip(ip_literal, hostname)
            return

        try:
            resolved = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise RuntimeError(f'Could not resolve the requested page hostname: {hostname}.') from exc

        if not resolved:
            raise RuntimeError(f'Could not resolve the requested page hostname: {hostname}.')

        for result in resolved:
            sockaddr = result[4]
            if not sockaddr:
                continue
            ip_text = str(sockaddr[0])
            try:
                resolved_ip = ipaddress.ip_address(ip_text)
            except ValueError:
                continue
            self._ensure_public_ip(resolved_ip, hostname)

    def _ensure_public_ip(self, address: ipaddress._BaseAddress, hostname: str) -> None:
        """Reject IPs that point to local, private, or non-public address space."""
        blocked = (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_unspecified
            or address.is_multicast
            or address.is_reserved
        )
        if blocked:
            raise RuntimeError(
                f'Blocked web fetch target: {hostname} resolved to internal or non-public address {address}. '
                'Localhost, private, link-local, multicast, and reserved network ranges are not allowed.'
            )


    def _get_response_charset(self, response) -> str:
        """Read the response charset in a way that works for real and mocked headers."""
        headers = getattr(response, 'headers', None)
        if headers is not None and hasattr(headers, 'get_content_charset'):
            try:
                return str(headers.get_content_charset('utf-8') or 'utf-8')
            except TypeError:
                pass

        content_type = ''
        if headers is not None:
            try:
                content_type = str(headers.get('Content-Type', ''))
            except AttributeError:
                content_type = str(getattr(headers, 'Content-Type', ''))
        for part in content_type.split(';')[1:]:
            key, _, value = part.partition('=')
            if key.strip().lower() == 'charset' and value.strip():
                return value.strip().strip('"').strip("'")
        return 'utf-8'

    def _read_response_with_limit(self, response) -> bytes:
        """Read the body incrementally and stop once the configured byte cap is exceeded."""
        remaining = self.max_response_bytes
        chunks: List[bytes] = []
        chunk_size = 64 * 1024

        while True:
            if remaining <= 0:
                raise RuntimeError(
                    'Web fetch blocked because the response body exceeded the configured byte limit '
                    f'of {self.max_response_bytes} bytes.'
                )
            chunk = response.read(min(chunk_size, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)

        return b''.join(chunks)

    def _normalize_domain_rules(self, rules: Sequence[str] | None) -> set[str]:
        """Normalize domain allow/block entries into a canonical lowercase set."""
        normalized: set[str] = set()
        for rule in rules or ():
            cleaned = str(rule).strip().lower().lstrip('.').rstrip('.')
            if cleaned:
                normalized.add(cleaned)
        return normalized

    def _matches_domain_rules(self, hostname: str, rules: Sequence[str] | set[str]) -> bool:
        """Return True when a hostname exactly matches or is a subdomain of a rule."""
        for rule in rules:
            normalized_rule = str(rule).strip().lower().lstrip('.').rstrip('.')
            if not normalized_rule:
                continue
            if hostname == normalized_rule or hostname.endswith(f'.{normalized_rule}'):
                return True
        return False

    def _normalize_content_type(self, value: str) -> str:
        """Reduce a Content-Type header to its MIME type only."""
        return str(value or '').split(';', 1)[0].strip().lower()

    def _safe_int(self, value: Any) -> int | None:
        """Convert a numeric-looking value to int or return None when absent/invalid."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
