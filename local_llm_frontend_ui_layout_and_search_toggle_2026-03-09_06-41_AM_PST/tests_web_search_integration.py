"""
tests_web_search_integration.py

Purpose:
    Regression tests for the SearXNG-backed web search integration and Ollama
    tool-calling bridge.

What this file does:
    - Verifies SearXNG results are normalized into the internal schema.
    - Verifies HTML fetch extraction returns readable page text.
    - Verifies the Ollama client executes a `search_web` tool call and then
      performs a second streamed request using the tool result.

How this file fits into the system:
    These tests cover the new web retrieval layer added to support grounded
    browsing through the local front end.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import error, parse, request

from services.config_service import ConfigService
from services.ollama_client import OllamaClient
from services.reranker_service import (
    NonLlmSearchRerankerService,
    SearchRerankerService,
    SentenceTransformersSearchRerankerService,
)
from services.search_service import SearchService
from services.web_fetch_service import WebFetchService

from controllers.chat_controller import ChatController
from models.generation_job import GenerationJob
from services.session_service import SessionService


class LiveSearchFetchPipelineTests(unittest.TestCase):
    """Run an opt-in live SearXNG -> rerank -> fetch pipeline test with trace logging.

    Why this test exists:
        The existing search/fetch tests are fast and deterministic because they
        use mocks. They protect control flow, but they do not show what happens
        against a real SearXNG instance and real fetched pages. This test is an
        explicit live smoke test for that pipeline.

    How to use it:
        Set ``LLM_FRONTEND_RUN_LIVE_WEB_PIPELINE_TEST=1`` before running the
        test suite. Optional environment variables let the reviewer override the
        SearXNG URL, query, fetch URL, and log-file path. The test writes a
        step-by-step trace to stdout and optionally to a file so the reviewer
        can inspect what actually happened in the live pipeline.
    """

    _ENABLE_ENV = 'LLM_FRONTEND_RUN_LIVE_WEB_PIPELINE_TEST'
    _BASE_URL_ENV = 'LLM_FRONTEND_LIVE_WEB_BASE_URL'
    _QUERY_ENV = 'LLM_FRONTEND_LIVE_WEB_QUERY'
    _MAX_RESULTS_ENV = 'LLM_FRONTEND_LIVE_WEB_MAX_RESULTS'
    _CATEGORY_ENV = 'LLM_FRONTEND_LIVE_WEB_CATEGORY'
    _LANGUAGE_ENV = 'LLM_FRONTEND_LIVE_WEB_LANGUAGE'
    _TIME_RANGE_ENV = 'LLM_FRONTEND_LIVE_WEB_TIME_RANGE'
    _SAFE_SEARCH_ENV = 'LLM_FRONTEND_LIVE_WEB_SAFE_SEARCH'
    _FETCH_URL_ENV = 'LLM_FRONTEND_LIVE_WEB_FETCH_URL'
    _LOG_PATH_ENV = 'LLM_FRONTEND_LIVE_WEB_LOG_PATH'
    _ALLOWLIST_ENV = 'LLM_FRONTEND_LIVE_WEB_FETCH_ALLOWLIST'
    _BLOCKLIST_ENV = 'LLM_FRONTEND_LIVE_WEB_FETCH_BLOCKLIST'

    def setUp(self) -> None:
        if not self._env_flag_enabled(self._ENABLE_ENV):
            self.skipTest(
                'Live web pipeline test is disabled. Set '
                'LLM_FRONTEND_RUN_LIVE_WEB_PIPELINE_TEST=1 to enable it.'
            )

        self.base_url = os.environ.get(self._BASE_URL_ENV, 'http://127.0.0.1:8080').strip()
        self.query = os.environ.get(self._QUERY_ENV, 'latest local llm tooling news').strip()
        self.max_results = max(1, min(10, self._safe_int_env(self._MAX_RESULTS_ENV, 5)))
        self.category = os.environ.get(self._CATEGORY_ENV, 'general').strip() or 'general'
        self.language = os.environ.get(self._LANGUAGE_ENV, 'all').strip() or 'all'
        self.time_range = os.environ.get(self._TIME_RANGE_ENV, 'none').strip() or 'none'
        self.safe_search = max(0, min(2, self._safe_int_env(self._SAFE_SEARCH_ENV, 1)))
        self.fetch_url_override = os.environ.get(self._FETCH_URL_ENV, '').strip()
        self.log_path = os.environ.get(self._LOG_PATH_ENV, '').strip()
        self.fetch_allowlist = self._split_csv_env(self._ALLOWLIST_ENV)
        self.fetch_blocklist = self._split_csv_env(self._BLOCKLIST_ENV)
        self.logger = self._build_logger()

        self._log(
            'Live pipeline test configuration: '
            f'base_url={self.base_url} query={self.query!r} category={self.category} '
            f'language={self.language} time_range={self.time_range} safe_search={self.safe_search} '
            f'max_results={self.max_results} fetch_url_override={self.fetch_url_override or "<auto>"} '
            f'allowlist={self.fetch_allowlist or ["<none>"]} blocklist={self.fetch_blocklist or ["<none>"]}'
        )

    def test_live_search_rerank_and_fetch_pipeline(self) -> None:
        search_service = SearchService(self.base_url, timeout_seconds=30)
        reranker_service = NonLlmSearchRerankerService()
        fetch_service = WebFetchService(
            timeout_seconds=30,
            max_characters=4000,
            max_response_bytes=1_500_000,
            max_redirects=4,
            domain_allowlist=self.fetch_allowlist,
            domain_blocklist=self.fetch_blocklist,
        )

        self._log('STEP 1: Running live SearXNG query.')
        search_payload = search_service.search(
            self.query,
            category=self.category,
            language=self.language,
            time_range=self.time_range,
            safe_search=self.safe_search,
            max_results=self.max_results,
        )
        self._log(
            f'Search returned {search_payload.get("result_count", 0)} normalized result(s) for query {self.query!r}.'
        )
        for item in search_payload.get('results', []):
            self._log(
                'SEARCH RESULT: '
                f'id={item.get("id")} position={item.get("position")} domain={item.get("domain")} '
                f'title={item.get("title", "")[:120]!r} url={item.get("url")} '
                f'published_date={item.get("published_date")!r}'
            )

        self.assertGreater(search_payload.get('result_count', 0), 0, 'Live search returned zero results.')

        self._log('STEP 2: Applying deterministic stage-1 reranker to the live search payload.')
        reranked_payload = reranker_service.rerank_search_payload(self.query, search_payload)
        for item in reranked_payload.get('results', []):
            self._log(
                'RERANKED RESULT: '
                f'id={item.get("id")} heuristic_score={item.get("heuristic_score")} '
                f'domain={item.get("domain")} title={item.get("title", "")[:120]!r} url={item.get("url")}'
            )

        candidate_results = list(reranked_payload.get('results', []))
        if self.fetch_url_override:
            override_match = next((item for item in candidate_results if item.get('url') == self.fetch_url_override), None)
            if override_match is None:
                candidate_results.insert(0, {
                    'id': 'manual-fetch-url',
                    'title': 'Manual live fetch override',
                    'url': self.fetch_url_override,
                    'domain': parse.urlparse(self.fetch_url_override).netloc,
                    'position': 0,
                })
            else:
                candidate_results.remove(override_match)
                candidate_results.insert(0, override_match)
            self._log(f'Fetch candidate order overridden to try {self.fetch_url_override} first.')

        self._log('STEP 3: Attempting live fetches in reranked order until one succeeds.')
        fetch_failures = []
        fetched_payload = None
        selected_candidate = None
        for candidate in candidate_results:
            candidate_url = str(candidate.get('url', '')).strip()
            if not candidate_url:
                continue
            self._log(
                'FETCH ATTEMPT: '
                f'id={candidate.get("id")} domain={candidate.get("domain")} url={candidate_url}'
            )
            try:
                fetched_payload = fetch_service.fetch(candidate_url)
                selected_candidate = candidate
                self._log(
                    'FETCH SUCCESS: '
                    f'final_url={fetched_payload.get("final_url")} title={fetched_payload.get("title")!r} '
                    f'content_type={fetched_payload.get("content_type")} text_length={len(fetched_payload.get("text", ""))}'
                )
                break
            except Exception as exc:  # pragma: no cover - exact live failure depends on external sites.
                failure_text = f'{candidate_url} -> {exc}'
                fetch_failures.append(failure_text)
                self._log(f'FETCH FAILURE: {failure_text}')

        if fetched_payload is None or selected_candidate is None:
            self.fail(
                'Live pipeline search succeeded, but every fetch candidate failed. '
                f'Failures observed: {fetch_failures}'
            )

        preview = fetched_payload.get('text', '').replace('\n', ' ')[:500]
        self._log(
            'STEP 4: Final live fetch preview: '
            f'selected_id={selected_candidate.get("id")} selected_url={selected_candidate.get("url")} '
            f'preview={preview!r}'
        )

        self.assertTrue(fetched_payload.get('title'))
        self.assertTrue(fetched_payload.get('text'))
        self.assertGreater(len(fetched_payload.get('text', '')), 50)

    def _build_logger(self) -> logging.Logger:
        logger_name = 'tests.live_web_pipeline'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        stream_handler_exists = any(
            isinstance(handler, logging.StreamHandler) and getattr(handler, '_live_pipeline_stream', False)
            for handler in logger.handlers
        )
        if not stream_handler_exists:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler._live_pipeline_stream = True
            stream_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
            logger.addHandler(stream_handler)

        if self.log_path:
            log_path = Path(self.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler_exists = any(
                isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
                for handler in logger.handlers
            )
            if not file_handler_exists:
                file_handler = logging.FileHandler(log_path, encoding='utf-8')
                file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
                logger.addHandler(file_handler)
        return logger

    def _log(self, message: str) -> None:
        self.logger.info(message)

    def _env_flag_enabled(self, env_name: str) -> bool:
        value = os.environ.get(env_name, '').strip().lower()
        return value in {'1', 'true', 'yes', 'on'}

    def _safe_int_env(self, env_name: str, default: int) -> int:
        try:
            return int(str(os.environ.get(env_name, default)).strip())
        except (TypeError, ValueError):
            return default

    def _split_csv_env(self, env_name: str) -> list[str]:
        raw_value = os.environ.get(env_name, '')
        return [part.strip() for part in raw_value.split(',') if part.strip()]


class SearchServiceTests(unittest.TestCase):
    """Validate normalized SearXNG search output."""

    def test_search_normalizes_result_shape(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict) -> None:
                self._payload = payload
            def read(self) -> bytes:
                return json.dumps(self._payload).encode('utf-8')
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        payload = {
            'results': [
                {
                    'title': 'Qdrant Docs',
                    'url': 'https://qdrant.tech/documentation/',
                    'content': 'Vector database documentation.',
                    'engines': ['brave'],
                    'category': 'it',
                }
            ]
        }

        with patch('services.search_service.request.urlopen', return_value=FakeResponse(payload)):
            result = SearchService('http://127.0.0.1:8080').search('qdrant')

        self.assertEqual(result['query'], 'qdrant')
        self.assertEqual(result['result_count'], 1)
        self.assertEqual(result['results'][0]['domain'], 'qdrant.tech')
        self.assertEqual(result['results'][0]['engine'], 'brave')



    def test_base_url_without_scheme_is_repaired(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict) -> None:
                self._payload = payload
            def read(self) -> bytes:
                return json.dumps(self._payload).encode('utf-8')
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        with patch('services.search_service.request.urlopen', return_value=FakeResponse({'results': []})) as mocked_open:
            SearchService('localhost:8080').search('qdrant')

        requested_url = mocked_open.call_args.args[0].full_url
        self.assertTrue(requested_url.startswith('http://localhost:8080/search?'))


    def test_full_search_endpoint_is_reduced_to_instance_root(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict) -> None:
                self._payload = payload
            def read(self) -> bytes:
                return json.dumps(self._payload).encode('utf-8')
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        with patch('services.search_service.request.urlopen', return_value=FakeResponse({'results': []})) as mocked_open:
            SearchService('http://127.0.0.1:8080/search?q=old&format=json').search('qdrant')

        requested_url = mocked_open.call_args.args[0].full_url
        self.assertTrue(requested_url.startswith('http://127.0.0.1:8080/search?'))
        self.assertNotIn('old', requested_url)

    def test_connection_refused_message_includes_actionable_host_and_port_hint(self) -> None:
        with patch(
            'services.search_service.request.urlopen',
            side_effect=error.URLError(ConnectionRefusedError('refused'))
        ):
            with self.assertRaises(RuntimeError) as exc_info:
                SearchService('http://127.0.0.1:8080').search('qdrant')

        message = str(exc_info.exception)
        self.assertIn('127.0.0.1:8080', message)
        self.assertIn('search?q=test&format=json', message)
        self.assertIn('Connection was refused', message)


class NonLlmSearchRerankerServiceTests(unittest.TestCase):
    """Validate deterministic first-pass reranking of normalized search candidates."""

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def test_non_llm_rerank_search_payload_prefers_direct_match_in_title_and_snippet(self) -> None:
        service = NonLlmSearchRerankerService()
        search_payload = {
            'query': 'qdrant docs',
            'result_count': 3,
            'results': [
                {'id': 'r1', 'title': 'Qdrant Homepage', 'url': 'https://qdrant.tech/', 'domain': 'qdrant.tech', 'snippet': 'Vector database home page.', 'position': 1},
                {'id': 'r2', 'title': 'Qdrant Documentation', 'url': 'https://qdrant.tech/documentation/', 'domain': 'qdrant.tech', 'snippet': 'Official docs and API reference.', 'position': 2},
                {'id': 'r3', 'title': 'Random Blog', 'url': 'https://blog.example/qdrant', 'domain': 'blog.example', 'snippet': 'Some notes about vectors.', 'position': 3},
            ],
        }

        result = service.rerank_search_payload('qdrant docs', search_payload)

        self.assertEqual([item['id'] for item in result['results']], ['r2', 'r1', 'r3'])
        self.assertTrue(result['heuristic_reranker']['applied'])
        self.assertEqual(result['heuristic_reranker']['method'], 'query_title_snippet_domain_scoring')
        self.assertGreater(result['results'][0]['heuristic_score'], result['results'][1]['heuristic_score'])


class SearchRerankerServiceTests(unittest.TestCase):
    """Validate optional Ollama reranking after deterministic screening."""

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def test_rerank_search_payload_reorders_results_and_appends_missing_ids(self) -> None:
        service = SearchRerankerService(host='127.0.0.1', port=11434, model_name='tiny-reranker')
        search_payload = {
            'query': 'latest cat news',
            'result_count': 3,
            'results': [
                {'id': 'r1', 'title': 'Result 1', 'url': 'https://a.example', 'domain': 'a.example', 'snippet': 'first', 'position': 1, 'heuristic_score': 1.5},
                {'id': 'r2', 'title': 'Result 2', 'url': 'https://b.example', 'domain': 'b.example', 'snippet': 'second', 'position': 2, 'heuristic_score': 1.2},
                {'id': 'r3', 'title': 'Result 3', 'url': 'https://c.example', 'domain': 'c.example', 'snippet': 'third', 'position': 3, 'heuristic_score': 0.2},
            ],
        }
        response_payload = {
            'response': json.dumps({'ordered_result_ids': ['r2', 'r1'], 'notes': 'Prefer fresher source'})
        }

        with patch('services.reranker_service.request.urlopen', return_value=self.FakeResponse(response_payload)):
            result = service.rerank_search_payload('latest cat news', search_payload)

        self.assertEqual([item['id'] for item in result['results']], ['r2', 'r1', 'r3'])
        self.assertTrue(result['reranker']['applied'])
        self.assertEqual(result['reranker']['model'], 'tiny-reranker')


class SentenceTransformersSearchRerankerServiceTests(unittest.TestCase):
    """Validate sentence-transformers reranking after deterministic screening."""

    def test_rerank_search_payload_reorders_results_from_predicted_scores(self) -> None:
        service = SentenceTransformersSearchRerankerService(model_name='BAAI/bge-reranker-base', device='cpu')
        search_payload = {
            'query': 'latest cat news',
            'result_count': 3,
            'results': [
                {'id': 'r1', 'title': 'Result 1', 'url': 'https://a.example', 'domain': 'a.example', 'snippet': 'first', 'position': 1, 'heuristic_score': 1.5},
                {'id': 'r2', 'title': 'Result 2', 'url': 'https://b.example', 'domain': 'b.example', 'snippet': 'second', 'position': 2, 'heuristic_score': 1.2},
                {'id': 'r3', 'title': 'Result 3', 'url': 'https://c.example', 'domain': 'c.example', 'snippet': 'third', 'position': 3, 'heuristic_score': 0.2},
            ],
        }

        with patch.object(service, '_score_pairs', return_value=[0.15, 0.91, 0.32]):
            result = service.rerank_search_payload('latest cat news', search_payload)

        self.assertEqual([item['id'] for item in result['results']], ['r2', 'r3', 'r1'])
        self.assertTrue(result['reranker']['applied'])
        self.assertEqual(result['reranker']['backend'], 'sentence_transformers')
        self.assertEqual(result['reranker']['model'], 'BAAI/bge-reranker-base')


class WebFetchServiceTests(unittest.TestCase):
    """Validate readable extraction from fetched HTML and safety controls."""

    class FakeHeaders(dict):
        """Minimal header object that mimics HTTPMessage methods used by the service."""

        def get_content_charset(self, default='utf-8'):
            content_type = str(self.get('Content-Type', ''))
            for part in content_type.split(';')[1:]:
                key, _, value = part.partition('=')
                if key.strip().lower() == 'charset' and value.strip():
                    return value.strip()
            return default

    class FakeResponse:
        """Simple fake HTTP response for WebFetchService tests."""

        def __init__(self, html: str, *, final_url: str = 'https://example.com/final', content_type: str = 'text/html; charset=utf-8', content_length: int | None = None) -> None:
            self._bytes = html.encode('utf-8')
            self._offset = 0
            self._final_url = final_url
            self.headers = WebFetchServiceTests.FakeHeaders({'Content-Type': content_type})
            if content_length is not None:
                self.headers['Content-Length'] = str(content_length)

        def read(self, amount: int = -1) -> bytes:
            if amount is None or amount < 0:
                amount = len(self._bytes) - self._offset
            chunk = self._bytes[self._offset:self._offset + amount]
            self._offset += len(chunk)
            return chunk

        def geturl(self) -> str:
            return self._final_url

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _patch_public_resolution(self):
        return patch('services.web_fetch_service.socket.getaddrinfo', return_value=[(None, None, None, None, ('93.184.216.34', 0))])

    def test_fetch_extracts_title_and_body_text(self) -> None:
        html = '<html><head><title>Example</title></head><body><p>' + ('A' * 60) + '</p></body></html>'
        with self._patch_public_resolution(), patch(
            'services.web_fetch_service.request.build_opener'
        ) as mocked_build_opener:
            mocked_build_opener.return_value.open.return_value = self.FakeResponse(html)
            result = WebFetchService().fetch('https://example.com/start')

        self.assertEqual(result['title'], 'Example')
        self.assertIn('A' * 40, result['text'])
        self.assertEqual(result['final_url'], 'https://example.com/final')

    def test_fetch_blocks_localhost(self) -> None:
        with self.assertRaises(RuntimeError) as exc_info:
            WebFetchService().fetch('http://localhost:8080/private')
        self.assertIn('localhost', str(exc_info.exception).lower())

    def test_fetch_blocks_private_ip_literal(self) -> None:
        with self.assertRaises(RuntimeError) as exc_info:
            WebFetchService().fetch('http://192.168.1.10/admin')
        self.assertIn('internal or non-public address', str(exc_info.exception))

    def test_fetch_blocks_hostname_that_resolves_private(self) -> None:
        with patch('services.web_fetch_service.socket.getaddrinfo', return_value=[(None, None, None, None, ('10.0.0.7', 0))]):
            with self.assertRaises(RuntimeError) as exc_info:
                WebFetchService().fetch('https://example.com/start')
        self.assertIn('10.0.0.7', str(exc_info.exception))

    def test_fetch_blocks_oversized_content_length_before_read(self) -> None:
        html = '<html><body><p>' + ('A' * 60) + '</p></body></html>'
        with self._patch_public_resolution(), patch('services.web_fetch_service.request.build_opener') as mocked_build_opener:
            mocked_build_opener.return_value.open.return_value = self.FakeResponse(
                html,
                content_length=2000,
            )
            with self.assertRaises(RuntimeError) as exc_info:
                WebFetchService(max_response_bytes=1024).fetch('https://example.com/start')
        self.assertIn('too large', str(exc_info.exception))

    def test_fetch_blocks_unsupported_content_type(self) -> None:
        html = '{"status": "ok"}'
        with self._patch_public_resolution(), patch('services.web_fetch_service.request.build_opener') as mocked_build_opener:
            mocked_build_opener.return_value.open.return_value = self.FakeResponse(
                html,
                content_type='application/json',
            )
            with self.assertRaises(RuntimeError) as exc_info:
                WebFetchService().fetch('https://example.com/start')
        self.assertIn('Unsupported content type', str(exc_info.exception))

    def test_fetch_enforces_domain_allowlist(self) -> None:
        service = WebFetchService(domain_allowlist=['reuters.com'])
        with self.assertRaises(RuntimeError) as exc_info:
            service.fetch('https://example.com/story')
        self.assertIn('allowlist', str(exc_info.exception))

    def test_redirect_handler_revalidates_targets(self) -> None:
        service = WebFetchService(max_redirects=1)
        handler = service._build_redirect_handler_for_test()
        req = request.Request('https://example.com/start', method='GET')
        setattr(req, '_safe_redirect_count', 0)
        with patch.object(service, '_validate_url_target', side_effect=RuntimeError('blocked redirect')):
            handler = service._build_redirect_handler_for_test()
            with self.assertRaises(RuntimeError) as exc_info:
                handler.redirect_request(req, None, 302, 'Found', {}, 'http://127.0.0.1/admin')
        self.assertIn('blocked redirect', str(exc_info.exception))


class GenerationJobWebConfigTests(unittest.TestCase):
    """Verify that in-flight requests preserve web-search settings."""

    def test_to_request_options_includes_web_search_settings(self) -> None:
        job = GenerationJob(
            request_id='job1',
            session_id='session1',
            session_title_at_start='Session 1',
            model_name='qwen3:14b',
            host='127.0.0.1',
            port=11434,
            timeout_seconds=30,
            system_prompt='prompt',
            thinking_mode=True,
            thinking_level='medium',
            enable_web_search=True,
            searxng_base_url='http://127.0.0.1:8081',
            web_search_category='news',
            web_search_language='all',
            web_search_time_range='day',
            web_safe_search=1,
            web_max_results=6,
            enable_search_reranker=True,
            search_reranker_backend='ollama',
            search_reranker_model='tiny-reranker',
            sentence_transformers_reranker_model='BAAI/bge-reranker-base',
            sentence_transformers_reranker_device='auto',
            web_max_pages=2,
            web_fetch_enabled=True,
            web_fetch_max_response_bytes=1500000,
            web_fetch_max_redirects=4,
            web_fetch_allowlist="reuters.com",
            web_fetch_blocklist="example.com",
        )

        options = job.to_request_options()

        self.assertTrue(options['enable_web_search'])
        self.assertEqual(options['searxng_base_url'], 'http://127.0.0.1:8081')
        self.assertEqual(options['web_search_category'], 'news')
        self.assertEqual(options['web_search_time_range'], 'day')
        self.assertEqual(options['web_max_results'], 6)
        self.assertTrue(options['enable_search_reranker'])
        self.assertEqual(options['search_reranker_backend'], 'ollama')
        self.assertEqual(options['search_reranker_model'], 'tiny-reranker')
        self.assertEqual(options['sentence_transformers_reranker_model'], 'BAAI/bge-reranker-base')
        self.assertEqual(options['sentence_transformers_reranker_device'], 'auto')
        self.assertTrue(options['web_fetch_enabled'])
        self.assertEqual(options['web_fetch_max_response_bytes'], 1500000)
        self.assertEqual(options['web_fetch_max_redirects'], 4)
        self.assertEqual(options['web_fetch_allowlist'], 'reuters.com')
        self.assertEqual(options['web_fetch_blocklist'], 'example.com')


class ChatControllerGenerationSnapshotTests(unittest.TestCase):
    """Verify the controller freezes web settings into each generation job."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config_service = ConfigService(config_path=root / 'config.json')
        self.config_service.update_config({
            'host': '127.0.0.1',
            'port': 11434,
            'model': 'qwen3:14b',
            'timeout_seconds': 30,
            'system_prompt': 'test prompt',
            'thinking_mode': True,
            'thinking_level': 'medium',
            'enable_web_search': True,
            'searxng_base_url': 'http://127.0.0.1:8081',
            'web_search_category': 'news',
            'web_search_language': 'all',
            'web_search_time_range': 'day',
            'web_safe_search': 1,
            'web_max_results': 7,
            'enable_search_reranker': True,
            'search_reranker_backend': 'ollama',
            'search_reranker_model': 'tiny-reranker',
            'sentence_transformers_reranker_model': 'BAAI/bge-reranker-base',
            'sentence_transformers_reranker_device': 'auto',
            'web_max_pages': 2,
            'web_fetch_enabled': True,
            'web_fetch_max_response_bytes': 1500000,
            'web_fetch_max_redirects': 4,
            'web_fetch_allowlist': 'reuters.com',
            'web_fetch_blocklist': 'example.com',
        })
        self.controller = ChatController(
            self.config_service,
            OllamaClient(self.config_service),
            SessionService(root / 'sessions'),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_generation_job_captures_web_search_config(self) -> None:
        job = self.controller._create_generation_job('latest cat news')

        self.assertTrue(job.enable_web_search)
        self.assertEqual(job.searxng_base_url, 'http://127.0.0.1:8081')
        self.assertEqual(job.web_search_category, 'news')
        self.assertEqual(job.web_search_time_range, 'day')
        self.assertEqual(job.web_max_results, 7)
        self.assertTrue(job.enable_search_reranker)
        self.assertEqual(job.search_reranker_backend, 'ollama')
        self.assertEqual(job.search_reranker_model, 'tiny-reranker')
        self.assertEqual(job.sentence_transformers_reranker_model, 'BAAI/bge-reranker-base')
        self.assertEqual(job.sentence_transformers_reranker_device, 'auto')
        self.assertTrue(job.web_fetch_enabled)
        self.assertEqual(job.web_fetch_max_response_bytes, 1500000)
        self.assertEqual(job.web_fetch_max_redirects, 4)
        self.assertEqual(job.web_fetch_allowlist, 'reuters.com')
        self.assertEqual(job.web_fetch_blocklist, 'example.com')



class OllamaToolCallingTests(unittest.TestCase):
    """Validate the search tool loop inside OllamaClient."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config_service = ConfigService(config_path=root / 'config.json')
        self.config_service.update_config({
            'host': '127.0.0.1',
            'port': 11434,
            'model': 'qwen3',
            'timeout_seconds': 30,
            'system_prompt': 'test prompt',
            'thinking_mode': False,
            'enable_web_search': True,
            'searxng_base_url': 'http://127.0.0.1:8080',
            'web_max_results': 5,
            'enable_search_reranker': True,
            'search_reranker_backend': 'ollama',
            'search_reranker_model': 'tiny-reranker',
            'sentence_transformers_reranker_model': 'BAAI/bge-reranker-base',
            'sentence_transformers_reranker_device': 'auto',
        })

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


    def test_chat_stream_adds_web_tool_guidance_system_message(self) -> None:
        client = OllamaClient(self.config_service)

        messages = client._build_chat_messages(
            'base prompt',
            [],
            web_tools_enabled=True,
        )

        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], 'base prompt')
        self.assertEqual(messages[1]['role'], 'system')
        self.assertIn('Web search tools are available', messages[1]['content'])

    def test_post_tool_grounding_message_explicitly_allows_post_cutoff_results(self) -> None:
        client = OllamaClient(self.config_service)

        message = client._build_post_tool_grounding_message()

        self.assertEqual(message["role"], "system")
        self.assertIn('training cutoff', message["content"])
        self.assertIn('current runtime evidence', message["content"])

    def test_chat_stream_forces_search_when_model_claims_no_internet_access(self) -> None:
        client = OllamaClient(self.config_service)
        captured_chunks = []
        initial_response = {
            'message': {
                'role': 'assistant',
                'content': "I do not have internet access, so I cannot search for cats.",
            }
        }
        final_stream = [
            {'message': {'content': 'Cats are small domesticated mammals.'}, 'done': False},
            {'message': {}, 'done': True, 'eval_count': 12, 'eval_duration': 1_000_000_000},
        ]

        with patch.object(client, '_request_json_with_request_options', return_value=initial_response):
            with patch.object(client, '_execute_tool_call', return_value={'query': 'cats', 'result_count': 1, 'results': []}) as execute_tool:
                with patch.object(client, '_stream_json_lines', return_value=final_stream):
                    result = client.chat_stream(
                        [type('Msg', (), {'role': 'user', 'content': 'search for cats'})()],
                        on_chunk=lambda kind, chunk: captured_chunks.append((kind, chunk)),
                    )

        execute_tool.assert_called_once()
        called_name = execute_tool.call_args.args[0]
        called_arguments = execute_tool.call_args.args[1]
        self.assertEqual(called_name, 'search_web')
        self.assertEqual(called_arguments['query'], 'search for cats')
        self.assertEqual(result['content'], 'Cats are small domesticated mammals.')
        self.assertEqual(captured_chunks, [('content', 'Cats are small domesticated mammals.')])

    def test_chat_stream_forces_search_for_latest_news_prompt_even_without_explicit_no_internet_phrase(self) -> None:
        client = OllamaClient(self.config_service)
        captured_chunks = []
        initial_response = {
            'message': {
                'role': 'assistant',
                'content': "I currently don't have access to real-time news updates, but I can share recent trends about cats.",
            }
        }
        final_stream = [
            {'message': {'content': 'Here are the latest cat news results from web search.'}, 'done': False},
            {'message': {}, 'done': True, 'eval_count': 14, 'eval_duration': 1_000_000_000},
        ]

        with patch.object(client, '_request_json_with_request_options', return_value=initial_response):
            with patch.object(client, '_execute_tool_call', return_value={'query': 'Give me the latest news about cats', 'result_count': 2, 'results': []}) as execute_tool:
                with patch.object(client, '_stream_json_lines', return_value=final_stream):
                    result = client.chat_stream(
                        [type('Msg', (), {'role': 'user', 'content': 'Give me the latest news about cats'})()],
                        on_chunk=lambda kind, chunk: captured_chunks.append((kind, chunk)),
                    )

        execute_tool.assert_called_once()
        self.assertEqual(execute_tool.call_args.args[0], 'search_web')
        self.assertEqual(execute_tool.call_args.args[1]['query'], 'Give me the latest news about cats')
        self.assertEqual(result['content'], 'Here are the latest cat news results from web search.')
        self.assertEqual(captured_chunks, [('content', 'Here are the latest cat news results from web search.')])

    def test_chat_stream_executes_search_tool_and_streams_final_answer(self) -> None:
        client = OllamaClient(self.config_service)
        captured_chunks = []
        tool_result = {'query': 'qdrant', 'result_count': 1, 'results': [{'id': 'r1', 'title': 'Qdrant'}]}
        initial_response = {
            'message': {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {'function': {'name': 'search_web', 'arguments': {'query': 'qdrant', 'max_results': 3}}}
                ],
            }
        }
        final_stream = [
            {'message': {'content': 'Grounded answer'}, 'done': False},
            {'message': {}, 'done': True, 'eval_count': 10, 'eval_duration': 1_000_000_000},
        ]

        with patch.object(client, '_request_json_with_request_options', return_value=initial_response):
            with patch.object(client, '_execute_tool_call', return_value=tool_result) as execute_tool:
                with patch.object(client, '_stream_json_lines', return_value=final_stream):
                    result = client.chat_stream([], on_chunk=lambda kind, chunk: captured_chunks.append((kind, chunk)))

        execute_tool.assert_called_once()
        self.assertEqual(result['content'], 'Grounded answer')
        self.assertEqual(result['output_tokens'], 10)
        self.assertAlmostEqual(result['tokens_per_second'], 10.0)
        self.assertEqual(captured_chunks, [('content', 'Grounded answer')])

    def test_fetch_tool_schema_mentions_per_request_page_limit(self) -> None:
        client = OllamaClient(self.config_service)

        tools = client._build_web_tools_schema({
            **self.config_service.get_config(),
            'web_fetch_enabled': True,
            'web_max_pages': 2,
        })

        fetch_tool = next(tool for tool in tools if tool['function']['name'] == 'fetch_url_content')
        description = fetch_tool['function']['description']
        self.assertIn('At most 2 page fetch(es)', description)

    def test_execute_tool_call_enforces_page_fetch_limit_with_structured_error(self) -> None:
        client = OllamaClient(self.config_service)
        config = {
            **self.config_service.get_config(),
            'web_fetch_enabled': True,
            'web_max_pages': 1,
        }
        runtime_state = client._build_tool_runtime_state(config)

        with patch('services.ollama_client.WebFetchService.fetch', return_value={'url': 'https://example.com', 'content': 'ok'}) as mocked_fetch:
            first_result = client._execute_tool_call(
                'fetch_url_content',
                {'url': 'https://example.com'},
                config,
                runtime_state=runtime_state,
            )
            second_result = client._execute_tool_call(
                'fetch_url_content',
                {'url': 'https://example.org'},
                config,
                runtime_state=runtime_state,
            )

        mocked_fetch.assert_called_once_with('https://example.com')
        self.assertEqual(first_result['url'], 'https://example.com')
        self.assertFalse(second_result['ok'])
        self.assertIn('Page fetch limit reached', second_result['error'])
        self.assertEqual(second_result['page_fetch_limit'], 1)
        self.assertEqual(second_result['page_fetch_count'], 1)

    def test_chat_stream_returns_tool_error_after_fetch_limit_is_hit(self) -> None:
        client = OllamaClient(self.config_service)
        captured_chunks = []
        initial_response = {
            'message': {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {'function': {'name': 'fetch_url_content', 'arguments': {'url': 'https://example.com/page-1'}}},
                    {'function': {'name': 'fetch_url_content', 'arguments': {'url': 'https://example.com/page-2'}}},
                ],
            }
        }
        final_stream = [
            {'message': {'content': 'I read one page and the second fetch was rejected by policy.'}, 'done': False},
            {'message': {}, 'done': True, 'eval_count': 11, 'eval_duration': 1_000_000_000},
        ]

        with patch.object(client, '_request_json_with_request_options', return_value=initial_response):
            with patch('services.ollama_client.WebFetchService.fetch', return_value={'url': 'https://example.com/page-1', 'content': 'first page'}) as mocked_fetch:
                with patch.object(client, '_stream_json_lines', return_value=final_stream):
                    request_options = {
                        **self.config_service.get_config(),
                        'web_max_pages': 1,
                        'web_fetch_enabled': True,
                        'enable_web_search': True,
                        'model': 'qwen3',
                    }
                    result = client.chat_stream(
                        [type('Msg', (), {'role': 'user', 'content': 'Read two pages'})()],
                        on_chunk=lambda kind, chunk: captured_chunks.append((kind, chunk)),
                        request_options=request_options,
                    )

        mocked_fetch.assert_called_once_with('https://example.com/page-1')
        self.assertEqual(result['content'], 'I read one page and the second fetch was rejected by policy.')
        self.assertEqual(captured_chunks, [('content', 'I read one page and the second fetch was rejected by policy.')])

    def test_execute_tool_call_search_web_applies_reranker_before_returning_results(self) -> None:
        client = OllamaClient(self.config_service)
        config = self.config_service.get_config()
        base_results = {
            'query': 'qdrant docs',
            'result_count': 2,
            'results': [
                {'id': 'r1', 'title': 'Homepage', 'url': 'https://example.com', 'domain': 'example.com', 'snippet': 'broad page', 'position': 1},
                {'id': 'r2', 'title': 'Docs', 'url': 'https://docs.example.com', 'domain': 'docs.example.com', 'snippet': 'direct docs', 'position': 2},
            ],
        }
        reranked = {
            'query': 'qdrant docs',
            'result_count': 2,
            'results': [base_results['results'][1], base_results['results'][0]],
            'reranker': {'enabled': True, 'applied': True, 'model': 'tiny-reranker'},
        }

        with patch('services.ollama_client.SearchService.search', return_value=base_results) as mocked_search:
            with patch('services.ollama_client.NonLlmSearchRerankerService.rerank_search_payload', return_value=base_results) as mocked_stage1:
                with patch('services.ollama_client.SearchRerankerService.rerank_search_payload', return_value=reranked) as mocked_rerank:
                    result = client._execute_tool_call('search_web', {'query': 'qdrant docs'}, config)

        mocked_search.assert_called_once()
        mocked_stage1.assert_called_once()
        mocked_rerank.assert_called_once()
        self.assertEqual([item['id'] for item in result['results']], ['r2', 'r1'])
        self.assertTrue(result['reranker']['applied'])

    def test_execute_tool_call_search_web_uses_non_llm_reranker_when_llm_reranker_disabled(self) -> None:
        client = OllamaClient(self.config_service)
        config = self.config_service.get_config()
        config['enable_search_reranker'] = False
        config['search_reranker_model'] = ''
        base_results = {
            'query': 'qdrant docs',
            'result_count': 2,
            'results': [
                {'id': 'r1', 'title': 'Homepage', 'url': 'https://example.com', 'domain': 'example.com', 'snippet': 'broad page', 'position': 1},
                {'id': 'r2', 'title': 'Docs', 'url': 'https://docs.example.com', 'domain': 'docs.example.com', 'snippet': 'direct docs', 'position': 2},
            ],
        }
        stage1_result = {
            'query': 'qdrant docs',
            'result_count': 2,
            'results': [base_results['results'][1], base_results['results'][0]],
            'heuristic_reranker': {'enabled': True, 'applied': True, 'method': 'query_title_snippet_domain_scoring'},
        }

        with patch('services.ollama_client.SearchService.search', return_value=base_results):
            with patch('services.ollama_client.NonLlmSearchRerankerService.rerank_search_payload', return_value=stage1_result) as mocked_stage1:
                result = client._execute_tool_call('search_web', {'query': 'qdrant docs'}, config)

        mocked_stage1.assert_called_once()
        self.assertEqual([item['id'] for item in result['results']], ['r2', 'r1'])
        self.assertTrue(result['heuristic_reranker']['applied'])
        self.assertFalse(result['reranker']['applied'])

    def test_execute_tool_call_search_web_preserves_original_order_when_reranker_fails(self) -> None:
        client = OllamaClient(self.config_service)
        config = self.config_service.get_config()
        base_results = {
            'query': 'qdrant docs',
            'result_count': 2,
            'results': [
                {'id': 'r1', 'title': 'Homepage', 'url': 'https://example.com', 'domain': 'example.com', 'snippet': 'broad page', 'position': 1},
                {'id': 'r2', 'title': 'Docs', 'url': 'https://docs.example.com', 'domain': 'docs.example.com', 'snippet': 'direct docs', 'position': 2},
            ],
        }

        with patch('services.ollama_client.SearchService.search', return_value=base_results):
            with patch('services.ollama_client.NonLlmSearchRerankerService.rerank_search_payload', return_value={**base_results, 'results': [base_results['results'][1], base_results['results'][0]], 'heuristic_reranker': {'enabled': True, 'applied': True}}):
                with patch('services.ollama_client.SearchRerankerService.rerank_search_payload', side_effect=RuntimeError('reranker down')):
                    result = client._execute_tool_call('search_web', {'query': 'qdrant docs'}, config)

        self.assertEqual([item['id'] for item in result['results']], ['r2', 'r1'])
        self.assertFalse(result['reranker']['applied'])
        self.assertIn('reranker down', result['reranker']['error'])

    def test_chat_stream_places_tool_results_immediately_after_assistant_tool_call(self) -> None:
        client = OllamaClient(self.config_service)
        initial_response = {
            'message': {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {'id': 'call-1', 'function': {'name': 'search_web', 'arguments': {'query': 'qdrant', 'max_results': 3}}}
                ],
            }
        }
        final_stream = [
            {'message': {'content': 'Grounded answer'}, 'done': False},
            {'message': {}, 'done': True, 'eval_count': 10, 'eval_duration': 1_000_000_000},
        ]
        captured_messages = {}

        def fake_stream(endpoint, payload, request_options):
            captured_messages['messages'] = payload['messages']
            return final_stream

        with patch.object(client, '_request_json_with_request_options', return_value=initial_response):
            with patch.object(client, '_execute_tool_call', return_value={'query': 'qdrant', 'result_count': 1, 'results': []}):
                with patch.object(client, '_stream_json_lines', side_effect=fake_stream):
                    client.chat_stream([], on_chunk=lambda *_args: None)

        messages = captured_messages['messages']
        assistant_index = next(index for index, item in enumerate(messages) if item.get('role') == 'assistant')
        self.assertEqual(messages[assistant_index + 1]['role'], 'tool')
        self.assertEqual(messages[assistant_index + 1]['tool_call_id'], 'call-1')
        self.assertEqual(messages[assistant_index + 1]['name'], 'search_web')

    def test_forced_search_fallback_reuses_runtime_state(self) -> None:
        client = OllamaClient(self.config_service)
        runtime_state = {'page_fetch_count': 1, 'page_fetch_limit': 1}
        assistant_message = {'role': 'assistant', 'content': 'I do not have internet access.'}
        chat_messages = [{'role': 'user', 'content': 'search for cats'}]

        with patch.object(client, '_execute_tool_call', return_value={'query': 'search for cats', 'results': []}) as execute_tool:
            client._maybe_force_web_search_fallback(chat_messages, assistant_message, self.config_service.get_config(), runtime_state=runtime_state)

        self.assertIs(execute_tool.call_args.kwargs['runtime_state'], runtime_state)

    def test_prompt_likely_requires_web_search_avoids_false_positive_for_offline_current_question(self) -> None:
        client = OllamaClient(self.config_service)

        self.assertFalse(client._prompt_likely_requires_web_search('What is the current capital of France?'))
        self.assertFalse(client._prompt_likely_requires_web_search('Explain the current design of TCP/IP.'))
        self.assertTrue(client._prompt_likely_requires_web_search('What is the latest news about cats?'))
        self.assertTrue(client._prompt_likely_requires_web_search('Look up the current weather in Davis.'))


if __name__ == '__main__':
    unittest.main()
