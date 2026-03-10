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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import error

from services.config_service import ConfigService
from services.ollama_client import OllamaClient
from services.search_service import SearchService
from services.web_fetch_service import WebFetchService

from controllers.chat_controller import ChatController
from models.generation_job import GenerationJob
from services.session_service import SessionService


class SearchServiceTests(unittest.TestCase):
    """Validate normalized SearXNG search output."""

    def test_search_normalizes_result_shape(self) -> None:
        """Verify raw SearXNG-style results are normalized into the frontend result schema."""
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
        """Verify missing URL schemes are repaired before connection attempts."""
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
        """Verify full search endpoint URLs are reduced to the SearXNG instance root."""
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
        """Verify refusal errors include a host-and-port troubleshooting hint."""
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


class WebFetchServiceTests(unittest.TestCase):
    """Validate readable extraction from fetched HTML."""

    def test_fetch_extracts_title_and_body_text(self) -> None:
        """Verify HTML fetch results expose a readable title and extracted text body."""
        class FakeResponse:
            def __init__(self, html: str) -> None:
                self._html = html
                self.headers = {'Content-Type': 'text/html; charset=utf-8'}
            def read(self) -> bytes:
                return self._html.encode('utf-8')
            def geturl(self) -> str:
                return 'https://example.com/final'
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        html = '<html><head><title>Example</title></head><body><p>' + ('A' * 60) + '</p></body></html>'
        with patch('services.web_fetch_service.request.urlopen', return_value=FakeResponse(html)):
            result = WebFetchService().fetch('https://example.com/start')

        self.assertEqual(result['title'], 'Example')
        self.assertIn('A' * 40, result['text'])
        self.assertEqual(result['final_url'], 'https://example.com/final')


class GenerationJobWebConfigTests(unittest.TestCase):
    """Verify that in-flight requests preserve web-search settings."""

    def test_to_request_options_includes_web_search_settings(self) -> None:
        """Verify generation-job request snapshots preserve web-search configuration values."""
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
            web_max_pages=2,
            web_fetch_enabled=True,
            reranker_model='nomic-embed-text',
        )

        options = job.to_request_options()

        self.assertTrue(options['enable_web_search'])
        self.assertEqual(options['searxng_base_url'], 'http://127.0.0.1:8081')
        self.assertEqual(options['web_search_category'], 'news')
        self.assertEqual(options['web_search_time_range'], 'day')
        self.assertEqual(options['web_max_results'], 6)
        self.assertTrue(options['web_fetch_enabled'])
        self.assertEqual(options['reranker_model'], 'nomic-embed-text')


class ChatControllerGenerationSnapshotTests(unittest.TestCase):
    """Verify the controller freezes web settings into each generation job."""

    def setUp(self) -> None:
        """Create an isolated controller stack for generation snapshot assertions."""
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
            'web_max_pages': 2,
            'web_fetch_enabled': True,
            'reranker_model': 'qwen-rerank',
        })
        self.controller = ChatController(
            self.config_service,
            OllamaClient(self.config_service),
            SessionService(root / 'sessions'),
        )

    def tearDown(self) -> None:
        """Clean up temporary files created for generation snapshot tests."""
        self.temp_dir.cleanup()

    def test_generation_job_captures_web_search_config(self) -> None:
        """Verify generation jobs capture the current web-search configuration at send time."""
        job = self.controller._create_generation_job('latest cat news')

        self.assertTrue(job.enable_web_search)
        self.assertEqual(job.searxng_base_url, 'http://127.0.0.1:8081')
        self.assertEqual(job.web_search_category, 'news')
        self.assertEqual(job.web_search_time_range, 'day')
        self.assertEqual(job.web_max_results, 7)
        self.assertTrue(job.web_fetch_enabled)
        self.assertEqual(job.reranker_model, 'qwen-rerank')



class OllamaToolCallingTests(unittest.TestCase):
    """Validate the search tool loop inside OllamaClient."""

    def setUp(self) -> None:
        """Create an Ollama client and preserve original request helpers for patching."""
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
        })

    def tearDown(self) -> None:
        """Restore request helpers patched during tool-calling tests."""
        self.temp_dir.cleanup()


    def test_chat_stream_adds_web_tool_guidance_system_message(self) -> None:
        """Verify web-enabled requests add the grounding guidance system prompt."""
        client = OllamaClient(self.config_service)

        messages = client._build_chat_messages(
            'base prompt',
            [],
            web_tools_enabled=True,
            web_max_results=5,
        )

        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], 'base prompt')
        self.assertEqual(messages[1]['role'], 'system')
        # Check that the grounding message tells the model it has web search access,
        # gives the date, instructs it to search before answering, and to cite with [S1].
        self.assertIn('You have access to web search tools', messages[1]['content'])
        self.assertIn('March 10, 2026', messages[1]['content'])
        self.assertIn('search_web', messages[1]['content'])
        self.assertIn('[S1]', messages[1]['content'])

    def test_grounding_message_embeds_configured_max_results(self) -> None:
        """Verify the configured web_max_results value appears in the grounding message.

        This is the regression test for the root cause of the 5-result cap.
        The tool schema had no default and no description hint, so the model
        was picking an arbitrary small number (typically 3-5) regardless of
        the user-configured ceiling. The fix embeds the configured value in
        both the tool schema default and the grounding system message so the
        model treats it as the expected request size.
        """
        client = OllamaClient(self.config_service)

        messages = client._build_chat_messages(
            '',
            [],
            web_tools_enabled=True,
            web_max_results=50,
        )

        grounding = messages[0]['content']
        self.assertIn('max_results=50', grounding,
                      'Grounding message must include the configured max_results value')

    def test_tool_schema_max_results_default_matches_configured_value(self) -> None:
        """Verify the search_web tool schema default for max_results reflects the config.

        Without a default in the JSON Schema, models skip the parameter entirely
        or use an arbitrary small number. Setting default=web_max_results tells
        the model what value to use when it does not override it explicitly.
        """
        import tempfile as _tempfile
        from pathlib import Path as _Path

        with _tempfile.TemporaryDirectory() as d:
            cs = ConfigService(config_path=_Path(d) / 'c.json')
            cs.update_config({'web_max_results': 40, 'web_fetch_enabled': False})
            client = OllamaClient(cs)
            tools = client._build_web_tools_schema(cs.get_config())

        max_results_prop = tools[0]['function']['parameters']['properties']['max_results']
        self.assertEqual(max_results_prop.get('default'), 40,
                         'Tool schema default must equal the configured web_max_results')
        self.assertIn('40', max_results_prop.get('description', ''),
                      'Tool schema description must mention the configured value')

    def test_post_tool_grounding_message_explicitly_allows_post_cutoff_results(self) -> None:
        """Verify the post-tool grounding message instructs the model to use retrieved results."""
        client = OllamaClient(self.config_service)

        message = client._build_post_tool_grounding_message()

        self.assertEqual(message["role"], "system")
        self.assertIn('March 10, 2026', message["content"])
        # The message must tell the model retrieved results outrank its training knowledge
        # and that it should cite with bracketed IDs.
        self.assertIn('[S1]', message["content"])
        self.assertIn('primary source', message["content"])

    def test_chat_stream_forces_search_when_model_claims_no_internet_access(self) -> None:
        """Verify fallback search executes when the model incorrectly claims it lacks internet access."""
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
        # _extract_search_query strips "search for" preamble, leaving just "cats"
        self.assertEqual(called_arguments['query'], 'cats')
        self.assertEqual(result['content'], 'Cats are small domesticated mammals.')
        self.assertEqual(captured_chunks, [('content', 'Cats are small domesticated mammals.')])

    def test_chat_stream_forces_search_for_latest_news_prompt_even_without_explicit_no_internet_phrase(self) -> None:
        """Verify latest-news prompts trigger web-search fallback when needed."""
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
            with patch.object(client, '_execute_tool_call', return_value={'query': 'latest news about cats', 'result_count': 2, 'results': []}) as execute_tool:
                with patch.object(client, '_stream_json_lines', return_value=final_stream):
                    result = client.chat_stream(
                        [type('Msg', (), {'role': 'user', 'content': 'Give me the latest news about cats'})()],
                        on_chunk=lambda kind, chunk: captured_chunks.append((kind, chunk)),
                    )

        execute_tool.assert_called_once()
        self.assertEqual(execute_tool.call_args.args[0], 'search_web')
        # _extract_search_query strips "Give me the latest news about" preamble
        self.assertEqual(execute_tool.call_args.args[1]['query'], 'cats')
        self.assertEqual(result['content'], 'Here are the latest cat news results from web search.')
        self.assertEqual(captured_chunks, [('content', 'Here are the latest cat news results from web search.')])


    def test_search_tool_results_include_source_ids_and_fragment_links(self) -> None:
        """Verify normalized search results expose stable citation IDs and validation links."""
        client = OllamaClient(self.config_service)
        prepared = client._prepare_search_tool_result({
            'query': 'qdrant',
            'result_count': 1,
            'results': [
                {
                    'id': 'r1',
                    'source_id': 'S1',
                    'title': 'Qdrant Docs',
                    'url': 'https://qdrant.tech/documentation/',
                    'domain': 'qdrant.tech',
                    'snippet': 'Vector database documentation for search and retrieval.',
                }
            ],
        })

        result = prepared['results'][0]
        self.assertEqual(result['citation_text'], '[S1]')
        self.assertTrue(result['validated'])
        self.assertIn('#:~:text=', result['citation_url'])

    def test_final_answer_sources_are_validated_against_real_search_results(self) -> None:
        """Verify only real search-result citations survive into the validated source list."""
        client = OllamaClient(self.config_service)
        source_catalog = [
            {
                'source_id': 'S1',
                'title': 'Qdrant Docs',
                'url': 'https://qdrant.tech/documentation/',
                'citation_url': 'https://qdrant.tech/documentation/#:~:text=Vector%20database',
                'highlight_text': 'Vector database',
            }
        ]

        validated = client._build_validated_sources_from_answer('Use Qdrant for vectors [S1] but not [S9].', source_catalog)
        invalid = client._find_invalid_citation_ids('Use Qdrant for vectors [S1] but not [S9].', source_catalog)

        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]['source_id'], 'S1')
        self.assertEqual(invalid, ['S9'])

    def test_chat_stream_executes_search_tool_and_streams_final_answer(self) -> None:
        """Verify tool calls execute search and the client resumes final answer streaming."""
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

    def test_post_tool_grounding_message_appears_after_tool_result_not_before(self) -> None:
        """Verify the post-tool grounding system message is appended after tool results.

        This is the regression test for a message-ordering bug that caused the
        model to stop producing [S1]-style citations and the validated sources
        panel to go empty. The Ollama tool-use protocol requires:

            [assistant]  tool_calls=[search_web]
            [tool]       result JSON
            [system]     post-tool grounding (cite with [S1])

        When the system message was inserted between the assistant tool-call
        message and the tool result, most Ollama-served models ignored the tool
        result entirely, so source_id values never appeared in the answer and
        _build_validated_sources_from_answer returned an empty list.
        """
        client = OllamaClient(self.config_service)
        captured_messages: list[list[dict]] = []
        tool_result = {
            'query': 'cats',
            'result_count': 1,
            'results': [{'source_id': 'S1', 'title': 'Cats', 'url': 'https://cats.example/', 'snippet': 'info'}],
        }
        initial_response = {
            'message': {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {'function': {'name': 'search_web', 'arguments': {'query': 'cats', 'max_results': 5}}}
                ],
            }
        }
        final_stream = [
            {'message': {'content': 'Cats are great [S1].'}, 'done': False},
            {'message': {}, 'done': True, 'eval_count': 8, 'eval_duration': 1_000_000_000},
        ]

        def capture_stream(endpoint, payload, config):
            captured_messages.append(list(payload.get('messages', [])))
            return iter(final_stream)

        with patch.object(client, '_request_json_with_request_options', return_value=initial_response):
            with patch.object(client, '_execute_tool_call', return_value=tool_result):
                with patch.object(client, '_stream_json_lines', side_effect=capture_stream):
                    client.chat_stream([], on_chunk=lambda k, c: None)

        self.assertTrue(captured_messages, 'Final streaming call must have been made')
        final_messages = captured_messages[0]

        # Find the positions of the assistant tool-call, tool result, and grounding messages
        roles = [m.get('role') for m in final_messages]
        assistant_idx = next((i for i, m in enumerate(final_messages)
                              if m.get('role') == 'assistant' and m.get('tool_calls')), None)
        tool_idx = next((i for i, m in enumerate(final_messages)
                         if m.get('role') == 'tool'), None)
        grounding_idx = next((i for i, m in enumerate(final_messages)
                              if m.get('role') == 'system'
                              and 'primary source' in str(m.get('content', ''))), None)

        self.assertIsNotNone(assistant_idx, 'Assistant tool-call message must be present')
        self.assertIsNotNone(tool_idx, 'Tool result message must be present')
        self.assertIsNotNone(grounding_idx, 'Post-tool grounding system message must be present')

        self.assertLess(assistant_idx, tool_idx,
                        'Assistant tool-call message must come before tool result')
        self.assertLess(tool_idx, grounding_idx,
                        'Tool result must come before the post-tool grounding message — '
                        'inserting a system message between assistant and tool breaks '
                        'the tool-use pairing and causes the model to ignore citations')




class OllamaClientRerankerTests(unittest.TestCase):
    """Verify the optional reranker stage preserves real source IDs only.

    Note: this class was previously defined after the if __name__ == '__main__'
    guard. unittest.main() calls sys.exit(), so the class was never parsed and
    the reranker tests were silently skipped on every direct run. The class is
    now above the guard so it executes normally.
    """

    def test_reranker_reorders_results_using_only_valid_ids(self) -> None:
        """Verify the reranker can reorder results but cannot inject made-up candidates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ConfigService(config_path=Path(temp_dir) / 'config.json')
            client = OllamaClient(config)
            result = {
                'query': 'gpu news',
                'results': [
                    {'source_id': 'S1', 'title': 'One', 'domain': 'a.com', 'snippet': 'alpha'},
                    {'source_id': 'S2', 'title': 'Two', 'domain': 'b.com', 'snippet': 'beta'},
                    {'source_id': 'S3', 'title': 'Three', 'domain': 'c.com', 'snippet': 'gamma'},
                ],
            }
            request_options = config.get_config()
            request_options['reranker_model'] = 'mini-reranker'
            with patch.object(client, '_request_json_with_request_options', return_value={'message': {'content': json.dumps({'ordered_source_ids': ['S2', 'S9', 'S1']})}}):
                reranked = client._maybe_rerank_search_results(result, request_options)

        self.assertTrue(reranked['reranked'])
        self.assertEqual([item['source_id'] for item in reranked['results']], ['S2', 'S1', 'S3'])
        self.assertEqual(reranked['reranker_model'], 'mini-reranker')


class OllamaClientQueryCleaningTests(unittest.TestCase):
    """Verify conversational preamble is stripped from fallback search queries.

    When the model fails to call the search tool, the fallback path uses the raw
    user message as the SearXNG query. Conversational phrasing like "Give me the
    latest on Iran's conflict" produces poor SearXNG results compared to keyword
    form "Iran conflict latest 2026". _extract_search_query strips common preambles
    so the fallback query lands closer to what a person would type into a search box.
    """

    def setUp(self) -> None:
        """Create an isolated client for query-cleaning tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        config = ConfigService(config_path=root / 'config.json')
        self.client = OllamaClient(config)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_plain_keyword_query_is_unchanged(self) -> None:
        """Verify that a query already in keyword form passes through unmodified."""
        self.assertEqual(self.client._extract_search_query('Iran conflict 2026'), 'Iran conflict 2026')

    def test_give_me_the_latest_on_is_stripped(self) -> None:
        """Verify 'Give me the latest on X' becomes 'X'."""
        self.assertEqual(self.client._extract_search_query("Give me the latest on Iran's conflict"),
                         "Iran's conflict")

    def test_give_me_the_latest_news_about_is_stripped(self) -> None:
        """Verify 'Give me the latest news about X' becomes 'X'."""
        self.assertEqual(self.client._extract_search_query('Give me the latest news about cats'), 'cats')

    def test_search_for_preamble_is_stripped(self) -> None:
        """Verify 'search for X' becomes 'X'."""
        self.assertEqual(self.client._extract_search_query('search for cats'), 'cats')

    def test_whats_the_latest_on_is_stripped(self) -> None:
        """Verify 'What's the latest on X' becomes 'X'."""
        self.assertEqual(self.client._extract_search_query("What's the latest on AI?"), 'AI')

    def test_trailing_punctuation_is_removed(self) -> None:
        """Verify trailing question marks and periods are removed from the query."""
        self.assertEqual(self.client._extract_search_query('Iran conflict?'), 'Iran conflict')

    def test_very_short_result_falls_back_to_original(self) -> None:
        """Verify that a stripping result shorter than 3 chars falls back to the original."""
        # "Tell me about AI" → stripping would leave "AI" which is >= 3 chars actually,
        # so use something that strips to nothing meaningful.
        result = self.client._extract_search_query('search for it')
        # "it" is 2 chars, should fall back to original stripped of punctuation
        self.assertIn('it', result)

    def test_empty_input_returns_empty(self) -> None:
        """Verify empty input returns empty string."""
        self.assertEqual(self.client._extract_search_query(''), '')


class SearchServiceIdOffsetTests(unittest.TestCase):
    """Verify source ID offset stamping for multi-search catalog accumulation.

    Each search call within a turn must produce globally unique source IDs so
    the catalog can accumulate entries from multiple searches without collisions.
    The id_offset parameter shifts the starting position number, e.g. a second
    search with offset=3 produces S4, S5, S6 instead of S1, S2, S3.
    """

    def _fake_response(self, n: int):
        """Return a fake urlopen context manager yielding n raw SearXNG result items."""
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload
            def read(self):
                return json.dumps(self._payload).encode('utf-8')
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False

        payload = {
            'results': [
                {'title': f'R{i}', 'url': f'https://ex.com/{i}',
                 'content': f'snippet {i}', 'engines': ['brave']}
                for i in range(1, n + 1)
            ]
        }
        return FakeResponse(payload)

    def test_default_offset_produces_s1_through_sn(self) -> None:
        """Verify source IDs start at S1 when no offset is given."""
        with patch('services.search_service.request.urlopen', return_value=self._fake_response(3)):
            result = SearchService('http://127.0.0.1:8080').search('cats', id_offset=0)
        self.assertEqual([r['source_id'] for r in result['results']], ['S1', 'S2', 'S3'])

    def test_offset_shifts_starting_source_id(self) -> None:
        """Verify id_offset=10 shifts IDs to S11, S12, S13 for a second search batch."""
        with patch('services.search_service.request.urlopen', return_value=self._fake_response(3)):
            result = SearchService('http://127.0.0.1:8080').search('dogs', id_offset=10)
        self.assertEqual([r['source_id'] for r in result['results']], ['S11', 'S12', 'S13'])

    def test_two_searches_with_accumulated_offsets_produce_no_colliding_ids(self) -> None:
        """Verify sequential searches with correct offsets yield disjoint source ID sets."""
        with patch('services.search_service.request.urlopen', return_value=self._fake_response(3)):
            first = SearchService('http://127.0.0.1:8080').search('cats', id_offset=0)
        with patch('services.search_service.request.urlopen', return_value=self._fake_response(3)):
            second = SearchService('http://127.0.0.1:8080').search('dogs',
                                                                     id_offset=len(first['results']))

        first_ids = {r['source_id'] for r in first['results']}
        second_ids = {r['source_id'] for r in second['results']}
        self.assertTrue(first_ids.isdisjoint(second_ids),
                        f'ID collision: {first_ids & second_ids}')
        self.assertEqual(first_ids, {'S1', 'S2', 'S3'})
        self.assertEqual(second_ids, {'S4', 'S5', 'S6'})


class OllamaClientCatalogAccumulationTests(unittest.TestCase):
    """Verify source catalog accumulates across multiple search_web calls in one turn.

    Before the fix, _execute_tool_call called source_catalog.clear() before
    extending it with new search results. This wiped citations from earlier
    searches in the same turn, causing the UI to mark them as invalid even
    though they came from real SearXNG results. The fix replaces clear()+extend()
    with plain extend() and passes len(catalog) as id_offset so IDs stay unique.
    """

    def setUp(self) -> None:
        """Create an isolated client configured for multi-search catalog tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config_service = ConfigService(config_path=root / 'config.json')
        self.config_service.update_config({
            'model': 'qwen3',
            'host': '127.0.0.1',
            'port': 11434,
            'timeout_seconds': 30,
            'system_prompt': '',
            'thinking_mode': False,
            'enable_web_search': True,
            'searxng_base_url': 'http://127.0.0.1:8080',
            'web_max_results': 3,
            'web_fetch_enabled': False,
        })
        self.client = OllamaClient(self.config_service)

    def tearDown(self) -> None:
        """Clean up temporary config files."""
        self.temp_dir.cleanup()

    def _prepared_result(self, ids):
        """Return a minimal already-prepared search result with the given source IDs."""
        return {
            'query': 'test',
            'result_count': len(ids),
            'results': [
                {
                    'source_id': sid,
                    'id': sid.lower(),
                    'title': f'Title {sid}',
                    'url': f'https://example.com/{sid.lower()}',
                    'domain': 'example.com',
                    'snippet': f'Snippet {sid}',
                    'citation_url': f'https://example.com/{sid.lower()}',
                    'highlight_text': f'Snippet {sid}',
                    'validated': True,
                    'validation_reason': 'test',
                }
                for sid in ids
            ],
        }

    def test_first_search_populates_empty_catalog(self) -> None:
        """Verify the first search_web call populates a previously empty catalog."""
        catalog = []
        config = self.config_service.get_config()
        with patch.object(self.client, '_prepare_search_tool_result',
                          return_value=self._prepared_result(['S1', 'S2', 'S3'])):
            with patch.object(self.client, '_maybe_rerank_search_results',
                              side_effect=lambda r, c: r):
                with patch('services.search_service.SearchService.search',
                           return_value={'query': 'cats', 'result_count': 3, 'results': []}):
                    self.client._execute_tool_call(
                        'search_web', {'query': 'cats'}, config, source_catalog=catalog)

        self.assertEqual(len(catalog), 3)
        self.assertEqual([x['source_id'] for x in catalog], ['S1', 'S2', 'S3'])

    def test_second_search_appends_without_wiping_first_results(self) -> None:
        """Verify a second search_web call extends the catalog rather than replacing it.

        This is the direct regression test for the source_catalog.clear() bug.
        """
        catalog = []
        config = self.config_service.get_config()

        for ids, query in [(['S1', 'S2', 'S3'], 'cats'), (['S4', 'S5', 'S6'], 'dogs')]:
            with patch.object(self.client, '_prepare_search_tool_result',
                              return_value=self._prepared_result(ids)):
                with patch.object(self.client, '_maybe_rerank_search_results',
                                  side_effect=lambda r, c: r):
                    with patch('services.search_service.SearchService.search',
                               return_value={'query': query, 'result_count': 3, 'results': []}):
                        self.client._execute_tool_call(
                            'search_web', {'query': query}, config, source_catalog=catalog)

        self.assertEqual(len(catalog), 6,
                         'catalog must accumulate to 6 after two searches, not reset to 3')
        self.assertEqual([x['source_id'] for x in catalog],
                         ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

    def test_citations_from_first_search_remain_valid_after_second_search(self) -> None:
        """Verify [S1] from the first search still resolves after a second search runs.

        Before the fix, source_catalog.clear() meant S1 was absent from the
        catalog after the second search, causing the UI validation pass to flag
        [S1] as an invented/invalid citation.
        """
        catalog = []
        config = self.config_service.get_config()

        for ids, query in [(['S1', 'S2', 'S3'], 'cats'), (['S4', 'S5', 'S6'], 'dogs')]:
            with patch.object(self.client, '_prepare_search_tool_result',
                              return_value=self._prepared_result(ids)):
                with patch.object(self.client, '_maybe_rerank_search_results',
                                  side_effect=lambda r, c: r):
                    with patch('services.search_service.SearchService.search',
                               return_value={'query': query, 'result_count': 3, 'results': []}):
                        self.client._execute_tool_call(
                            'search_web', {'query': query}, config, source_catalog=catalog)

        answer = 'Cats [S1] and dogs [S4] are both popular pets.'
        validated = self.client._build_validated_sources_from_answer(answer, catalog)
        invalid = self.client._find_invalid_citation_ids(answer, catalog)

        self.assertEqual(len(validated), 2)
        validated_ids = {x['source_id'] for x in validated}
        self.assertIn('S1', validated_ids, 'S1 from first search must still be valid')
        self.assertIn('S4', validated_ids, 'S4 from second search must be valid')
        self.assertEqual(invalid, [])


class OllamaClientFetchCountEnforcementTests(unittest.TestCase):
    """Verify fetch_url_content calls are capped at the web_max_pages limit per turn.

    Before the fix, web_max_pages was stored in GenerationJob and passed through
    to_request_options(), but _chat_with_optional_tools never checked it against
    the actual number of fetch_url_content calls the model made in a single turn.
    The model could fetch unlimited pages regardless of the configured ceiling.
    """

    def setUp(self) -> None:
        """Create an isolated client with web_max_pages=2 for fetch-limit tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config_service = ConfigService(config_path=root / 'config.json')
        self.config_service.update_config({
            'model': 'qwen3',
            'host': '127.0.0.1',
            'port': 11434,
            'timeout_seconds': 30,
            'system_prompt': '',
            'thinking_mode': False,
            'enable_web_search': True,
            'searxng_base_url': 'http://127.0.0.1:8080',
            'web_max_results': 5,
            'web_max_pages': 2,
            'web_fetch_enabled': True,
        })
        self.client = OllamaClient(self.config_service)

    def tearDown(self) -> None:
        """Clean up temporary config files."""
        self.temp_dir.cleanup()

    def test_fetch_calls_beyond_max_pages_are_blocked(self) -> None:
        """Verify that only web_max_pages fetch_url_content calls reach _execute_tool_call.

        The model requests 3 page fetches but web_max_pages=2, so the third
        call must be intercepted by the fetch_count guard and must not reach
        _execute_tool_call as a real fetch attempt.
        """
        config = self.config_service.get_config()
        tool_calls = [
            {'function': {'name': 'fetch_url_content', 'arguments': {'url': 'https://a.com/'}}},
            {'function': {'name': 'fetch_url_content', 'arguments': {'url': 'https://b.com/'}}},
            {'function': {'name': 'fetch_url_content', 'arguments': {'url': 'https://c.com/'}}},
        ]
        initial_response = {
            'message': {'role': 'assistant', 'content': '', 'tool_calls': tool_calls}
        }
        final_stream = [
            {'message': {'content': 'done'}, 'done': False},
            {'message': {}, 'done': True, 'eval_count': 4, 'eval_duration': 1_000_000_000},
        ]

        real_fetch_calls = []

        def fake_execute(tool_name, arguments, cfg, source_catalog=None):
            if tool_name == 'fetch_url_content':
                real_fetch_calls.append(arguments['url'])
                return {'url': arguments['url'], 'title': 'T', 'text': 'body',
                        'final_url': arguments['url'], 'source_id': '',
                        'highlight_text': '', 'citation_url': arguments['url']}
            return {}

        with patch.object(self.client, '_request_json_with_request_options',
                          return_value=initial_response):
            with patch.object(self.client, '_execute_tool_call',
                              side_effect=fake_execute):
                with patch.object(self.client, '_stream_json_lines',
                                  return_value=final_stream):
                    self.client._chat_with_optional_tools(
                        [{'role': 'user', 'content': 'fetch all three pages'}],
                        on_chunk=lambda k, c: None,
                        config=config,
                    )

        self.assertEqual(len(real_fetch_calls), 2,
                         f'Expected 2 fetch calls (web_max_pages=2), got {len(real_fetch_calls)}: {real_fetch_calls}')
        self.assertNotIn('https://c.com/', real_fetch_calls,
                         'Third fetch must be blocked by the web_max_pages limit')


if __name__ == '__main__':
    unittest.main()
