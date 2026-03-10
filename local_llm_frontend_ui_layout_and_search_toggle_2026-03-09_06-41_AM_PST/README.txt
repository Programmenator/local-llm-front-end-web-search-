Local LLM Front End Chat Interface
==================================

Purpose
-------
This project is a modular Tkinter desktop interface for chatting with a local
Ollama server. It supports saved chat sessions, streaming assistant output,
optional reasoning/thinking display, optional SearXNG-backed web search tool
calling, an always-on deterministic non-LLM reranker stage immediately after
SearXNG retrieval, an optional dedicated second-stage reranker after that first
screening pass (either an Ollama model or a sentence-transformers BAAI-style
cross-encoder), URL content fetch for selected search results, GPU/VRAM live
readout, and explicit model unload requests intended to free VRAM.

This README was rewritten after a file-by-file code-to-documentation audit of
the current repository contents. Statements here are based on the code now
present in this package and on the bundled automated tests that passed in this
environment. The latest observed repository test run for this revision was
48 passing tests plus 1 opt-in live pipeline test that is skipped by default
unless explicitly enabled through environment variables.

What the Program Actually Does
------------------------------
Core runtime behavior confirmed from the codebase:

1. Launches a Tkinter desktop chat window from `main.py`.
2. Connects to a local Ollama server over HTTP using Python standard-library
   `urllib`, not the `requests` library.
3. Streams assistant content and optional thinking output into separate UI
   regions.
4. Maintains true multi-turn conversation history per saved session.
5. Saves sessions to JSON files under `sessions/` using atomic replace writes.
6. Loads, lists, renames, deletes, and creates sessions through the controller.
7. Lets the user choose an installed Ollama model from the main window and the
   settings window.
8. Lets the user request an Ollama model unload through the settings window
   using the "Flush GPU VRAM" control.
9. Polls AMD GPU/VRAM usage in the background using `amd-smi` or `rocm-smi`
   without blocking the Tkinter UI thread.
10. Displays a live token-per-second estimate during streaming and prefers final
    Ollama eval metrics when they are available in the last stream item.
11. Optionally exposes `search_web` and `fetch_url_content` tools to the model
    when web search is enabled.
12. Always passes normalized SearXNG candidates through a deterministic
    first-pass reranker that scores query/title/snippet/domain relevance and
    returns a ranked top-N candidate list.
13. Can optionally pass those first-pass-ranked candidates through a dedicated
    second-stage reranker before the main model sees them.
14. The optional second-stage reranker can run either through an Ollama model
    or through a sentence-transformers cross-encoder such as
    `BAAI/bge-reranker-base`, without routing through Ollama.
14. Uses SearXNG search results as tool output and can optionally fetch page
    text from selected URLs.
15. Applies explicit web-fetch safety controls before retrieving page content:
    blocks localhost and non-public IP targets, resolves hostnames before
    connecting, re-validates redirect targets, enforces strict content-type
    checks, and rejects oversized response bodies.
16. Supports optional public-domain allowlist and blocklist rules for fetched
    pages through the settings window.
17. Injects explicit post-tool grounding so the model is told to treat retrieved
    search results as current runtime evidence even when those results are newer
    than the model's pretraining cutoff.
18. Preserves protocol-safe tool sequencing for Ollama follow-up calls by
    ensuring tool result messages immediately follow the assistant message that
    emitted `tool_calls`, and includes `tool_call_id` when available.
19. Includes fallback logic that can force a web-search pass when the prompt
    clearly requires current external information or when the model explicitly
    claims it lacks internet access. The heuristic is intentionally narrow and
    no longer fires on generic standalone words such as `current` or `recent`.
20. Enforces the Settings value "Max Pages To Read" at runtime across both the
    normal model-emitted tool path and the forced-search fallback path, so one
    answer cannot fetch more pages than the user allowed for that request.

Architecture
------------
The program uses a layered structure.

Entry Layer
    `main.py`
        Minimal startup entry point. Creates the application object and runs the
        Tkinter event loop.

Composition Root
    `app.py`
        Builds ConfigService, SessionService, OllamaClient, ChatController, and
        MainWindow, then wires the controller to the view.

Controller Layer
    `controllers/chat_controller.py`
        Main orchestration layer. Owns the active session, owns the active
        GenerationJob snapshot, coordinates persistence and UI updates, blocks
        state-changing actions during streaming, and routes requests to the
        Ollama and search-related services.

Service Layer
    `services/config_service.py`
        Loads, normalizes, and atomically saves configuration in `config.json`.

    `services/session_service.py`
        Creates, saves, lists, loads, and deletes saved conversation sessions.
        Single-session loads now use direct filename lookup instead of a full
        scan of every saved session file.

    `services/ollama_client.py`
        Handles Ollama HTTP communication, model discovery, unload requests,
        streamed chat, tool-call execution, deterministic first-pass reranking,
        optional Ollama reranking, protocol-safe tool result injection,
        post-cutoff grounding injection, request-scoped web page fetch limits,
        and limited forced-search fallback behavior.

    `services/search_service.py`
        Talks to SearXNG, normalizes result shape, repairs some malformed base
        URL input, and provides a direct connection test.

    `services/reranker_service.py`
        Houses both reranking stages. The first is a deterministic non-LLM
        reranker that scores candidates using query/title/snippet/domain
        overlap and returns a ranked top-N list. The optional second stage can
        either use an Ollama-served reranker model or a local
        sentence-transformers cross-encoder while preserving all original
        candidates when backend output is partial or unavailable.

    `services/web_fetch_service.py`
        Fetches one URL and extracts readable title/body text for model use.

    `services/gpu_monitor_service.py`
        Collects live AMD GPU and VRAM metrics from `amd-smi` or `rocm-smi` and
        normalizes the output for the UI.

Model Layer
    `models/chat_message.py`
        Structured chat message model with serialization helpers.

    `models/conversation_session.py`
        Structured saved session model with serialization helpers.

    `models/generation_job.py`
        Frozen per-request snapshot used so an in-flight reply stays bound to
        the session/config state that existed when Send was pressed.

UI Layer
    `ui/main_window.py`
        Main chat window with transcript, input area, settings access, model
        dropdown, session manager, thinking panel, GPU/VRAM display, and token
        meter. The thinking panel state is reset correctly when switching into
        a session that has no thinking output.

    `ui/settings_window.py`
        Settings popup for Ollama connection, model behavior, SearXNG behavior,
        dedicated search-reranker configuration, model list refresh,
        connectivity testing, and model unload.

Utility Layer
    `utils/threading_helpers.py`
        Small helper to run background work on daemon threads.

Package Marker Files
    `controllers/__init__.py`
    `models/__init__.py`
    `services/__init__.py`
    `ui/__init__.py`
    `utils/__init__.py`
        These files are package markers plus package-role documentation. They do
        not contain business logic.

Main UI Features Confirmed from Code
------------------------------------
The main window currently includes:

- Settings button
- Installed-model dropdown in the sidebar
- Model detail text showing host, port, and current model
- GPU live readout panel with GPU%, VRAM%, and Tok/s labels
- Session list with New, Rename, Delete, and Refresh actions
- Chat transcript area
- Collapsible thinking panel
- Multi-line input box
- Send button
- Status bar
- Ctrl+Enter send shortcut
- Ctrl+mouse-wheel text zoom in the transcript, thinking, input, and session-list text regions
- Draggable pane dividers so the chat, thinking, and input text boxes can be resized to suit the user's preferred layout
- Main-window Internet Search ON/OFF toggle synchronized with the saved settings state

Main-Window Usability Updates
---------------------------
The main chat screen now exposes three direct usability controls requested after
the earlier search/reranker work:

1. Text zoom
   The transcript, thinking panel, input field, and session list now respond to
   Ctrl + mouse wheel. Scroll up while holding Ctrl to enlarge the text or
   scroll down while holding Ctrl to reduce it. This is implemented at the UI
   widget level so the control works without opening Settings.

2. Resizable text regions
   The central chat layout now uses a vertical paned window. The user can drag
   the divider lines between the transcript, thinking panel, and input area to
   allocate more or less space to each section during a session.

3. Main-window internet-search toggle
   The main header now includes a visible Internet Search ON/OFF toggle. It is
   tied directly to the same saved `enable_web_search` flag used by the
   Settings window, so the user can immediately verify whether web-search tools
   are available to the model and can change that state without opening the
   settings dialog.

Live Search/Fetch Pipeline Test
------------------------------
The repository now includes one explicit opt-in live integration test in
`tests_web_search_integration.py` that exercises the real pipeline:

    SearXNG search -> deterministic stage-1 rerank -> real page fetch

This test exists because the normal search/fetch tests are mock-driven. Those
mocked tests are still useful for deterministic regression coverage, but they
do not show what actually happens against a running SearXNG instance and live
web pages.

Default behavior:
- The live pipeline test is skipped during ordinary `python -m unittest -v`
  runs so packaging and offline validation remain stable.

Enable the live test:
- Set `LLM_FRONTEND_RUN_LIVE_WEB_PIPELINE_TEST=1`

Useful optional environment variables:
- `LLM_FRONTEND_LIVE_WEB_BASE_URL`
- `LLM_FRONTEND_LIVE_WEB_QUERY`
- `LLM_FRONTEND_LIVE_WEB_MAX_RESULTS`
- `LLM_FRONTEND_LIVE_WEB_CATEGORY`
- `LLM_FRONTEND_LIVE_WEB_LANGUAGE`
- `LLM_FRONTEND_LIVE_WEB_TIME_RANGE`
- `LLM_FRONTEND_LIVE_WEB_SAFE_SEARCH`
- `LLM_FRONTEND_LIVE_WEB_FETCH_URL`
- `LLM_FRONTEND_LIVE_WEB_LOG_PATH`
- `LLM_FRONTEND_LIVE_WEB_FETCH_ALLOWLIST`
- `LLM_FRONTEND_LIVE_WEB_FETCH_BLOCKLIST`

What gets logged:
- normalized search result summaries
- deterministic reranker ordering and heuristic scores
- each fetch attempt in reranked order
- fetch failures for rejected or unreachable pages
- the first successful fetch metadata and a short text preview

Example command:
    LLM_FRONTEND_RUN_LIVE_WEB_PIPELINE_TEST=1     LLM_FRONTEND_LIVE_WEB_BASE_URL=http://127.0.0.1:8080     LLM_FRONTEND_LIVE_WEB_QUERY="latest local llm tooling news"     LLM_FRONTEND_LIVE_WEB_LOG_PATH=./live_web_pipeline.log     python -m unittest -v tests_web_search_integration.LiveSearchFetchPipelineTests

Settings Window Features Confirmed from Code
--------------------------------------------
The settings popup currently edits or controls:

- Ollama host
- Ollama port
- Request timeout
- Installed model selection
- Thinking mode enable/disable
- Thinking level (`low`, `medium`, `high`)
- System prompt text
- Enable/disable model web search tools
- SearXNG base URL
- Search category (`general`, `news`, `science`, `it`)
- Search language
- Search time range (`none`, `day`, `week`, `month`, `year`)
- Safe search value
- Max search results
- Enable/disable dedicated search reranker before the main model
- Reranker backend (`disabled`, `ollama`, `sentence_transformers`)
- Ollama reranker model name
- Sentence-transformers reranker model name
- Sentence-transformers device preference (`auto`, `cpu`, `cuda`, `mps`)
- Max pages to read
- Enable/disable fetched page content after search
- Max fetched response size in bytes
- Max redirect hops for fetched pages
- Optional fetched-page domain allowlist
- Optional fetched-page domain blocklist
- User-visible fetch policy summary describing the enforced safety rules
- Test Ollama connection
- Test SearXNG connection
- Refresh model list
- Flush GPU VRAM via model unload request
- Save settings

Session Persistence
-------------------
Session files are stored in the `sessions/` directory. The filename format is:

    <sanitized_title>__<session_id>.json

The session service uses atomic temp-file replacement when saving, removes
stale duplicate files for the same session ID after a rename or resave, and
loads a specific session by direct filename lookup instead of deserializing the
entire session directory.

Configuration Persistence
-------------------------
The configuration file is `config.json` in the project root. The config service
creates it automatically if it does not exist and normalizes invalid or damaged
numeric/boolean values back to safe defaults instead of crashing startup.

Current saved config fields confirmed from `services/config_service.py`:

- host
- port
- model
- timeout_seconds
- system_prompt
- thinking_mode
- thinking_level
- enable_web_search
- searxng_base_url
- web_search_category
- web_search_language
- web_search_time_range
- web_safe_search
- web_max_results
- enable_search_reranker
- search_reranker_backend
- search_reranker_model
- sentence_transformers_reranker_model
- sentence_transformers_reranker_device
- web_max_pages (hard runtime cap on fetched pages per answer)
- web_fetch_enabled
- web_fetch_max_response_bytes
- web_fetch_max_redirects
- web_fetch_allowlist
- web_fetch_blocklist

Web Search and Grounding Behavior
---------------------------------
When web search is enabled, the Ollama client can expose two tools to the
model:

1. `search_web`
2. `fetch_url_content`

The tool path is not a naive "take the first N results" flow. The model can
call the search tool, inspect normalized results, optionally call the fetch tool
for selected pages, and then receive a second chat pass with tool output and an
extra grounding message. When the dedicated search reranker is enabled, the
search tool first returns candidates from SearXNG, then routes those normalized
results through the configured second-stage backend, and only then exposes the
reordered candidates to the main model. This makes the pipeline closer to:

    SearXNG -> deterministic reranker -> optional stage-2 reranker -> main model

Fetched-page safety policy now enforced by `services/web_fetch_service.py`:
- only `http` and `https` fetches are allowed
- `localhost`, `127.0.0.1`, `::1`, private IP ranges, link-local ranges,
  multicast ranges, and reserved ranges are blocked
- hostnames are resolved before connection and rejected when they resolve to
  non-public addresses
- redirects are capped and every redirect target is re-validated before follow
- only `text/html` and `application/xhtml+xml` content types are accepted
- oversized responses are rejected before the full body is read
- optional public-domain allowlist and blocklist rules can narrow fetch scope

Important operational note:
The localhost/private-address block applies to the page-fetch stage, not the
SearXNG search stage. The automated test suite still passes with a SearXNG base
URL on `127.0.0.1:8080`, so this safety hardening does not interfere with the
existing local SearXNG search integration.

The grounding message explicitly states that retrieved search results are live
runtime evidence gathered by the application at answer time. This is the code
path intended to fix the earlier failure mode where the model rejected relevant
post-cutoff web information only because the dates were newer than training.

The client also contains fallback logic that can trigger a search-based retry
when:
- the prompt appears to request current or latest information, or
- the first model reply indicates it lacks internet access.

Dependencies
------------
Third-party dependencies actually present in `requirements.txt`:

- `beautifulsoup4>=4.12`
- `sentence-transformers>=3.0`

Operational note:
The sentence-transformers dependency is used only when the user selects the
`sentence_transformers` reranker backend in Settings. That backend is intended
for models such as `BAAI/bge-reranker-base` and does not route through Ollama.

Standard-library modules do substantial work in this project, including:
- `tkinter`
- `urllib`
- `json`
- `threading`
- `pathlib`
- `subprocess`

Run Instructions
----------------
1. Ensure Python 3 is available.
2. Install dependencies:

   pip install -r requirements.txt

3. Ensure Ollama is running locally.
4. Launch the app with either:

   ./run_local_llm_frontend.sh

or:

   python3 main.py

Automated Validation
--------------------
Bundled automated tests passed in this environment during this audit pass:

- `tests_web_search_integration.py`
- `tests_test_generation_job_flow.py`

Observed result from this pass:
- 47 tests passed

The automated tests cover, among other things:
- generation-job request/session snapshot behavior
- web-search tool configuration propagation
- dedicated search-reranker config propagation and result reordering
- fallback preservation of original SearXNG order when reranking fails
- post-cutoff grounding injection
- search normalization
- page fetch extraction and fetch-policy enforcement
- runtime `web_max_pages` enforcement
- GPU parsing behavior
- live throughput metric handling
- UI shutdown safety

Scope and Limits of This README
-------------------------------
This document is based on the code and tests inside this package. It is not a
claim that every external runtime dependency on the user's machine is already
validated. Real-user confirmation is still required for:
- actual local Ollama availability
- actual SearXNG instance behavior
- actual GPU tool availability and output shape on the target machine
- live reranker-model quality and latency on the target machine
