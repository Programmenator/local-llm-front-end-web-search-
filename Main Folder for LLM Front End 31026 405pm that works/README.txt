Local LLM Front End - PyQt6 Documentation Sweep Revision
=======================================================

What this package is
--------------------
This package is the PyQt6 desktop front end for the local LLM workflow you have
been building. It keeps the existing controller, service, and model layers, and
uses PyQt6 for the user-facing desktop interface. The main goal of this package
is to preserve the original frontend behavior while giving you a more scalable
GUI foundation for future plug-ins and workflow growth.

Current documentation/compliance status
---------------------------------------
This revision includes a full documentation sweep focused on your current
standards:
- top-of-file module purpose documentation is present across the Python source
- function and method docstrings are present across the Python source,
  including helper/test support code that previously had documentation gaps
- README, file log, change log, and test report were revised to remove stale or
  misleading wording and to reflect the current PyQt6 implementation state
- the package was cleaned for delivery so compiled cache folders are not needed
  in the final ZIP

What this frontend does
-----------------------
The program provides a desktop interface for a local-model workflow with these
major responsibilities:
- manage sessions and conversation history
- select and refresh Ollama models
- send user prompts and display streamed assistant replies
- display a separate thinking panel
- display a validated sources pane under the thinking panel
- expose configuration controls through a settings dialog
- surface GPU utilization, VRAM utilization, and token throughput readouts
- support optional web-search configuration through the existing service layer
- render clickable in-text citations and validated source links for grounded answers

High-level architecture
-----------------------
main.py
- Thin launcher.
- Starts the application entry point.

app.py
- Composition root.
- Creates QApplication, services, controller, and the main window.
- Wires the UI to the controller and starts the event loop.

controllers/chat_controller.py
- Behavioral center of the application.
- Coordinates session state, configuration state, persistence, model refresh,
  streamed replies, and UI updates.

models/*
- Small durable data structures shared across the application.
- Preserve session, message, and request snapshot state.

services/*
- Deterministic infrastructure layer.
- Handles config persistence, session persistence, Ollama transport, GPU metric
  parsing, SearXNG search calls, and HTML/text extraction for fetched pages.

ui/main_window.py
- Main desktop window.
- Renders the session rail, transcript, prompt area, footer metrics, and
  thinking panel.
- Exposes controller-facing compatibility methods that the rest of the program
  still relies on.

ui/settings_window.py
- Settings dialog.
- Lets the user edit config, test connections, refresh models, and request
  model unload / VRAM flush operations.
- After a successful flush request, it now also triggers a model refresh so the
  visible loaded-model state and the main window metrics can catch up quickly.

utils/threading_helpers.py
- Small helper for background-thread execution.

How the UI is arranged
----------------------
Left side:
- session controls and session list
- top-level model selection/readout area
- status area and supporting controls

Center:
- conversation transcript
- bottom prompt input
- send workflow
- main-page web-search toggle
- bottom-left footer metrics strip for GPU %, VRAM %, and token throughput

Right side:
- dedicated thinking panel
- validated sources pane directly under the thinking panel
- hide/show workflow
- separate scrollable areas for streamed reasoning text and source validation links

Why PyQt6 is being used here
----------------------------
PyQt6 gives this project a better desktop layout foundation than Tkinter for
this specific use case. In this program that matters for:
- split-pane resizing
- richer desktop widget behavior
- easier growth if more panes or tool modules are added later
- cleaner handling of a multi-region desktop interface

Documentation note about function coverage
------------------------------------------
This revision specifically closes documentation gaps that remained after the
earlier PyQt6 repair passes. The gaps were mostly in:
- small compatibility helper classes in ui/main_window.py
- fake test harness classes used to validate controller and UI behavior
- several smoke/integration test methods that previously had behavior but no
  docstring traceability

That means the package now has both behavior coverage and traceability coverage
for the parts that are most likely to be revisited later during maintenance.

Temporal grounding behavior for web search
-----------------------------------------
This revision strengthens the web-grounding instructions passed into the Ollama
client so current web results are treated correctly during 2026-era queries.
The program now explicitly tells the model that:
- the live runtime date is March 10, 2026
- 2026 is the present application context, not a hypothetical future date
- retrieved web evidence outranks internal model knowledge for current or
  time-sensitive facts
- internal knowledge versus retrieved current evidence should not be described
  as a conflict
- conflicts should only be reported when the retrieved sources disagree with
  each other or remain genuinely ambiguous

Verification summary
--------------------
This package was source-verified after the temporal-grounding revision:
- all Python files compile with py_compile
- the web-search integration test suite passes
- full PyQt6-dependent test execution still requires PyQt6 to be installed in
  the runtime environment used for verification

The automated verification available in this environment covers:
- controller generation/session behavior
- config persistence and normalization
- session persistence lifecycle
- Ollama streaming/non-streaming behaviors
- GPU metric parsing across AMD/ROCm output shapes
- SearXNG URL normalization and result shaping
- web page text extraction
- main-window workflow helpers and settings-window workflows
- prompt send shortcut behavior
- unavailable-model preservation behavior
- threading helper behavior

What to review first
--------------------
1. ui/main_window.py
2. ui/settings_window.py
3. controllers/chat_controller.py
4. services/config_service.py
5. services/gpu_monitor_service.py
6. TEST_REPORT.txt

Manual verification checklist
-----------------------------
- Launch the UI successfully under your target Linux desktop.
- Confirm a model can be selected and refreshed.
- Confirm sessions can be created, loaded, renamed, and deleted.
- Confirm the prompt input stays at the bottom of the main workspace.
- Confirm Ctrl+Enter sends from inside the prompt editor.
- Confirm transcript/thinking/prompt panes respond to Ctrl + mouse wheel zoom.
- Confirm the thinking panel can be hidden and shown.
- Confirm footer GPU/VRAM/Tok/s readouts update during use.
- Confirm settings save/load behavior matches your expected runtime setup.
- Confirm the Flush GPU VRAM action drops VRAM usage for the selected model or,
  if it does not, verify whether another model/process is still holding memory.
- Confirm Ollama and SearXNG connection tests behave correctly on your machine.

Revision history inside this package
------------------------------------
The package already includes prior PyQt6 migration, repair, stabilization, and
verification work. This documentation sweep adds one more pass focused on
compliance, traceability, and correction of stale documentation rather than a
new behavioral redesign.

Grounded sources and citation behavior
-------------------------------------
This revision adds a second validation surface for web-grounded answers:
- search results returned by SearXNG are stamped with stable source IDs such as
  S1, S2, and S3
- the Ollama grounding prompt now tells the model to cite factual claims with
  those exact bracketed IDs, for example [S1]
- the program validates cited source IDs against the real SearXNG results for
  that request instead of trusting model-invented URLs
- the sources pane shows only validated sources from the real result list
- source links are clickable and use browser text-fragment anchors when enough
  snippet or fetched-page text is available
- invalid or hallucinated citation IDs are ignored and flagged in the sources
  status text instead of being treated as real evidence

Limit of the current highlight behavior
--------------------------------------
The program uses best-effort browser text-fragment links (the `#:~:text=`
format). On supported browsers and pages, this jumps to and highlights the
relevant quoted snippet or fetched excerpt. Some sites disable or break browser
text fragments, so the link can still open the correct page without visually
highlighting the exact text. In those cases the source is still validated
against the real SearXNG result list, but page-side highlighting depends on the
site and browser.


Search result pipeline fix (2026-03-10)
----------------------------------------
Two independent bottlenecks were silently capping search results below the
UI-configured value:

1. services/search_service.py contained a hard min(20, ...) clamp that
   discarded any results above 20 regardless of the UI setting.

2. services/config_service.py clamped web_max_results at 20 and web_max_pages
   at 5 during config normalization, so values of 50 or 100 were silently
   reduced before they ever reached the search or fetch layers.

3. services/ollama_client.py carried web_max_pages in GenerationJob and
   to_request_options() but never enforced it as a per-turn limit. The model
   could call fetch_url_content an unlimited number of times regardless of the
   configured ceiling.

After the fix, a UI setting of 50 results flows through config normalization
(ceiling now 200), through search_service (hard clamp removed), to SearXNG,
and back to the reranker and LLM without truncation. The fetch_url_content
tool call count is now tracked and capped at the configured web_max_pages
value per turn.

Source catalog accumulation fix (2026-03-10)
--------------------------------------------
The validated sources panel was showing fewer sources than expected even after
the search pipeline was unblocked. The root cause was in the source catalog
management inside services/ollama_client.py.

When a model issues a second search_web tool call in the same turn, the old
code called source_catalog.clear() before extending it with the new results.
This wiped all citations from the first search. Any [S1], [S2] etc. that the
model had cited from the first search were no longer in the catalog, so the
validation pass marked them invalid and the UI dropped them silently.

The clear() also masked a secondary bug: each search call assigned source IDs
starting at S1, so two searches produced colliding IDs pointing at different
results. The clear() was the only thing preventing the wrong result from being
displayed for a reused ID.

Both problems are fixed together:
- services/search_service.py now accepts an id_offset parameter. The offset
  shifts the starting source ID number so the first search produces S1..S10
  and a second search produces S11..S20, making all IDs globally unique within
  the turn.
- services/ollama_client.py now passes len(source_catalog) as id_offset when
  calling search_web, then uses extend() instead of clear() + extend(). The
  catalog accumulates results from all searches in the turn, and citations from
  any search call resolve correctly at validation time.


- The PyQt6 footer and lower-left GPU readouts now accept both the canonical service keys (`gpu_percent`, `vram_percent`) and the older compatibility keys (`gpu_percent_text`, `vram_percent_text`).
- Root cause: the GPU monitor service returned canonical keys, while the PyQt6 refresh path still read only the older `*_text` keys, which forced the UI to display `--` even when valid metrics existed.
- Result: live GPU utilization and VRAM utilization values now render correctly when the service returns either naming style.


VRAM flush fix (2026-03-10):
- Root cause was split across two layers. The unload action itself had become too brittle because it relied only on one API unload path, and the PyQt6 settings dialog did not force a fast visible refresh after a successful unload.
- The Ollama client now prefers `ollama stop <model>` when the CLI is available, then falls back to `/api/generate` with `keep_alive=0`, and then `/api/chat` with `keep_alive=0`.
- The controller now asks the main window for an immediate GPU metrics refresh after a successful unload, and the settings dialog refreshes the model list right away.
- Result: the program now distinguishes more cleanly between a true unload failure and a stale UI readout.


RERANKER UPDATE
- Added an optional Ollama-served reranker stage between SearXNG result retrieval and the main answer model.
- The main window now shows a reranker selector beside the primary model selector.
- The reranker uses only real returned source IDs, so it can reorder results but cannot invent candidates.
