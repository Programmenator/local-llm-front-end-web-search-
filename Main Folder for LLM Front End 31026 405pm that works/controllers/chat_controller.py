"""
controllers/chat_controller.py

Purpose:
    Application logic for sending messages, streaming assistant output, and
    managing saved sessions.

What this file does:
    - Receives user input from the UI.
    - Stores true rolling conversation history.
    - Calls the Ollama client in a background thread.
    - Streams thinking and answer tokens back into the view.
    - Saves, loads, creates, and deletes conversation sessions.
    - Applies settings changes and model selections.
    - Exposes a settings action that unloads the selected Ollama model from memory.
    - Persists the optional reranker model used between search retrieval and answer generation.
    - Tracks one explicit GenerationJob so in-flight requests stay bound to the
      session and config snapshot that existed when the request started.
    - Surfaces assistant throughput metrics so the UI can show a token/second
      performance meter during and after generation.

How this file fits into the system:
    This is the coordinator between the interface and the service layer. It keeps
    business logic out of the widgets and keeps network and persistence work out
    of the UI thread.
"""

from __future__ import annotations

from datetime import datetime
from functools import partial
from threading import RLock
from typing import List, Optional
from uuid import uuid4

from models.chat_message import ChatMessage
from models.conversation_session import ConversationSession
from models.generation_job import GenerationJob
from services.config_service import ConfigService
from services.gpu_monitor_service import GPUMonitorService
from services.ollama_client import OllamaClient
from services.search_service import SearchService
from services.session_service import SessionService
from utils.threading_helpers import run_in_background


class ChatController:
    """Coordinate message flow between the UI, sessions, and the Ollama client.

    Responsibility boundary:
        - Receives user intent from the UI layer.
        - Converts UI actions into service calls.
        - Owns active session state while the program is running.
        - Pushes safe, UI-ready updates back into the view.
        - Owns the active GenerationJob so in-flight request state is not taken
          from mutable controller fields after the request starts.

    Important traceability note:
        This class is the main orchestration point in the project. When someone
        wants to understand *why* the application did something, this is usually
        the first file to inspect because it shows the sequence from button click
        or typed message to persistence, Ollama request, and UI refresh.
    """

    def __init__(
        self,
        config_service: ConfigService,
        ollama_client: OllamaClient,
        session_service: SessionService,
    ) -> None:
        """Store dependencies and initialize runtime session state.

        System interaction summary:
            - Reads the currently selected model from ConfigService.
            - Creates an initial blank ConversationSession through SessionService.
            - Keeps references to services so later UI actions can be routed
              without the widgets needing to know how networking or disk I/O work.
            - Creates a lock-protected GenerationJob slot so one in-flight request
              can be tracked safely across the UI thread and worker thread.
        """
        self.config_service = config_service
        self.ollama_client = ollama_client
        self.session_service = session_service
        self.gpu_monitor_service = GPUMonitorService()
        self.view: Optional[object] = None
        current_model = str(self.config_service.get_config().get("model", ""))
        self.active_session: ConversationSession = self.session_service.create_session(model_name=current_model)
        self.is_generating = False
        self.active_job: Optional[GenerationJob] = None
        self._job_lock = RLock()

    def attach_view(self, view: object) -> None:
        """Attach the active UI view so the controller can push updates to it.

        Call chain:
            app.py -> MainWindow created -> controller.attach_view(main_window)

        The controller intentionally stores only a generic object reference here
        so the service layer remains unaware of Tkinter specifics.
        """
        self.view = view

    def bootstrap_state(self) -> None:
        """Initialize the UI with current sessions, config, and installed models.

        Startup sequence driven by this method:
            1. Push saved session metadata into the session list.
            2. Render the current in-memory conversation in the transcript.
            3. Update the title and current model display.
            4. Start a background model refresh so the UI does not freeze while
               contacting the local Ollama server.
        """
        if not self.view:
            return
        self._push_session_list_to_view()
        self.view.render_full_conversation(self.active_session.messages)
        self.view.update_session_title(self.active_session.title)
        self.view.refresh_model_display(self.config_service.get_config())
        run_in_background(self._refresh_models_background)

    def get_config(self) -> dict:
        """Return the current saved application configuration."""
        return self.config_service.get_config()

    def save_settings(self, new_settings: dict) -> dict:
        """Persist updated settings, refresh models, and return saved config.

        Data flow:
            SettingsWindow -> ChatController.save_settings -> ConfigService.update_config
            -> active session model update -> session resave -> MainWindow refresh

        This method is a good example of why the controller exists: one UI action
        changes both persistent settings and active runtime state. The method now
        refuses to mutate config while a GenerationJob is active because those
        edits should apply only to future requests, not retroactively to the one
        already in flight.
        """
        self._raise_if_generation_active("save settings")
        saved = self.config_service.update_config(new_settings)
        self.active_session.model_name = str(saved.get("model", ""))
        self._persist_active_session()
        if self.view:
            self.view.safe_ui_call(lambda: self.view.refresh_model_display(saved))
        run_in_background(self._refresh_models_background)
        return saved

    def update_selected_model(self, model_name: str) -> None:
        """Save a newly selected model from the UI dropdown.

        Side effects:
            - Updates config.json through ConfigService.
            - Updates the active in-memory session so future reloads remember
              which model was associated with the conversation.
            - Triggers a visible status update in the main window.

        This action is blocked during generation so the active request keeps the
        model snapshot captured in its GenerationJob.
        """
        self._raise_if_generation_active("change the model")
        cleaned_model = model_name.strip()
        if not cleaned_model:
            return
        saved = self.config_service.update_config({"model": cleaned_model})
        self.active_session.model_name = cleaned_model
        self._persist_active_session()
        if self.view:
            self.view.refresh_model_display(saved)
            self.view.set_status(f"Model set to {cleaned_model}")


    def update_selected_reranker(self, model_name: str) -> None:
        """Save the optional reranker model selected from the main window.

        System interaction summary:
            This updates persistent config only. The reranker is a global
            retrieval-stage helper rather than a per-session transcript model, so
            it is not written into the ConversationSession model_name field. An
            empty selection disables reranking and restores direct SearXNG result
            order.
        """
        self._raise_if_generation_active("change the reranker model")
        cleaned_model = model_name.strip()
        saved = self.config_service.update_config({"reranker_model": cleaned_model})
        if self.view:
            self.view.refresh_model_display(saved)
            if cleaned_model:
                self.view.set_status(f"Reranker set to {cleaned_model}")
            else:
                self.view.set_status("Reranker disabled")

    def test_connection(self) -> dict:
        """Run a synchronous Ollama connectivity check for the settings window."""
        return self.ollama_client.test_connection()


    def test_web_search_connection(self, *, base_url: str | None = None, timeout_seconds: int | None = None) -> dict:
        """Run a synchronous SearXNG connectivity check for the settings window.

        This gives the settings dialog a direct operational test for the web
        retrieval backend without forcing the user to send a chat message first.
        The settings popup may pass unsaved field values so the user can verify
        a new endpoint before committing it to disk.
        """
        config = self.config_service.get_config()
        selected_base_url = str(base_url if base_url is not None else config.get("searxng_base_url", "http://127.0.0.1:8080"))
        selected_timeout = int(timeout_seconds if timeout_seconds is not None else config.get("timeout_seconds", 30))
        try:
            service = SearchService(
                base_url=selected_base_url,
                timeout_seconds=selected_timeout,
            )
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}
        return service.test_connection()

    def list_models(self) -> List[str]:
        """Fetch the locally available Ollama model names."""
        return self.ollama_client.list_models()

    def unload_selected_model(self, model_name: str) -> dict:
        """Request immediate unload of one Ollama model from memory.

        Args:
            model_name: Model name selected in the settings window.

        Returns:
            A small status dictionary suitable for driving a popup message.

        UI synchronization note:
            A successful unload can still *look* broken if the footer metrics are
            not refreshed after the unload request finishes. This method now asks
            the attached main window to refresh GPU metrics immediately when the
            unload succeeds so the user can see whether VRAM usage actually fell.
        """
        cleaned_model = model_name.strip()
        if not cleaned_model:
            return {
                "ok": False,
                "message": "No model is selected. Pick a model before trying to unload it.",
            }

        if self.is_generating:
            return {
                "ok": False,
                "message": (
                    "A response is currently streaming. Stop waiting for the current generation first, "
                    "then unload the model."
                ),
            }

        try:
            response = self.ollama_client.unload_model(cleaned_model)
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

        done_flag = bool(response.get("done", False))
        status_text = str(response.get("done_reason", "")).strip() or "completed"
        returned_model = str(response.get("model", cleaned_model)).strip() or cleaned_model

        if done_flag:
            self._request_post_unload_ui_refresh()
            return {
                "ok": True,
                "message": (
                    f"Unload request completed for '{returned_model}'. "
                    f"Ollama reported done_reason='{status_text}'."
                ),
            }

        self._request_post_unload_ui_refresh()
        return {
            "ok": True,
            "message": (
                f"Unload request was sent for '{returned_model}'. "
                "Ollama did not report an explicit done flag, so the interface refreshed GPU metrics "
                "immediately for verification. If VRAM remains allocated, check whether another model "
                "or process is still holding the memory."
            ),
        }

    def _request_post_unload_ui_refresh(self) -> None:
        """Ask the attached view to refresh GPU metrics after a VRAM flush action.

        This keeps the UI from appearing stale after a successful unload request.
        The method is intentionally best-effort so it never turns a successful
        unload into a controller error just because the visible window is absent.
        """
        if not self.view:
            return
        refresh_helper = getattr(self.view, "request_immediate_gpu_metrics_refresh", None)
        if callable(refresh_helper):
            try:
                refresh_helper()
            except Exception:
                pass

    def get_gpu_metrics_snapshot(self) -> dict:
        """Return one normalized GPU metric snapshot for the main window.

        Why this is exposed through the controller:
            The main UI needs a polling-safe read operation, but the UI should
            not know how ROCm commands are executed or parsed. The controller
            forwards the normalized snapshot from the service layer and also
            preserves the legacy ``*_text`` keys expected by the PyQt6 footer
            refresh path.

        Compatibility detail:
            The GPU service returns canonical ``gpu_percent`` and
            ``vram_percent`` fields. Earlier UI refresh code still read
            ``gpu_percent_text`` and ``vram_percent_text``. This method now
            normalizes both naming styles so the metrics stay visible even if a
            caller still expects the older keys.
        """
        snapshot = dict(self.gpu_monitor_service.get_live_metrics())
        gpu_value = snapshot.get("gpu_percent") or snapshot.get("gpu_percent_text") or "N/A"
        vram_value = snapshot.get("vram_percent") or snapshot.get("vram_percent_text") or "N/A"
        snapshot["gpu_percent"] = gpu_value
        snapshot["vram_percent"] = vram_value
        snapshot["gpu_percent_text"] = gpu_value
        snapshot["vram_percent_text"] = vram_value
        return snapshot

    def request_model_refresh(self) -> None:
        """Refresh installed model metadata without blocking the UI thread."""
        run_in_background(self._refresh_models_background)

    def create_new_session(self) -> None:
        """Create a blank session, persist it, and update the UI.

        Call chain:
            MainWindow button/menu -> controller -> SessionService.create_session
            -> SessionService.save_session -> MainWindow re-render

        This resets only the active conversation workspace. It does not delete
        any previously saved sessions. The action is blocked while a GenerationJob
        is active so the running request cannot lose its target session binding.
        """
        self._raise_if_generation_active("create a new session")
        current_model = str(self.config_service.get_config().get("model", ""))
        self.active_session = self.session_service.create_session(model_name=current_model)
        self._persist_active_session()
        if self.view:
            self.view.render_full_conversation([])
            self.view.update_session_title(self.active_session.title)
            self.view.set_status("Created a new session")
        self._push_session_list_to_view()

    def load_session(self, session_id: str) -> None:
        """Load a saved session into the active workspace and refresh the UI.

        Traceability detail:
            Loading a session also restores its remembered model into the config
            layer. That keeps the model dropdown and saved conversation aligned.

        This action is blocked while a GenerationJob is active so a background
        worker cannot finish into a session that was not part of the request.
        """
        self._raise_if_generation_active("switch sessions")
        session = self.session_service.load_session(session_id)
        if not session:
            if self.view:
                self.view.set_status("Could not load session")
            return

        self.active_session = session
        if session.model_name:
            self.config_service.update_config({"model": session.model_name})
        if self.view:
            self.view.render_full_conversation(self.active_session.messages)
            self.view.update_session_title(self.active_session.title)
            self.view.refresh_model_display(self.config_service.get_config())
            self.view.highlight_session(session.session_id)
            self.view.set_status(f"Loaded session: {session.title}")

    def delete_session(self, session_id: str) -> None:
        """Delete a saved session and replace it if it was active.

        This action is blocked while a GenerationJob is active so the response
        being streamed cannot lose the session file it is meant to update.
        """
        self._raise_if_generation_active("delete a session")
        deleted = self.session_service.delete_session(session_id)
        if not deleted:
            if self.view:
                self.view.set_status("Session delete failed")
            return

        if self.active_session.session_id == session_id:
            current_model = str(self.config_service.get_config().get("model", ""))
            self.active_session = self.session_service.create_session(model_name=current_model)
            self._persist_active_session()
            if self.view:
                self.view.render_full_conversation([])
                self.view.update_session_title(self.active_session.title)
        self._push_session_list_to_view()
        if self.view:
            self.view.set_status("Session deleted")

    def rename_active_session(self, new_title: str) -> None:
        """Rename the active session and persist the change.

        The action is blocked while a GenerationJob is active so any later auto-
        title or persistence step still applies to the same request target.
        """
        self._raise_if_generation_active("rename the active session")
        cleaned = new_title.strip()
        if not cleaned:
            return
        self.active_session.title = cleaned
        self._persist_active_session()
        self._push_session_list_to_view()
        if self.view:
            self.view.update_session_title(cleaned)
            self.view.set_status("Session renamed")

    def send_user_message(self, text: str) -> None:
        """Add a user message, update the UI, and stream an assistant reply.

        This is the main runtime path of the whole application. The sequence is:
            1. Validate and normalize the typed text.
            2. Append a ChatMessage(role="user") to the active session.
            3. Persist the session to disk.
            4. Build a GenerationJob that freezes the request session/config state.
            5. Ask the UI to render the new user message immediately.
            6. Prepare an empty assistant placeholder in the transcript.
            7. Start a background Ollama request so Tkinter stays responsive.

        The actual network streaming work continues in background helper methods
        later in this file.
        """
        cleaned = text.strip()
        if not cleaned:
            return
        with self._job_lock:
            if self.is_generating:
                return

            self.is_generating = True
            user_message = ChatMessage(role="user", content=cleaned, timestamp=datetime.now())
            self.active_session.messages.append(user_message)
            self._persist_active_session()
            self.active_job = self._create_generation_job(user_message_text=cleaned)
            job = self.active_job

        if self.view:
            self.view.append_message(user_message)
            self.view.begin_assistant_stream()
            self.view.set_busy_state(True, "Streaming response from Ollama...")

        run_in_background(partial(self._generate_assistant_reply, job))

    def _generate_assistant_reply(self, job: GenerationJob) -> None:
        """Background worker that streams a model response for one GenerationJob.

        Critical design rule:
            The worker uses only the data stored inside the GenerationJob after
            the request starts. It does not read mutable controller session/config
            state to decide where the response belongs or which model generated it.
        """
        assistant_message = ChatMessage(role="assistant", content="", thinking="", timestamp=datetime.now())
        with self._job_lock:
            job.status = "streaming"

        try:
            result = self.ollama_client.chat_stream(
                job.message_history,
                on_chunk=partial(self._handle_stream_chunk, job),
                request_options=job.to_request_options(),
            )
            job.response_text = result.get("content", "") or "[Model returned an empty response.]"
            job.thinking_text = result.get("thinking", "")
            job.actual_output_tokens = int(result.get("output_tokens", 0) or 0)
            job.tokens_per_second = float(result.get("tokens_per_second", 0.0) or 0.0)
            job.throughput_source = str(result.get("throughput_source", "") or "")
            assistant_message.content = job.response_text
            assistant_message.thinking = job.thinking_text
            assistant_message.sources = [item for item in result.get("sources_used", []) if isinstance(item, dict)]
            assistant_message.timestamp = datetime.now()
            self._commit_job_result(job, assistant_message)
            with self._job_lock:
                job.status = "completed"
            if self.view and self._is_active_job(job.request_id):
                self.view.safe_ui_call(
                    lambda: self.view.finalize_assistant_stream(
                        assistant_message.content,
                        assistant_message.thinking,
                        {
                            "output_tokens": job.actual_output_tokens,
                            "tokens_per_second": job.tokens_per_second,
                            "throughput_source": job.throughput_source,
                            "sources_used": assistant_message.sources,
                            "invalid_citations": result.get("invalid_citations", []),
                        },
                    )
                )
        except Exception as exc:  # Broad catch keeps the UI alive on unexpected failures.
            job.error_text = str(exc)
            with self._job_lock:
                job.status = "failed"
            self._record_job_error(job)
        finally:
            self._finish_job(job)

    def _commit_job_result(self, job: GenerationJob, assistant_message: ChatMessage) -> None:
        """Persist a completed assistant reply back into the correct session.

        System interaction summary:
            This method resolves the target by job.session_id rather than by the
            mutable active-session pointer. That is the core session-race fix.
        """
        target_session = self._get_session_for_job(job)
        if target_session is None:
            raise RuntimeError(
                f"The target session for request {job.request_id} no longer exists, so the reply could not be saved."
            )

        target_session.messages.append(assistant_message)
        self._auto_title_if_needed(target_session)
        self._persist_session(target_session)

        with self._job_lock:
            if self.active_session.session_id == target_session.session_id:
                self.active_session = target_session

    def _record_job_error(self, job: GenerationJob) -> None:
        """Persist a failed job as a system message inside the correct session.

        The application still keeps UI continuity by surfacing an error message
        in the transcript, but the message is attached to the session identified
        by the GenerationJob rather than whichever session happens to be active.
        """
        error_message = ChatMessage(
            role="system",
            content=f"Error: {job.error_text}",
            timestamp=datetime.now(),
        )
        target_session = self._get_session_for_job(job)
        if target_session is not None:
            target_session.messages.append(error_message)
            self._persist_session(target_session)
            with self._job_lock:
                if self.active_session.session_id == target_session.session_id:
                    self.active_session = target_session
        if self.view and self._is_active_job(job.request_id):
            self.view.safe_ui_call(lambda: self.view.append_message(error_message))

    def _handle_stream_chunk(self, job: GenerationJob, chunk_type: str, text_chunk: str) -> None:
        """Route streamed model chunks back to the UI for the active job only.

        This guard prevents late or stale worker callbacks from writing into the
        current transcript if they do not belong to the active GenerationJob.
        """
        if not self.view or not self._is_active_job(job.request_id):
            return
        self.view.safe_ui_call(lambda: self.view.append_stream_chunk(chunk_type, text_chunk))

    def _refresh_models_background(self) -> None:
        """Background worker that fetches installed models and updates the UI."""
        try:
            models = self.ollama_client.list_models()
        except Exception as exc:
            if self.view:
                self.view.safe_ui_call(lambda: self.view.set_status(f"Model refresh failed: {exc}"))
            return

        config = self.config_service.get_config()
        if not config.get("model") and models:
            config = self.config_service.update_config({"model": models[0]})
            self.active_session.model_name = models[0]
            self._persist_active_session()

        if self.view:
            self.view.safe_ui_call(lambda: self.view.update_model_choices(models, str(config.get("model", "")), str(config.get("reranker_model", ""))))
            self.view.safe_ui_call(lambda: self.view.refresh_model_display(config))

    def _push_session_list_to_view(self) -> None:
        """Refresh the UI session list from disk persistence."""
        if not self.view:
            return
        sessions = self.session_service.list_sessions()
        self.view.safe_ui_call(lambda: self.view.update_session_list(sessions, self.active_session.session_id))

    def _persist_active_session(self) -> None:
        """Save the active session through the session service.

        The model name written to the session is current long-lived config state
        for future requests. In-flight request metadata is stored separately in
        the GenerationJob so the save does not retroactively redefine the model
        used by an already running request.
        """
        self.active_session.model_name = str(self.config_service.get_config().get("model", ""))
        self._persist_session(self.active_session)

    def _persist_session(self, session: ConversationSession) -> None:
        """Save one specific session object through the session service."""
        self.session_service.save_session(session)

    def _auto_title_if_needed(self, session: ConversationSession) -> None:
        """Replace a generic default title with one derived from the first user message.

        The title logic now accepts an explicit session object so auto-titling is
        bound to the session identified by the GenerationJob rather than the
        mutable active-session field.
        """
        if not session.title.startswith("Session "):
            return
        for message in session.messages:
            if message.role == "user" and message.content.strip():
                summary = message.content.strip().replace("\n", " ")[:48].strip()
                session.title = summary or session.title
                if self.view and self.active_session.session_id == session.session_id:
                    self.view.safe_ui_call(lambda: self.view.update_session_title(session.title))
                break

    def _create_generation_job(self, user_message_text: str) -> GenerationJob:
        """Capture a frozen request snapshot for one new assistant generation.

        Snapshot contents:
            - target session identity
            - current session title
            - current host/port/timeout
            - current model and prompt settings
            - current web-search settings, so tool availability cannot silently
              disappear between the Settings window and the actual chat request
            - frozen message history including the newly appended user message

        This snapshot is the main architectural fix for the session race bug.
        """
        config = self.config_service.get_config()
        return GenerationJob(
            request_id=uuid4().hex,
            session_id=self.active_session.session_id,
            session_title_at_start=self.active_session.title,
            model_name=str(config.get("model", "")),
            host=str(config.get("host", "127.0.0.1")),
            port=int(config.get("port", 11434)),
            timeout_seconds=int(config.get("timeout_seconds", 300)),
            system_prompt=str(config.get("system_prompt", "")),
            thinking_mode=bool(config.get("thinking_mode", True)),
            thinking_level=str(config.get("thinking_level", "medium")),
            reranker_model=str(config.get("reranker_model", "")),
            enable_web_search=bool(config.get("enable_web_search", False)),
            searxng_base_url=str(config.get("searxng_base_url", "")),
            web_search_category=str(config.get("web_search_category", "general")),
            web_search_language=str(config.get("web_search_language", "all")),
            web_search_time_range=str(config.get("web_search_time_range", "none")),
            web_safe_search=int(config.get("web_safe_search", 1)),
            web_max_results=int(config.get("web_max_results", 8)),
            web_max_pages=int(config.get("web_max_pages", 2)),
            web_fetch_enabled=bool(config.get("web_fetch_enabled", True)),
            message_history=self._snapshot_messages(self.active_session.messages),
            user_message_text=user_message_text,
        )

    def _snapshot_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Create a detached copy of message history for one GenerationJob.

        Why a deep snapshot matters:
            The worker thread must send the exact conversation history that
            existed at request start. Copying the messages prevents later UI or
            controller mutations from altering the request while it is running.
        """
        return [ChatMessage.from_dict(message.to_dict()) for message in messages]

    def _get_session_for_job(self, job: GenerationJob) -> Optional[ConversationSession]:
        """Resolve the session targeted by one GenerationJob.

        Resolution order:
            1. Use the in-memory active session if it is the same session ID.
            2. Otherwise load the persisted session by ID from SessionService.
        """
        with self._job_lock:
            if self.active_session.session_id == job.session_id:
                return self.active_session
        return self.session_service.load_session(job.session_id)

    def _is_active_job(self, request_id: str) -> bool:
        """Return True when the supplied request ID is still the active job."""
        with self._job_lock:
            return bool(self.active_job and self.active_job.request_id == request_id)

    def _finish_job(self, job: GenerationJob) -> None:
        """Clear controller busy state only if the finished job is still active."""
        should_clear_ui = False
        with self._job_lock:
            if self.active_job and self.active_job.request_id == job.request_id:
                self.active_job = None
                self.is_generating = False
                should_clear_ui = True

        self._push_session_list_to_view()
        if self.view and should_clear_ui:
            self.view.safe_ui_call(lambda: self.view.set_busy_state(False, "Ready"))

    def _raise_if_generation_active(self, action_description: str) -> None:
        """Block state-changing actions while a GenerationJob is active.

        Why this exists:
            The GenerationJob fixes request ownership, but command locking still
            matters because switching sessions or mutating config during a stream
            is confusing to users even if the worker no longer corrupts state.
        """
        with self._job_lock:
            if not self.is_generating:
                return
        message = f"Cannot {action_description} while a response is streaming. Wait for the current generation to finish."
        if self.view:
            self.view.set_status(message)
        raise RuntimeError(message)
