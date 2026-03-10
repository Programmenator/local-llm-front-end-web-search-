"""
ui/main_window.py

Purpose:
    Primary desktop chat interface implemented with PyQt6.

What this file does:
    - Builds the main application window in PyQt6.
    - Displays conversation history.
    - Accepts user input from a prompt box anchored at the bottom.
    - Exposes the settings button plus primary-model and reranker selectors in a left sidebar.
    - Manages saved sessions.
    - Streams assistant text live into the transcript.
    - Shows optional thinking output in a dedicated dockable panel.
    - Shows live GPU and throughput readouts in both the sidebar and a
      bottom-left footer strip near the prompt controls.
    - Keeps layout resizable through Qt splitters.
    - Supports Ctrl+mouse-wheel transcript and input zoom for readability.

How this file fits into the system:
    This is the user-facing shell of the application. It delegates application
    logic to the controller and focuses on layout, widget wiring, and safe UI
    presentation of state updates pushed from background worker threads.
"""

from __future__ import annotations

import html
import re
import time
from typing import Iterable

from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QTextCursor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from controllers.chat_controller import ChatController
from models.chat_message import ChatMessage
from models.conversation_session import ConversationSession
from services.config_service import ConfigService
from ui.settings_window import SettingsWindow
from utils.threading_helpers import run_in_background


class ZoomableTextEdit(QTextEdit):
    """QTextEdit with Ctrl+wheel zoom support for readability."""

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        """Zoom the text when Ctrl is held, otherwise keep normal scrolling."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoomIn(1)
            elif delta < 0:
                self.zoomOut(1)
            event.accept()
            return
        super().wheelEvent(event)


class RichTextBrowser(QTextBrowser):
    """Read-only rich-text browser with Ctrl+wheel zoom and external links."""

    def __init__(self) -> None:
        """Initialize the read-only browser used for transcript-style panes."""
        super().__init__()
        self.setReadOnly(True)
        self.setOpenExternalLinks(True)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        """Zoom the text when Ctrl is held, otherwise keep normal scrolling."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoomIn(1)
            elif delta < 0:
                self.zoomOut(1)
            event.accept()
            return
        super().wheelEvent(event)


class PromptInputTextEdit(ZoomableTextEdit):
    """Prompt editor that emits a send signal when Ctrl+Enter is pressed.

    Why this exists:
        Overriding QMainWindow.keyPressEvent is not enough to catch Ctrl+Enter
        while focus is inside the QTextEdit itself. This widget restores the
        advertised shortcut behavior directly at the input control.
    """

    send_requested = pyqtSignal()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        """Emit send_requested when Ctrl+Enter is pressed inside the editor."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.send_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class TextValueProxy:
    """Small StringVar-like compatibility wrapper for legacy tests and helpers.

    Why this exists:
        The earlier Tkinter window exposed several `*_var` objects with `set()`
        and `get()` methods. The PyQt6 rewrite moved to QLabel widgets, but a
        few tests and support paths still expect the old interface shape. This
        proxy preserves that lightweight contract without reintroducing Tkinter.
    """

    def __init__(self, initial: str = "") -> None:
        """Initialize a simple string proxy used to preserve legacy Tk-style compatibility paths."""
        self.value = initial

    def set(self, value: str) -> None:
        """Store the latest string value."""
        self.value = value

    def get(self) -> str:
        """Return the current stored value."""
        return self.value


class BoolValueProxy:
    """Small BooleanVar-like compatibility wrapper for legacy tests and helpers."""

    def __init__(self, initial: bool = False) -> None:
        """Initialize a simple boolean proxy used to preserve legacy compatibility state."""
        self.value = bool(initial)

    def set(self, value: bool) -> None:
        """Store the latest boolean value."""
        self.value = bool(value)

    def get(self) -> bool:
        """Return the current stored boolean value."""
        return self.value


class UICallbackBridge(QObject):
    """Thread-safe bridge for scheduling arbitrary callbacks on the Qt UI thread."""

    invoke = pyqtSignal(object)

    def __init__(self) -> None:
        """Initialize the bridge that marshals worker-thread callbacks back onto the UI thread."""
        super().__init__()
        self.invoke.connect(self._execute)

    def _execute(self, callback: object) -> None:
        """Run a queued callback after Qt delivers it on the owning thread."""
        if callable(callback):
            callback()


class MainWindow(QMainWindow):
    """Main desktop window for chatting with the local Ollama instance."""

    GPU_POLL_INTERVAL_MS = 1500

    def __init__(self, controller: ChatController, config_service: ConfigService) -> None:
        """Create the Qt window, state holders, and the interface layout."""
        super().__init__()
        self.controller = controller
        self.config_service = config_service

        self.setWindowTitle("Local LLM Front End")
        self.resize(1440, 900)
        self.setMinimumSize(1160, 760)

        self.status_var = TextValueProxy("Ready")
        self.model_var = TextValueProxy("")
        self.reranker_var = TextValueProxy("")
        self.session_title_var = TextValueProxy("Untitled Session")
        self.thinking_visible_var = BoolValueProxy(True)
        self.gpu_percent_var = TextValueProxy("GPU%: --")
        self.vram_percent_var = TextValueProxy("VRAM%: --")
        self.tokens_per_second_var = TextValueProxy("Tok/s: --")
        self.sources_status_var = TextValueProxy("Sources: none validated yet")

        self.active_stream_has_thinking = False
        self.streaming_in_progress = False
        self._gpu_poll_inflight = False
        self._gpu_poll_generation = 0
        self._stream_started_monotonic = 0.0
        self._stream_output_token_estimate = 0
        self._is_closing = False
        self._current_assistant_stream_open = False
        self._suspend_session_signal = False
        self._suspend_model_signal = False
        self._settings_window: SettingsWindow | None = None
        self._ui_callback_bridge = UICallbackBridge()
        self._latest_validated_sources: list[dict] = []
        self._latest_invalid_citations: list[str] = []

        self._build_layout()
        self.refresh_model_display(self.config_service.get_config())
        self._schedule_gpu_metrics_refresh()

    def _build_layout(self) -> None:
        """Create all primary widgets and arrange them in the main window."""
        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(main_splitter)

        sidebar = self._build_sidebar()
        main_splitter.addWidget(sidebar)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(content_splitter)

        center_panel = self._build_center_panel()
        content_splitter.addWidget(center_panel)

        thinking_panel = self._build_thinking_panel()
        content_splitter.addWidget(thinking_panel)
        content_splitter.setSizes([940, 300])

        main_splitter.setSizes([300, 1140])

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.set_status("Ready")

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings_window)
        self.addAction(settings_action)

    def _build_sidebar(self) -> QWidget:
        """Build the left control rail with models, GPU status, and sessions."""
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        controls_label = QLabel("Controls")
        controls_label.setProperty("sectionHeader", True)
        layout.addWidget(controls_label)

        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings_window)
        layout.addWidget(self.settings_button)

        selectors_header = QLabel("Models")
        selectors_header.setProperty("sectionHeader", True)
        layout.addWidget(selectors_header)

        selector_row = QHBoxLayout()

        model_column = QVBoxLayout()
        model_label = QLabel("Primary")
        model_column.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._handle_model_selected)
        model_column.addWidget(self.model_combo)
        selector_row.addLayout(model_column, stretch=1)

        reranker_column = QVBoxLayout()
        reranker_label = QLabel("Reranker")
        reranker_column.addWidget(reranker_label)
        self.reranker_combo = QComboBox()
        self.reranker_combo.currentTextChanged.connect(self._handle_reranker_selected)
        reranker_column.addWidget(self.reranker_combo)
        selector_row.addLayout(reranker_column, stretch=1)

        layout.addLayout(selector_row)

        self.model_detail_label = QLabel("")
        self.model_detail_label.setWordWrap(True)
        layout.addWidget(self.model_detail_label)

        metrics_header = QLabel("GPU Live Readout")
        metrics_header.setProperty("sectionHeader", True)
        layout.addWidget(metrics_header)

        self.gpu_percent_label = QLabel(self.gpu_percent_var.get())
        self.vram_percent_label = QLabel(self.vram_percent_var.get())
        self.tokens_per_second_label = QLabel(self.tokens_per_second_var.get())
        layout.addWidget(self.gpu_percent_label)
        layout.addWidget(self.vram_percent_label)
        layout.addWidget(self.tokens_per_second_label)

        sessions_header = QLabel("Sessions")
        sessions_header.setProperty("sectionHeader", True)
        layout.addWidget(sessions_header)

        self.session_title_label = QLabel("Untitled Session")
        self.session_title_label.setWordWrap(True)
        layout.addWidget(self.session_title_label)

        session_button_row = QHBoxLayout()
        self.new_session_button = QPushButton("New")
        self.new_session_button.clicked.connect(self._handle_new_session)
        session_button_row.addWidget(self.new_session_button)

        self.rename_session_button = QPushButton("Rename")
        self.rename_session_button.clicked.connect(self._rename_session)
        session_button_row.addWidget(self.rename_session_button)

        self.delete_session_button = QPushButton("Delete")
        self.delete_session_button.clicked.connect(self._delete_selected_session)
        session_button_row.addWidget(self.delete_session_button)
        layout.addLayout(session_button_row)

        self.session_list = QListWidget()
        self.session_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.session_list.currentItemChanged.connect(self._handle_session_selected)
        layout.addWidget(self.session_list, stretch=1)

        return panel

    def _build_center_panel(self) -> QWidget:
        """Build the primary conversation area with bottom-anchored prompt input."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header_row = QHBoxLayout()
        conversation_label = QLabel("Conversation")
        conversation_label.setProperty("sectionHeader", True)
        header_row.addWidget(conversation_label)
        header_row.addStretch(1)

        self.refresh_models_button = QPushButton("Refresh Models")
        self.refresh_models_button.clicked.connect(self.controller.request_model_refresh)
        header_row.addWidget(self.refresh_models_button)
        layout.addLayout(header_row)

        self.chat_history = RichTextBrowser()
        self.chat_history.setPlaceholderText("Conversation transcript will appear here.")
        layout.addWidget(self.chat_history, stretch=1)

        prompt_header = QHBoxLayout()
        prompt_label = QLabel("Prompt Input")
        prompt_label.setProperty("sectionHeader", True)
        prompt_header.addWidget(prompt_label)
        prompt_header.addStretch(1)

        self.main_web_search_checkbox = QCheckBox("Web search enabled")
        self.main_web_search_checkbox.stateChanged.connect(self._handle_main_web_search_toggle)
        prompt_header.addWidget(self.main_web_search_checkbox)
        layout.addLayout(prompt_header)

        prompt_row = QHBoxLayout()
        self.input_text = PromptInputTextEdit()
        self.input_text.setPlaceholderText("Type a message. Press Ctrl+Enter to send.")
        self.input_text.setMinimumHeight(130)
        self.input_text.send_requested.connect(self._send_current_input)
        prompt_row.addWidget(self.input_text, stretch=1)

        send_column = QVBoxLayout()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._send_current_input)
        send_column.addWidget(self.send_button)
        send_column.addStretch(1)
        prompt_row.addLayout(send_column)

        layout.addLayout(prompt_row)

        footer_metrics_row = QHBoxLayout()
        footer_metrics_row.setSpacing(16)

        self.footer_gpu_percent_label = QLabel(self.gpu_percent_var.get())
        self.footer_vram_percent_label = QLabel(self.vram_percent_var.get())
        self.footer_tokens_per_second_label = QLabel(self.tokens_per_second_var.get())
        footer_metrics_row.addWidget(self.footer_gpu_percent_label)
        footer_metrics_row.addWidget(self.footer_vram_percent_label)
        footer_metrics_row.addWidget(self.footer_tokens_per_second_label)
        footer_metrics_row.addStretch(1)
        layout.addLayout(footer_metrics_row)
        return panel

    def _build_thinking_panel(self) -> QWidget:
        """Build the optional thinking stream panel shown beside the transcript."""
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        header_row = QHBoxLayout()
        thinking_label = QLabel("Thinking Stream")
        thinking_label.setProperty("sectionHeader", True)
        header_row.addWidget(thinking_label)
        header_row.addStretch(1)

        self.toggle_thinking_button = QToolButton()
        self.toggle_thinking_button.setText("Hide")
        self.toggle_thinking_button.clicked.connect(self._toggle_thinking_visibility)
        header_row.addWidget(self.toggle_thinking_button)
        layout.addLayout(header_row)

        pane_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(pane_splitter, stretch=1)

        thinking_container = QWidget()
        thinking_layout = QVBoxLayout(thinking_container)
        thinking_layout.setContentsMargins(0, 0, 0, 0)
        thinking_layout.setSpacing(6)
        self.thinking_text = RichTextBrowser()
        self.thinking_text.setPlaceholderText("Reasoning-capable models may stream thinking text here.")
        thinking_layout.addWidget(self.thinking_text)
        pane_splitter.addWidget(thinking_container)

        sources_container = QWidget()
        sources_layout = QVBoxLayout(sources_container)
        sources_layout.setContentsMargins(0, 0, 0, 0)
        sources_layout.setSpacing(6)
        sources_label = QLabel("Validated Sources")
        sources_label.setProperty("sectionHeader", True)
        sources_layout.addWidget(sources_label)
        self.sources_status_label = QLabel(self.sources_status_var.get())
        self.sources_status_label.setWordWrap(True)
        sources_layout.addWidget(self.sources_status_label)
        self.sources_text = RichTextBrowser()
        self.sources_text.setPlaceholderText("Validated source links used by the current assistant answer will appear here.")
        sources_layout.addWidget(self.sources_text)
        pane_splitter.addWidget(sources_container)
        pane_splitter.setSizes([360, 220])

        return panel

    def show(self) -> None:  # type: ignore[override]
        """Show the window and apply the first-pass focus behavior."""
        super().show()
        self.input_text.setFocus()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Stop future UI polling once the window is closing."""
        self._is_closing = True
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        """Send the prompt when Ctrl+Enter is pressed anywhere relevant."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._send_current_input()
            event.accept()
            return
        super().keyPressEvent(event)

    def safe_ui_call(self, callback) -> None:
        """Schedule a UI callback on the main Qt thread when the window is alive."""
        if self._is_closing:
            return

        def guarded_callback() -> None:
            if self._is_closing:
                return
            callback()

        self._ui_callback_bridge.invoke.emit(guarded_callback)

    def set_status(self, text: str) -> None:
        """Show one short status line in the status bar."""
        self.status_var.set(text)
        self.status_bar.showMessage(text)

    def set_busy_state(self, is_busy: bool, status_text: str) -> None:
        """Toggle busy widgets so the user cannot start conflicting actions."""
        self.streaming_in_progress = is_busy
        self.send_button.setEnabled(not is_busy)
        self.input_text.setReadOnly(is_busy)
        self.settings_button.setEnabled(not is_busy)
        self.model_combo.setEnabled(not is_busy)
        self.new_session_button.setEnabled(not is_busy)
        self.rename_session_button.setEnabled(not is_busy)
        self.delete_session_button.setEnabled(not is_busy)
        self.session_list.setEnabled(not is_busy)
        self.refresh_models_button.setEnabled(not is_busy)
        self.set_status(status_text)

        if is_busy:
            self._stream_started_monotonic = time.monotonic()
            self._stream_output_token_estimate = 0
            self._current_assistant_stream_open = False
            self.active_stream_has_thinking = False
            self.thinking_text.clear()
            self._latest_validated_sources = []
            self._latest_invalid_citations = []
            self._update_sources_panel([])
        else:
            self._current_assistant_stream_open = False
            self.input_text.setFocus()

    def render_full_conversation(self, messages: Iterable[ChatMessage]) -> None:
        """Replace the transcript and thinking panes from complete session state."""
        self.chat_history.clear()
        self.thinking_text.clear()
        self._current_assistant_stream_open = False
        self.active_stream_has_thinking = False
        latest_sources: list[dict] = []

        for message in messages:
            self._append_message_to_transcript(message)
            if message.thinking.strip():
                self.active_stream_has_thinking = True
                self._append_text_block(self.thinking_text, f"[{message.role}]\n{message.thinking.strip()}\n\n")
            if getattr(message, "sources", None):
                latest_sources = [item for item in message.sources if isinstance(item, dict)]

        self._latest_validated_sources = latest_sources
        self._update_sources_panel(latest_sources)
        self._scroll_text_to_end(self.chat_history)
        self._scroll_text_to_end(self.thinking_text)

    def append_message(self, message: ChatMessage) -> None:
        """Append one completed message to the transcript and any thinking trace."""
        self._current_assistant_stream_open = False
        self._append_message_to_transcript(message)
        if message.thinking.strip():
            self.active_stream_has_thinking = True
            self._append_text_block(self.thinking_text, f"[{message.role}]\n{message.thinking.strip()}\n\n")
            self._scroll_text_to_end(self.thinking_text)
        if getattr(message, "sources", None):
            self._latest_validated_sources = [item for item in message.sources if isinstance(item, dict)]
            self._update_sources_panel(self._latest_validated_sources)

    def begin_assistant_stream(self) -> None:
        """Prepare transcript widgets and throughput tracking for a new stream.

        This method restores the controller-facing contract used by the
        Tkinter version so the existing ChatController can start a streamed
        assistant reply without needing any PyQt-specific logic.
        """
        self.streaming_in_progress = True
        self.active_stream_has_thinking = False
        self._stream_started_monotonic = time.monotonic()
        self._stream_output_token_estimate = 0
        self._current_assistant_stream_open = False
        self._set_tokens_per_second_text("Tok/s: measuring...")
        self.thinking_text.clear()
        self._latest_validated_sources = []
        self._latest_invalid_citations = []
        self._update_sources_panel([])

        if self.chat_history.toPlainText() and not self.chat_history.toPlainText().endswith("\n\n"):
            self._append_text_block(self.chat_history, "\n\n")
        self._append_text_block(self.chat_history, "Assistant\n---------\n")
        self._current_assistant_stream_open = True
        self._scroll_text_to_end(self.chat_history)

    def finalize_assistant_stream(
        self,
        final_content: str,
        final_thinking: str,
        performance_stats: dict | None = None,
    ) -> None:
        """Finish the streamed assistant block, validate sources, and re-render citations.

        Streaming uses simple plain-text insertion for responsiveness. Once the
        final answer is known, this method re-renders the persisted transcript so
        validated citations become clickable links and the sources pane reflects
        the latest assistant answer.
        """
        _ = final_content
        if not self.streaming_in_progress:
            return

        if self.chat_history.toPlainText() and not self.chat_history.toPlainText().endswith("\n\n"):
            self._append_text_block(self.chat_history, "\n\n")
        self._scroll_text_to_end(self.chat_history)
        self.streaming_in_progress = False
        self._current_assistant_stream_open = False

        stats = performance_stats or {}
        self._latest_validated_sources = [item for item in stats.get("sources_used", []) if isinstance(item, dict)]
        self._latest_invalid_citations = [str(item) for item in stats.get("invalid_citations", []) if str(item).strip()]
        self._update_sources_panel(self._latest_validated_sources, self._latest_invalid_citations)
        self._apply_final_performance_stats(stats)
        self._refresh_transcript_from_controller()

        if final_thinking.strip():
            self.active_stream_has_thinking = True
            self._scroll_text_to_end(self.thinking_text)

    def highlight_session(self, session_id: str) -> None:
        """Select the visible session row that matches the controller session id."""
        self._suspend_session_signal = True
        try:
            for index in range(self.session_list.count()):
                item = self.session_list.item(index)
                if item is not None and str(item.data(Qt.ItemDataRole.UserRole) or "") == session_id:
                    self.session_list.setCurrentItem(item)
                    break
        finally:
            self._suspend_session_signal = False

    def append_stream_chunk(self, chunk_type: str, text_chunk: str) -> None:
        """Append streamed assistant content or thinking text incrementally."""
        if not text_chunk:
            return

        if chunk_type == "thinking":
            self.active_stream_has_thinking = True
            self._append_text_block(self.thinking_text, text_chunk)
            self._scroll_text_to_end(self.thinking_text)
            return

        if not self._current_assistant_stream_open:
            if self.chat_history.toPlainText() and not self.chat_history.toPlainText().endswith("\n\n"):
                self._append_text_block(self.chat_history, "\n\n")
            self._append_text_block(self.chat_history, "Assistant\n---------\n")
            self._current_assistant_stream_open = True

        self._append_text_block(self.chat_history, text_chunk)
        self._scroll_text_to_end(self.chat_history)

        self._stream_output_token_estimate += self._estimate_token_count(text_chunk)
        self._refresh_estimated_tokens_per_second()

    def _estimate_token_count(self, text: str) -> int:
        """Return a rough token estimate for live throughput display."""
        pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return len(pieces)

    def _refresh_estimated_tokens_per_second(self) -> None:
        """Update the live token/second label using local wall-clock estimation."""
        if not self.streaming_in_progress or self._stream_started_monotonic <= 0:
            return
        elapsed = max(time.monotonic() - self._stream_started_monotonic, 0.001)
        if self._stream_output_token_estimate <= 0:
            token_text = "Tok/s: measuring..."
        else:
            rate = self._stream_output_token_estimate / elapsed
            token_text = f"Tok/s: {rate:.1f} (est)"

        if hasattr(self, "_set_tokens_per_second_text"):
            self._set_tokens_per_second_text(token_text)
            return
        if hasattr(self, "tokens_per_second_var"):
            self.tokens_per_second_var.set(token_text)
        if hasattr(self, "tokens_per_second_label"):
            self.tokens_per_second_label.setText(token_text)
        if hasattr(self, "footer_tokens_per_second_label"):
            self.footer_tokens_per_second_label.setText(token_text)

    def _apply_final_performance_stats(self, performance_stats: dict) -> None:
        """Replace the live estimate with final metrics when they are available."""
        final_rate = float(performance_stats.get("tokens_per_second", 0.0) or 0.0)
        final_tokens = int(performance_stats.get("output_tokens", 0) or 0)
        source = str(performance_stats.get("throughput_source", "") or "")

        if final_rate > 0:
            suffix = "" if source == "ollama_eval" else " (est)"
            token_hint = f", {final_tokens} tok" if final_tokens > 0 else ""
            token_text = f"Tok/s: {final_rate:.1f}{suffix}{token_hint}"
            if hasattr(self, "_set_tokens_per_second_text"):
                self._set_tokens_per_second_text(token_text)
                return
            if hasattr(self, "tokens_per_second_var"):
                self.tokens_per_second_var.set(token_text)
            if hasattr(self, "tokens_per_second_label"):
                self.tokens_per_second_label.setText(token_text)
            if hasattr(self, "footer_tokens_per_second_label"):
                self.footer_tokens_per_second_label.setText(token_text)
            return

        self._refresh_estimated_tokens_per_second()

    def update_session_title(self, title: str) -> None:
        """Refresh the visible title of the active session."""
        self.session_title_var.set(title)
        self.session_title_label.setText(title)

    def _load_model_choices(self, models: list[str], selected_model: str, selected_reranker: str = "") -> None:
        """Populate the primary-model and reranker combos while preserving saved selections.

        Stability note:
            The saved primary model or reranker may be temporarily unavailable
            from a refresh result. This helper preserves the selection visibly so
            later saves do not silently blank it. The reranker combo also keeps a
            dedicated disabled option at index 0.
        """
        normalized_models = [str(model).strip() for model in models if str(model).strip()]
        cleaned_selected = str(selected_model).strip()
        cleaned_reranker = str(selected_reranker).strip()
        if cleaned_selected and cleaned_selected not in normalized_models:
            normalized_models.insert(0, cleaned_selected)
        if cleaned_reranker and cleaned_reranker not in normalized_models:
            normalized_models.append(cleaned_reranker)

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(normalized_models)
        if cleaned_selected and self.model_combo.findText(cleaned_selected) >= 0:
            self.model_combo.setCurrentText(cleaned_selected)
        elif self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
        self.model_combo.blockSignals(False)

        self.reranker_combo.blockSignals(True)
        self.reranker_combo.clear()
        self.reranker_combo.addItem("Disabled")
        self.reranker_combo.setItemData(0, "")
        for model_name in normalized_models:
            self.reranker_combo.addItem(model_name)
        if cleaned_reranker and self.reranker_combo.findText(cleaned_reranker) >= 0:
            self.reranker_combo.setCurrentText(cleaned_reranker)
        else:
            self.reranker_combo.setCurrentIndex(0)
        self.reranker_combo.blockSignals(False)

    def refresh_model_display(self, config: dict) -> None:
        """Refresh the model summary text and main-page search toggle state."""
        model_name = str(config.get("model", "")).strip() or "No model selected"
        thinking_mode = "On" if config.get("thinking_mode", True) else "Off"
        thinking_level = str(config.get("thinking_level", "medium")).strip().lower()
        web_search = "On" if config.get("enable_web_search", False) else "Off"
        timeout_seconds = config.get("timeout_seconds", 300)
        reranker_name = str(config.get("reranker_model", "")).strip() or "Disabled"
        self.model_detail_label.setText(
            f"Selected: {model_name}\nReranker: {reranker_name}\nThinking: {thinking_mode} ({thinking_level})\n"
            f"Web search: {web_search}\nTimeout: {timeout_seconds}s"
        )

        self._suspend_model_signal = True
        current_model = str(config.get("model", ""))
        current_reranker = str(config.get("reranker_model", ""))
        self.model_var.set(current_model)
        self.reranker_var.set(current_reranker)
        existing_models = [self.model_combo.itemText(index) for index in range(self.model_combo.count())]
        self._load_model_choices(existing_models, current_model, current_reranker)
        self._suspend_model_signal = False

        self.main_web_search_checkbox.blockSignals(True)
        self.main_web_search_checkbox.setChecked(bool(config.get("enable_web_search", False)))
        self.main_web_search_checkbox.blockSignals(False)

    def update_model_choices(self, models: list[str], selected_model: str, selected_reranker: str = "") -> None:
        """Replace the primary-model and reranker dropdown contents while keeping selections.

        The reranker selector intentionally reuses the same Ollama model list so
        the user can manage both stages from one runtime instead of a second
        model-management toolchain.
        """
        self._suspend_model_signal = True
        self.model_var.set(selected_model)
        self.reranker_var.set(selected_reranker)
        self._load_model_choices(models, selected_model, selected_reranker)
        self._suspend_model_signal = False

    def update_session_list(self, sessions: list[ConversationSession], active_session_id: str) -> None:
        """Replace the session list with controller-provided session metadata."""
        self._suspend_session_signal = True
        self.session_list.clear()
        active_item: QListWidgetItem | None = None
        for session in sessions:
            label = session.title.strip() or "Untitled Session"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, session.session_id)
            item.setToolTip(f"Model: {session.model_name or 'unset'}")
            self.session_list.addItem(item)
            if session.session_id == active_session_id:
                active_item = item
        if active_item is not None:
            self.session_list.setCurrentItem(active_item)
        self._suspend_session_signal = False

    def open_settings_window(self) -> None:
        """Open the settings dialog, reusing the existing instance when possible."""
        if self._settings_window is not None and self._settings_window.isVisible():
            self._settings_window.raise_()
            self._settings_window.activateWindow()
            return

        self._settings_window = SettingsWindow(
            controller=self.controller,
            config_service=self.config_service,
            on_save_callback=self.refresh_model_display,
            parent=self,
        )
        self._settings_window.show()

    def _handle_model_selected(self, model_name: str) -> None:
        """Propagate a user-chosen model from the dropdown into saved config."""
        if self._suspend_model_signal:
            return
        cleaned = model_name.strip()
        if not cleaned:
            return
        try:
            self.controller.update_selected_model(cleaned)
        except RuntimeError as exc:
            self._show_warning("Model Busy", str(exc))


    def _handle_reranker_selected(self, model_name: str) -> None:
        """Persist the reranker model selected beside the primary model selector."""
        if self._suspend_model_signal:
            return
        cleaned = model_name.strip()
        if cleaned == "Disabled":
            cleaned = ""
        try:
            self.controller.update_selected_reranker(cleaned)
        except RuntimeError as exc:
            self._show_warning("Reranker Busy", str(exc))

    def _handle_main_web_search_toggle(self, state: int) -> None:
        """Allow quick enabling or disabling of web search from the main page."""
        try:
            saved = self.controller.save_settings({"enable_web_search": bool(state)})
        except RuntimeError as exc:
            self._show_warning("Settings Busy", str(exc))
            self.main_web_search_checkbox.blockSignals(True)
            self.main_web_search_checkbox.setChecked(not bool(state))
            self.main_web_search_checkbox.blockSignals(False)
            return
        self.refresh_model_display(saved)
        self.set_status("Web search setting updated.")

    def _handle_new_session(self) -> None:
        """Ask the controller to create a fresh blank session."""
        try:
            self.controller.create_new_session()
        except RuntimeError as exc:
            self._show_warning("Session Busy", str(exc))

    def _handle_session_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None = None) -> None:
        """Load the selected session when the user changes the active list item."""
        del previous
        if self._suspend_session_signal or current is None:
            return
        session_id = str(current.data(Qt.ItemDataRole.UserRole) or "")
        if not session_id:
            return
        try:
            self.controller.load_session(session_id)
        except RuntimeError as exc:
            self._show_warning("Session Busy", str(exc))

    def _rename_session(self) -> None:
        """Prompt for a new title and send it to the controller."""
        current_title = self.session_title_label.text().strip() or "Untitled Session"
        new_title, accepted = QInputDialog.getText(self, "Rename Session", "New session title:", text=current_title)
        if not accepted:
            return
        cleaned = new_title.strip()
        if not cleaned:
            self._show_warning("Rename Session", "Session title cannot be empty.")
            return
        try:
            self.controller.rename_active_session(cleaned)
        except RuntimeError as exc:
            self._show_warning("Session Busy", str(exc))

    def _delete_selected_session(self) -> None:
        """Delete the currently selected session after a confirmation dialog."""
        item = self.session_list.currentItem()
        if item is None:
            self._show_warning("Delete Session", "Select a session to delete.")
            return
        session_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not session_id:
            return
        result = QMessageBox.question(
            self,
            "Delete Session",
            "Delete the selected session? This removes its saved transcript from disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return
        try:
            self.controller.delete_session(session_id)
        except RuntimeError as exc:
            self._show_warning("Session Busy", str(exc))

    def _send_current_input(self) -> None:
        """Send the current prompt text to the controller."""
        text = self.input_text.toPlainText().strip()
        if not text:
            self._show_warning("Empty Prompt", "Type a prompt before sending.")
            return
        self.input_text.clear()
        self.thinking_text.clear()
        self.active_stream_has_thinking = False
        try:
            self.controller.send_user_message(text)
        except RuntimeError as exc:
            self._show_warning("Generation Busy", str(exc))
            self.input_text.setPlainText(text)

    def _toggle_thinking_visibility(self) -> None:
        """Show or hide the thinking panel without altering underlying data."""
        is_visible = self.thinking_text.isVisible()
        self.thinking_text.setVisible(not is_visible)
        self.thinking_visible_var.set(not is_visible)
        self.toggle_thinking_button.setText("Show" if is_visible else "Hide")

    def request_immediate_gpu_metrics_refresh(self) -> None:
        """Force one immediate GPU metrics refresh outside the normal timer loop.

        This is used after actions such as a manual VRAM flush so the footer can
        reflect the post-action state right away instead of waiting for the next
        scheduled poll. If a poll is already running, the request is ignored so
        overlapping metric subprocesses are not spawned.
        """
        if self._is_closing or self._gpu_poll_inflight:
            return
        self._gpu_poll_inflight = True
        generation = self._gpu_poll_generation = self._gpu_poll_generation + 1
        run_in_background(lambda: self._refresh_gpu_metrics_background(generation))

    def _schedule_gpu_metrics_refresh(self) -> None:
        """Poll GPU metrics on a timer without blocking the Qt UI thread."""
        if self._is_closing:
            return
        if not self._gpu_poll_inflight:
            self._gpu_poll_inflight = True
            generation = self._gpu_poll_generation = self._gpu_poll_generation + 1
            run_in_background(lambda: self._refresh_gpu_metrics_background(generation))
        QTimer.singleShot(self.GPU_POLL_INTERVAL_MS, self._schedule_gpu_metrics_refresh)

    def _refresh_gpu_metrics_background(self, generation: int) -> None:
        """Fetch GPU metrics in a worker thread and push them back safely.

        Snapshot compatibility note:
            The controller now normalizes both canonical metric keys
            (``gpu_percent`` / ``vram_percent``) and the older UI compatibility
            keys (``gpu_percent_text`` / ``vram_percent_text``). This refresh
            helper still accepts either naming style so footer readouts do not
            regress if one side of the interface changes first.
        """
        try:
            metrics = self.controller.get_gpu_metrics_snapshot()
        except Exception:
            metrics = {}
        finally:
            self._gpu_poll_inflight = False

        if generation != self._gpu_poll_generation or self._is_closing:
            return

        def apply_metrics() -> None:
            gpu_value = metrics.get('gpu_percent_text') or metrics.get('gpu_percent') or '--'
            vram_value = metrics.get('vram_percent_text') or metrics.get('vram_percent') or '--'
            gpu_text = f"GPU%: {gpu_value}"
            vram_text = f"VRAM%: {vram_value}"
            self._set_gpu_text(gpu_text)
            self._set_vram_text(vram_text)
            if not self.streaming_in_progress and metrics.get("tokens_per_second_text"):
                token_text = f"Tok/s: {metrics.get('tokens_per_second_text')}"
                self._set_tokens_per_second_text(token_text)

        self.safe_ui_call(apply_metrics)

    def _set_gpu_text(self, text: str) -> None:
        """Update both visible GPU labels and the legacy text proxy."""
        self.gpu_percent_var.set(text)
        self.gpu_percent_label.setText(text)
        self.footer_gpu_percent_label.setText(text)

    def _set_vram_text(self, text: str) -> None:
        """Update both visible VRAM labels and the legacy text proxy."""
        self.vram_percent_var.set(text)
        self.vram_percent_label.setText(text)
        self.footer_vram_percent_label.setText(text)

    def _set_tokens_per_second_text(self, text: str) -> None:
        """Update both visible throughput labels and the legacy text proxy."""
        self.tokens_per_second_var.set(text)
        if hasattr(self, "tokens_per_second_label"):
            self.tokens_per_second_label.setText(text)
        if hasattr(self, "footer_tokens_per_second_label"):
            self.footer_tokens_per_second_label.setText(text)

    def _refresh_transcript_from_controller(self) -> None:
        """Re-render from controller session state so finalized citations become clickable."""
        messages = getattr(getattr(self, 'controller', None), 'active_session', None)
        if messages is None:
            return
        conversation = getattr(messages, 'messages', None)
        if conversation is None:
            return
        self.render_full_conversation(conversation)

    def _update_sources_panel(self, sources: list[dict], invalid_citations: list[str] | None = None) -> None:
        """Render the validated-source pane for the latest assistant answer."""
        invalid_citations = invalid_citations or []
        self.sources_text.clear()
        if not sources:
            status = "Sources: no validated citations in the current answer."
            if invalid_citations:
                status += f" Ignored invalid citation IDs: {', '.join(invalid_citations)}."
            self.sources_status_var.set(status)
            if hasattr(self, 'sources_status_label'):
                self.sources_status_label.setText(status)
            return

        status = f"Sources: {len(sources)} validated against returned SearXNG results."
        if invalid_citations:
            status += f" Ignored invalid citation IDs: {', '.join(invalid_citations)}."
        self.sources_status_var.set(status)
        if hasattr(self, 'sources_status_label'):
            self.sources_status_label.setText(status)

        blocks = []
        for source in sources:
            source_id = html.escape(str(source.get('source_id', '')))
            title = html.escape(str(source.get('title') or source.get('fetched_title') or source.get('domain') or source_id))
            link = html.escape(str(source.get('citation_url') or source.get('url') or ''), quote=True)
            raw_url = html.escape(str(source.get('url') or ''), quote=True)
            domain = html.escape(str(source.get('domain') or ''))
            snippet = html.escape(str(source.get('highlight_text') or source.get('snippet') or ''))
            blocks.append(
                f'<div style="margin-bottom: 14px;">'
                f'<div><b>{source_id}</b> — <a href="{link}">{title}</a></div>'
                f'<div><a href="{raw_url}">{domain or raw_url}</a></div>'
                f'<div>{snippet}</div>'
                f'<div style="color:#666;">Validated against real SearXNG result list for this answer.</div>'
                f'</div>'
            )
        self.sources_text.setHtml(''.join(blocks))

    def _append_message_to_transcript(self, message: ChatMessage) -> None:
        """Render one completed chat message with readable HTML and validated citations."""
        role_label = html.escape(message.role.capitalize())
        body_html = self._format_message_body_html(message)
        separator = html.escape('-' * max(len(message.role), 4))
        block = (
            f'<div style="margin-bottom: 16px;">'
            f'<div><b>{role_label}</b></div>'
            f'<div style="color: #666;">{separator}</div>'
            f'<div>{body_html}</div>'
            f'</div>'
        )
        self._append_html_block(self.chat_history, block)
        self._scroll_text_to_end(self.chat_history)

    def _format_message_body_html(self, message: ChatMessage) -> str:
        """Convert message content into safe HTML with clickable validated citations."""
        escaped = html.escape(message.content.strip() or "").replace("\n", "<br>")
        source_lookup = {
            str(item.get("source_id", "")).strip(): item
            for item in getattr(message, "sources", [])
            if isinstance(item, dict)
        }

        def replace_citation(match: re.Match[str]) -> str:
            source_id = match.group(1)
            source = source_lookup.get(source_id)
            if not source:
                return f'<span>[{html.escape(source_id)}]</span>'
            link = html.escape(str(source.get("citation_url") or source.get("url") or ""), quote=True)
            title = html.escape(str(source.get("title") or source.get("domain") or source_id))
            return f'<a href="{link}" title="{title}">[{html.escape(source_id)}]</a>'

        return re.sub(r"\[(S\d{1,3})\]", replace_citation, escaped)

    def _append_html_block(self, widget: QTextEdit, html_block: str) -> None:
        """Append HTML to the end of a rich text widget without resetting state."""
        cursor = widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html_block)
        cursor.insertHtml("<br>")
        widget.setTextCursor(cursor)

    def _append_text_block(self, widget: QTextEdit, text: str) -> None:
        """Append plain text to the end of a QTextEdit without resetting state."""
        cursor = widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        widget.setTextCursor(cursor)

    def _scroll_text_to_end(self, widget: QTextEdit) -> None:
        """Keep the given text widget pinned to its latest appended content."""
        scrollbar = widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _show_warning(self, title: str, text: str) -> None:
        """Show a non-fatal warning dialog for user-correctable issues."""
        QMessageBox.warning(self, title, text)
