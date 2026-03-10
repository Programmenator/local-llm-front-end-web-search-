"""
ui/main_window.py

Purpose:
    Primary desktop chat interface.

What this file does:
    - Builds the main application window.
    - Displays conversation history.
    - Accepts user input.
    - Exposes the settings button in the side panel.
    - Provides a lightweight conversation session manager.
    - Streams assistant text live into the chat window.
    - Shows optional thinking output inside a dedicated collapsible panel.
    - Lets the user switch installed models from a dropdown.
    - Shows live GPU% and VRAM% readings in the main UI.
    - Shows a token/second performance meter for the currently streaming model
      response and preserves the last completed rate.
    - Polls GPU metrics in a background thread so the Tkinter interface stays
      responsive while ROCm commands run.
    - Lets the user resize the chat, thinking, and input text regions with
      draggable pane dividers.
    - Lets the user zoom the text widgets with Ctrl + mouse wheel for easier
      reading on dense or high-DPI displays.
    - Mirrors the web-search enabled/disabled setting directly on the main
      screen so the user can tell at a glance whether internet tools are active.

How this file fits into the system:
    This is the user-facing shell of the application. It delegates application
    logic to the controller and focuses on widget layout and presentation.
"""

from __future__ import annotations

import re
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, scrolledtext, simpledialog, ttk
from typing import Dict, List

from controllers.chat_controller import ChatController
from models.chat_message import ChatMessage
from models.conversation_session import ConversationSession
from services.config_service import ConfigService
from ui.settings_window import SettingsWindow
from utils.threading_helpers import run_in_background


class MainWindow:
    """Main desktop window for chatting with the local Ollama instance."""

    GPU_POLL_INTERVAL_MS = 1500
    MIN_TEXT_FONT_SIZE = 8
    MAX_TEXT_FONT_SIZE = 32
    DEFAULT_TEXT_FONT_SIZE = 11

    def __init__(self, controller: ChatController, config_service: ConfigService) -> None:
        """Create the root window and build the interface layout."""
        self.controller = controller
        self.config_service = config_service
        self.root = tk.Tk()
        self.root.title("Local LLM Front End")
        self.root.geometry("1320x820")
        self.root.minsize(1100, 700)

        self.status_var = tk.StringVar(value="Ready")
        self.model_var = tk.StringVar()
        self.session_title_var = tk.StringVar(value="Untitled Session")
        self.thinking_visible_var = tk.BooleanVar(value=False)
        self.internet_search_enabled_var = tk.BooleanVar(value=False)
        self.internet_search_status_var = tk.StringVar(value="Internet Search: OFF")
        self.gpu_percent_var = tk.StringVar(value='GPU%: --')
        self.vram_percent_var = tk.StringVar(value='VRAM%: --')
        self.tokens_per_second_var = tk.StringVar(value='Tok/s: --')

        self.session_index_to_id: Dict[int, str] = {}
        self.active_stream_has_thinking = False
        self.streaming_in_progress = False
        self._gpu_poll_inflight = False
        self._gpu_poll_generation = 0
        self._stream_started_monotonic = 0.0
        self._stream_output_token_estimate = 0
        self._gpu_poll_after_id: str | None = None
        self._is_closing = False
        self._thinking_pane_added = True
        self._text_zoom_size = self.DEFAULT_TEXT_FONT_SIZE
        self._text_widgets: List[tk.Text | tk.Listbox] = []
        self._text_font = tkfont.nametofont("TkTextFont").copy()
        self._text_font.configure(size=self._text_zoom_size)

        self._apply_initial_style()
        self._build_layout()
        self.refresh_model_display(self.config_service.get_config())
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)
        self._schedule_gpu_metrics_refresh()

    def _apply_initial_style(self) -> None:
        """Apply a simple ttk theme for a clean lightweight appearance."""
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

    def _build_layout(self) -> None:
        """Create all primary widgets and arrange them in the main window."""
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, padding=12)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(6, weight=1)

        ttk.Label(sidebar, text="Controls", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.settings_button = ttk.Button(sidebar, text="Settings", command=self.open_settings_window)
        self.settings_button.grid(row=1, column=0, sticky="ew", pady=(10, 6))

        ttk.Label(sidebar, text="Model", font=("TkDefaultFont", 10, "bold")).grid(row=2, column=0, sticky="w", pady=(10, 4))
        self.model_combo = ttk.Combobox(sidebar, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=3, column=0, sticky="ew")
        self.model_combo.bind("<<ComboboxSelected>>", self._handle_model_selected)

        self.model_detail_label = ttk.Label(sidebar, text="", wraplength=220, justify="left")
        self.model_detail_label.grid(row=4, column=0, sticky="w", pady=(8, 10))

        gpu_frame = ttk.LabelFrame(sidebar, text="GPU Live Readout", padding=8)
        gpu_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        gpu_frame.columnconfigure(0, weight=1)
        ttk.Label(gpu_frame, textvariable=self.gpu_percent_var).grid(row=0, column=0, sticky='w')
        ttk.Label(gpu_frame, textvariable=self.vram_percent_var).grid(row=1, column=0, sticky='w', pady=(4, 0))
        ttk.Label(gpu_frame, textvariable=self.tokens_per_second_var).grid(row=2, column=0, sticky='w', pady=(4, 0))

        sessions_frame = ttk.LabelFrame(sidebar, text="Sessions", padding=8)
        sessions_frame.grid(row=6, column=0, sticky="nsew")
        sessions_frame.columnconfigure(0, weight=1)
        sessions_frame.rowconfigure(2, weight=1)

        ttk.Label(sessions_frame, textvariable=self.session_title_var, wraplength=220).grid(row=0, column=0, sticky="w")
        session_button_row = ttk.Frame(sessions_frame)
        session_button_row.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self.new_session_button = ttk.Button(session_button_row, text="New", command=self._handle_new_session)
        self.new_session_button.pack(side="left")
        self.rename_session_button = ttk.Button(session_button_row, text="Rename", command=self._rename_session)
        self.rename_session_button.pack(side="left", padx=(6, 0))
        self.delete_session_button = ttk.Button(session_button_row, text="Delete", command=self._delete_selected_session)
        self.delete_session_button.pack(side="left", padx=(6, 0))

        self.session_listbox = tk.Listbox(sessions_frame, height=18, font=self._text_font)
        self.session_listbox.grid(row=2, column=0, sticky="nsew")
        self.session_listbox.bind("<<ListboxSelect>>", self._handle_session_selected)
        self._register_zoom_widget(self.session_listbox)

        content = ttk.Frame(self.root, padding=12)
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        header = ttk.Frame(content)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(1, weight=1)
        header.columnconfigure(3, weight=1)

        ttk.Label(header, text="Conversation", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.internet_search_toggle = ttk.Checkbutton(
            header,
            textvariable=self.internet_search_status_var,
            variable=self.internet_search_enabled_var,
            command=self._handle_main_search_toggle,
        )
        self.internet_search_toggle.grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.refresh_models_button = ttk.Button(header, text="Refresh Models", command=self.controller.request_model_refresh)
        self.refresh_models_button.grid(row=0, column=4, sticky="e")

        self.content_paned = ttk.Panedwindow(content, orient=tk.VERTICAL)
        self.content_paned.grid(row=1, column=0, sticky="nsew")

        chat_frame = ttk.LabelFrame(self.content_paned, text="Chat Transcript", padding=8)
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        self.chat_history = scrolledtext.ScrolledText(chat_frame, wrap="word", state="disabled", height=24, font=self._text_font)
        self.chat_history.grid(row=0, column=0, sticky="nsew")
        self._register_zoom_widget(self.chat_history)
        self.content_paned.add(chat_frame, weight=5)

        self.thinking_frame = ttk.LabelFrame(self.content_paned, text="Thinking Stream", padding=8)
        self.thinking_frame.columnconfigure(0, weight=1)
        self.thinking_frame.rowconfigure(1, weight=1)

        toggle_row = ttk.Frame(self.thinking_frame)
        toggle_row.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(
            toggle_row,
            text="Show thinking panel",
            variable=self.thinking_visible_var,
            command=self._toggle_thinking_visibility,
        ).pack(side="left")

        self.thinking_text = scrolledtext.ScrolledText(self.thinking_frame, wrap="word", state="disabled", height=10, font=self._text_font)
        self.thinking_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self._register_zoom_widget(self.thinking_text)
        self.content_paned.add(self.thinking_frame, weight=2)

        input_frame = ttk.LabelFrame(self.content_paned, text="Input", padding=8)
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)

        self.input_text = tk.Text(input_frame, height=6, wrap="word", font=self._text_font)
        self.input_text.grid(row=0, column=0, sticky="nsew")
        self.input_text.bind("<Control-Return>", self._handle_send_shortcut)
        self._register_zoom_widget(self.input_text)

        send_button = ttk.Button(input_frame, text="Send", command=self._send_current_input)
        send_button.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self.send_button = send_button
        self.content_paned.add(input_frame, weight=1)

        self._toggle_thinking_visibility()

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief="sunken")
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _register_zoom_widget(self, widget: tk.Text | tk.Listbox) -> None:
        """Track one text-bearing widget and bind Ctrl+wheel zoom handlers.

        Linux Tk builds often emit Button-4/Button-5 wheel events instead of a
        MouseWheel delta value, so both forms are bound here.
        """
        self._text_widgets.append(widget)
        widget.bind("<Control-MouseWheel>", self._handle_text_zoom)
        widget.bind("<Control-Button-4>", self._handle_text_zoom)
        widget.bind("<Control-Button-5>", self._handle_text_zoom)

    def _handle_text_zoom(self, event: tk.Event) -> str:
        """Increase or decrease text-widget font size from a Ctrl+wheel gesture."""
        direction = 0
        delta = getattr(event, "delta", 0)
        if delta > 0 or getattr(event, "num", None) == 4:
            direction = 1
        elif delta < 0 or getattr(event, "num", None) == 5:
            direction = -1
        if direction == 0:
            return "break"
        self._text_zoom_size = max(
            self.MIN_TEXT_FONT_SIZE,
            min(self.MAX_TEXT_FONT_SIZE, self._text_zoom_size + direction),
        )
        self._text_font.configure(size=self._text_zoom_size)
        return "break"

    def start(self) -> None:
        """Start the Tkinter event loop."""
        self.root.mainloop()

    def _handle_close(self) -> None:
        """Shut down the Tkinter window without leaving UI callbacks behind.

        Stability note:
            Background GPU polling and worker-thread UI callbacks can otherwise
            try to target widgets after the root window has been destroyed. This
            method marks the UI as closing, cancels the repeating GPU timer, and
            then destroys the root safely.
        """
        self._is_closing = True
        if self._gpu_poll_after_id is not None:
            try:
                self.root.after_cancel(self._gpu_poll_after_id)
            except tk.TclError:
                pass
            self._gpu_poll_after_id = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def open_settings_window(self) -> None:
        """Open the settings popup populated with current configuration values."""
        try:
            SettingsWindow(
                parent=self.root,
                controller=self.controller,
                current_config=self.controller.get_config(),
                on_save_callback=self._on_settings_saved,
            )
        except RuntimeError as exc:
            messagebox.showerror("Settings", str(exc))

    def _on_settings_saved(self, new_config: dict) -> None:
        """Update main-window state that depends on saved configuration."""
        self.refresh_model_display(new_config)
        self.set_status("Settings updated")

    def refresh_model_display(self, config: dict) -> None:
        """Refresh the sidebar label showing the active model and search state."""
        self.model_var.set(str(config.get("model", "")))
        self.internet_search_enabled_var.set(bool(config.get("enable_web_search", False)))
        self._refresh_search_toggle_text()
        self.model_detail_label.config(
            text=(
                f"Connected target:\n{config.get('host')}:{config.get('port')}\n"
                f"Model: {config.get('model') or '[none selected]'}"
            )
        )

    def _refresh_search_toggle_text(self) -> None:
        """Update the main-window internet-search toggle label for clear visibility."""
        if self.internet_search_enabled_var.get():
            self.internet_search_status_var.set("Internet Search: ON")
        else:
            self.internet_search_status_var.set("Internet Search: OFF")

    def _handle_main_search_toggle(self) -> None:
        """Persist the main-window internet-search toggle immediately.

        The main-window toggle exists as a fast operational check and edit path.
        It writes directly to config through the controller so the user does not
        need to open Settings just to verify whether the model may search.
        """
        enabled = bool(self.internet_search_enabled_var.get())
        try:
            self.controller.update_web_search_enabled(enabled)
        except RuntimeError as exc:
            self.internet_search_enabled_var.set(not enabled)
            self._refresh_search_toggle_text()
            messagebox.showerror("Internet Search", str(exc))
            return
        self._refresh_search_toggle_text()
        self.set_status(f"Internet search {'enabled' if enabled else 'disabled'}")

    def update_model_choices(self, models: List[str], selected_model: str) -> None:
        """Populate the installed-model dropdown and preserve the active selection."""
        self.model_combo["values"] = models
        if selected_model:
            self.model_var.set(selected_model)
        elif models:
            self.model_var.set(models[0])

    def update_session_list(self, sessions: List[ConversationSession], active_session_id: str) -> None:
        """Rebuild the session manager listbox from saved session metadata."""
        self.session_listbox.delete(0, tk.END)
        self.session_index_to_id.clear()
        active_index = None

        for index, session in enumerate(sessions):
            display_name = f"{session.title} [{session.updated_at.strftime('%m/%d %I:%M %p')}]"
            self.session_listbox.insert(tk.END, display_name)
            self.session_index_to_id[index] = session.session_id
            if session.session_id == active_session_id:
                active_index = index

        if active_index is not None:
            self.session_listbox.selection_clear(0, tk.END)
            self.session_listbox.selection_set(active_index)
            self.session_listbox.see(active_index)

    def highlight_session(self, session_id: str) -> None:
        """Highlight a session row in the listbox by ID."""
        for index, mapped_id in self.session_index_to_id.items():
            if mapped_id == session_id:
                self.session_listbox.selection_clear(0, tk.END)
                self.session_listbox.selection_set(index)
                self.session_listbox.see(index)
                break

    def update_session_title(self, title: str) -> None:
        """Refresh the displayed title of the active session."""
        self.session_title_var.set(title)

    def render_full_conversation(self, messages: List[ChatMessage]) -> None:
        """Replace the transcript with the full active session message history."""
        self.chat_history.configure(state="normal")
        self.chat_history.delete("1.0", tk.END)
        self.chat_history.configure(state="disabled")
        self._clear_thinking_panel()
        self.thinking_visible_var.set(False)
        self._toggle_thinking_visibility()
        self.streaming_in_progress = False
        self.active_stream_has_thinking = False
        self.tokens_per_second_var.set("Tok/s: --")

        for message in messages:
            self.append_message(message)

    def append_message(self, message: ChatMessage) -> None:
        """Render one completed chat message into the conversation history widget."""
        time_label = message.timestamp.strftime("%I:%M:%S %p")
        role_label = message.role.upper()
        formatted = f"[{time_label}] {role_label}\n{message.content}\n\n"

        self.chat_history.configure(state="normal")
        self.chat_history.insert(tk.END, formatted)
        self.chat_history.see(tk.END)
        self.chat_history.configure(state="disabled")

        if message.role == "assistant" and message.thinking.strip():
            self._ensure_thinking_panel_visible()
            self._append_to_thinking_panel(
                f"[{time_label}] ASSISTANT THINKING\n{message.thinking}\n\n"
            )

    def begin_assistant_stream(self) -> None:
        """Prepare transcript widgets and throughput tracking for a new stream.

        Performance-meter note:
            The UI starts a local timer and rough token estimate here so the
            token/second label can move during streaming. When the final Ollama
            response includes true eval metrics, finalize_assistant_stream
            replaces the estimate with the more accurate server-reported value.
        """
        self.streaming_in_progress = True
        self.active_stream_has_thinking = False
        self._stream_started_monotonic = time.monotonic()
        self._stream_output_token_estimate = 0
        self.tokens_per_second_var.set("Tok/s: measuring...")
        self._clear_thinking_panel()

        self.chat_history.configure(state="normal")
        self.chat_history.insert(tk.END, f"[{self._now_label()}] ASSISTANT\n")
        self.chat_history.see(tk.END)
        self.chat_history.configure(state="disabled")

    def append_stream_chunk(self, chunk_type: str, text_chunk: str) -> None:
        """Append one streamed token chunk to the appropriate visible UI region.

        Channel routing:
            - content -> main transcript under the current assistant response
            - thinking -> collapsible thinking panel

        This separation is what allows the application to show reasoning traces
        without mixing them directly into the final visible answer text.
        """
        if chunk_type == "thinking":
            if not self.active_stream_has_thinking:
                self.active_stream_has_thinking = True
                self.thinking_visible_var.set(True)
                self._toggle_thinking_visibility()
                self._append_to_thinking_panel(f"[{self._now_label()}] ASSISTANT THINKING\n")
            self._append_to_thinking_panel(text_chunk)
            return

        self.chat_history.configure(state="normal")
        self.chat_history.insert(tk.END, text_chunk)
        self.chat_history.see(tk.END)
        self.chat_history.configure(state="disabled")
        self._stream_output_token_estimate += self._estimate_token_count(text_chunk)
        self._refresh_estimated_tokens_per_second()

    def finalize_assistant_stream(self, final_content: str, final_thinking: str, performance_stats: dict | None = None) -> None:
        """Finish the streamed assistant block and lock in final throughput stats.

        Args:
            final_content: Final assistant text. Present for API symmetry with
                the controller even though the live stream already rendered it.
            final_thinking: Final assistant thinking text.
            performance_stats: Optional final metrics from Ollama such as
                output_tokens and tokens_per_second. When absent, the UI keeps
                the local estimate captured during streaming.
        """
        _ = final_content
        if not self.streaming_in_progress:
            return

        current_end = self.chat_history.get("end-3c", "end-1c")
        self.chat_history.configure(state="normal")
        if current_end != "\n\n":
            self.chat_history.insert(tk.END, "\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.configure(state="disabled")
        self.streaming_in_progress = False
        self._apply_final_performance_stats(performance_stats or {})

        if final_thinking.strip():
            self._ensure_thinking_panel_visible()

    def _estimate_token_count(self, text: str) -> int:
        """Return a rough token estimate for live throughput display.

        Accuracy note:
            This is intentionally labeled as an estimate because true model token
            counts depend on the tokenizer used by the selected model. The goal
            here is to keep the UI moving during streaming. The final value is
            replaced with Ollama eval metrics when the server provides them.
        """
        pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return len(pieces)

    def _refresh_estimated_tokens_per_second(self) -> None:
        """Update the live token/second label using local wall-clock estimation."""
        if not self.streaming_in_progress or self._stream_started_monotonic <= 0:
            return
        elapsed = max(time.monotonic() - self._stream_started_monotonic, 0.001)
        if self._stream_output_token_estimate <= 0:
            self.tokens_per_second_var.set("Tok/s: measuring...")
            return
        rate = self._stream_output_token_estimate / elapsed
        self.tokens_per_second_var.set(f"Tok/s: {rate:.1f} (est)")

    def _apply_final_performance_stats(self, performance_stats: dict) -> None:
        """Replace the live estimate with final metrics when they are available."""
        final_rate = float(performance_stats.get("tokens_per_second", 0.0) or 0.0)
        final_tokens = int(performance_stats.get("output_tokens", 0) or 0)
        source = str(performance_stats.get("throughput_source", "") or "")

        if final_rate > 0:
            suffix = "" if source == "ollama_eval" else " (est)"
            token_hint = f", {final_tokens} tok" if final_tokens > 0 else ""
            self.tokens_per_second_var.set(f"Tok/s: {final_rate:.1f}{suffix}{token_hint}")
            return

        self._refresh_estimated_tokens_per_second()

    def set_busy_state(self, is_busy: bool, status_text: str) -> None:
        """Update status text and enable or disable input widgets as needed.

        Busy mode now also locks session-management and settings controls so the
        user cannot trigger state changes that conflict with an in-flight
        GenerationJob. The controller still enforces the same rule so the UI is
        convenience protection rather than the only safety layer.
        """
        self.status_var.set(status_text)
        if is_busy:
            self.input_text.configure(state="disabled")
            self.send_button.configure(state="disabled")
            self.model_combo.configure(state="disabled")
            self.settings_button.configure(state="disabled")
            self.new_session_button.configure(state="disabled")
            self.rename_session_button.configure(state="disabled")
            self.delete_session_button.configure(state="disabled")
            self.refresh_models_button.configure(state="disabled")
            self.session_listbox.configure(state="disabled")
            self.internet_search_toggle.configure(state="disabled")
        else:
            self.input_text.configure(state="normal")
            self.send_button.configure(state="normal")
            self.model_combo.configure(state="readonly")
            self.settings_button.configure(state="normal")
            self.new_session_button.configure(state="normal")
            self.rename_session_button.configure(state="normal")
            self.delete_session_button.configure(state="normal")
            self.refresh_models_button.configure(state="normal")
            self.session_listbox.configure(state="normal")
            self.internet_search_toggle.configure(state="normal")
            self.input_text.focus_set()

    def set_status(self, status_text: str) -> None:
        """Update only the status-bar text without toggling widget state."""
        self.status_var.set(status_text)

    def safe_ui_call(self, callback) -> bool:
        """Schedule a callback on the Tkinter UI thread when the window is alive.

        Returns:
            True when the callback was successfully queued. False when the
            window is already closing or Tk rejected the callback.
        """
        if self._is_closing:
            return False
        try:
            self.root.after(0, callback)
            return True
        except tk.TclError:
            self._is_closing = True
            return False

    def _schedule_gpu_metrics_refresh(self) -> None:
        """Schedule one non-blocking GPU poll and then queue the next poll.

        Why this method changed:
            The earlier implementation executed ROCm subprocess calls directly on
            the Tkinter thread. Even when those calls returned, they still tied
            visible metric freshness to UI-thread timing. The polling loop now
            launches the probe in a background thread and applies results back on
            the UI thread only after the sample is ready.
        """
        if self._is_closing:
            return
        if not self._gpu_poll_inflight:
            self._gpu_poll_inflight = True
            self._gpu_poll_generation += 1
            generation = self._gpu_poll_generation
            run_in_background(lambda: self._refresh_gpu_metrics(generation))
        try:
            self._gpu_poll_after_id = self.root.after(self.GPU_POLL_INTERVAL_MS, self._schedule_gpu_metrics_refresh)
        except tk.TclError:
            self._is_closing = True
            self._gpu_poll_after_id = None

    def _refresh_gpu_metrics(self, generation: int) -> None:
        """Fetch one GPU snapshot off-thread and update labels on the UI thread."""
        snapshot = self.controller.get_gpu_metrics_snapshot()

        def apply_snapshot() -> None:
            self._gpu_poll_inflight = False
            if self._is_closing or generation != self._gpu_poll_generation:
                return
            self.gpu_percent_var.set(f"GPU%: {snapshot.get('gpu_percent', 'N/A')}")
            self.vram_percent_var.set(f"VRAM%: {snapshot.get('vram_percent', 'N/A')}")

        if not self.safe_ui_call(apply_snapshot):
            self._gpu_poll_inflight = False

    def _handle_send_shortcut(self, _event: tk.Event) -> str:
        """Send the current message when the user presses Ctrl+Enter."""
        self._send_current_input()
        return "break"

    def _send_current_input(self) -> None:
        """Read text from the input box, clear it, and hand it to the controller."""
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            return
        self.input_text.delete("1.0", tk.END)
        self.controller.send_user_message(text)

    def _handle_model_selected(self, _event: tk.Event) -> None:
        """Persist a new model chosen from the installed-model dropdown."""
        selected_model = self.model_var.get().strip()
        try:
            self.controller.update_selected_model(selected_model)
        except RuntimeError as exc:
            messagebox.showerror("Model Change", str(exc))

    def _handle_session_selected(self, _event: tk.Event) -> None:
        """Load a clicked session from the session manager list."""
        selection = self.session_listbox.curselection()
        if not selection:
            return
        session_id = self.session_index_to_id.get(selection[0])
        if session_id:
            try:
                self.controller.load_session(session_id)
            except RuntimeError as exc:
                messagebox.showerror("Load Session", str(exc))

    def _rename_session(self) -> None:
        """Prompt the user for a new active-session title and send it to the controller."""
        new_title = simpledialog.askstring("Rename Session", "Enter a new session title:", parent=self.root)
        if new_title:
            try:
                self.controller.rename_active_session(new_title)
            except RuntimeError as exc:
                messagebox.showerror("Rename Session", str(exc))

    def _delete_selected_session(self) -> None:
        """Delete the currently selected session after a confirmation prompt."""
        selection = self.session_listbox.curselection()
        if not selection:
            messagebox.showinfo("Delete Session", "Select a session to delete first.")
            return
        session_id = self.session_index_to_id.get(selection[0])
        if not session_id:
            return
        confirmed = messagebox.askyesno("Delete Session", "Delete the selected session?")
        if confirmed:
            try:
                self.controller.delete_session(session_id)
            except RuntimeError as exc:
                messagebox.showerror("Delete Session", str(exc))

    def _handle_new_session(self) -> None:
        """Create a new blank session through the controller with UI-safe error handling."""
        try:
            self.controller.create_new_session()
        except RuntimeError as exc:
            messagebox.showerror("New Session", str(exc))

    def _ensure_thinking_panel_visible(self) -> None:
        """Show the thinking panel only when it is currently hidden."""
        if self.thinking_visible_var.get():
            return
        self.thinking_visible_var.set(True)
        self._toggle_thinking_visibility()

    def _toggle_thinking_visibility(self) -> None:
        """Show or hide the dedicated thinking panel based on the checkbox state."""
        if self.thinking_visible_var.get():
            if not self._thinking_pane_added:
                self.content_paned.insert(1, self.thinking_frame, weight=2)
                self._thinking_pane_added = True
            self.thinking_text.grid()
        else:
            self.thinking_text.grid_remove()
            if self._thinking_pane_added:
                self.content_paned.forget(self.thinking_frame)
                self._thinking_pane_added = False

    def _append_to_thinking_panel(self, text_chunk: str) -> None:
        """Append text into the dedicated thinking display."""
        self.thinking_text.configure(state="normal")
        self.thinking_text.insert(tk.END, text_chunk)
        self.thinking_text.see(tk.END)
        self.thinking_text.configure(state="disabled")

    def _clear_thinking_panel(self) -> None:
        """Clear the thinking panel before a new response is streamed."""
        self.thinking_text.configure(state="normal")
        self.thinking_text.delete("1.0", tk.END)
        self.thinking_text.configure(state="disabled")

    def _now_label(self) -> str:
        """Return the current local time label used in transcript headings."""
        import datetime as _datetime

        return _datetime.datetime.now().strftime("%I:%M:%S %p")
