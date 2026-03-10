"""
ui/settings_window.py

Purpose:
    Secondary window for configuring the local Ollama connection, model
    behavior, and optional SearXNG-backed web browsing settings.

What this file does:
    - Displays editable settings fields.
    - Saves host, port, model, timeout, system prompt, and thinking options.
    - Tests the connection to the configured Ollama server.
    - Refreshes and displays the locally installed model list.
    - Lets the user pick a model directly from the discovered model list.
    - Provides a Flush GPU VRAM button that requests model unload from Ollama.
    - Provides a SearXNG settings section with connectivity testing, more
      forgiving URL handling, and web search limits used by the model
      tool-calling path.
    - Lets the user choose whether the optional second-stage search reranker
      runs through Ollama or through a local sentence-transformers BAAI-style
      cross-encoder.
    - Uses a resizable scrollable layout so the full settings form remains usable
      on different monitor sizes.

How this file fits into the system:
    This window isolates configuration concerns from the main chat interface so
    the core chat screen stays simple and focused.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Dict

from controllers.chat_controller import ChatController


class SettingsWindow(tk.Toplevel):
    """Popup window for editing and testing local runtime settings."""

    def __init__(
        self,
        parent: tk.Tk,
        controller: ChatController,
        current_config: Dict[str, object],
        on_save_callback: Callable[[Dict[str, object]], None],
    ) -> None:
        """Build the settings window and populate it from saved configuration."""
        super().__init__(parent)
        self.controller = controller
        self.on_save_callback = on_save_callback
        self.title("Settings")
        self.geometry("760x860")
        self.minsize(560, 420)
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.host_var = tk.StringVar(value=str(current_config.get("host", "127.0.0.1")))
        self.port_var = tk.StringVar(value=str(current_config.get("port", 11434)))
        self.model_var = tk.StringVar(value=str(current_config.get("model", "")))
        self.timeout_var = tk.StringVar(value=str(current_config.get("timeout_seconds", 300)))
        self.thinking_mode_var = tk.BooleanVar(value=bool(current_config.get("thinking_mode", True)))
        self.thinking_level_var = tk.StringVar(value=str(current_config.get("thinking_level", "medium")))
        self.enable_web_search_var = tk.BooleanVar(value=bool(current_config.get("enable_web_search", False)))
        self.searxng_base_url_var = tk.StringVar(value=str(current_config.get("searxng_base_url", "http://127.0.0.1:8080")))
        self.web_search_category_var = tk.StringVar(value=str(current_config.get("web_search_category", "general")))
        self.web_search_language_var = tk.StringVar(value=str(current_config.get("web_search_language", "all")))
        self.web_search_time_range_var = tk.StringVar(value=str(current_config.get("web_search_time_range", "none")))
        self.web_safe_search_var = tk.StringVar(value=str(current_config.get("web_safe_search", 1)))
        self.web_max_results_var = tk.StringVar(value=str(current_config.get("web_max_results", 8)))
        self.enable_search_reranker_var = tk.BooleanVar(value=bool(current_config.get("enable_search_reranker", False)))
        self.search_reranker_backend_var = tk.StringVar(value=str(current_config.get("search_reranker_backend", "disabled")))
        self.search_reranker_model_var = tk.StringVar(value=str(current_config.get("search_reranker_model", "")))
        self.sentence_transformers_reranker_model_var = tk.StringVar(
            value=str(current_config.get("sentence_transformers_reranker_model", "BAAI/bge-reranker-base"))
        )
        self.sentence_transformers_reranker_device_var = tk.StringVar(
            value=str(current_config.get("sentence_transformers_reranker_device", "auto"))
        )
        self.web_max_pages_var = tk.StringVar(value=str(current_config.get("web_max_pages", 2)))
        self.web_fetch_enabled_var = tk.BooleanVar(value=bool(current_config.get("web_fetch_enabled", True)))
        self.web_fetch_max_response_bytes_var = tk.StringVar(value=str(current_config.get("web_fetch_max_response_bytes", 1500000)))
        self.web_fetch_max_redirects_var = tk.StringVar(value=str(current_config.get("web_fetch_max_redirects", 4)))
        self.web_fetch_allowlist_var = tk.StringVar(value=str(current_config.get("web_fetch_allowlist", "")))
        self.web_fetch_blocklist_var = tk.StringVar(value=str(current_config.get("web_fetch_blocklist", "")))

        self._build_widgets(current_config)
        self._handle_refresh_models(select_current=True)

    def _build_widgets(self, current_config: Dict[str, object]) -> None:
        """Create and arrange all settings window widgets."""
        outer = ttk.Frame(self, padding=0)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vertical_scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vertical_scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")

        container = ttk.Frame(canvas, padding=16)
        canvas_window = canvas.create_window((0, 0), window=container, anchor="nw")

        def _sync_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _fit_inner_width(event) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        container.bind("<Configure>", _sync_scroll_region)
        canvas.bind("<Configure>", _fit_inner_width)
        self.bind("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        container.columnconfigure(0, weight=1)

        form = ttk.Frame(container)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        self._add_labeled_entry(form, "Host", self.host_var, 0)
        self._add_labeled_entry(form, "Port", self.port_var, 1)
        self._add_labeled_entry(form, "Timeout (seconds)", self.timeout_var, 2)

        ttk.Label(form, text="Installed Model").grid(row=3, column=0, sticky="w", padx=(0, 12), pady=6)
        self.model_combo = ttk.Combobox(form, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=3, column=1, sticky="ew", pady=6)

        ttk.Checkbutton(
            form,
            text="Enable thinking stream when supported",
            variable=self.thinking_mode_var,
        ).grid(row=4, column=1, sticky="w", pady=(8, 6))

        ttk.Label(form, text="Thinking Level").grid(row=5, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Combobox(
            form,
            textvariable=self.thinking_level_var,
            state="readonly",
            values=["low", "medium", "high"],
        ).grid(row=5, column=1, sticky="ew", pady=6)

        ttk.Label(form, text="System Prompt").grid(row=6, column=0, sticky="nw", pady=(10, 0))
        self.system_prompt_text = tk.Text(form, height=9, wrap="word")
        self.system_prompt_text.grid(row=6, column=1, sticky="ew", pady=(10, 0))
        self.system_prompt_text.insert("1.0", str(current_config.get("system_prompt", "")))

        web_frame = ttk.LabelFrame(container, text="SearXNG Web Search", padding=12)
        web_frame.pack(fill="x", pady=(16, 8))
        web_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            web_frame,
            text="Enable model web search tools",
            variable=self.enable_web_search_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self._add_labeled_entry(web_frame, "SearXNG Base URL", self.searxng_base_url_var, 1)

        ttk.Label(web_frame, text="Search Category").grid(row=2, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Combobox(
            web_frame,
            textvariable=self.web_search_category_var,
            state="readonly",
            values=["general", "news", "science", "it"],
        ).grid(row=2, column=1, sticky="ew", pady=6)

        ttk.Label(web_frame, text="Language").grid(row=3, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Entry(web_frame, textvariable=self.web_search_language_var, width=42).grid(row=3, column=1, sticky="ew", pady=6)

        ttk.Label(web_frame, text="Time Range").grid(row=4, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Combobox(
            web_frame,
            textvariable=self.web_search_time_range_var,
            state="readonly",
            values=["none", "day", "week", "month", "year"],
        ).grid(row=4, column=1, sticky="ew", pady=6)

        self._add_labeled_entry(web_frame, "Safe Search (0-2)", self.web_safe_search_var, 5)
        self._add_labeled_entry(web_frame, "Max Search Results", self.web_max_results_var, 6)
        ttk.Checkbutton(
            web_frame,
            text="Enable dedicated search reranker before main model selection",
            variable=self.enable_search_reranker_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(web_frame, text="Reranker Backend").grid(row=8, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Combobox(
            web_frame,
            textvariable=self.search_reranker_backend_var,
            state="readonly",
            values=["disabled", "ollama", "sentence_transformers"],
        ).grid(row=8, column=1, sticky="ew", pady=6)
        self._add_labeled_entry(web_frame, "Ollama Reranker Model", self.search_reranker_model_var, 9)
        self._add_labeled_entry(
            web_frame,
            "Sentence-Transformers Model",
            self.sentence_transformers_reranker_model_var,
            10,
        )
        ttk.Label(web_frame, text="Sentence-Transformers Device").grid(row=11, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Combobox(
            web_frame,
            textvariable=self.sentence_transformers_reranker_device_var,
            state="readonly",
            values=["auto", "cpu", "cuda", "mps"],
        ).grid(row=11, column=1, sticky="ew", pady=6)
        ttk.Label(
            web_frame,
            text=(
                "Stage 1 always uses the built-in deterministic reranker. Stage 2 is optional. Choose 'ollama' to "
                "use a small Ollama reranker model, or choose 'sentence_transformers' to run a local Hugging Face "
                "cross-encoder such as BAAI/bge-reranker-base through the sentence-transformers package. "
                "The sentence-transformers backend does not go through Ollama. Use device='auto' unless you need "
                "to force cpu or another backend manually."
            ),
            wraplength=650,
            justify="left",
        ).grid(row=12, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._add_labeled_entry(web_frame, "Max Pages To Read", self.web_max_pages_var, 13)
        ttk.Checkbutton(
            web_frame,
            text="Allow page fetch after search selection",
            variable=self.web_fetch_enabled_var,
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(8, 0))

        fetch_policy_frame = ttk.LabelFrame(container, text="Web Fetch Safety Policy", padding=12)
        fetch_policy_frame.pack(fill="x", pady=(8, 8))
        fetch_policy_frame.columnconfigure(1, weight=1)

        ttk.Label(
            fetch_policy_frame,
            text=(
                "Page fetches only allow public http/https targets. Localhost, private IP ranges, and link-local "
                "targets are blocked. Redirects are capped and re-validated. Only approved content types are "
                "accepted, and oversized responses are rejected before the full body is read. Use allow/block lists "
                "to narrow which public domains may be fetched. Enter multiple domains separated by commas. Example: "
                "reuters.com, bbc.com"
            ),
            wraplength=650,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self._add_labeled_entry(fetch_policy_frame, "Max Response Bytes", self.web_fetch_max_response_bytes_var, 1)
        self._add_labeled_entry(fetch_policy_frame, "Max Redirects", self.web_fetch_max_redirects_var, 2)
        self._add_labeled_entry(fetch_policy_frame, "Domain Allowlist", self.web_fetch_allowlist_var, 3)
        self._add_labeled_entry(fetch_policy_frame, "Domain Blocklist", self.web_fetch_blocklist_var, 4)

        button_row = ttk.Frame(container)
        button_row.pack(fill="x", pady=(14, 8))

        ttk.Button(button_row, text="Test Ollama", command=self._handle_test_connection).pack(side="left")
        ttk.Button(button_row, text="Test SearXNG", command=self._handle_test_web_search_connection).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Refresh Models", command=self._handle_refresh_models).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Flush GPU VRAM", command=self._handle_flush_gpu_vram).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Save", command=self._handle_save).pack(side="right")

        ttk.Label(container, text="Installed Ollama Models").pack(anchor="w", pady=(8, 4))
        listbox_frame = ttk.Frame(container)
        listbox_frame.pack(fill="both", expand=True)
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)

        self.models_listbox = tk.Listbox(listbox_frame, height=10)
        self.models_listbox.grid(row=0, column=0, sticky="nsew")
        list_scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.models_listbox.yview)
        list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.models_listbox.configure(yscrollcommand=list_scrollbar.set)
        self.models_listbox.bind("<<ListboxSelect>>", self._handle_model_list_select)

    def _add_labeled_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int) -> None:
        """Create one labeled entry row in the settings form."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Entry(parent, textvariable=variable, width=42).grid(row=row, column=1, sticky="ew", pady=6)

    def _handle_save(self) -> None:
        """Validate form values, persist settings, and close the popup."""
        try:
            settings = {
                "host": self.host_var.get().strip(),
                "port": int(self.port_var.get().strip()),
                "model": self.model_var.get().strip(),
                "timeout_seconds": int(self.timeout_var.get().strip()),
                "system_prompt": self.system_prompt_text.get("1.0", "end").strip(),
                "thinking_mode": bool(self.thinking_mode_var.get()),
                "thinking_level": self.thinking_level_var.get().strip().lower(),
                "enable_web_search": bool(self.enable_web_search_var.get()),
                "searxng_base_url": self.searxng_base_url_var.get().strip(),
                "web_search_category": self.web_search_category_var.get().strip().lower(),
                "web_search_language": self.web_search_language_var.get().strip(),
                "web_search_time_range": self.web_search_time_range_var.get().strip().lower(),
                "web_safe_search": int(self.web_safe_search_var.get().strip()),
                "web_max_results": int(self.web_max_results_var.get().strip()),
                "enable_search_reranker": bool(self.enable_search_reranker_var.get()),
                "search_reranker_backend": self.search_reranker_backend_var.get().strip().lower(),
                "search_reranker_model": self.search_reranker_model_var.get().strip(),
                "sentence_transformers_reranker_model": self.sentence_transformers_reranker_model_var.get().strip(),
                "sentence_transformers_reranker_device": self.sentence_transformers_reranker_device_var.get().strip().lower(),
                "web_max_pages": int(self.web_max_pages_var.get().strip()),
                "web_fetch_enabled": bool(self.web_fetch_enabled_var.get()),
                "web_fetch_max_response_bytes": int(self.web_fetch_max_response_bytes_var.get().strip()),
                "web_fetch_max_redirects": int(self.web_fetch_max_redirects_var.get().strip()),
                "web_fetch_allowlist": self.web_fetch_allowlist_var.get().strip(),
                "web_fetch_blocklist": self.web_fetch_blocklist_var.get().strip(),
            }
        except ValueError:
            messagebox.showerror(
                "Invalid Settings",
                "Port, timeout, safe search, max search results, max pages, max response bytes, and max redirects must be whole numbers.",
            )
            return

        try:
            saved_config = self.controller.save_settings(settings)
        except RuntimeError as exc:
            messagebox.showerror("Settings Busy", str(exc))
            return

        self.on_save_callback(saved_config)
        messagebox.showinfo("Settings Saved", "Configuration saved successfully.")
        self.destroy()

    def _handle_test_connection(self) -> None:
        """Ask the controller to test Ollama connectivity and show the result."""
        result = self.controller.test_connection()
        if result.get("ok"):
            messagebox.showinfo("Connection Result", str(result.get("message", "Connected.")))
        else:
            messagebox.showerror("Connection Result", str(result.get("message", "Connection failed.")))

    def _handle_test_web_search_connection(self) -> None:
        """Ask the controller to test SearXNG connectivity and show the result.

        This uses the values currently typed into the settings form, not only the
        last-saved configuration. That lets the user validate a new SearXNG URL
        before pressing Save.
        """
        timeout_text = self.timeout_var.get().strip() or "30"
        try:
            timeout_seconds = int(timeout_text)
        except ValueError:
            messagebox.showerror("SearXNG Result", "Timeout must be a whole number before testing SearXNG.")
            return

        result = self.controller.test_web_search_connection(
            base_url=self.searxng_base_url_var.get().strip(),
            timeout_seconds=timeout_seconds,
        )
        if result.get("ok"):
            messagebox.showinfo("SearXNG Result", str(result.get("message", "Connected.")))
        else:
            messagebox.showerror("SearXNG Result", str(result.get("message", "Connection failed.")))

    def _handle_flush_gpu_vram(self) -> None:
        """Request that Ollama unload the selected model to free VRAM and show the result."""
        selected_model = self.model_var.get().strip()
        result = self.controller.unload_selected_model(selected_model)
        if result.get("ok"):
            messagebox.showinfo("Flush GPU VRAM Result", str(result.get("message", "Flush completed.")))
        else:
            messagebox.showerror("Flush GPU VRAM Result", str(result.get("message", "Flush failed.")))

    def _handle_refresh_models(self, select_current: bool = False) -> None:
        """Load available model names from Ollama into the listbox and combo box."""
        models = self.controller.list_models()
        self.model_combo["values"] = models
        self.models_listbox.delete(0, tk.END)
        for model_name in models:
            self.models_listbox.insert(tk.END, model_name)

        if select_current and self.model_var.get().strip() in models:
            current_model = self.model_var.get().strip()
        elif models:
            current_model = models[0]
            self.model_var.set(current_model)
        else:
            current_model = ""

        if current_model:
            try:
                selected_index = models.index(current_model)
                self.models_listbox.selection_clear(0, tk.END)
                self.models_listbox.selection_set(selected_index)
                self.models_listbox.see(selected_index)
            except ValueError:
                pass

    def _handle_model_list_select(self, _event: tk.Event) -> None:
        """Mirror the listbox model selection into the saved-model combobox."""
        selection = self.models_listbox.curselection()
        if not selection:
            return
        self.model_var.set(str(self.models_listbox.get(selection[0])))
