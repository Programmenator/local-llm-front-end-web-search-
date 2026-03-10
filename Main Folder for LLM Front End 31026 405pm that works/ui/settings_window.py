"""
ui/settings_window.py

Purpose:
    PyQt6 settings dialog for the local LLM front end.

What this file does:
    - Exposes editable Ollama connection settings.
    - Exposes model, prompt, thinking, and SearXNG settings.
    - Lets the user test Ollama and SearXNG connectivity.
    - Lets the user refresh the installed model list.
    - Lets the user request a GPU VRAM flush for the selected model.

How this file fits into the system:
    This dialog is the configuration surface for persistent settings. It stays in
    the UI layer and delegates all mutation and system operations to the
    controller.
"""

from __future__ import annotations

from typing import Callable, Dict

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from controllers.chat_controller import ChatController
from services.config_service import ConfigService


class SettingsWindow(QDialog):
    """Popup dialog for editing persistent application settings."""

    def __init__(
        self,
        controller: ChatController,
        config_service: ConfigService,
        on_save_callback: Callable[[Dict[str, object]], None],
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the dialog with current config values and widget state."""
        super().__init__(parent)
        self.controller = controller
        self.config_service = config_service
        self.on_save_callback = on_save_callback
        self.available_models: list[str] = []

        self.setWindowTitle("Settings")
        self.resize(860, 760)
        self.setModal(False)

        current_config = self.config_service.get_config()
        self._build_widgets(current_config)
        self._handle_refresh_models(select_current=True)

    def _build_widgets(self, current_config: Dict[str, object]) -> None:
        """Create the full settings form and button layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        core_group = QGroupBox("Core Settings")
        core_form = QFormLayout(core_group)
        self.host_input = QLineEdit(str(current_config.get("host", "127.0.0.1")))
        self.port_input = QLineEdit(str(current_config.get("port", 11434)))
        self.timeout_input = QLineEdit(str(current_config.get("timeout_seconds", 300)))
        self.model_combo = QComboBox()
        self.thinking_mode_checkbox = QCheckBox("Enable thinking stream when supported")
        self.thinking_mode_checkbox.setChecked(bool(current_config.get("thinking_mode", True)))
        self.thinking_level_combo = QComboBox()
        self.thinking_level_combo.addItems(["low", "medium", "high"])
        self.thinking_level_combo.setCurrentText(str(current_config.get("thinking_level", "medium")))
        self.system_prompt_text = QTextEdit()
        self.system_prompt_text.setPlainText(str(current_config.get("system_prompt", "")))
        self.system_prompt_text.setMinimumHeight(160)

        core_form.addRow("Host", self.host_input)
        core_form.addRow("Port", self.port_input)
        core_form.addRow("Timeout (seconds)", self.timeout_input)
        core_form.addRow("Installed Model", self.model_combo)
        core_form.addRow("Thinking", self.thinking_mode_checkbox)
        core_form.addRow("Thinking Level", self.thinking_level_combo)
        core_form.addRow("System Prompt", self.system_prompt_text)
        layout.addWidget(core_group)

        web_group = QGroupBox("SearXNG Web Search")
        web_form = QFormLayout(web_group)
        self.enable_web_search_checkbox = QCheckBox("Enable model web search tools")
        self.enable_web_search_checkbox.setChecked(bool(current_config.get("enable_web_search", False)))
        self.searxng_base_url_input = QLineEdit(str(current_config.get("searxng_base_url", "http://127.0.0.1:8080")))
        self.web_search_category_combo = QComboBox()
        self.web_search_category_combo.addItems(["general", "news", "science", "it"])
        self.web_search_category_combo.setCurrentText(str(current_config.get("web_search_category", "general")))
        self.web_search_language_input = QLineEdit(str(current_config.get("web_search_language", "all")))
        self.web_search_time_range_combo = QComboBox()
        self.web_search_time_range_combo.addItems(["none", "day", "week", "month", "year"])
        self.web_search_time_range_combo.setCurrentText(str(current_config.get("web_search_time_range", "none")))
        self.web_safe_search_input = QLineEdit(str(current_config.get("web_safe_search", 1)))
        self.web_max_results_input = QLineEdit(str(current_config.get("web_max_results", 8)))
        self.web_max_pages_input = QLineEdit(str(current_config.get("web_max_pages", 2)))
        self.web_fetch_enabled_checkbox = QCheckBox("Allow page fetch after search selection")
        self.web_fetch_enabled_checkbox.setChecked(bool(current_config.get("web_fetch_enabled", True)))

        web_form.addRow(self.enable_web_search_checkbox)
        web_form.addRow("SearXNG Base URL", self.searxng_base_url_input)
        web_form.addRow("Search Category", self.web_search_category_combo)
        web_form.addRow("Language", self.web_search_language_input)
        web_form.addRow("Time Range", self.web_search_time_range_combo)
        web_form.addRow("Safe Search (0-2)", self.web_safe_search_input)
        web_form.addRow("Max Search Results", self.web_max_results_input)
        web_form.addRow("Max Pages To Read", self.web_max_pages_input)
        web_form.addRow(self.web_fetch_enabled_checkbox)
        layout.addWidget(web_group)

        button_row = QHBoxLayout()
        self.test_ollama_button = QPushButton("Test Ollama")
        self.test_ollama_button.clicked.connect(self._handle_test_connection)
        button_row.addWidget(self.test_ollama_button)

        self.test_searxng_button = QPushButton("Test SearXNG")
        self.test_searxng_button.clicked.connect(self._handle_test_web_search_connection)
        button_row.addWidget(self.test_searxng_button)

        self.refresh_models_button = QPushButton("Refresh Models")
        self.refresh_models_button.clicked.connect(self._handle_refresh_models)
        button_row.addWidget(self.refresh_models_button)

        self.flush_gpu_button = QPushButton("Flush GPU VRAM")
        self.flush_gpu_button.clicked.connect(self._handle_flush_gpu_vram)
        button_row.addWidget(self.flush_gpu_button)

        button_row.addStretch(1)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._handle_save)
        button_row.addWidget(self.save_button)
        layout.addLayout(button_row)

        models_group = QGroupBox("Installed Ollama Models")
        models_layout = QVBoxLayout(models_group)
        models_layout.addWidget(QLabel("The list below mirrors locally available models returned by Ollama."))
        self.models_list = QListWidget()
        self.models_list.itemSelectionChanged.connect(self._handle_model_list_select)
        models_layout.addWidget(self.models_list)
        layout.addWidget(models_group, stretch=1)

    def _handle_save(self) -> None:
        """Validate form values, persist settings, and close the dialog."""
        try:
            settings = {
                "host": self.host_input.text().strip(),
                "port": int(self.port_input.text().strip()),
                "model": self.model_combo.currentText().strip(),
                "timeout_seconds": int(self.timeout_input.text().strip()),
                "system_prompt": self.system_prompt_text.toPlainText().strip(),
                "thinking_mode": self.thinking_mode_checkbox.isChecked(),
                "thinking_level": self.thinking_level_combo.currentText().strip().lower(),
                "enable_web_search": self.enable_web_search_checkbox.isChecked(),
                "searxng_base_url": self.searxng_base_url_input.text().strip(),
                "web_search_category": self.web_search_category_combo.currentText().strip().lower(),
                "web_search_language": self.web_search_language_input.text().strip(),
                "web_search_time_range": self.web_search_time_range_combo.currentText().strip().lower(),
                "web_safe_search": int(self.web_safe_search_input.text().strip()),
                "web_max_results": int(self.web_max_results_input.text().strip()),
                "web_max_pages": int(self.web_max_pages_input.text().strip()),
                "web_fetch_enabled": self.web_fetch_enabled_checkbox.isChecked(),
            }
        except ValueError:
            QMessageBox.critical(
                self,
                "Invalid Settings",
                "Port, timeout, safe search, max search results, and max pages must be whole numbers.",
            )
            return

        try:
            saved_config = self.controller.save_settings(settings)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Settings Busy", str(exc))
            return

        self.on_save_callback(saved_config)
        QMessageBox.information(self, "Settings Saved", "Configuration saved successfully.")
        self.accept()

    def _handle_test_connection(self) -> None:
        """Ask the controller to test Ollama connectivity and show the result."""
        result = self.controller.test_connection()
        if result.get("ok"):
            QMessageBox.information(self, "Connection Result", str(result.get("message", "Connected.")))
        else:
            QMessageBox.critical(self, "Connection Result", str(result.get("message", "Connection failed.")))

    def _handle_test_web_search_connection(self) -> None:
        """Test SearXNG using the current unsaved field values in the dialog."""
        timeout_text = self.timeout_input.text().strip() or "30"
        try:
            timeout_seconds = int(timeout_text)
        except ValueError:
            QMessageBox.critical(self, "SearXNG Result", "Timeout must be a whole number before testing SearXNG.")
            return

        result = self.controller.test_web_search_connection(
            base_url=self.searxng_base_url_input.text().strip(),
            timeout_seconds=timeout_seconds,
        )
        if result.get("ok"):
            QMessageBox.information(self, "SearXNG Result", str(result.get("message", "Connected.")))
        else:
            QMessageBox.critical(self, "SearXNG Result", str(result.get("message", "Connection failed.")))

    def _handle_refresh_models(self, select_current: bool = False) -> None:
        """Fetch available models and refresh both list and combo box widgets."""
        try:
            self.available_models = self.controller.list_models()
        except Exception as exc:
            QMessageBox.critical(self, "Refresh Models", f"Could not load models: {exc}")
            return

        current_config = self.config_service.get_config()
        current_model = str(current_config.get("model", "")).strip()
        desired_model = current_model if select_current else self.model_combo.currentText().strip() or current_model

        visible_models = [model for model in self.available_models if str(model).strip()]
        if desired_model and desired_model not in visible_models:
            visible_models.insert(0, desired_model)

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(visible_models)
        if desired_model and desired_model in visible_models:
            self.model_combo.setCurrentText(desired_model)
        self.model_combo.blockSignals(False)

        self.models_list.clear()
        selected_row = -1
        for index, model_name in enumerate(visible_models):
            item = QListWidgetItem(model_name)
            if model_name == desired_model and model_name not in self.available_models:
                item.setToolTip("Saved model is not in the latest Ollama refresh results.")
            self.models_list.addItem(item)
            if model_name == desired_model:
                selected_row = index
        if selected_row >= 0:
            self.models_list.setCurrentRow(selected_row)

    def _handle_model_list_select(self) -> None:
        """Mirror list selection into the primary model combo box."""
        item = self.models_list.currentItem()
        if item is None:
            return
        self.model_combo.setCurrentText(item.text())

    def _handle_flush_gpu_vram(self) -> None:
        """Unload the selected model from Ollama memory and refresh visible state.

        Why this changed:
            The earlier settings action only showed a popup. Even when unload
            succeeded, the user could still see stale VRAM readouts until the
            next background poll, which made the feature look broken. The dialog
            now also triggers a model refresh after a successful unload so both
            the loaded-model list and the main-window metrics can catch up.
        """
        model_name = self.model_combo.currentText().strip()
        if not model_name:
            QMessageBox.warning(self, "Flush GPU VRAM", "Select a model before trying to unload it.")
            return

        result = self.controller.unload_selected_model(model_name)
        if result.get("ok"):
            self._handle_refresh_models()
            QMessageBox.information(self, "Flush GPU VRAM", str(result.get("message", "Model unload request sent.")))
        else:
            QMessageBox.warning(self, "Flush GPU VRAM", str(result.get("message", "Model unload failed.")))
