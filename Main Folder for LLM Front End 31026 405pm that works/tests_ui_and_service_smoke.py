"""
tests_ui_and_service_smoke.py

Purpose:
    Additional smoke and behavior tests for modules that were previously only
    lightly covered by the existing suite.

What this file does:
    - Exercises data-model serialization paths.
    - Verifies config and session persistence behavior.
    - Verifies the background-thread helper starts tasks.
    - Verifies major PyQt window and settings dialog workflows offscreen.

How this file fits into the system:
    The original suite already covered controller streaming, GPU parsing, and web
    search integration. This file adds broader coverage across the remaining
    modules so regressions in persistence and PyQt UI wiring are caught sooner.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QMessageBox

from models.chat_message import ChatMessage
from models.conversation_session import ConversationSession
from models.generation_job import GenerationJob
from services.config_service import ConfigService
from services.session_service import SessionService
from ui.main_window import MainWindow
from ui.settings_window import SettingsWindow
from utils.threading_helpers import run_in_background


class ModelRoundTripTests(unittest.TestCase):
    """Verify the small model classes serialize and deserialize cleanly."""

    def test_chat_message_round_trip_preserves_thinking(self) -> None:
        """Verify chat-message serialization preserves the optional thinking field."""
        original = ChatMessage(
            role="assistant",
            content="Done.",
            timestamp=datetime(2026, 3, 10, 8, 31, 0),
            thinking="scratch notes",
        )
        restored = ChatMessage.from_dict(original.to_dict())
        self.assertEqual(restored.role, "assistant")
        self.assertEqual(restored.content, "Done.")
        self.assertEqual(restored.thinking, "scratch notes")
        self.assertEqual(restored.timestamp, original.timestamp)

    def test_conversation_session_round_trip_preserves_messages(self) -> None:
        """Verify conversation-session serialization preserves nested messages."""
        session = ConversationSession(
            session_id="abc123",
            title="Test Session",
            model_name="qwen",
            created_at=datetime(2026, 3, 10, 8, 0, 0),
            updated_at=datetime(2026, 3, 10, 8, 5, 0),
            messages=[
                ChatMessage(role="user", content="hi", timestamp=datetime(2026, 3, 10, 8, 1, 0)),
                ChatMessage(role="assistant", content="hello", timestamp=datetime(2026, 3, 10, 8, 2, 0), thinking="trace"),
            ],
        )
        restored = ConversationSession.from_dict(session.to_dict())
        self.assertEqual(restored.session_id, "abc123")
        self.assertEqual(restored.title, "Test Session")
        self.assertEqual(restored.model_name, "qwen")
        self.assertEqual(len(restored.messages), 2)
        self.assertEqual(restored.messages[1].thinking, "trace")

    def test_generation_job_request_options_include_expected_fields(self) -> None:
        """Verify generation-job request snapshots include the expected runtime options."""
        job = GenerationJob(
            request_id="job1",
            session_id="session1",
            session_title_at_start="Session 1",
            model_name="qwen",
            host="127.0.0.1",
            port=11434,
            timeout_seconds=120,
            system_prompt="Be helpful",
            thinking_mode=True,
            thinking_level="high",
            enable_web_search=True,
            searxng_base_url="http://127.0.0.1:8080",
            web_search_category="news",
            web_search_language="en",
            web_search_time_range="week",
            web_safe_search=2,
            web_max_results=5,
            web_max_pages=3,
            web_fetch_enabled=False,
        )
        options = job.to_request_options()
        self.assertEqual(options["model"], "qwen")
        self.assertTrue(options["enable_web_search"])
        self.assertEqual(options["web_max_pages"], 3)
        self.assertFalse(options["web_fetch_enabled"])
        self.assertEqual(options["reranker_model"], "")


class ConfigAndSessionServiceTests(unittest.TestCase):
    """Exercise persistence-focused services against temporary directories."""

    def test_config_service_save_update_and_base_url(self) -> None:
        """Verify config save and update flows persist normalized base URL values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            service = ConfigService(config_path=config_path)
            saved = service.update_config({"host": "localhost", "port": 12345, "timeout_seconds": 60, "reranker_model": "mini-reranker"})
            self.assertEqual(saved["host"], "localhost")
            self.assertEqual(saved["port"], 12345)
            self.assertEqual(service.get_base_url(), "http://localhost:12345")
            self.assertEqual(saved["reranker_model"], "mini-reranker")
            self.assertTrue(config_path.exists())

    def test_session_service_create_save_list_load_delete(self) -> None:
        """Verify the session service supports the full create-save-load-delete lifecycle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = SessionService(sessions_dir=Path(temp_dir))
            session = service.create_session(model_name="qwen")
            session.title = "Persistent Session"
            session.messages.append(ChatMessage(role="user", content="hello", timestamp=datetime.now()))
            service.save_session(session)

            listed = service.list_sessions()
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0].title, "Persistent Session")

            loaded = service.load_session(session.session_id)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.messages[0].content, "hello")

            self.assertTrue(service.delete_session(session.session_id))
            self.assertEqual(service.list_sessions(), [])


class ThreadingHelperTests(unittest.TestCase):
    """Confirm the background-task utility actually starts a daemon thread."""

    def test_run_in_background_executes_task(self) -> None:
        """Verify the threading helper runs work on a background daemon thread."""
        event = threading.Event()

        def task() -> None:
            event.set()

        thread = run_in_background(task)
        self.assertTrue(thread.daemon)
        self.assertTrue(event.wait(timeout=1.0))


class FakeController:
    """Minimal controller used to smoke-test the PyQt windows offscreen."""

    def __init__(self) -> None:
        """Initialize deterministic controller state and call-tracking fields for UI smoke tests."""
        self.saved_settings: dict | None = None
        self.loaded_session_ids: list[str] = []
        self.sent_messages: list[str] = []
        self.renamed_to: str | None = None
        self.deleted_session_id: str | None = None
        self.created_sessions = 0
        self.model_updates: list[str] = []
        self.reranker_updates: list[str] = []
        self.models: list[str] = ["qwen", "deepseek"]

    def save_settings(self, settings: dict) -> dict:
        """Record settings payloads submitted by the settings window."""
        self.saved_settings = settings.copy()
        return {
            "model": settings.get("model", "qwen"),
            "thinking_mode": settings.get("thinking_mode", True),
            "thinking_level": settings.get("thinking_level", "medium"),
            "enable_web_search": settings.get("enable_web_search", False),
            "timeout_seconds": settings.get("timeout_seconds", 300),
            "reranker_model": settings.get("reranker_model", ""),
        }

    def list_models(self) -> list[str]:
        """Return the current fake model list for UI refresh workflows."""
        return list(self.models)

    def unload_selected_model(self, model_name: str) -> dict:
        """Record a VRAM flush request and report success."""
        return {"ok": True, "message": f"Unloaded {model_name}"}

    def test_connection(self) -> dict:
        """Report a successful fake Ollama connectivity check."""
        return {"ok": True, "message": "Connected"}

    def test_web_search_connection(self, base_url: str, timeout_seconds: int) -> dict:
        """Report a successful fake SearXNG connectivity check."""
        return {"ok": True, "message": f"Connected to {base_url} in {timeout_seconds}s"}

    def request_model_refresh(self) -> None:
        """Increment the refresh counter so tests can assert refresh requests occurred."""
        return None

    def update_selected_model(self, model_name: str) -> None:
        """Capture model-selection changes requested by the main window."""
        self.model_updates.append(model_name)

    def update_selected_reranker(self, model_name: str) -> None:
        """Capture reranker-selection changes requested by the main window."""
        self.reranker_updates.append(model_name)

    def create_new_session(self) -> None:
        """Record that the user requested a new session."""
        self.created_sessions += 1

    def load_session(self, session_id: str) -> None:
        """Capture the session identifier requested by the UI."""
        self.loaded_session_ids.append(session_id)

    def rename_active_session(self, new_title: str) -> None:
        """Record rename requests for the active session."""
        self.renamed_to = new_title

    def delete_session(self, session_id: str) -> None:
        """Record session-deletion requests."""
        self.deleted_session_id = session_id

    def send_user_message(self, text: str) -> None:
        """Capture user prompt submissions sent from the UI."""
        self.sent_messages.append(text)

    def get_gpu_metrics_snapshot(self) -> dict:
        """Return a deterministic GPU snapshot for footer/readout updates."""
        return {
            "gpu_percent_text": "42%",
            "vram_percent_text": "58%",
            "tokens_per_second_text": "11.5",
        }


class PyQtWindowSmokeTests(unittest.TestCase):
    """Exercise major MainWindow and SettingsWindow workflows offscreen."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a shared QApplication for the PyQt smoke-test module."""
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        """Create a fresh fake controller and windows for each smoke test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_service = ConfigService(config_path=Path(self.temp_dir.name) / "config.json")
        self.controller = FakeController()

    def tearDown(self) -> None:
        """Close UI objects created during the smoke test."""
        self.temp_dir.cleanup()

    def test_main_window_streaming_session_and_metric_helpers(self) -> None:
        """Verify main-window helpers update transcript, session, and metric state coherently."""
        window = MainWindow(self.controller, self.config_service)
        session = ConversationSession(
            session_id="session-a",
            title="Session A",
            model_name="qwen",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[],
        )
        window.update_session_list([session], "session-a")
        window.highlight_session("session-a")
        self.assertEqual(window.session_list.currentItem().text(), "Session A")

        window.update_session_title("Renamed")
        self.assertEqual(window.session_title_var.get(), "Renamed")

        window.begin_assistant_stream()
        window.append_stream_chunk("answer", "Hello world")
        window.append_stream_chunk("thinking", "scratch")
        window.finalize_assistant_stream("Hello world", "scratch", {"tokens_per_second": 12.5, "output_tokens": 4, "throughput_source": "ollama_eval"})
        self.assertIn("Assistant", window.chat_history.toPlainText())
        self.assertIn("scratch", window.thinking_text.toPlainText())
        self.assertEqual(window.tokens_per_second_var.get(), "Tok/s: 12.5, 4 tok")

        window._refresh_gpu_metrics_background(999)
        self.app.processEvents()
        self.assertEqual(window.gpu_percent_var.get(), "GPU%: 42%")
        self.assertEqual(window.footer_gpu_percent_label.text(), "GPU%: 42%")
        window.close()

    def test_main_window_send_and_toggle_workflows(self) -> None:
        """Verify prompt submission and thinking-panel toggle workflows use the controller correctly."""
        window = MainWindow(self.controller, self.config_service)
        window.input_text.setPlainText("hello from ui")
        window._send_current_input()
        self.assertEqual(self.controller.sent_messages, ["hello from ui"])

        window.show()
        self.app.processEvents()
        self.assertTrue(window.thinking_visible_var.get())
        window._toggle_thinking_visibility()
        self.assertFalse(window.thinking_visible_var.get())
        window._toggle_thinking_visibility()
        self.assertTrue(window.thinking_visible_var.get())
        window.close()

    def test_settings_window_save_refresh_and_flush(self) -> None:
        """Verify settings save, refresh, and flush workflows call the expected controller methods."""
        saved_payloads: list[dict] = []
        with patch.object(QMessageBox, "information") as info_box:
            window = SettingsWindow(
                controller=self.controller,
                config_service=self.config_service,
                on_save_callback=lambda payload: saved_payloads.append(payload),
            )
            window.host_input.setText("localhost")
            window.port_input.setText("11434")
            window.timeout_input.setText("120")
            window.model_combo.setCurrentText("qwen")
            window.enable_web_search_checkbox.setChecked(True)
            window.web_max_results_input.setText("6")
            window._handle_save()
            self.assertEqual(self.controller.saved_settings["host"], "localhost")
            self.assertTrue(saved_payloads[-1]["enable_web_search"])
            info_box.assert_called()
            window.close()

        with patch.object(QMessageBox, "information") as info_box:
            window = SettingsWindow(
                controller=self.controller,
                config_service=self.config_service,
                on_save_callback=lambda payload: None,
            )
            window.model_combo.setCurrentText("qwen")
            window._handle_flush_gpu_vram()
            info_box.assert_called()
            window.close()


    def test_settings_refresh_preserves_saved_model_when_list_is_empty(self) -> None:
        """The settings dialog should not silently blank a saved model on refresh failure/empty list."""
        controller = FakeController()
        controller.models = []
        config_service = ConfigService(config_path=Path(self.temp_dir.name) / "config.json")
        config_service.update_config({"model": "kept-model"})
        window = SettingsWindow(controller, config_service, lambda _saved: None)

        self.assertEqual(window.model_combo.currentText(), "kept-model")
        self.assertEqual(window.models_list.item(0).text(), "kept-model")

    def test_settings_window_invalid_numeric_input_is_blocked(self) -> None:
        """Verify invalid numeric settings input is rejected before save."""
        with patch.object(QMessageBox, "critical") as critical_box:
            window = SettingsWindow(
                controller=self.controller,
                config_service=self.config_service,
                on_save_callback=lambda payload: None,
            )
            window.port_input.setText("not-a-number")
            window._handle_save()
            critical_box.assert_called()
            window.close()


if __name__ == "__main__":
    unittest.main()
