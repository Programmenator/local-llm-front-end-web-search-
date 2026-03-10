"""
tests_test_generation_job_flow.py

Purpose:
    Focused regression tests for the GenerationJob / RequestContext session-race
    fix, GPU metric normalization, and GPU VRAM flush control.

What this file does:
    - Verifies that an in-flight reply is committed to the session captured when
      the request started, even if the controller's active session pointer later changes.
    - Verifies that the Ollama request uses the frozen model/config snapshot from
      request start rather than live config edited later.
    - Verifies that the non-streaming model refresh path uses live config safely.
    - Verifies that GPU metric parsing returns the GPU% and VRAM% values needed
      by the main UI and updates correctly across repeated live samples.
    - Verifies that the flush GPU VRAM control still calls the unload path.

How this file fits into the system:
    These tests validate the architectural fix introduced for in-flight request
    isolation plus the later GPU metrics and VRAM flush revisions. They are
    lightweight unit tests that run without a live Ollama server or a Tkinter window.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from controllers.chat_controller import ChatController
from services.config_service import ConfigService
from services.gpu_monitor_service import GPUMonitorService
from services.session_service import SessionService
from ui.main_window import MainWindow


class FakeView:
    """Minimal view stub used to satisfy controller callbacks during tests."""

    def __init__(self) -> None:
        """Initialize the fake view state used by controller-side generation tests."""
        self.status_messages = []
        self.immediate_gpu_refresh_requests = 0

    def safe_ui_call(self, callback):
        """Execute the callback immediately to mimic a cooperative UI test harness."""
        callback()

    def append_message(self, _message):
        """Record a completed chat message in the fake transcript."""
        return None

    def begin_assistant_stream(self):
        """Mark the start of a streamed assistant response for controller tests."""
        return None

    def set_busy_state(self, _is_busy, status_text):
        """Capture busy-state transitions requested by the controller."""
        self.status_messages.append(status_text)

    def finalize_assistant_stream(self, _content, _thinking, _performance_stats=None):
        """Store the final streamed assistant response payload for assertions."""
        return None

    def append_stream_chunk(self, _chunk_type, _text_chunk):
        """Append an incremental stream chunk to the fake transcript state."""
        return None

    def update_session_list(self, _sessions, _active_session_id):
        """Capture the latest session-list payload pushed by the controller."""
        return None

    def update_session_title(self, _title):
        """Capture active-session title updates for assertions."""
        return None

    def render_full_conversation(self, _messages):
        """Store a full conversation render request from the controller."""
        return None

    def refresh_model_display(self, _config):
        """Capture the model label text shown to the user."""
        return None

    def highlight_session(self, _session_id):
        """Record which session the controller asked the UI to highlight."""
        return None

    def set_status(self, status_text):
        """Capture status-bar text updates requested by the controller."""
        self.status_messages.append(status_text)

    def request_immediate_gpu_metrics_refresh(self):
        """Record controller requests for an immediate post-action GPU metrics refresh."""
        self.immediate_gpu_refresh_requests += 1



class FakeOllamaClient:
    """Small Ollama client stub that records the frozen request snapshot."""

    def __init__(self) -> None:
        """Initialize the fake Ollama client with canned responses and call tracking."""
        self.last_request_options = None
        self.calls = 0
        self.unload_calls = []

    def chat_stream(self, messages, on_chunk, request_options=None):
        """Yield a deterministic response stream for controller generation tests."""
        self.calls += 1
        self.last_request_options = request_options.copy() if request_options else None
        on_chunk("content", "assistant partial")
        return {
            "content": "assistant final",
            "thinking": "",
            "output_tokens": 12,
            "tokens_per_second": 24.0,
            "throughput_source": "ollama_eval",
        }

    def list_models(self):
        """Return the configured fake model list."""
        return ["model-a", "model-b"]

    def test_connection(self):
        """Report a successful fake Ollama connectivity check."""
        return {"ok": True, "message": "ok"}

    def unload_model(self, model_name):
        """Record a model unload request and report success."""
        self.unload_calls.append(model_name)
        return {"done": True, "done_reason": "completed", "model": model_name}


class GenerationJobFlowTests(unittest.TestCase):
    """Regression tests for request/session isolation in ChatController."""

    def setUp(self) -> None:
        """Create isolated temp directories and controller dependencies for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config_service = ConfigService(config_path=root / "config.json")
        self.config_service.update_config({
            "host": "127.0.0.1",
            "port": 11434,
            "model": "model-a",
            "timeout_seconds": 90,
            "system_prompt": "test prompt",
            "thinking_mode": True,
            "thinking_level": "medium",
        })
        self.session_service = SessionService(sessions_dir=root / "sessions")
        self.ollama_client = FakeOllamaClient()
        self.controller = ChatController(self.config_service, self.ollama_client, self.session_service)
        self.view = FakeView()
        self.controller.attach_view(self.view)
        self.controller._persist_active_session()

    def tearDown(self) -> None:
        """Remove temporary directories created for the test case."""
        self.temp_dir.cleanup()

    def test_generation_job_commits_reply_to_original_session_id(self) -> None:
        """The assistant reply should be saved to the request's session, not the later active one."""
        queued_tasks = []

        def capture_task(task):
            queued_tasks.append(task)
            return None

        original_session_id = self.controller.active_session.session_id
        other_session = self.session_service.create_session(model_name="model-b", title="Other Session")
        self.session_service.save_session(other_session)

        with patch("controllers.chat_controller.run_in_background", side_effect=capture_task):
            self.controller.send_user_message("hello from original session")

        self.assertEqual(len(queued_tasks), 1)

        self.controller.active_session = other_session
        queued_tasks[0]()

        original_session = self.session_service.load_session(original_session_id)
        reloaded_other_session = self.session_service.load_session(other_session.session_id)

        self.assertIsNotNone(original_session)
        self.assertIsNotNone(reloaded_other_session)
        self.assertEqual([msg.role for msg in original_session.messages], ["user", "assistant"])
        self.assertEqual(original_session.messages[-1].content, "assistant final")
        self.assertEqual(reloaded_other_session.messages, [])

    def test_generation_job_uses_frozen_request_model_snapshot(self) -> None:
        """The Ollama request should use the model captured when Send was pressed."""
        queued_tasks = []

        def capture_task(task):
            queued_tasks.append(task)
            return None

        with patch("controllers.chat_controller.run_in_background", side_effect=capture_task):
            self.controller.send_user_message("hello snapshot")

        self.assertEqual(len(queued_tasks), 1)
        self.config_service.update_config({"model": "model-b", "timeout_seconds": 5})
        queued_tasks[0]()

        self.assertIsNotNone(self.ollama_client.last_request_options)
        self.assertEqual(self.ollama_client.last_request_options["model"], "model-a")
        self.assertEqual(self.ollama_client.last_request_options["timeout_seconds"], 90)

    def test_flush_gpu_vram_still_calls_unload_path(self) -> None:
        """The settings-side VRAM flush control should still route to unload_model."""
        result = self.controller.unload_selected_model("model-a")
        self.assertTrue(result["ok"])
        self.assertEqual(self.ollama_client.unload_calls, ["model-a"])

    def test_successful_flush_requests_immediate_gpu_refresh(self) -> None:
        """A successful unload should ask the attached view to refresh GPU metrics immediately."""
        result = self.controller.unload_selected_model("model-a")
        self.assertTrue(result["ok"])
        self.assertEqual(self.view.immediate_gpu_refresh_requests, 1)


class OllamaClientNonStreamingTests(unittest.TestCase):
    """Regression tests for non-streaming Ollama client actions after the GenerationJob refactor."""

    def setUp(self) -> None:
        """Build a shared configuration object for non-streaming Ollama client tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config_service = ConfigService(config_path=root / "config.json")
        self.config_service.update_config({
            "host": "127.0.0.1",
            "port": 11434,
            "model": "model-a",
            "timeout_seconds": 45,
        })

    def tearDown(self) -> None:
        """Restore patched globals used by the non-streaming Ollama client tests."""
        self.temp_dir.cleanup()

    def test_list_models_uses_live_config_timeout_without_name_error(self) -> None:
        """Model refresh should use live config timeout and must not reference an undefined request_options name."""
        from services.ollama_client import OllamaClient

        class FakeResponse:
            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["timeout"] = timeout
            return FakeResponse({"models": [{"name": "qwen3:8b"}, {"name": "qwen3:4b"}]})

        client = OllamaClient(self.config_service)
        with patch("services.ollama_client.request.urlopen", side_effect=fake_urlopen):
            models = client.list_models()

        self.assertEqual(models, ["qwen3:4b", "qwen3:8b"])
        self.assertEqual(captured["timeout"], 45)
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/tags")


class ConfigServiceStabilityTests(unittest.TestCase):
    """Validate config normalization paths that protect startup stability."""

    def test_load_config_recovers_from_invalid_numeric_and_boolean_values(self) -> None:
        """Malformed user-edited config values should fall back to safe defaults instead of crashing startup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(json.dumps({
                "host": "",
                "port": "not-a-port",
                "timeout_seconds": "bad",
                "thinking_mode": "false",
                "thinking_level": "extreme",
            }), encoding="utf-8")

            service = ConfigService(config_path=config_path)
            loaded = service.get_config()

        self.assertEqual(loaded["host"], "127.0.0.1")
        self.assertEqual(loaded["port"], 11434)
        self.assertEqual(loaded["timeout_seconds"], 300)
        self.assertFalse(loaded["thinking_mode"])
        self.assertEqual(loaded["thinking_level"], "medium")


class MainWindowShutdownSafetyTests(unittest.TestCase):
    """Validate shutdown guards for background UI callbacks."""

    def test_safe_ui_call_becomes_noop_when_window_is_closing(self) -> None:
        """Worker threads should not schedule Tk callbacks after the UI has begun shutting down."""

        class DummyRoot:
            def after(self, _delay, _callback):
                raise AssertionError("after should not be called once the window is closing")

        class DummyWindow:
            def __init__(self):
                self._is_closing = True
                self.root = DummyRoot()

        MainWindow.safe_ui_call(DummyWindow(), lambda: None)


class GPUMonitorServiceTests(unittest.TestCase):
    """Validate the normalized GPU metric parsing used by the main UI."""

    def test_get_live_metrics_prefers_amd_smi_json_and_computes_vram_percent_from_used_and_total_values(self) -> None:
        """amd-smi JSON should compute VRAM% even when the CLI does not expose a direct percent field."""
        service = GPUMonitorService()
        amd_json_output = json.dumps(
            {
                "gpu_metrics": [
                    {
                        "GPU Activity": 9,
                        "VRAM Total": "16384 MiB",
                        "VRAM Used": "8192 MiB",
                    }
                ]
            }
        )

        with patch("services.gpu_monitor_service.shutil.which", side_effect=lambda name: "/usr/bin/amd-smi" if name == "amd-smi" else None):
            with patch.object(service, "_run_command", return_value=amd_json_output):
                snapshot = service.get_live_metrics()

        self.assertTrue(snapshot["ok"])
        self.assertEqual(snapshot["source"], "amd-smi-monitor")
        self.assertEqual(snapshot["gpu_percent"], "9%")
        self.assertEqual(snapshot["vram_percent"], "50%")

    """Validate the normalized GPU metric parsing used by the main UI."""

    def test_get_live_metrics_prefers_amd_smi_and_extracts_gpu_and_vram_percent(self) -> None:
        """amd-smi output should be normalized to the two visible percentage fields."""
        service = GPUMonitorService()
        amd_output = "GPU Utilization (%) : 67%\nVRAM Usage (%) : 42%\n"

        with patch("services.gpu_monitor_service.shutil.which", side_effect=lambda name: "/usr/bin/amd-smi" if name == "amd-smi" else None):
            with patch.object(service, "_run_command", return_value=amd_output):
                snapshot = service.get_live_metrics()

        self.assertTrue(snapshot["ok"])
        self.assertEqual(snapshot["source"], "amd-smi-monitor")
        self.assertEqual(snapshot["gpu_percent"], "67%")
        self.assertEqual(snapshot["vram_percent"], "42%")

    def test_get_live_metrics_falls_back_to_rocm_smi_and_extracts_gpu_and_vram_percent(self) -> None:
        """rocm-smi output should also normalize to the same two visible percentage fields."""
        service = GPUMonitorService()
        rocm_output = "GPU use (%) : 83%\nGPU memory use (%) : 51%\n"

        def fake_which(name: str):
            mapping = {
                "amd-smi": None,
                "rocm-smi": "/opt/rocm/bin/rocm-smi",
            }
            return mapping.get(name)

        with patch("services.gpu_monitor_service.shutil.which", side_effect=fake_which):
            with patch.object(service, "_run_command", return_value=rocm_output):
                snapshot = service.get_live_metrics()

        self.assertTrue(snapshot["ok"])
        self.assertEqual(snapshot["source"], "rocm-smi")
        self.assertEqual(snapshot["gpu_percent"], "83%")
        self.assertEqual(snapshot["vram_percent"], "51%")

    def test_get_live_metrics_prefers_amd_smi_monitor_json_for_live_updates(self) -> None:
        """Repeated monitor samples should reflect changed VRAM state instead of staying stuck on one metric snapshot."""
        service = GPUMonitorService()
        outputs = [
            json.dumps({"gpu_metrics": [{"GPU Activity": 3, "VRAM Total": "16384 MiB", "VRAM Used": "0 MiB"}]}),
            json.dumps({"gpu_metrics": [{"GPU Activity": 41, "VRAM Total": "16384 MiB", "VRAM Used": "8192 MiB"}]}),
        ]

        with patch("services.gpu_monitor_service.shutil.which", side_effect=lambda name: "/usr/bin/amd-smi" if name == "amd-smi" else None):
            with patch.object(service, "_run_command", side_effect=outputs):
                first = service.get_live_metrics()
                second = service.get_live_metrics()

        self.assertEqual(first["source"], "amd-smi-monitor")
        self.assertEqual(first["gpu_percent"], "3%")
        self.assertEqual(first["vram_percent"], "0%")
        self.assertEqual(second["source"], "amd-smi-monitor")
        self.assertEqual(second["gpu_percent"], "41%")
        self.assertEqual(second["vram_percent"], "50%")

    def test_get_live_metrics_parses_amd_smi_monitor_table_output(self) -> None:
        """Monitor table output should compute VRAM% from VRAM_USED and VRAM_TOTAL when needed."""
        service = GPUMonitorService()
        monitor_output = (
            "GPU  VRAM_USED  VRAM_FREE  VRAM_TOTAL  VRAM%\n"
            "0    8192 MiB   8192 MiB   16384 MiB   50 %\n"
            "GPU use (%) : 22%\n"
        )

        with patch("services.gpu_monitor_service.shutil.which", side_effect=lambda name: "/usr/bin/amd-smi" if name == "amd-smi" else None):
            with patch.object(service, "_run_command", side_effect=["not-json", "not-json", monitor_output]):
                snapshot = service.get_live_metrics()

        self.assertTrue(snapshot["ok"])
        self.assertEqual(snapshot["source"], "amd-smi-monitor")
        self.assertEqual(snapshot["gpu_percent"], "22%")
        self.assertEqual(snapshot["vram_percent"], "50%")


if __name__ == "__main__":
    unittest.main()


class OllamaClientStreamingMetricTests(unittest.TestCase):
    """Validate final token/second metric extraction from Ollama stream events."""

    def setUp(self) -> None:
        """Create a client instance and capture original request plumbing for restoration."""
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config_service = ConfigService(config_path=root / "config.json")
        self.config_service.update_config({
            "host": "127.0.0.1",
            "port": 11434,
            "model": "model-a",
            "timeout_seconds": 45,
            "system_prompt": "",
            "thinking_mode": False,
            "thinking_level": "medium",
        })

    def tearDown(self) -> None:
        """Restore the original request helper patched during streaming metric tests."""
        self.temp_dir.cleanup()

    def test_chat_stream_returns_eval_count_and_tokens_per_second_from_final_stream_item(self) -> None:
        """Verify final streaming metrics are exposed to callers after generation completes."""
        from services.ollama_client import OllamaClient

        client = OllamaClient(self.config_service)
        fake_stream = [
            {"message": {"content": "Hello "}, "done": False},
            {"message": {"content": "world"}, "done": False},
            {"message": {}, "done": True, "eval_count": 20, "eval_duration": 2_000_000_000},
        ]

        with patch.object(client, "_stream_json_lines", return_value=fake_stream):
            result = client.chat_stream([], on_chunk=lambda *_args: None)

        self.assertEqual(result["content"], "Hello world")
        self.assertEqual(result["output_tokens"], 20)
        self.assertAlmostEqual(result["tokens_per_second"], 10.0)
        self.assertEqual(result["throughput_source"], "ollama_eval")


class MainWindowTokenMeterTests(unittest.TestCase):
    """Validate the UI-side token/second estimation and final metric display helpers."""

    def test_apply_final_performance_stats_prefers_final_ollama_value(self) -> None:
        """Verify final Ollama throughput replaces transient estimated token rates."""
        from ui.main_window import MainWindow

        class FakeVar:
            def __init__(self):
                self.value = ""
            def set(self, value):
                self.value = value

        class DummyWindow:
            def __init__(self):
                self.tokens_per_second_var = FakeVar()
                self.streaming_in_progress = False
                self._stream_started_monotonic = 0.0
                self._stream_output_token_estimate = 0
            _refresh_estimated_tokens_per_second = MainWindow._refresh_estimated_tokens_per_second
            _apply_final_performance_stats = MainWindow._apply_final_performance_stats

        dummy = DummyWindow()
        MainWindow._apply_final_performance_stats(dummy, {
            "tokens_per_second": 33.3,
            "output_tokens": 120,
            "throughput_source": "ollama_eval",
        })
        self.assertEqual(dummy.tokens_per_second_var.value, "Tok/s: 33.3, 120 tok")

    def test_estimate_token_count_returns_nonzero_for_stream_text(self) -> None:
        """Verify token estimation returns a useful nonzero value for normal stream text."""
        from ui.main_window import MainWindow

        estimated = MainWindow._estimate_token_count(object(), "Hello, world!")
        self.assertGreaterEqual(estimated, 3)


class ControllerGpuSnapshotCompatibilityTests(unittest.TestCase):
    """Verify the controller preserves the GPU metric key contract used by the UI."""

    def test_get_gpu_metrics_snapshot_adds_legacy_text_keys(self) -> None:
        """Normalize canonical GPU service fields into the older ``*_text`` names too."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_service = ConfigService(config_path=Path(temp_dir) / "config.json")
            session_service = SessionService(sessions_dir=Path(temp_dir) / "sessions")
            controller = ChatController(config_service, FakeOllamaClient(), session_service)

            class StubGpuService:
                """Provide a canonical GPU snapshot without the legacy UI keys."""

                def get_live_metrics(self):
                    """Return one deterministic canonical GPU snapshot."""
                    return {
                        "gpu_percent": "73%",
                        "vram_percent": "64%",
                        "source": "stub",
                        "ok": True,
                        "message": "ok",
                    }

            controller.gpu_monitor_service = StubGpuService()
            snapshot = controller.get_gpu_metrics_snapshot()

            self.assertEqual(snapshot["gpu_percent"], "73%")
            self.assertEqual(snapshot["vram_percent"], "64%")
            self.assertEqual(snapshot["gpu_percent_text"], "73%")
            self.assertEqual(snapshot["vram_percent_text"], "64%")
