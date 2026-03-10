"""
tests_vram_flush_regression.py

Purpose:
    Regression tests for the VRAM flush path after the PyQt6 migration.

What this file does:
    - Verifies that ChatController still routes the flush action into the Ollama
      unload path.
    - Verifies that a successful unload asks the attached view to refresh GPU
      metrics immediately so the user does not mistake stale footer values for
      a failed unload.
    - Verifies that the direct Ollama CLI stop helper reports success and
      failure cleanly without forcing the caller to crash.

How this file fits into the system:
    These tests cover the controller/service behavior behind the settings-side
    "Flush GPU VRAM" action without requiring a live PyQt6 runtime.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from controllers.chat_controller import ChatController
from services.config_service import ConfigService
from services.ollama_client import OllamaClient
from services.session_service import SessionService


class _FakeView:
    """Minimal controller-facing view stub for post-unload refresh tests."""

    def __init__(self) -> None:
        """Initialize refresh tracking used by the controller regression tests."""
        self.immediate_gpu_refresh_requests = 0

    def request_immediate_gpu_metrics_refresh(self) -> None:
        """Record one immediate GPU refresh request from the controller."""
        self.immediate_gpu_refresh_requests += 1


class _FakeOllamaClient:
    """Small unload-only Ollama client stub used by controller tests."""

    def __init__(self) -> None:
        """Initialize unload-call tracking for the regression tests."""
        self.unload_calls: list[str] = []

    def unload_model(self, model_name: str) -> dict:
        """Record the unload target and return a successful completion payload."""
        self.unload_calls.append(model_name)
        return {"done": True, "done_reason": "completed", "model": model_name}


class VRAMFlushControllerTests(unittest.TestCase):
    """Verify controller-side behavior for the settings VRAM flush action."""

    def setUp(self) -> None:
        """Create isolated config/session state for each regression test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config_service = ConfigService(config_path=root / "config.json")
        self.config_service.update_config({"model": "model-a"})
        self.session_service = SessionService(sessions_dir=root / "sessions")
        self.ollama_client = _FakeOllamaClient()
        self.controller = ChatController(self.config_service, self.ollama_client, self.session_service)
        self.view = _FakeView()
        self.controller.attach_view(self.view)

    def tearDown(self) -> None:
        """Remove the isolated temp directory used by the test case."""
        self.temp_dir.cleanup()

    def test_successful_flush_requests_immediate_gpu_refresh(self) -> None:
        """A successful unload should refresh GPU metrics immediately in the UI."""
        result = self.controller.unload_selected_model("model-a")
        self.assertTrue(result["ok"])
        self.assertEqual(self.ollama_client.unload_calls, ["model-a"])
        self.assertEqual(self.view.immediate_gpu_refresh_requests, 1)


class OllamaClientUnloadFallbackTests(unittest.TestCase):
    """Verify the direct CLI-based unload helper reports usable status."""

    def setUp(self) -> None:
        """Create a minimal config service for the Ollama client under test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_service = ConfigService(config_path=Path(self.temp_dir.name) / "config.json")
        self.client = OllamaClient(self.config_service)

    def tearDown(self) -> None:
        """Clean up temporary files created for the Ollama client tests."""
        self.temp_dir.cleanup()

    @patch("services.ollama_client.shutil.which", return_value="/usr/bin/ollama")
    @patch("services.ollama_client.subprocess.run")
    def test_cli_unload_helper_reports_success(self, run_mock, _which_mock) -> None:
        """The CLI helper should normalize a successful `ollama stop` result."""
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = "stopped model-a\n"
        run_mock.return_value.stderr = ""
        result = self.client._unload_model_via_cli("model-a")
        self.assertTrue(result["ok"])
        self.assertEqual(result["unload_method"], "ollama_cli_stop")

    @patch("services.ollama_client.shutil.which", return_value=None)
    def test_cli_unload_helper_reports_missing_cli(self, _which_mock) -> None:
        """The CLI helper should report a missing executable without raising."""
        result = self.client._unload_model_via_cli("model-a")
        self.assertFalse(result["ok"])
        self.assertIn("not installed", result["message"])


if __name__ == "__main__":
    unittest.main()
