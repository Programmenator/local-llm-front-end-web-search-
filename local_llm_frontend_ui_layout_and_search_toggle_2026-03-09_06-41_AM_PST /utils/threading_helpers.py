"""
utils/threading_helpers.py

Purpose:
    Small helper for running background work without freezing the UI.

What this file does:
    - Starts daemon threads for non-UI work.

How this file fits into the system:
    The main window must stay responsive while Ollama streams text. This helper
    centralizes the thread creation pattern used by the controller.
"""

from __future__ import annotations

import threading
from typing import Callable


def run_in_background(task: Callable[[], None]) -> threading.Thread:
    """Start a daemon thread for a no-argument background task."""
    thread = threading.Thread(target=task, daemon=True)
    thread.start()
    return thread
