"""
app.py

Purpose:
    Application bootstrap and dependency wiring.

What this file does:
    - Builds the service layer.
    - Builds the controller layer.
    - Creates the main window.
    - Connects the UI to the controller.

How this file fits into the system:
    This file is the composition root for the project. It keeps object creation in
    one place so the rest of the code can focus on one responsibility at a time.
"""

from controllers.chat_controller import ChatController
from services.config_service import ConfigService
from services.ollama_client import OllamaClient
from services.session_service import SessionService
from ui.main_window import MainWindow


class LocalLLMFrontendApp:
    """Build and run the local desktop chat interface."""

    def __init__(self) -> None:
        """Create all major application components and connect them together."""
        self.config_service = ConfigService()
        self.session_service = SessionService()
        self.ollama_client = OllamaClient(self.config_service)
        self.chat_controller = ChatController(
            config_service=self.config_service,
            ollama_client=self.ollama_client,
            session_service=self.session_service,
        )
        self.main_window = MainWindow(self.chat_controller, self.config_service)
        self.chat_controller.attach_view(self.main_window)
        self.chat_controller.bootstrap_state()

    def run(self) -> None:
        """Start the Tkinter main loop for the desktop application."""
        self.main_window.start()
