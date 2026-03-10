"""
main.py

Purpose:
    Entry point for the local LLM desktop interface.

What this file does:
    - Starts the Tkinter application.
    - Creates the main application object.
    - Hands control to the UI event loop.

How this file fits into the system:
    This file is intentionally small. It exists so the project has a clear launch
    point and so startup concerns remain separate from UI layout, controller
    logic, persistence, and Ollama communication.
"""

from app import LocalLLMFrontendApp


def main() -> None:
    """Create the application and start the desktop event loop."""
    app = LocalLLMFrontendApp()
    app.run()


if __name__ == "__main__":
    main()
