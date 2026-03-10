"""
models/__init__.py

Purpose:
    Package marker for shared data-model code.

What this file does:
    - Marks the models directory as a Python package.
    - Identifies this directory as the location for structured data objects used
      across the UI, controller, and persistence layers.

How this file fits into the system:
    This file has no runtime behavior by itself, but it documents the role of
    the models package so the project layout is easier to understand and audit.
"""

# generation_job.py
#     Explicit in-flight request model used to bind one streaming generation to
#     the session and config snapshot captured when the user pressed Send.
