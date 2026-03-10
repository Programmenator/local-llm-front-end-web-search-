#!/usr/bin/env bash
#
# run_local_llm_frontend.sh
#
# Purpose:
#     Simple launcher script for the modular local LLM desktop interface.
#
# What this file does:
#     - Changes into the project directory.
#     - Verifies Python 3 is available.
#     - Starts the main GUI entry point.
#
# How this file fits into the system:
#     This script gives the project a single-click or terminal-friendly launcher
#     on Kubuntu so the user does not need to manually cd into the folder and run
#     python each time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is not installed or not available in PATH."
    exit 1
fi

python3 main.py
