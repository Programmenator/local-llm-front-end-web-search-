"""
services/gpu_monitor_service.py

Purpose:
    Read live AMD GPU utilization and VRAM usage metrics for display in the
    main chat window.

What this file does:
    - Detects whether `amd-smi` or legacy `rocm-smi` is installed.
    - Polls live monitoring commands intended for changing utilization values,
      not just generic one-off metric summaries.
    - Prefers machine-readable AMD SMI JSON output so the UI does not depend
      on one fragile text layout.
    - Parses only the GPU% and VRAM% style values needed by the UI.
    - Computes VRAM% from used/total memory values when a CLI exposes raw
      memory sizes instead of a direct percentage field.
    - Returns a small normalized snapshot dictionary that the controller and UI
      can poll repeatedly.

How this file fits into the system:
    This service keeps machine-specific ROCm probing and text parsing out of the
    controller and Tkinter layers. The rest of the application only needs the
    normalized snapshot fields returned here.

Latest revision note:
    The earlier GPU readout pass used `amd-smi metric -u -m` and polled it from
    the Tkinter thread. That was too weak in two ways for this project:
    1. It did not reliably expose changing VRAM state on the user's system.
    2. The blocking subprocess call lived on the UI thread.

    This revision switches the AMD path to the live `amd-smi monitor` family
    and is designed so the UI can poll without freezing while model loads are
    happening.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class GPUMetricSnapshot:
    """Normalized single-sample GPU metric snapshot.

    Fields:
        gpu_percent: Current GPU utilization percentage as display text.
        vram_percent: Current VRAM usage percentage as display text.
        source: Which backend produced the reading.
        ok: True when a usable reading was collected.
        message: Human-readable status detail for troubleshooting.
    """

    gpu_percent: str
    vram_percent: str
    source: str
    ok: bool
    message: str


class GPUMonitorService:
    """Collect live GPU and VRAM percentage readings from ROCm tooling.

    The UI polls this service on a timer. Each poll runs one short command and
    parses only the two values the user asked to see in the main interface.
    """

    def get_live_metrics(self) -> Dict[str, str | bool]:
        """Return one normalized GPU metric snapshot for the current machine.

        Returns:
            Dictionary containing gpu_percent, vram_percent, source, ok, and
            message. The percentage fields are always present so the UI can
            render predictable text even when the probe fails.

        Polling behavior note:
            The main UI calls this repeatedly. This helper therefore prefers
            fast single-iteration monitoring commands instead of long-running
            live monitors that would block forever.
        """
        try:
            if shutil.which("amd-smi"):
                snapshot = self._collect_from_amd_smi()
                return snapshot.__dict__

            if shutil.which("rocm-smi"):
                output = self._run_command(["rocm-smi", "--showuse", "--showmemuse"])
                snapshot = self._parse_rocm_smi_output(output)
                return snapshot.__dict__

            return GPUMetricSnapshot(
                gpu_percent="N/A",
                vram_percent="N/A",
                source="unavailable",
                ok=False,
                message="Neither amd-smi nor rocm-smi was found on this machine.",
            ).__dict__
        except Exception as exc:
            return GPUMetricSnapshot(
                gpu_percent="N/A",
                vram_percent="N/A",
                source="error",
                ok=False,
                message=str(exc),
            ).__dict__

    def _collect_from_amd_smi(self) -> GPUMetricSnapshot:
        """Collect AMD SMI metrics using the live monitor command first.

        Why this helper changed:
            Official AMD SMI documentation and changelog notes show that
            `amd-smi monitor -v` is the command path that exposes live VRAM_USED,
            VRAM_TOTAL, and VRAM% style output. The earlier `metric -u -m` path
            could parse successfully yet still fail to reflect changing model
            allocations on the user's machine.

        Strategy:
            1. Prefer `amd-smi monitor -u -v --json -i 1` for one live sample.
            2. Fall back to text monitor output.
            3. Fall back to metric output for older environments.
        """
        monitor_json_commands = [
            ["amd-smi", "monitor", "-u", "-v", "--json", "-i", "1"],
            ["amd-smi", "monitor", "-u", "-v", "--json"],
        ]
        for command in monitor_json_commands:
            try:
                json_output = self._run_command(command)
                json_snapshot = self._parse_amd_smi_json_output(json_output)
                if json_snapshot.ok:
                    return GPUMetricSnapshot(
                        gpu_percent=json_snapshot.gpu_percent,
                        vram_percent=json_snapshot.vram_percent,
                        source="amd-smi-monitor",
                        ok=json_snapshot.ok,
                        message=json_snapshot.message,
                    )
            except Exception:
                pass

        monitor_text_commands = [
            ["amd-smi", "monitor", "-u", "-v", "-i", "1"],
            ["amd-smi", "monitor", "-u", "-v"],
        ]
        for command in monitor_text_commands:
            try:
                text_output = self._run_command(command)
                text_snapshot = self._parse_amd_smi_monitor_output(text_output)
                if text_snapshot.ok:
                    return text_snapshot
            except Exception:
                pass

        metric_commands = [
            ["amd-smi", "metric", "-u", "-m", "--json"],
            ["amd-smi", "metric", "-u", "-m"],
        ]
        last_error_message = "No AMD SMI command produced a usable reading."
        for command in metric_commands:
            try:
                output = self._run_command(command)
                if "--json" in command:
                    snapshot = self._parse_amd_smi_json_output(output)
                else:
                    snapshot = self._parse_amd_smi_output(output)
                if snapshot.ok:
                    return snapshot
                last_error_message = snapshot.message
            except Exception as exc:
                last_error_message = str(exc)

        return GPUMetricSnapshot(
            gpu_percent="N/A",
            vram_percent="N/A",
            source="amd-smi",
            ok=False,
            message=last_error_message,
        )

    def _run_command(self, command: List[str]) -> str:
        """Run one GPU metric command and return decoded stdout text."""
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout

    def _parse_amd_smi_json_output(self, output: str) -> GPUMetricSnapshot:
        """Parse AMD SMI JSON and normalize GPU% plus VRAM%.

        This parser is intentionally tolerant because AMD SMI JSON keys can vary
        by ROCm version. It recursively flattens the JSON structure, then looks
        for either direct percentage fields or enough used/total memory data to
        compute VRAM utilization.
        """
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return GPUMetricSnapshot(
                gpu_percent="N/A",
                vram_percent="N/A",
                source="amd-smi-json",
                ok=False,
                message="amd-smi JSON output could not be decoded.",
            )

        flattened = list(self._flatten_key_value_pairs(payload))
        gpu_value = self._extract_gpu_percent_from_pairs(flattened)
        vram_value = self._extract_vram_percent_from_pairs(flattened)

        if not vram_value:
            vram_value = self._compute_vram_percent_from_pairs(flattened)

        return GPUMetricSnapshot(
            gpu_percent=gpu_value or "N/A",
            vram_percent=vram_value or "N/A",
            source="amd-smi",
            ok=bool(gpu_value or vram_value),
            message=(
                "amd-smi JSON metric snapshot collected."
                if (gpu_value or vram_value)
                else "amd-smi JSON did not contain recognizable GPU/VRAM metrics."
            ),
        )

    def _parse_amd_smi_output(self, output: str) -> GPUMetricSnapshot:
        """Parse amd-smi text output into normalized GPU and VRAM percentages.

        The exact CLI presentation can differ by ROCm version, so this parser is
        intentionally tolerant. It scans for multiple likely label patterns and
        returns the first percentage value found for each category.
        """
        gpu_value = self._extract_percent(
            output,
            [
                r"GPU[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
                r"gfx[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
                r"GPU\s+use[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
                r"GPU\s+activity[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
            ],
        )
        vram_value = self._extract_percent(
            output,
            [
                r"VRAM[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
                r"memory[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
                r"GPU\s+memory\s+use[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
                r"memory\s+activity[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
                r"MEM\s+use[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
            ],
        )
        return GPUMetricSnapshot(
            gpu_percent=gpu_value or "N/A",
            vram_percent=vram_value or "N/A",
            source="amd-smi",
            ok=bool(gpu_value or vram_value),
            message=(
                "amd-smi metric snapshot collected."
                if (gpu_value or vram_value)
                else "amd-smi output did not contain recognizable GPU/VRAM percentages."
            ),
        )

    def _parse_amd_smi_monitor_output(self, output: str) -> GPUMetricSnapshot:
        """Parse amd-smi monitor text output into live GPU and VRAM percentages.

        The monitor command is the preferred live path because official AMD SMI
        documentation notes that `amd-smi monitor -v` includes VRAM_USED,
        VRAM_TOTAL, and VRAM% output. This parser accepts either a direct VRAM%
        column or used/total memory columns from which VRAM% can be computed.
        """
        gpu_value = self._extract_percent(
            output,
            [
                r"GPU(?:\s+use|\s+util(?:ization)?)?[^\n\r]*?(\d+(?:\.\d+)?)\s*%",
                r"(?:^|\s)GFX%[^\n\r]*?(\d+(?:\.\d+)?)\s*%",
                r"(?:^|\s)USE%[^\n\r]*?(\d+(?:\.\d+)?)\s*%",
            ],
        )
        vram_value = self._extract_percent(
            output,
            [
                r"VRAM%[^\n\r]*?(\d+(?:\.\d+)?)\s*%",
                r"VRAM[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
            ],
        )
        if not vram_value:
            vram_value = self._compute_vram_percent_from_monitor_text(output)
        return GPUMetricSnapshot(
            gpu_percent=gpu_value or "N/A",
            vram_percent=vram_value or "N/A",
            source="amd-smi-monitor",
            ok=bool(gpu_value or vram_value),
            message=(
                "amd-smi monitor snapshot collected."
                if (gpu_value or vram_value)
                else "amd-smi monitor output did not contain recognizable GPU/VRAM metrics."
            ),
        )

    def _parse_rocm_smi_output(self, output: str) -> GPUMetricSnapshot:
        """Parse rocm-smi output into normalized GPU and VRAM percentages."""
        gpu_value = self._extract_percent(
            output,
            [
                r"GPU[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
                r"use[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
            ],
        )
        vram_value = self._extract_percent(
            output,
            [
                r"GPU\s+memory[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
                r"mem(?:ory)?[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
                r"VRAM[^\n\r]*?:\s*(\d+(?:\.\d+)?)\s*%",
            ],
        )
        return GPUMetricSnapshot(
            gpu_percent=gpu_value or "N/A",
            vram_percent=vram_value or "N/A",
            source="rocm-smi",
            ok=bool(gpu_value or vram_value),
            message=(
                "rocm-smi metric snapshot collected."
                if (gpu_value or vram_value)
                else "rocm-smi output did not contain recognizable GPU/VRAM percentages."
            ),
        )

    def _extract_percent(self, text: str, patterns: List[str]) -> Optional[str]:
        """Return the first percentage match for any candidate regular expression."""
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return f"{match.group(1)}%"
        return None

    def _flatten_key_value_pairs(self, value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
        """Yield flattened `(path, value)` pairs from nested JSON-like data.

        This helper converts arbitrarily nested dict/list payloads into a flat
        stream of searchable key paths so version-specific AMD SMI JSON layouts
        can still be parsed without hard-coding one exact schema.
        """
        if isinstance(value, dict):
            for key, child in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from self._flatten_key_value_pairs(child, next_prefix)
            return

        if isinstance(value, list):
            for index, child in enumerate(value):
                next_prefix = f"{prefix}[{index}]"
                yield from self._flatten_key_value_pairs(child, next_prefix)
            return

        yield prefix, value

    def _extract_gpu_percent_from_pairs(self, pairs: List[tuple[str, Any]]) -> Optional[str]:
        """Return a GPU-usage percentage from flattened AMD SMI JSON fields."""
        for path, value in pairs:
            path_lower = path.lower()
            if any(excluded in path_lower for excluded in ["memory", "mem", "vram"]):
                continue
            if not any(keyword in path_lower for keyword in ["gpu", "gfx", "usage", "activity", "util"]):
                continue
            percent_value = self._coerce_percent_value(value)
            if percent_value is not None:
                return self._format_percent(percent_value)
        return None

    def _extract_vram_percent_from_pairs(self, pairs: List[tuple[str, Any]]) -> Optional[str]:
        """Return a direct VRAM percentage from flattened AMD SMI JSON fields."""
        for path, value in pairs:
            path_lower = path.lower()
            if not any(keyword in path_lower for keyword in ["vram", "memory", "mem"]):
                continue
            if not any(keyword in path_lower for keyword in ["percent", "usage", "use", "activity", "util"]):
                continue
            percent_value = self._coerce_percent_value(value)
            if percent_value is not None:
                return self._format_percent(percent_value)
        return None

    def _compute_vram_percent_from_pairs(self, pairs: List[tuple[str, Any]]) -> Optional[str]:
        """Compute VRAM% from used/total memory values when no direct percent exists."""
        used_value: Optional[float] = None
        total_value: Optional[float] = None

        for path, value in pairs:
            path_lower = path.lower()
            if not any(keyword in path_lower for keyword in ["vram", "memory", "mem"]):
                continue
            if any(excluded in path_lower for excluded in ["clock", "temp", "bandwidth", "ecc", "cache"]):
                continue
            numeric_value = self._coerce_memory_quantity(value)
            if numeric_value is None:
                continue

            if used_value is None and any(keyword in path_lower for keyword in ["used", "usage", "allocated", "in_use"]):
                used_value = numeric_value
                continue

            if total_value is None and any(keyword in path_lower for keyword in ["total", "max", "capacity"]):
                total_value = numeric_value

        if used_value is None or total_value in (None, 0):
            return None
        if used_value > total_value:
            return None

        return self._format_percent((used_value / total_value) * 100.0)

    def _compute_vram_percent_from_monitor_text(self, output: str) -> Optional[str]:
        """Compute VRAM% from amd-smi monitor text when only used/total columns exist.

        Example documented monitor output includes columns such as VRAM_USED,
        VRAM_TOTAL, and VRAM%. Some installations may omit the explicit VRAM%
        column or place values in a table layout that is easier to interpret
        through the raw used/total numbers. This helper handles that case.
        """
        normalized = output.replace(',', '')
        table_match = re.search(
            r"VRAM_USED\s+VRAM_FREE\s+VRAM_TOTAL(?:\s+VRAM%)?.*?\n\s*\d+\s+(\d+(?:\.\d+)?)\s*([KMGTP]i?B|B)\s+(?:\d+(?:\.\d+)?)\s*(?:[KMGTP]i?B|B)\s+(\d+(?:\.\d+)?)\s*([KMGTP]i?B|B)",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if table_match:
            used = self._coerce_memory_quantity(f"{table_match.group(1)} {table_match.group(2)}")
            total = self._coerce_memory_quantity(f"{table_match.group(3)} {table_match.group(4)}")
            if used is not None and total not in (None, 0) and used <= total:
                return self._format_percent((used / total) * 100.0)

        line_match = re.search(
            r"VRAM[_ ]USED[^\n\r]*?(\d+(?:\.\d+)?)\s*([KMGTP]i?B|B).*?VRAM[_ ]TOTAL[^\n\r]*?(\d+(?:\.\d+)?)\s*([KMGTP]i?B|B)",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if line_match:
            used = self._coerce_memory_quantity(f"{line_match.group(1)} {line_match.group(2)}")
            total = self._coerce_memory_quantity(f"{line_match.group(3)} {line_match.group(4)}")
            if used is not None and total not in (None, 0) and used <= total:
                return self._format_percent((used / total) * 100.0)

        return None

    def _coerce_percent_value(self, value: Any) -> Optional[float]:
        """Normalize a raw percentage-like value into a float from 0 to 100."""
        if isinstance(value, (int, float)):
            numeric = float(value)
            if 0.0 <= numeric <= 100.0:
                return numeric
            return None

        if not isinstance(value, str):
            return None

        percent_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", value)
        if percent_match:
            numeric = float(percent_match.group(1))
            if 0.0 <= numeric <= 100.0:
                return numeric
            return None

        if re.fullmatch(r"-?\d+(?:\.\d+)?", value.strip()):
            numeric = float(value.strip())
            if 0.0 <= numeric <= 100.0:
                return numeric
        return None

    def _coerce_memory_quantity(self, value: Any) -> Optional[float]:
        """Convert a memory quantity into a comparable byte-like float.

        The exact unit is not important as long as both used and total values
        are converted consistently. This helper supports plain numbers and the
        common IEC/SI byte suffixes seen in CLI output.
        """
        if isinstance(value, (int, float)):
            return float(value)

        if not isinstance(value, str):
            return None

        cleaned = value.strip().replace(",", "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*([KMGTP]?i?B)?", cleaned, flags=re.IGNORECASE)
        if not match:
            return None

        amount = float(match.group(1))
        unit = (match.group(2) or "").upper()
        multipliers = {
            "": 1.0,
            "B": 1.0,
            "KB": 1000.0,
            "MB": 1000.0 ** 2,
            "GB": 1000.0 ** 3,
            "TB": 1000.0 ** 4,
            "KIB": 1024.0,
            "MIB": 1024.0 ** 2,
            "GIB": 1024.0 ** 3,
            "TIB": 1024.0 ** 4,
        }
        multiplier = multipliers.get(unit)
        if multiplier is None:
            return None
        return amount * multiplier

    def _format_percent(self, value: float) -> str:
        """Format a float percentage as compact UI-ready text."""
        rounded = round(value, 1)
        if rounded.is_integer():
            return f"{int(rounded)}%"
        return f"{rounded}%"
