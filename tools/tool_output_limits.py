"""Configurable tool-output truncation limits.

Ported from anomalyco/opencode PR #23770 (``feat(truncate): allow
configuring tool output truncation limits``).

OpenCode hardcoded ``MAX_LINES = 2000`` and ``MAX_BYTES = 50 * 1024``
as tool-output truncation thresholds. Hermes-agent had the same
hardcoded constants in two places:

* ``tools/terminal_tool.py`` — ``MAX_OUTPUT_CHARS = 50000`` (terminal
  stdout/stderr cap)
* ``tools/file_operations.py`` — ``MAX_LINES = 2000`` /
  ``MAX_LINE_LENGTH = 2000`` (read_file pagination cap + per-line cap)

This module centralises those values behind a single config section
(``tool_output`` in ``config.yaml``) so power users can tune them
without patching the source. The existing hardcoded numbers remain as
defaults, so behaviour is unchanged when the config key is absent.

Example ``config.yaml``::

    tool_output:
      max_bytes: 100000        # terminal output cap (chars)
      max_lines: 5000          # read_file pagination + truncation cap
      max_line_length: 2000    # per-line length cap before '... [truncated]'

The limits reader is defensive: any error (missing config file, invalid
value type, etc.) falls back to the built-in defaults so tools never
fail because of a malformed config.
"""

from __future__ import annotations

from typing import Any, Dict

# Hardcoded defaults — these match the pre-existing values, so adding
# this module is behaviour-preserving for users who don't set
# ``tool_output`` in config.yaml.
DEFAULT_MAX_BYTES = 50_000       # terminal_tool.MAX_OUTPUT_CHARS
DEFAULT_MAX_LINES = 2000         # file_operations.MAX_LINES
DEFAULT_MAX_LINE_LENGTH = 2000   # file_operations.MAX_LINE_LENGTH

# Tool-result truncation defaults (Phase 1 of the hard-cap rollout).
# Kept intentionally independent from terminal/file caps so tool payload limits are
# stable even if those other limits are tuned via config.
DEFAULT_TOOL_RESULT_MAX_BYTES = 12_000
DEFAULT_TOOL_RESULT_MAX_LINES = 200
TOOL_RESULT_TRUNCATION_MARKER = "[truncated by Hermes tool output budget]"


def _coerce_positive_int(value: Any, default: int) -> int:
    """Return ``value`` as a positive int, or ``default`` on any issue."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    if iv <= 0:
        return default
    return iv


def get_tool_output_limits() -> Dict[str, int]:
    """Return resolved tool-output limits, reading ``tool_output`` from config.

    Keys: ``max_bytes``, ``max_lines``, ``max_line_length``. Missing or
    invalid entries fall through to the ``DEFAULT_*`` constants. This
    function NEVER raises.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        section = cfg.get("tool_output") if isinstance(cfg, dict) else None
        if not isinstance(section, dict):
            section = {}
    except Exception:
        section = {}

    return {
        "max_bytes": _coerce_positive_int(section.get("max_bytes"), DEFAULT_MAX_BYTES),
        "max_lines": _coerce_positive_int(section.get("max_lines"), DEFAULT_MAX_LINES),
        "max_line_length": _coerce_positive_int(
            section.get("max_line_length"), DEFAULT_MAX_LINE_LENGTH
        ),
    }


def get_max_bytes() -> int:
    """Shortcut for terminal-tool callers that only need the byte cap."""
    return get_tool_output_limits()["max_bytes"]


def get_max_lines() -> int:
    """Shortcut for file-ops callers that only need the line cap."""
    return get_tool_output_limits()["max_lines"]


def get_max_line_length() -> int:
    """Shortcut for file-ops callers that only need the per-line cap."""
    return get_tool_output_limits()["max_line_length"]


def get_tool_result_max_bytes() -> int:
    """Shortcut for tool-result callers that only need the tool payload byte cap."""
    return DEFAULT_TOOL_RESULT_MAX_BYTES


def get_tool_result_max_lines() -> int:
    """Shortcut for tool-result callers that only need the tool payload line cap."""
    return DEFAULT_TOOL_RESULT_MAX_LINES


def truncate_tool_output(
    text: str,
    max_chars: int | None = None,
    max_lines: int | None = None,
) -> str:
    """Return text truncated with a marker inserted in the middle.

    Hard-cap policy for Phase 1:
    - default cap: 12,000 chars OR 200 lines, whichever is reached first
    - preserve head/tail and emit a marker in between
    - plain-string-only (callers decide whether to pass non-string payloads through)
    """
    if not text:
        return text

    effective_max_chars = max_chars if max_chars is not None else DEFAULT_TOOL_RESULT_MAX_BYTES
    effective_max_lines = max_lines if max_lines is not None else DEFAULT_TOOL_RESULT_MAX_LINES

    if len(text) <= effective_max_chars:
        lines = text.splitlines()
        if len(lines) <= effective_max_lines:
            return text

    marker = TOOL_RESULT_TRUNCATION_MARKER
    marker_len = len(marker)

    # If char cap is the stricter immediate constraint, truncate by bytes first.
    if len(text) > effective_max_chars:
        available = max(0, effective_max_chars - marker_len)
        if available <= 0:
            return text[:effective_max_chars]
        head_chars = available // 2
        tail_chars = available - head_chars
        text = text[:head_chars] + marker + text[-tail_chars:]

    # Apply line-aware truncation if lines are still too many.
    lines = text.splitlines()
    if len(lines) <= effective_max_lines:
        return text

    keep = max(0, (effective_max_lines - 1) // 2)
    head = lines[:keep]
    tail = lines[-keep:] if keep > 0 else []

    return "\n".join(head) + "\n" + marker + "\n" + "\n".join(tail)
