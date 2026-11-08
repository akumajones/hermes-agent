"""Structured diagnostics for execute_code failures.

Three classes stay distinct:

* ``command`` — the cell raised. ``exception_type`` and ``exception_message``
  come from the runner, which still holds the exception object. The traceback
  text is extra detail, not the only copy of the type.
* ``postprocessing`` — the cell already finished and the host failed while
  formatting the reply. A command failure already captured on the payload is
  kept under ``command_failure`` and is not relabeled as the host error.
* ``transport`` — the kernel died or the frame never arrived. Stderr is kept
  head-and-tail so the exception line is not the part dropped. If the runner
  wrote a ``HERMES_CELL_EXCEPTION`` line before the reply failed, that command
  failure is nested and the top-level class stays ``transport``.

``error`` is one sanitized line that leads with the class and type, so a
head-truncated log still shows them. Secrets are redacted. A missing exception
line sets ``diagnostics_incomplete`` instead of guessing a type.
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from typing import Any

logger = logging.getLogger(__name__)

CELL_EXCEPTION_MARKER = "HERMES_CELL_EXCEPTION\t"

_EXC_LINE = re.compile(r"^([A-Za-z_][\w.]*):(.*)$")
_LEAD_KEYS = (
    "failure_class",
    "error",
    "exception_type",
    "exception_message",
    "traceback",
    "traceback_truncated",
    "diagnostics_incomplete",
    "command_failure",
    "runner_stderr",
)


def sanitize_diagnostic(text: str) -> str:
    """Redact secrets and strip ANSI. On sanitizer failure, drop the text."""
    if not text:
        return ""
    try:
        from agent.redact import redact_sensitive_text
        from tools.ansi_strip import strip_ansi

        cleaned = redact_sensitive_text(strip_ansi(str(text)), code_file=True)
    except Exception:
        logger.exception("execute_code diagnostic sanitizer failed")
        return "[diagnostic omitted: sanitizer failed]"
    if not isinstance(cleaned, str):
        return "[diagnostic omitted: sanitizer failed]"
    return cleaned.encode("utf-8", errors="replace").decode("utf-8")


def _one_line(text: str, limit: int = 500) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def parse_traceback_exception(trace: str) -> tuple[str, str]:
    """Return ``(type, message)`` from the last ``Type: message`` line."""
    for line in reversed((trace or "").splitlines()):
        match = _EXC_LINE.fullmatch(line.strip())
        if match:
            return match.group(1), match.group(2).strip()
    return "", ""


def parse_cell_marker(stderr: str) -> tuple[str, str] | None:
    """A runner line written before the reply, so a dead kernel can still name the cell."""
    for line in (stderr or "").splitlines():
        if not line.startswith(CELL_EXCEPTION_MARKER):
            continue
        exc_type, _sep, message = line[len(CELL_EXCEPTION_MARKER):].partition("\t")
        exc_type = exc_type.strip()
        if exc_type:
            return exc_type, message.strip()
    return None


def command_failure_fields(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Command-failure fields from a runner payload. Does not include stdout."""
    payload = payload or {}
    trace = sanitize_diagnostic(str(payload.get("traceback") or ""))
    exc_type = str(payload.get("exception_type") or "").strip()
    message = str(payload.get("exception_message") or "")
    if not exc_type or not str(message).strip():
        parsed_type, parsed_message = parse_traceback_exception(trace)
        exc_type = exc_type or parsed_type
        message = message or parsed_message
    exc_type = sanitize_diagnostic(exc_type).strip()
    message = sanitize_diagnostic(str(message)).strip()
    truncated = bool(
        payload.get("traceback_truncated") or payload.get("exception_message_truncated")
    )
    incomplete = not exc_type
    if exc_type:
        summary = f"command failure: {exc_type}: {message}"
    else:
        summary = "command failure: exception type was not captured"
    if incomplete:
        summary += "; diagnostics incomplete"
    return {
        "failure_class": "command",
        "error": _one_line(summary),
        "exception_type": exc_type or None,
        "exception_message": message,
        "traceback": trace,
        "traceback_truncated": truncated,
        "diagnostics_incomplete": incomplete,
    }


def host_exception_fields(exc: BaseException, *, failure_class: str) -> dict[str, Any]:
    """Sanitized fields for an exception raised in this process. Call from ``except``."""
    exc_type = type(exc).__name__
    message = sanitize_diagnostic(str(exc))
    trace = sanitize_diagnostic(traceback.format_exc())
    return {
        "failure_class": failure_class,
        "error": _one_line(f"{failure_class} failure: {exc_type}: {message}"),
        "exception_type": exc_type,
        "exception_message": message,
        "traceback": trace,
        "traceback_truncated": False,
        "diagnostics_incomplete": False,
    }


def postprocessing_failure_fields(
    exc: BaseException, payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Host formatting failed. Nested ``command_failure`` is set only if the cell had already raised."""
    fields = host_exception_fields(exc, failure_class="postprocessing")
    command = None
    if isinstance(payload, dict) and payload.get("status") == "error":
        command = command_failure_fields(payload)
        fields["error"] = _one_line(f"{fields['error']}; {command['error']}")
    fields["command_failure"] = command
    return fields


def transport_failure_fields(stderr: str) -> dict[str, Any]:
    """Kernel died or the frame never arrived. Preserves a prior command failure if the runner noted one."""
    clean = sanitize_diagnostic(stderr or "")
    marker = parse_cell_marker(clean)
    command = None
    exc_type, message = "", ""
    if marker:
        exc_type, message = marker
        command = {
            "failure_class": "command",
            "error": _one_line(f"command failure: {exc_type}: {message}"),
            "exception_type": exc_type,
            "exception_message": message,
            "traceback": "",
            "traceback_truncated": False,
            "diagnostics_incomplete": False,
        }
    elif clean:
        exc_type, message = parse_traceback_exception(clean)
    summary = "transport failure: session kernel died while running the cell"
    if command:
        summary += f"; {command['error']}"
    elif exc_type:
        summary += f": {exc_type}: {message}"
    else:
        summary += "; diagnostics incomplete"
    return {
        "failure_class": "transport",
        "error": _one_line(summary),
        "exception_type": exc_type or None,
        "exception_message": message,
        "traceback": "",
        "runner_stderr": clean,
        "traceback_truncated": "STDERR TRUNCATED" in clean,
        "diagnostics_incomplete": command is None and not exc_type,
        "command_failure": command,
    }


def place_failure_fields(result: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    """Put the failure summary ahead of stdout so a head-truncated log still shows the type."""
    ordered: dict[str, Any] = {"status": result.get("status", "error")}
    for key in _LEAD_KEYS:
        if key in fields:
            ordered[key] = fields[key]
    for key, value in result.items():
        if key not in ordered:
            ordered[key] = value
    for key, value in fields.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def dumps_result(result: dict[str, Any]) -> str:
    """JSON for a tool result. A serialization failure is itself postprocessing, not a command error."""
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({
            "status": "error",
            "failure_class": "postprocessing",
            "error": "postprocessing failure: result was not JSON-serializable",
            "diagnostics_incomplete": True,
            "command_failure": None,
        })
