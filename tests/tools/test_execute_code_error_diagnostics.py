"""execute_code reports command failures and postprocessing failures separately.

No live network. One case runs a local kernel cell; the rest use payloads.
"""

from __future__ import annotations

import json
import os
import threading
from unittest.mock import MagicMock, patch

os.environ.setdefault("TERMINAL_ENV", "local")

from tools.code_execution_errors import (
    command_failure_fields,
    postprocessing_failure_fields,
    transport_failure_fields,
)
from tools.code_kernel import _BoundedBuffer, _run_cell

_SECRET = "sk-abcdefghijklmnopqrstuvwxyz"


def _command_payload():
    trace = (
        'Traceback (most recent call last):\n'
        '  File "/tmp/hermes_kernel_x/hermes_kernel_runner.py", line 84, in run_cell\n'
        "    exec(compile(request['code'], '<cell>', 'exec'), GLOBALS)\n"
        '  File "<cell>", line 2, in <module>\n'
        "    A().missing\n"
        "AttributeError: 'A' object has no attribute 'missing'\n"
    )
    return {
        "status": "error",
        "exception_type": "AttributeError",
        "exception_message": f"missing {_SECRET}",
        "traceback": trace + f"token {_SECRET}\n",
        "traceback_truncated": False,
    }


def test_command_failure_leads_with_the_exception_type_and_redacts():
    fields = command_failure_fields(_command_payload())
    rendered = json.dumps({"status": "error", **fields}, ensure_ascii=False)
    assert fields["failure_class"] == "command"
    assert fields["exception_type"] == "AttributeError"
    assert "AttributeError" in rendered[:200]
    assert "has no attribute" in fields["traceback"]
    assert _SECRET not in rendered
    assert fields["diagnostics_incomplete"] is False


def test_postprocessing_failure_keeps_the_command_exception():
    try:
        raise RuntimeError(f"formatter exploded {_SECRET}")
    except RuntimeError as exc:
        fields = postprocessing_failure_fields(exc, _command_payload())
    assert fields["failure_class"] == "postprocessing"
    assert fields["exception_type"] == "RuntimeError"
    assert fields["command_failure"]["failure_class"] == "command"
    assert fields["command_failure"]["exception_type"] == "AttributeError"
    blob = json.dumps(fields)
    assert "RuntimeError" in blob[:400]
    assert "AttributeError" in fields["error"]
    assert _SECRET not in blob


def test_postprocessing_of_a_successful_cell_is_not_a_command_failure():
    try:
        raise ValueError("could not encode stdout")
    except ValueError as exc:
        fields = postprocessing_failure_fields(exc, {"status": "ok", "stdout": "hi"})
    assert fields["failure_class"] == "postprocessing"
    assert fields["command_failure"] is None
    assert "command failure" not in fields["error"]
    assert fields["exception_type"] == "ValueError"


def test_transport_marker_preserves_the_command_exception():
    stderr = "partial runner noise\nHERMES_CELL_EXCEPTION\tAttributeError\tmissing\n"
    fields = transport_failure_fields(stderr)
    assert fields["failure_class"] == "transport"
    assert fields["command_failure"]["exception_type"] == "AttributeError"
    assert fields["diagnostics_incomplete"] is False
    assert "transport failure" in fields["error"]
    assert "AttributeError" in fields["error"]


def test_truncated_runner_stderr_without_an_exception_line_is_incomplete():
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/tmp/hermes_kernel_x/hermes_kernel_runner.py", line 84, in run_cell\n'
    )
    fields = transport_failure_fields(stderr)
    assert fields["failure_class"] == "transport"
    assert fields["diagnostics_incomplete"] is True
    assert not fields["exception_type"]
    assert fields["command_failure"] is None
    assert "line 84, in run_cell" in fields["runner_stderr"]


def test_stderr_buffer_keeps_the_exception_tail():
    buf = _BoundedBuffer(keep_tail=True)
    buf.append(b"N" * 8000, 1000)
    buf.append(b"\nAttributeError: tail\n", 1000)
    text = buf.drain()
    assert "AttributeError: tail" in text
    assert "STDERR TRUNCATED" in text


def test_stderr_buffer_under_the_cap_is_unchanged():
    buf = _BoundedBuffer(keep_tail=True)
    buf.append(b"AttributeError: whole\n", 1000)
    assert buf.drain() == "AttributeError: whole\n"


def test_run_cell_postprocessing_exception_is_not_a_command_failure():
    kernel = MagicMock()
    kernel.proc = MagicMock()
    kernel.proc.stdin = MagicMock()
    kernel.lock = threading.Lock()
    kernel.tool_call_counter = [0]
    kernel.tool_call_log = []
    kernel.raw.drain.return_value = ""
    kernel.stderr.drain.return_value = ""
    payload = _command_payload()
    with patch("tools.code_kernel._await_cell", return_value=("success", payload)), \
         patch("tools.code_kernel._cell_result", side_effect=RuntimeError("formatter exploded")):
        raw = _run_cell(
            kernel, ("owner",), "A().missing", task_id="t", child_python="python",
            child_cwd="", sandbox_tools=frozenset(), timeout=5, max_tool_calls=1,
            is_interrupted=lambda: False, exec_start=0.0, state_reset=False,
        )
    result = json.loads(raw)
    assert result["failure_class"] == "postprocessing"
    assert result["exception_type"] == "RuntimeError"
    assert result["command_failure"]["exception_type"] == "AttributeError"
    assert "AttributeError" in result["error"]
    # The formatter failed after the cell returned; the kernel is not discarded here.
    assert kernel.teardown.called is False
    assert payload["exception_type"] == "AttributeError"


def test_live_cell_reports_attribute_error_as_a_command_failure():
    from tools.code_execution_tool import execute_code

    code = "class A:\n    pass\nA().missing\n"
    config = {"mode": "strict", "timeout": 30}
    with patch("tools.code_execution_tool._load_config", return_value=config):
        result = json.loads(execute_code(code, task_id="diag-kernel"))
    try:
        assert result["failure_class"] == "command"
        assert result["exception_type"] == "AttributeError"
        assert "AttributeError" in json.dumps(result)[:200]
        assert "has no attribute" in result["traceback"]
        assert result["diagnostics_incomplete"] is False
    finally:
        from tools.code_kernel import shutdown_all_kernels
        shutdown_all_kernels()
