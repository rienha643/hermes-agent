import json
from unittest.mock import patch

from gateway.run import (
    _extract_explicit_worker_label,
    _extract_explicit_worker_labels,
    _explicit_worker_direct_handling_allowed,
    _run_explicit_worker_delegation,
    _strip_routing_context_prefixes,
)


class DummyAgent:
    def __init__(self):
        self.tool_progress_callback = None


def test_extract_explicit_worker_label():
    assert _extract_explicit_worker_label("[WORKER: Eclipse]\nDo it") == "Eclipse"
    assert _extract_explicit_worker_label("hello") is None


def test_extract_explicit_worker_label_ignores_reply_and_thread_context_prefixes():
    message = (
        '[Replying to: "[WORKER: Palette] old"]\n\n'
        "[Thread context — prior messages in this thread (not yet in conversation history):]\n"
        "[thread parent] 두목: [WORKER: Palette] 이전 지시\n"
        "[End of thread context]\n\n"
        "[WORKER: Eclipse]\n테스트"
    )
    assert _extract_explicit_worker_label(message) == "Eclipse"
    assert _extract_explicit_worker_labels(message) == ["Eclipse"]


def test_extract_explicit_worker_labels_dedupes_same_worker():
    message = "[WORKER: Eclipse]\n[WORKER: Eclipse]\n테스트"
    assert _extract_explicit_worker_labels(message) == ["Eclipse"]
    assert _extract_explicit_worker_label(message) == "Eclipse"


def test_extract_explicit_worker_labels_returns_multiple_distinct_workers():
    message = "[WORKER: Eclipse]\n[WORKER: Palette]\n테스트"
    assert _extract_explicit_worker_labels(message) == ["Eclipse", "Palette"]
    assert _extract_explicit_worker_label(message) is None


def test_strip_routing_context_prefixes_preserves_current_body():
    message = (
        '[Replying to: "[WORKER: Palette] old"]\n\n'
        "[Thread context — prior messages in this thread (not yet in conversation history):]\n"
        "[thread parent] 두목: [WORKER: Palette] 이전 지시\n"
        "[End of thread context]\n\n"
        "[WORKER: Eclipse]\n테스트"
    )
    assert _strip_routing_context_prefixes(message) == "[WORKER: Eclipse]\n테스트"


def test_direct_handling_escape_detected():
    assert _explicit_worker_direct_handling_allowed("[WORKER: Eclipse]\n직접 처리해") is True
    assert _explicit_worker_direct_handling_allowed("[WORKER: Eclipse]\n현재 시간 알려줘") is False


def test_explicit_worker_eclipse_forces_delegate_task():
    agent = DummyAgent()
    payload = {
        "results": [
            {
                "status": "completed",
                "summary": "UX_VALIDATION_OK",
                "api_calls": 1,
                "artifacts": [],
            }
        ]
    }
    with patch("tools.delegate_tool.delegate_task", return_value=json.dumps(payload)) as mocked:
        result = _run_explicit_worker_delegation(agent, "[WORKER: Eclipse]\n현재 시간 한 줄로 알려줘")
    assert result is not None
    mocked.assert_called_once()
    assert mocked.call_args.kwargs["profile"] == "coder"
    assert "[WORKER:" not in mocked.call_args.kwargs["goal"]
    assert "[WORKER RESULT: Eclipse]" in result["final_response"]
    assert "UX_VALIDATION_OK" in result["final_response"]
    assert result["completed"] is True


def test_explicit_worker_palette_forces_delegate_task():
    agent = DummyAgent()
    payload = {
        "results": [
            {
                "status": "completed",
                "summary": "PALETTE_OK",
                "api_calls": 1,
                "artifacts": [],
            }
        ]
    }
    with patch("tools.delegate_tool.delegate_task", return_value=json.dumps(payload)) as mocked:
        result = _run_explicit_worker_delegation(agent, "[WORKER: Palette]\n한 줄만 답해")
    assert result is not None
    mocked.assert_called_once()
    assert mocked.call_args.kwargs["profile"] == "artist"
    assert "[WORKER RESULT: Palette]" in result["final_response"]
    assert "PALETTE_OK" in result["final_response"]


def test_non_directive_worker_mention_stays_direct():
    agent = DummyAgent()
    with patch("tools.delegate_tool.delegate_task") as mocked:
        result = _run_explicit_worker_delegation(agent, "Eclipse가 예전에 뭐 했지?")
    assert result is None
    mocked.assert_not_called()


def test_explicit_worker_unknown_returns_delegation_failed():
    agent = DummyAgent()
    with patch("tools.delegate_tool.delegate_task") as mocked:
        result = _run_explicit_worker_delegation(agent, "[WORKER: UnknownWorker]\n뭔가 해줘")
    assert result is not None
    mocked.assert_not_called()
    assert "[WORKER RESULT: UnknownWorker]" in result["final_response"]
    assert "delegation failed" in result["final_response"]
    assert result["completed"] is False


def test_explicit_worker_direct_handling_escape_bypasses_delegation():
    agent = DummyAgent()
    with patch("tools.delegate_tool.delegate_task") as mocked:
        result = _run_explicit_worker_delegation(agent, "[WORKER: Eclipse]\n직접 처리해")
    assert result is None
    mocked.assert_not_called()


def test_reply_context_worker_directive_does_not_trigger_delegation():
    agent = DummyAgent()
    message = '[Replying to: "[WORKER: Eclipse] old"]\n\n오늘 상태 알려줘'
    with patch("tools.delegate_tool.delegate_task") as mocked:
        result = _run_explicit_worker_delegation(agent, message)
    assert result is None
    mocked.assert_not_called()


def test_reply_context_direct_handling_uses_current_body_only():
    agent = DummyAgent()
    message = '[Replying to: "[WORKER: Palette] old"]\n\n직접 처리해'
    with patch("tools.delegate_tool.delegate_task") as mocked:
        result = _run_explicit_worker_delegation(agent, message)
    assert result is None
    mocked.assert_not_called()


def test_duplicate_same_worker_directives_delegate_once():
    agent = DummyAgent()
    payload = {
        "results": [
            {
                "status": "completed",
                "summary": "ECLIPSE_OK",
                "api_calls": 1,
                "artifacts": [],
            }
        ]
    }
    with patch("tools.delegate_tool.delegate_task", return_value=json.dumps(payload)) as mocked:
        result = _run_explicit_worker_delegation(agent, "[WORKER: Eclipse]\n[WORKER: Eclipse]\n테스트")
    assert result is not None
    mocked.assert_called_once()
    assert mocked.call_args.kwargs["profile"] == "coder"
    assert "[WORKER RESULT: Eclipse]" in result["final_response"]


def test_multi_worker_directives_return_unsupported_message_without_delegation():
    agent = DummyAgent()
    with patch("tools.delegate_tool.delegate_task") as mocked:
        result = _run_explicit_worker_delegation(agent, "[WORKER: Eclipse]\n[WORKER: Palette]\n테스트")
    assert result is not None
    mocked.assert_not_called()
    assert result["completed"] is False
    assert "[WORKER RESULT: MULTI_WORKER_UNSUPPORTED]" in result["final_response"]
    assert "multi-worker delegation is not supported in one turn" in result["final_response"]
