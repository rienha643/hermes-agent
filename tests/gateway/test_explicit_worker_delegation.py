import json
from unittest.mock import patch

from gateway.run import (
    _extract_explicit_worker_label,
    _explicit_worker_direct_handling_allowed,
    _run_explicit_worker_delegation,
)


class DummyAgent:
    def __init__(self):
        self.tool_progress_callback = None


def test_extract_explicit_worker_label():
    assert _extract_explicit_worker_label("[WORKER: Eclipse]\nDo it") == "Eclipse"
    assert _extract_explicit_worker_label("hello") is None


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
