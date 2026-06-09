"""Tests for transcript growth guardrails in SessionStore.append_to_transcript.

Phase 1 objective: keep context stored in session transcripts compact while
preserving user-facing assistant behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from tools.tool_output_limits import get_tool_result_max_bytes


def _store_with_mocked_db(tmp_path: Path):
    """Build a SessionStore with a mocked SessionDB append target."""
    config = GatewayConfig()

    with patch("hermes_state.DEFAULT_DB_PATH", tmp_path / "state.db"):
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)

    mocked_db = MagicMock()
    store._db = mocked_db
    return store, mocked_db


def test_tool_non_text_payload_is_stored_as_summary(tmp_path):
    store, db = _store_with_mocked_db(tmp_path)

    giant_tool_payload = [{"text": "x" * 2048} for _ in range(300)]

    store.append_to_transcript(
        "session-1",
        {
            "role": "tool",
            "tool_name": "terminal",
            "content": giant_tool_payload,
        },
    )

    db.append_message.assert_called_once()
    saved = db.append_message.call_args.kwargs["content"]
    assert isinstance(saved, str)
    assert len(saved) <= get_tool_result_max_bytes()
    assert "tool_payload" in saved


def test_worker_long_result_is_compacted_before_store(tmp_path):
    store, db = _store_with_mocked_db(tmp_path)

    worker_body = "[WORKER RESULT: test]\n"
    worker_body += "\n".join(
        [f"artifact file-{i}: /tmp/artifact_{i}.txt" for i in range(50)]
    )
    worker_body += "\noutput_tail: " + ("x" * 2000)

    store.append_to_transcript(
        "session-1",
        {
            "role": "assistant",
            "content": worker_body,
        },
    )

    db.append_message.assert_called_once()
    saved = db.append_message.call_args.kwargs["content"]
    assert isinstance(saved, str)
    assert "[WORKER RESULT:" in saved
    assert "artifacts" in saved or "output_tail" in saved
    assert len(saved) < len(worker_body)


def test_assistant_response_content_stays_unmodified_for_large_text(tmp_path):
    store, db = _store_with_mocked_db(tmp_path)
    assistant_text = "assistant response " * 200

    store.append_to_transcript(
        "session-1",
        {
            "role": "assistant",
            "content": assistant_text,
        },
    )

    db.append_message.assert_called_once()
    saved = db.append_message.call_args.kwargs["content"]
    assert saved == assistant_text


def test_session_transcript_size_declines_with_tool_and_worker_payloads(tmp_path):
    store, db = _store_with_mocked_db(tmp_path)

    big_tool = "x" * (get_tool_result_max_bytes() + 5_000)
    worker_text = "[WORKER RESULT: test]\noutput_tail: " + ("y" * 8000)
    assistant_text = "assistant response unaffected"

    total_raw = len(big_tool) + len(worker_text) + len(assistant_text)

    store.append_to_transcript(
        "session-1",
        {
            "role": "tool",
            "tool_name": "terminal",
            "content": big_tool,
        },
    )
    store.append_to_transcript(
        "session-1",
        {
            "role": "assistant",
            "content": worker_text,
        },
    )
    store.append_to_transcript(
        "session-1",
        {
            "role": "assistant",
            "content": assistant_text,
        },
    )

    saved_contents = [call.kwargs["content"] for call in db.append_message.call_args_list]
    total_stored = sum(len(item or "") for item in saved_contents)
    assert total_stored < total_raw


def test_transcript_store_metrics_are_logged(tmp_path, caplog):
    store, db = _store_with_mocked_db(tmp_path)

    with caplog.at_level(logging.INFO):
        store.append_to_transcript(
            "session-1",
            {
                "role": "tool",
                "tool_name": "terminal",
                "content": "x" * 5000,
            },
        )

    assert any(
        "transcript_store: role=tool" in rec.message and "chars=" in rec.message
        for rec in caplog.records
    )
