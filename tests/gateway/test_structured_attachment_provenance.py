import json
from pathlib import Path
from typing import Any

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import _collect_turn_scoped_structured_attachments
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, error=None, message_id="send")

    async def get_chat_info(self, chat_id):
        return {}


def _tool_message(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_name": tool_name,
        "content": json.dumps(payload),
    }


def test_search_files_files_are_not_structured_attachments(tmp_path):
    first = tmp_path / "a.md"
    second = tmp_path / "b.docx"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    agent_result = {
        "messages": [
            _tool_message("search_files", {"total_count": 2, "files": [str(first), str(second)]})
        ]
    }

    collected, debug = _collect_turn_scoped_structured_attachments(agent_result, history_len=0)

    assert collected == []
    assert debug["collected_count"] == 0


def test_explicit_document_files_are_collected(tmp_path):
    final_doc = tmp_path / "final.docx"
    final_doc.write_text("doc", encoding="utf-8")
    agent_result = {
        "messages": [
            _tool_message("document_generate", {"document_files": [str(final_doc)]})
        ]
    }

    collected, debug = _collect_turn_scoped_structured_attachments(agent_result, history_len=0)

    assert collected == [str(final_doc)]
    assert debug["collected_count"] == 1


@pytest.mark.parametrize("tool_name", ["search_files", "read_file", "terminal", "execute_code"])
def test_untrusted_tool_output_paths_are_blocked(tool_name, tmp_path):
    candidate = tmp_path / "leaked.md"
    candidate.write_text("leaked", encoding="utf-8")
    agent_result = {
        "messages": [
            _tool_message(
                tool_name,
                {
                    "files": [str(candidate)],
                    "content": f"ordinary stdout mentions {candidate}",
                    "output": f"ordinary stdout mentions {candidate}",
                },
            )
        ]
    }

    collected, _debug = _collect_turn_scoped_structured_attachments(agent_result, history_len=0)

    assert collected == []


def test_files_field_requires_explicit_delivery_intent_for_artifact_tool(tmp_path):
    final_doc = tmp_path / "final.docx"
    final_doc.write_text("doc", encoding="utf-8")
    blocked_doc = tmp_path / "blocked.docx"
    blocked_doc.write_text("blocked", encoding="utf-8")
    agent_result = {
        "messages": [
            _tool_message("search_files", {"files": [str(blocked_doc)]}),
            _tool_message(
                "document_generate",
                {"user_requested_delivery": True, "files": [str(final_doc)]},
            ),
        ]
    }

    collected, _debug = _collect_turn_scoped_structured_attachments(agent_result, history_len=0)

    assert collected == [str(final_doc)]


@pytest.mark.asyncio
async def test_excessive_structured_attachments_do_not_publish_documents(monkeypatch, tmp_path):
    structured_paths = []
    for idx in range(200):
        path = tmp_path / f"polluted-{idx}.md"
        path.write_text("polluted", encoding="utf-8")
        structured_paths.append(str(path))

    publish_calls: list[str] = []
    sent_documents: list[str] = []
    sent_texts: list[str] = []

    adapter = _StubAdapter(PlatformConfig(enabled=True, token="test"), Platform.SLACK)

    async def _handler(_event: MessageEvent) -> str:
        return "done"

    async def _send_with_retry(
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: Any = None,
        max_retries: int = 2,
        base_delay: float = 2,
    ) -> SendResult:
        sent_texts.append(content)
        return SendResult(success=True, error=None, message_id="text")

    async def _send_document(
        chat_id: str,
        file_path: str,
        caption: str | None = None,
        file_name: str | None = None,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> SendResult:
        sent_documents.append(file_path)
        return SendResult(success=True, error=None, message_id="doc")

    def _publish_document_artifact(path: Path, *, folder_name=None):
        publish_calls.append(str(path))
        return path

    monkeypatch.setattr("gateway.platforms.base.validate_media_delivery_path", lambda path: str(Path(path).resolve()))
    monkeypatch.setattr("gateway.platforms.base.publish_document_artifact", _publish_document_artifact)

    adapter._message_handler = _handler
    adapter._send_with_retry = _send_with_retry
    adapter.send_document = _send_document
    event = MessageEvent(
        text="trigger",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id="C123", chat_type="channel"),
    )
    setattr(event, "_structured_attachment_paths", structured_paths)

    await adapter._process_message_background(event, build_session_key(event.source))

    assert sent_texts == ["done"]
    assert publish_calls == []
    assert sent_documents == []
