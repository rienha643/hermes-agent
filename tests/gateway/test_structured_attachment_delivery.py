from __future__ import annotations

from typing import Any

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import (
    GatewayRunner,
    _collect_structured_attachment_paths,
    _collect_turn_scoped_structured_attachments,
    _slice_turn_messages,
)
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


class _MiniRunner:
    _deliver_media_from_response = GatewayRunner._deliver_media_from_response
    _thread_metadata_for_source = GatewayRunner._thread_metadata_for_source
    _reply_anchor_for_event = staticmethod(GatewayRunner._reply_anchor_for_event)


def _make_event(*, thread_id: str | None = None) -> MessageEvent:
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="channel",
        thread_id=thread_id,
    )
    event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=source)
    event.message_id = "m-1"
    return event


def _make_adapter(response_text: str) -> _StubAdapter:
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="test"), Platform.SLACK)
    adapter.sent_texts: list[str] = []
    adapter.sent_images: list[dict[str, Any]] = []
    adapter.sent_documents: list[dict[str, Any]] = []

    async def _handler(_event: MessageEvent) -> str:
        return response_text

    async def _send_with_retry(
        chat_id: str,
        content: str,
        reply_to=None,
        metadata: Any = None,
        max_retries: int = 2,
        base_delay: float = 2,
    ) -> SendResult:
        adapter.sent_texts.append(content)
        return SendResult(success=True, error=None, message_id="text")

    async def _send_multiple_images(
        chat_id: str,
        images,
        caption=None,
        reply_to=None,
        metadata=None,
        human_delay: float = 0,
    ) -> SendResult:
        adapter.sent_images.append({"images": list(images), "metadata": metadata})
        return SendResult(success=True, error=None, message_id="img")

    async def _send_document(
        chat_id: str,
        file_path: str,
        caption=None,
        file_name=None,
        reply_to=None,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        adapter.sent_documents.append({"file_path": file_path, "metadata": metadata})
        return SendResult(success=True, error=None, message_id="doc")

    adapter._message_handler = _handler
    adapter._send_with_retry = _send_with_retry
    adapter.send_multiple_images = _send_multiple_images
    adapter.send_document = _send_document
    return adapter


@pytest.fixture
def allow_tmp_delivery(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_structured_attachment_without_media_tag_is_uploaded_once_with_thread_metadata(allow_tmp_delivery):
    image_path = allow_tmp_delivery / "avatar.png"
    image_path.write_bytes(b"png")

    adapter = _make_adapter("완료했습니다.")
    event = _make_event(thread_id="1779980356.888399")
    event._structured_attachment_paths = [str(image_path)]

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent_texts == ["완료했습니다."]
    assert len(adapter.sent_images) == 1
    assert adapter.sent_images[0]["metadata"] == {"thread_id": "1779980356.888399", "notify": True}
    batch_urls = [item[0] for item in adapter.sent_images[0]["images"]]
    assert len(batch_urls) == 1
    assert str(image_path) in batch_urls[0]


@pytest.mark.asyncio
async def test_structured_attachment_and_media_tag_are_deduped(allow_tmp_delivery):
    image_path = allow_tmp_delivery / "avatar.png"
    image_path.write_bytes(b"png")

    adapter = _make_adapter(f"완료했습니다.\nMEDIA:{image_path}")
    event = _make_event()
    event._structured_attachment_paths = [str(image_path)]

    await adapter._process_message_background(event, build_session_key(event.source))

    assert len(adapter.sent_images) == 1
    batch_urls = [item[0] for item in adapter.sent_images[0]["images"]]
    assert len(batch_urls) == 1


@pytest.mark.asyncio
async def test_media_only_response_still_works(allow_tmp_delivery):
    image_path = allow_tmp_delivery / "avatar.png"
    image_path.write_bytes(b"png")

    adapter = _make_adapter(f"완료했습니다.\nMEDIA:{image_path}")
    event = _make_event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert len(adapter.sent_images) == 1
    assert adapter.sent_texts == ["완료했습니다."]


@pytest.mark.asyncio
async def test_text_only_response_does_not_attempt_attachment_send(allow_tmp_delivery):
    adapter = _make_adapter("텍스트만 보냅니다.")
    event = _make_event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent_texts == ["텍스트만 보냅니다."]
    assert adapter.sent_images == []
    assert adapter.sent_documents == []


@pytest.mark.asyncio
async def test_invalid_structured_attachment_path_is_ignored_safely(allow_tmp_delivery):
    adapter = _make_adapter("완료했습니다.")
    event = _make_event()
    event._structured_attachment_paths = [str(allow_tmp_delivery / "missing.png")]

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent_texts == ["완료했습니다."]
    assert adapter.sent_images == []
    assert adapter.sent_documents == []


@pytest.mark.asyncio
async def test_post_stream_structured_attachment_without_media_tag_is_uploaded(allow_tmp_delivery):
    image_path = allow_tmp_delivery / "stream-avatar.png"
    image_path.write_bytes(b"png")

    adapter = _make_adapter("unused")
    event = _make_event(thread_id="1779980356.888399")
    event._structured_attachment_paths = [str(image_path)]

    runner = _MiniRunner()
    await runner._deliver_media_from_response("스트리밍 완료", event, adapter)

    assert len(adapter.sent_images) == 1
    assert adapter.sent_images[0]["metadata"] == {"thread_id": "1779980356.888399"}
    batch_urls = [item[0] for item in adapter.sent_images[0]["images"]]
    assert len(batch_urls) == 1
    assert str(image_path) in batch_urls[0]


def test_collect_structured_attachment_paths_finds_delegate_and_image_results(tmp_path):
    image_path = tmp_path / "avatar.png"
    doc_path = tmp_path / "report.pdf"
    image_path.write_bytes(b"png")
    doc_path.write_bytes(b"%PDF")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "content": (
                    "{\"results\":[{\"summary\":\"done\",\"artifacts\":[\""
                    + str(image_path)
                    + "\"]}],\"image\":\""
                    + str(image_path)
                    + "\"}"
                ),
            },
            {"role": "tool", "content": {"file_path": str(doc_path)}},
        ]
    }

    assert _collect_structured_attachment_paths(agent_result) == [str(image_path), str(doc_path)]


def test_collect_structured_attachment_paths_supports_artifact_files_array(tmp_path):
    artifact_a = tmp_path / "artifact-a.png"
    artifact_b = tmp_path / "artifact-b.txt"
    artifact_a.write_bytes(b"png")
    artifact_b.write_bytes(b"txt")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "content": {
                    "artifact_files": [str(artifact_a), str(artifact_b)],
                },
            },
        ]
    }

    assert _collect_structured_attachment_paths(agent_result) == [str(artifact_a), str(artifact_b)]


def test_slice_turn_messages_uses_history_offset_boundary():
    messages = [
        {"role": "user", "content": "old user"},
        {"role": "tool", "content": {"path": "/tmp/old.png"}},
        {"role": "user", "content": "new user"},
        {"role": "tool", "content": {"path": "/tmp/new.png"}},
    ]

    assert _slice_turn_messages({"messages": messages}, 2) == messages[2:]


def test_turn_scoped_structured_attachments_ignore_old_tool_artifacts(tmp_path):
    old_a = tmp_path / "oldA.png"
    old_b = tmp_path / "oldB.png"
    current = tmp_path / "current.png"
    for path in (old_a, old_b, current):
        path.write_bytes(b"png")

    agent_result = {
        "messages": [
            {"role": "tool", "tool_name": "delegate_task", "content": {"artifacts": [str(old_a)]}},
            {"role": "tool", "tool_name": "image_generate", "content": {"image": str(old_b)}},
            {"role": "user", "content": "current request"},
            {"role": "tool", "tool_name": "delegate_task", "content": {"artifacts": [str(current)]}},
        ]
    }

    collected, debug = _collect_turn_scoped_structured_attachments(agent_result, 3)

    assert collected == [str(current)]
    assert debug["turn_messages_count"] == 1
    assert debug["turn_tool_messages_count"] == 1
    assert debug["collected_count"] == 1


def test_turn_scoped_structured_attachments_empty_when_only_history_has_artifacts(tmp_path):
    old_a = tmp_path / "oldA.png"
    old_b = tmp_path / "oldB.png"
    old_a.write_bytes(b"png")
    old_b.write_bytes(b"png")

    agent_result = {
        "messages": [
            {"role": "tool", "tool_name": "delegate_task", "content": {"artifacts": [str(old_a)]}},
            {"role": "tool", "tool_name": "image_generate", "content": {"image": str(old_b)}},
            {"role": "assistant", "content": "no new artifacts this turn"},
        ]
    }

    collected, debug = _collect_turn_scoped_structured_attachments(agent_result, 2)

    assert collected == []
    assert debug["turn_messages_count"] == 1
    assert debug["turn_tool_messages_count"] == 0
    assert debug["collected_count"] == 0


def test_turn_scoped_structured_attachments_collect_delegate_fields_from_current_tool_message(tmp_path):
    image_path = tmp_path / "delegate.png"
    image_path.write_bytes(b"png")

    agent_result = {
        "messages": [
            {"role": "user", "content": "request"},
            {
                "role": "tool",
                "tool_name": "delegate_task",
                "content": {
                    "results": [
                        {
                            "summary": "done",
                            "artifacts": [str(image_path)],
                            "image": str(image_path),
                            "file_path": str(image_path),
                            "path": str(image_path),
                        }
                    ]
                },
            },
        ]
    }

    collected, debug = _collect_turn_scoped_structured_attachments(agent_result, 1)

    assert collected == [str(image_path)]
    assert debug["turn_tool_messages_count"] == 1


def test_turn_scoped_structured_attachments_collect_image_generate_fields_from_current_tool_message(tmp_path):
    image_path = tmp_path / "generated.png"
    output_path = tmp_path / "generated-final.png"
    image_path.write_bytes(b"png")
    output_path.write_bytes(b"png")

    agent_result = {
        "messages": [
            {"role": "assistant", "content": "draft"},
            {
                "role": "tool",
                "tool_name": "image_generate",
                "content": {
                    "local_path": str(image_path),
                    "output_path": str(output_path),
                },
            },
        ]
    }

    collected, debug = _collect_turn_scoped_structured_attachments(agent_result, 1)

    assert collected == [str(image_path), str(output_path)]
    assert debug["turn_tool_messages_count"] == 1


def test_turn_scoped_structured_attachments_dedups_send_message_delivered_media_markers(tmp_path):
    image_path = tmp_path / "duplicate.png"
    image_path.write_bytes(b"png")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "send_message",
                "content": {
                    "success": True,
                    "media_delivery_completed": True,
                    "delivered_media_files": [str(image_path)],
                    "delivered_media_realpaths": [str(image_path)],
                },
            },
            {
                "role": "tool",
                "tool_name": "image_generate",
                "content": {"image": str(image_path)},
            },
        ]
    }

    collected, debug = _collect_turn_scoped_structured_attachments(agent_result, 0)

    assert collected == []
    assert debug["turn_messages_count"] == 2
    assert debug["turn_tool_messages_count"] == 2
    assert debug["collected_count"] == 0


def test_turn_scoped_structured_attachments_preserves_non_matching_artifact_candidates(tmp_path):
    delivered = tmp_path / "delivered.png"
    kept = tmp_path / "kept.png"
    delivered.write_bytes(b"png")
    kept.write_bytes(b"png")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "send_message",
                "content": {
                    "success": True,
                    "media_delivery_completed": True,
                    "delivered_media_files": [str(delivered)],
                },
            },
            {
                "role": "tool",
                "tool_name": "image_generate",
                "content": {"image": str(kept)},
            },
        ]
    }

    collected, debug = _collect_turn_scoped_structured_attachments(agent_result, 0)

    assert collected == [str(kept)]
    assert debug["turn_messages_count"] == 2
    assert debug["turn_tool_messages_count"] == 2
    assert debug["collected_count"] == 1



def test_turn_scoped_structured_attachments_ignore_user_and_assistant_textual_paths(tmp_path):
    old_path = tmp_path / "old.png"
    current_path = tmp_path / "current.png"
    old_path.write_bytes(b"png")
    current_path.write_bytes(b"png")

    agent_result = {
        "messages": [
            {"role": "tool", "tool_name": "delegate_task", "content": {"artifacts": [str(old_path)]}},
            {"role": "user", "content": f"Please mention {current_path}"},
            {"role": "assistant", "content": f"I see {current_path}"},
        ]
    }

    collected, debug = _collect_turn_scoped_structured_attachments(agent_result, 1)

    assert collected == []
    assert debug["turn_messages_count"] == 2
    assert debug["turn_tool_messages_count"] == 0
