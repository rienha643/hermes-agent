from __future__ import annotations

from typing import Any

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import (
    GatewayRunner,
    _collect_current_turn_media_tags_from_tool_results,
    _collect_structured_attachment_paths,
    _collect_turn_scoped_structured_attachments,
    _slice_turn_messages,
    _validate_gateway_delivery_candidates,
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
async def test_post_stream_response_body_paths_are_suppressed_when_media_tags_are_present(allow_tmp_delivery):
    media_path = allow_tmp_delivery / "media.png"
    body_path = allow_tmp_delivery / "body.pdf"
    media_path.write_bytes(b"png")
    body_path.write_bytes(b"pdf")

    adapter = _make_adapter("unused")
    event = _make_event(thread_id="1779980356.888399")
    runner = _MiniRunner()

    await runner._deliver_media_from_response(
        f"완료했습니다.\nMEDIA:{media_path}\n{body_path}",
        event,
        adapter,
    )

    assert len(adapter.sent_images) == 1
    assert adapter.sent_documents == []


@pytest.mark.asyncio
async def test_post_stream_response_body_paths_are_suppressed_when_structured_attachments_exist(allow_tmp_delivery):
    structured_path = allow_tmp_delivery / "structured.png"
    body_path = allow_tmp_delivery / "body.pdf"
    structured_path.write_bytes(b"png")
    body_path.write_bytes(b"pdf")

    adapter = _make_adapter("unused")
    event = _make_event(thread_id="1779980356.888399")
    event._structured_attachment_paths = [str(structured_path)]
    runner = _MiniRunner()

    await runner._deliver_media_from_response(
        f"완료했습니다.\n{body_path}",
        event,
        adapter,
    )

    assert len(adapter.sent_images) == 1
    assert adapter.sent_documents == []


@pytest.mark.asyncio
async def test_post_stream_png_only_mode_skips_non_png_response_body_paths(allow_tmp_delivery):
    body_path = allow_tmp_delivery / "body.pdf"
    body_path.write_bytes(b"pdf")

    adapter = _make_adapter(f"완료했습니다.\n{body_path}")
    event = _make_event(thread_id="1779980356.888399")
    runner = _MiniRunner()

    await runner._deliver_media_from_response(
        f"완료했습니다.\n{body_path}",
        event,
        adapter,
    )

    assert adapter.sent_images == []
    assert adapter.sent_documents == []


@pytest.mark.asyncio
async def test_slack_rca_response_body_md_path_is_text_only(allow_tmp_delivery):
    report_path = allow_tmp_delivery / "artifacts_v1.md"
    report_path.write_text("rca")

    adapter = _make_adapter(f"Root Cause\n산출물:\n- {report_path}")
    event = _make_event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent_texts == [f"Root Cause\n산출물:\n- {report_path}"]
    assert adapter.sent_images == []
    assert adapter.sent_documents == []


@pytest.mark.asyncio
async def test_slack_test_report_body_paths_are_text_only(allow_tmp_delivery):
    md_path = allow_tmp_delivery / "slack_v3.md"
    txt_path = allow_tmp_delivery / "report.txt"
    json_path = allow_tmp_delivery / "result.json"
    for path in (md_path, txt_path, json_path):
        path.write_text("report")

    response = f"Test Result\n{md_path}\n{txt_path}\n{json_path}"
    adapter = _make_adapter(response)
    event = _make_event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent_texts == [response]
    assert adapter.sent_images == []
    assert adapter.sent_documents == []


@pytest.mark.asyncio
async def test_explicit_document_files_pdf_allows_document_attachment(allow_tmp_delivery):
    doc_path = allow_tmp_delivery / "final.pdf"
    doc_path.write_bytes(b"pdf")

    adapter = _make_adapter("문서 전달합니다.")
    event = _make_event()
    event._structured_attachment_paths = [str(doc_path)]

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent_texts == ["문서 전달합니다."]
    assert adapter.sent_documents == [{"file_path": str(doc_path), "metadata": {"notify": True}}]
    assert adapter.sent_images == []


@pytest.mark.asyncio
async def test_explicit_media_files_png_allows_image_attachment(allow_tmp_delivery):
    image_path = allow_tmp_delivery / "final.png"
    image_path.write_bytes(b"png")

    adapter = _make_adapter("이미지 전달합니다.")
    event = _make_event()
    event._structured_attachment_paths = [str(image_path)]

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent_texts == ["이미지 전달합니다."]
    assert len(adapter.sent_images) == 1
    assert adapter.sent_documents == []


@pytest.mark.asyncio
async def test_structured_nai_published_image_000_is_not_treated_as_document_intermediate(allow_tmp_delivery):
    image_path = (
        allow_tmp_delivery
        / "HermesWork"
        / "Image"
        / "NAI"
        / "20260617_121900_a72428d7"
        / "image_000.png"
    )
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")

    adapter = _make_adapter("이미지 전달합니다.")
    event = _make_event(thread_id="1779980356.888399")
    event._structured_attachment_paths = [str(image_path)]

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent_texts == ["이미지 전달합니다."]
    assert len(adapter.sent_images) == 1
    assert adapter.sent_documents == []
    batch_urls = [item[0] for item in adapter.sent_images[0]["images"]]
    assert len(batch_urls) == 1
    assert "image_000.png" in batch_urls[0]


def test_published_image_artifact_survives_document_delivery_prioritization(allow_tmp_delivery):
    image_path = (
        allow_tmp_delivery
        / "HermesWork"
        / "Image"
        / "NAI"
        / "20260617_121900_a72428d7"
        / "image_000.png"
    )
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")

    filtered = BasePlatformAdapter.filter_local_delivery_paths([str(image_path)])

    assert len(filtered) == 1
    assert BasePlatformAdapter.looks_like_document_intermediate_path(filtered[0]) is False
    assert BasePlatformAdapter.prioritize_document_delivery_paths(filtered) == filtered


@pytest.mark.parametrize(
    "relative_path",
    [
        "HermesWork/Documents/run/page_001.png",
        "HermesWork/Documents/run/ocr_page.png",
        "HermesWork/Documents/run/render_page.png",
        "HermesWork/Documents/run/tmp_image.png",
        "HermesWork/Documents/run/extracted_page.png",
        "HermesWork/Image/NAI/20260617_121900_a72428d7/metadata_image.png",
        "HermesWork/Image/NAI/20260617_121900_a72428d7/manifest_image.png",
        "HermesWork/Image/NAI/20260617_121900_a72428d7/integrity_image.png",
        "HermesWork/Image/NAI/20260617_121900_a72428d7/sidecar/image_000.png",
    ],
)
def test_document_and_sidecar_intermediate_images_remain_suppressed(allow_tmp_delivery, relative_path):
    image_path = allow_tmp_delivery / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png")

    filtered = BasePlatformAdapter.filter_local_delivery_paths([str(image_path)])

    assert BasePlatformAdapter.looks_like_document_intermediate_path(str(image_path)) is True
    assert BasePlatformAdapter.prioritize_document_delivery_paths(filtered) == []


@pytest.mark.asyncio
async def test_sensitive_json_document_files_are_always_blocked(allow_tmp_delivery):
    sensitive = allow_tmp_delivery / "token.json"
    sensitive.write_text("{}")

    adapter = _make_adapter("민감 파일은 차단합니다.")
    event = _make_event()
    event._structured_attachment_paths = [str(sensitive)]

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent_texts == ["민감 파일은 차단합니다."]
    assert adapter.sent_images == []
    assert adapter.sent_documents == []


@pytest.mark.asyncio
async def test_text_only_response_does_not_attempt_attachment_send(allow_tmp_delivery):
    adapter = _make_adapter("텍스트만 보냅니다.")
    event = _make_event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent_texts == ["텍스트만 보냅니다."]  # type: ignore[attr-defined]
    assert adapter.sent_images == []  # type: ignore[attr-defined]
    assert adapter.sent_documents == []  # type: ignore[attr-defined]


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


def test_collect_structured_attachment_paths_finds_explicit_delivery_results(tmp_path):
    image_path = tmp_path / "avatar.png"
    doc_path = tmp_path / "report.pdf"
    image_path.write_bytes(b"png")
    doc_path.write_bytes(b"%PDF")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "delegate_task",
                "content": (
                    "{\"results\":[{\"summary\":\"done\",\"user_requested_delivery\":true,\"artifacts\":[\""
                    + str(image_path)
                    + "\"]}],\"media_files\":[\""
                    + str(image_path)
                    + "\"]}"
                ),
            },
            {"role": "tool", "content": {"document_files": [str(doc_path)]}},
        ]
    }

    assert _collect_structured_attachment_paths(agent_result) == [str(image_path), str(doc_path)]


def test_worker_result_artifact_paths_without_delivery_intent_are_not_collected(tmp_path):
    report = tmp_path / "artifacts_v1.md"
    report.write_text("rca")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "delegate_task",
                "content": {
                    "results": [
                        {
                            "summary": f"산출물\n- {report}",
                            "artifacts": [str(report)],
                        }
                    ]
                },
            }
        ]
    }

    assert _collect_structured_attachment_paths(agent_result) == []


def test_delegate_result_hermeswork_image_artifact_is_collected_without_artifact_files(tmp_path):
    image_path = tmp_path / "HermesWork" / "Image" / "260617_HermesWork_Image" / "angelica_windows_remote_comfyui_fresh_e2e_v4.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "delegate_task",
                "content": {
                    "results": [
                        {
                            "summary": "Angelica result",
                            "artifacts": [str(image_path)],
                            "artifact_files": [],
                        }
                    ]
                },
            }
        ]
    }

    assert _collect_structured_attachment_paths(agent_result) == [str(image_path)]


def test_delegate_result_hermeswork_image_artifact_becomes_slack_candidate(allow_tmp_delivery):
    image_path = allow_tmp_delivery / "HermesWork" / "Image" / "260617_HermesWork_Image" / "angelica_windows_remote_comfyui_fresh_e2e_v4.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    adapter = _make_adapter("이미지 전달합니다.")

    allowed, blocked = _validate_gateway_delivery_candidates(adapter, [str(image_path)])

    assert allowed == [str(image_path)]
    assert blocked == []


def test_delegate_result_repo_tmp_png_artifact_without_delivery_intent_is_not_collected(tmp_path):
    image_path = tmp_path / "tmp" / "windows_remote_smoke" / "sdxl_00001_.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "delegate_task",
                "content": {
                    "results": [
                        {
                            "summary": "ad-hoc smoke only",
                            "artifacts": [str(image_path)],
                            "artifact_files": [],
                        }
                    ]
                },
            }
        ]
    }

    assert _collect_structured_attachment_paths(agent_result) == []


def test_delegate_result_user_requested_delivery_false_blocks_hermeswork_image_artifact(tmp_path):
    image_path = tmp_path / "HermesWork" / "Image" / "blocked" / "blocked.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "delegate_task",
                "content": {"results": [{"user_requested_delivery": False, "artifacts": [str(image_path)]}]},
            }
        ]
    }

    assert _collect_structured_attachment_paths(agent_result) == []


def test_user_requested_delivery_false_does_not_enable_legacy_artifacts(tmp_path):
    report = tmp_path / "test_report.txt"
    report.write_text("test")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "delegate_task",
                "content": {
                    "user_requested_delivery": False,
                    "artifacts": [str(report)],
                },
            }
        ]
    }

    assert _collect_structured_attachment_paths(agent_result) == []


def test_user_requested_delivery_false_blocks_media_files_too(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"png")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "delegate_task",
                "content": {
                    "user_requested_delivery": False,
                    "media_files": [str(image)],
                },
            }
        ]
    }

    assert _collect_structured_attachment_paths(agent_result) == []


def test_bare_local_path_in_body_is_not_collected_without_explicit_delivery_intent(tmp_path):
    report = tmp_path / "body-only-report.md"
    report.write_text("rca")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "delegate_task",
                "content": f"RCA complete\n산출물\n- {report}",
            }
        ]
    }

    assert _collect_structured_attachment_paths(agent_result) == []


def test_explicit_document_files_are_collected(tmp_path):
    report = tmp_path / "final.pdf"
    report.write_bytes(b"pdf")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "document_generate",
                "content": {"document_files": [str(report)]},
            }
        ]
    }

    assert _collect_structured_attachment_paths(agent_result) == [str(report)]


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
            {
                "role": "tool",
                "tool_name": "delegate_task",
                "content": {"user_requested_delivery": True, "artifacts": [str(current)]},
            },
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
                            "user_requested_delivery": True,
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
                    "media_files": [str(image_path)],
                    "output_path": str(output_path),
                },
            },
        ]
    }

    collected, debug = _collect_turn_scoped_structured_attachments(agent_result, 1)

    assert collected == [str(image_path), str(output_path)]
    assert debug["turn_tool_messages_count"] == 1


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


def test_media_tag_fallback_ignores_session_search_history(tmp_path):
    old_image = tmp_path / "old.png"
    old_image.write_bytes(b"old")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "session_search",
                "content": f"historical result MEDIA:{old_image}",
            }
        ]
    }

    media_tags, has_voice = _collect_current_turn_media_tags_from_tool_results(
        agent_result,
        history_len=0,
        history_media_paths=set(),
    )

    assert media_tags == []
    assert has_voice is False


def test_media_tag_fallback_collects_trusted_current_tool_media(tmp_path):
    current_audio = tmp_path / "voice.mp3"
    current_audio.write_bytes(b"audio")

    agent_result = {
        "messages": [
            {
                "role": "tool",
                "tool_name": "text_to_speech",
                "content": f"[[audio_as_voice]]\nMEDIA:{current_audio}",
            }
        ]
    }

    media_tags, has_voice = _collect_current_turn_media_tags_from_tool_results(
        agent_result,
        history_len=0,
        history_media_paths=set(),
    )

    assert media_tags == [f"MEDIA:{current_audio}"]
    assert has_voice is True
