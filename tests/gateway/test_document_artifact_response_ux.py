from pathlib import Path
from typing import Any

import pytest

import gateway.document_artifacts as document_artifacts
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    sent_texts: list[str]
    sent_documents: list[str]

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, error=None, message_id="send")

    async def get_chat_info(self, chat_id):
        return {}


def _make_event(text: str = "generate doc", chat_id: str = "C123") -> MessageEvent:
    source = SessionSource(platform=Platform.SLACK, chat_id=chat_id, chat_type="channel")
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


def _make_adapter(response_text: str):
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="test"), Platform.SLACK)
    adapter.sent_texts = []
    adapter.sent_documents = []

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

    async def _send_document(
        chat_id: str,
        file_path: str,
        caption=None,
        file_name=None,
        reply_to=None,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        adapter.sent_documents.append(file_path)
        return SendResult(success=True, error=None, message_id="doc")

    adapter._message_handler = _handler
    adapter._send_with_retry = _send_with_retry
    adapter.send_document = _send_document
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_template", "source_subdir", "published_subdir"),
    [
        ("완료했습니다.\nMEDIA:{source}", "planning", ("Documents", "planning")),
        ("완료했습니다.\n{source}", "worldbuilding", ("Story", "worldbuilding")),
    ],
)
async def test_direct_document_response_appends_standard_artifact_block(
    monkeypatch,
    tmp_path,
    response_template,
    source_subdir,
    published_subdir,
):
    work_root = tmp_path / "HermesWork"
    source_dir = tmp_path / source_subdir
    source_dir.mkdir(parents=True, exist_ok=True)
    source = source_dir / "report.docx"
    source.write_text("placeholder", encoding="utf-8")

    published_path = work_root.joinpath(*published_subdir, source.name)
    published_path.parent.mkdir(parents=True, exist_ok=True)
    published_path.write_text("published", encoding="utf-8")

    monkeypatch.setattr(
        document_artifacts,
        "get_hermes_work_dir",
        lambda *parts: work_root.joinpath(*parts),
    )
    monkeypatch.setattr(
        "gateway.platforms.base.publish_document_artifact",
        lambda path, *, folder_name=None: published_path,
    )

    response = response_template.format(source=source)
    adapter = _make_adapter(response)
    event = _make_event()
    session_key = build_session_key(event.source)

    await adapter._process_message_background(event, session_key)

    expected_block = document_artifacts.format_document_artifact_block(published_path)
    assert adapter.sent_texts == [f"완료했습니다.\n\n{expected_block}"]
    assert adapter.sent_documents == [str(published_path)]
