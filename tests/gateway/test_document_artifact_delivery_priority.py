from pathlib import Path
from typing import Any

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
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


@pytest.mark.asyncio
async def test_final_docx_pdf_suppress_sidecar_uploads(monkeypatch, tmp_path):
    final_docx = tmp_path / "final.docx"
    final_pdf = tmp_path / "final.pdf"
    sidecars = tmp_path / "_sidecars"
    sidecars.mkdir()
    ocr = sidecars / "ocr_all.txt"
    page = sidecars / "page_1.png"
    for path in (final_docx, final_pdf, ocr, page):
        path.write_bytes(b"x")

    sent_documents: list[str] = []
    sent_images: list[Any] = []
    sent_texts: list[str] = []

    adapter = _StubAdapter(PlatformConfig(enabled=True, token="test"), Platform.SLACK)

    async def _handler(_event: MessageEvent) -> str:
        return "done"

    async def _send_with_retry(chat_id: str, content: str, **kwargs) -> SendResult:
        sent_texts.append(content)
        return SendResult(success=True, error=None, message_id="text")

    async def _send_document(chat_id: str, file_path: str, **kwargs) -> SendResult:
        sent_documents.append(file_path)
        return SendResult(success=True, error=None, message_id="doc")

    async def _send_multiple_images(chat_id: str, images, **kwargs) -> None:
        sent_images.extend(images)

    monkeypatch.setattr("gateway.platforms.base.validate_media_delivery_path", lambda path: str(Path(path).resolve()))
    monkeypatch.setattr("gateway.platforms.base.publish_document_artifact", lambda path, folder_name=None: path)
    monkeypatch.setattr("gateway.platforms.base.format_document_artifact_block", lambda path: f"ARTIFACT:{path}")

    adapter._message_handler = _handler
    adapter._send_with_retry = _send_with_retry
    adapter.send_document = _send_document
    adapter.send_multiple_images = _send_multiple_images

    event = MessageEvent(
        text="trigger",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id="C123", chat_type="channel"),
    )
    setattr(event, "_structured_attachment_paths", [str(ocr), str(final_docx), str(page), str(final_pdf)])

    await adapter._process_message_background(event, build_session_key(event.source))

    assert sent_documents == [str(final_docx), str(final_pdf)]
    assert sent_images == []
    assert sent_texts and "ARTIFACT:" in sent_texts[0]
    assert str(ocr) not in sent_documents
    assert str(page) not in sent_documents


def test_intermediate_priority_without_final_docs_keeps_root_ocr_but_excludes_sidecar_dir(tmp_path):
    root_ocr = tmp_path / "ocr_all.txt"
    sidecar_dir = tmp_path / "_sidecars"
    sidecar_dir.mkdir()
    hidden_page = sidecar_dir / "page_1.png"
    for path in (root_ocr, hidden_page):
        path.write_bytes(b"x")

    prioritized = BasePlatformAdapter.prioritize_document_delivery_paths([str(root_ocr), str(hidden_page)])

    assert prioritized == [str(root_ocr)]
