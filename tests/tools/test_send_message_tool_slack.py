from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import importlib
import sys

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult


class _DummyAsyncWebClient:
    def __init__(self, *_, **__):
        pass

    async def chat_postMessage(self, **kwargs):
        return {"ok": True, "ts": "1781023592.888399"}

    async def files_upload_v2(self, **kwargs):
        return {"ok": True}


# Keep this test module robust even when optional Slack SDK modules are absent.
def _ensure_slack_mocks():
    if "slack_sdk" not in sys.modules:
        async_client_mod = ModuleType("slack_sdk.web.async_client")
        async_client_mod.AsyncWebClient = _DummyAsyncWebClient

        sys.modules.setdefault("slack_sdk", ModuleType("slack_sdk"))
        sys.modules.setdefault("slack_sdk.web", ModuleType("slack_sdk.web"))
        sys.modules.setdefault("slack_sdk.web.async_client", async_client_mod)
    else:
        client_mod = importlib.import_module("slack_sdk.web.async_client")
        if not hasattr(client_mod, "AsyncWebClient"):
            client_mod.AsyncWebClient = _DummyAsyncWebClient


class _DummySlackAdapter:
    def __init__(self, pconfig):
        self.pconfig = pconfig
        self._app = SimpleNamespace()

    async def send(self, chat_id: str, content: str, metadata=None):
        return SendResult(success=True, error=None, message_id="text-msg")

    async def send_multiple_images(self, chat_id, images, metadata=None, human_delay=0):
        self.multi_calls = getattr(self, "multi_calls", 0) + 1
        self.last_multi = (chat_id, list(images), metadata, human_delay)
        return None

    async def send_document(self, chat_id: str, file_path: str, metadata=None, file_name=None):
        self.doc_calls = getattr(self, "doc_calls", 0) + 1
        self.last_doc = (chat_id, file_path, metadata, file_name)
        return SendResult(success=True, error=None, message_id="doc-msg")

    async def send_video(self, chat_id: str, file_path: str, metadata=None, caption=None):
        self.video_calls = getattr(self, "video_calls", 0) + 1
        return SendResult(success=True, error=None, message_id="video-msg")

    async def send_voice(self, chat_id: str, file_path: str, metadata=None, caption=None):
        self.voice_calls = getattr(self, "voice_calls", 0) + 1
        return SendResult(success=True, error=None, message_id="voice-msg")


@pytest.mark.asyncio
async def test_send_slack_via_adapter_image_path_uses_send_multiple_images(monkeypatch, tmp_path):
    _ensure_slack_mocks()
    import tools.send_message_tool as smt
    import gateway.platforms.slack as slack_mod
    import slack_sdk.web.async_client as async_client_mod

    adapter = _DummySlackAdapter(SimpleNamespace())
    monkeypatch.setattr(slack_mod, "SlackAdapter", lambda *_: adapter, raising=False)
    monkeypatch.setattr(async_client_mod, "AsyncWebClient", _DummyAsyncWebClient, raising=False)
    monkeypatch.setattr(slack_mod, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),), raising=False)
    monkeypatch.setattr(slack_mod, "_SLACK_BLOCKED_SOURCE_ROOTS", (), raising=False)
    monkeypatch.setattr(smt, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),), raising=False)
    monkeypatch.setattr(smt, "_SLACK_BLOCKED_SOURCE_ROOTS", (), raising=False)

    image = tmp_path / "smoke_ssd_hmw_1781023592_v1.png"
    image.write_bytes(b"png")

    pconfig = SimpleNamespace(token="x-token", extra={})
    result = await smt._send_slack_via_adapter(
        pconfig,
        chat_id="C123",
        message="Slack image smoke",
        media_files=[(str(image), False)],
        thread_id="1781023592.888399",
    )

    assert result.get("success")
    assert result["platform"] == "slack"
    assert getattr(adapter, "multi_calls", 0) == 1
    chat_id, urls, metadata, _ = adapter.last_multi
    assert chat_id == "C123"
    assert urls[0][0].startswith("file://")
    assert metadata is not None
    assert metadata["thread_id"] == "1781023592.888399"


@pytest.mark.asyncio
async def test_send_slack_via_adapter_document_path_is_blocked(monkeypatch, tmp_path):
    _ensure_slack_mocks()
    import tools.send_message_tool as smt
    import gateway.platforms.slack as slack_mod
    import slack_sdk.web.async_client as async_client_mod

    adapter = _DummySlackAdapter(SimpleNamespace())
    monkeypatch.setattr(slack_mod, "SlackAdapter", lambda *_: adapter, raising=False)
    monkeypatch.setattr(async_client_mod, "AsyncWebClient", _DummyAsyncWebClient, raising=False)
    monkeypatch.setattr(slack_mod, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),), raising=False)
    monkeypatch.setattr(slack_mod, "_SLACK_BLOCKED_SOURCE_ROOTS", (), raising=False)
    monkeypatch.setattr(smt, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),), raising=False)
    monkeypatch.setattr(smt, "_SLACK_BLOCKED_SOURCE_ROOTS", (), raising=False)

    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"pdf")

    pconfig = SimpleNamespace(token="x-token", extra={})
    result = await smt._send_slack_via_adapter(
        pconfig,
        chat_id="C123",
        message="",
        media_files=[(str(doc), False)],
        thread_id="1781023592.888399",
    )

    assert "error" in result
    assert "PNG-only" in result["error"]
    assert getattr(adapter, "multi_calls", 0) == 0
    assert getattr(adapter, "doc_calls", 0) == 0


@pytest.mark.asyncio
async def test_send_to_platform_slack_text_only_routes_to_chat_post(monkeypatch):
    _ensure_slack_mocks()
    import tools.send_message_tool as smt

    slack_send = AsyncMock(return_value={"success": True, "platform": "slack", "chat_id": "C123"})
    via_adapter = AsyncMock(return_value={"error": "should-not"})

    monkeypatch.setattr(smt, "_send_slack", slack_send)
    monkeypatch.setattr(smt, "_send_slack_via_adapter", via_adapter)

    await smt._send_to_platform(
        Platform.SLACK,
        SimpleNamespace(token="x-token", extra={}),
        chat_id="C123",
        message="text only",
        thread_id="1781023592.888399",
        media_files=[],
    )

    assert slack_send.call_count == 1
    assert via_adapter.await_count == 0
