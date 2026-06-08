import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from tools.send_message_tool import send_message_tool


def _run_async_immediately(coro):
    import asyncio
    return asyncio.run(coro)


def _make_slack_config(chat_id=None):
    slack_cfg = SimpleNamespace(enabled=True, token="xoxb-test", extra={})
    home = SimpleNamespace(chat_id=chat_id) if chat_id else None
    return SimpleNamespace(
        platforms={Platform.SLACK: slack_cfg},
        get_home_channel=lambda _platform: home,
    ), slack_cfg


def test_slack_channel_target_auto_threads_worker_probe_in_thread_scoped_session():
    config, slack_cfg = _make_slack_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.session_context.get_session_env") as session_env, \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        session_env.side_effect = lambda name, default="": {
            "HERMES_SESSION_PLATFORM": "slack",
            "HERMES_SESSION_CHAT_ID": "C0B5W21GF8A",
            "HERMES_SESSION_THREAD_ID": "1780857153.315049",
            "HERMES_SESSION_USER_ID": "U123",
        }.get(name, default)
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "slack:C0B5W21GF8A",
                    "message": "[WORKER: Eclipse]\nReplay Verification A",
                }
            )
        )

    assert result["success"] is True
    assert result["target_kind"] == "thread_reply"
    assert result["thread_ts"] == "1780857153.315049"
    assert result["auto_threaded"] is True
    send_mock.assert_awaited_once_with(
        Platform.SLACK,
        slack_cfg,
        "C0B5W21GF8A",
        "[WORKER: Eclipse]\nReplay Verification A",
        thread_id="1780857153.315049",
        media_files=[],
        force_document=False,
    )


def test_slack_worker_probe_top_level_blocked_without_thread_context():
    config, _slack_cfg = _make_slack_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.session_context.get_session_env", return_value=""), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock:
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "slack:C0B5W21GF8A",
                    "message": "[WORKER: Eclipse]\nReplay Verification A",
                }
            )
        )

    assert result["blocked_top_level_worker_probe"] is True
    assert result["target_kind"] == "top_level"
    assert "Blocked unsafe Slack top-level send" in result["error"]
    send_mock.assert_not_awaited()


def test_slack_worker_probe_top_level_allowed_with_explicit_override():
    config, slack_cfg = _make_slack_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.session_context.get_session_env", return_value=""), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "slack:C0B5W21GF8A",
                    "message": "[[allow_top_level]]\n[WORKER: Eclipse]\nReplay Verification A",
                }
            )
        )

    assert result["success"] is True
    assert result["target_kind"] == "top_level"
    assert result["blocked_top_level_worker_probe"] is False
    send_mock.assert_awaited_once_with(
        Platform.SLACK,
        slack_cfg,
        "C0B5W21GF8A",
        "[WORKER: Eclipse]\nReplay Verification A",
        thread_id=None,
        media_files=[],
        force_document=False,
    )


def test_slack_regular_top_level_message_still_allowed():
    config, slack_cfg = _make_slack_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.session_context.get_session_env", return_value=""), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "slack:C0B5W21GF8A",
                    "message": "일반 공지 메시지",
                }
            )
        )

    assert result["success"] is True
    assert result["target_kind"] == "top_level"
    assert result["blocked_top_level_worker_probe"] is False
    send_mock.assert_awaited_once_with(
        Platform.SLACK,
        slack_cfg,
        "C0B5W21GF8A",
        "일반 공지 메시지",
        thread_id=None,
        media_files=[],
        force_document=False,
    )


def test_slack_home_channel_target_auto_threads_worker_probe_in_thread_scoped_session():
    config, slack_cfg = _make_slack_config(chat_id="C0B5W21GF8A")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.session_context.get_session_env") as session_env, \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        session_env.side_effect = lambda name, default="": {
            "HERMES_SESSION_PLATFORM": "slack",
            "HERMES_SESSION_CHAT_ID": "C0B5W21GF8A",
            "HERMES_SESSION_THREAD_ID": "1780857153.315049",
            "HERMES_SESSION_USER_ID": "U123",
        }.get(name, default)
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "slack",
                    "message": "[WORKER: Eclipse]\nReplay Verification A",
                }
            )
        )

    assert result["success"] is True
    assert result["target_kind"] == "thread_reply"
    assert result["thread_ts"] == "1780857153.315049"
    send_mock.assert_awaited_once_with(
        Platform.SLACK,
        slack_cfg,
        "C0B5W21GF8A",
        "[WORKER: Eclipse]\nReplay Verification A",
        thread_id="1780857153.315049",
        media_files=[],
        force_document=False,
    )
