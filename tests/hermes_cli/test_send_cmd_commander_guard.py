from __future__ import annotations

import pytest

from hermes_cli import send_cmd


def test_generic_slack_send_refuses_commander_dispatch():
    with pytest.raises(SystemExit) as exc:
        send_cmd._refuse_commander_dispatch_via_generic_send(
            "slack:C0B5W21GF8A:1782388012.254229",
            "[COMMANDER_DISPATCH]\nhello\n[/COMMANDER_DISPATCH]",
        )

    assert exc.value.code == 1


def test_generic_non_slack_send_does_not_refuse_commander_marker():
    send_cmd._refuse_commander_dispatch_via_generic_send(
        "telegram:-100123",
        "[COMMANDER_DISPATCH]\nhello\n[/COMMANDER_DISPATCH]",
    )


def test_generic_slack_send_allows_regular_message():
    send_cmd._refuse_commander_dispatch_via_generic_send(
        "slack:C0B5W21GF8A",
        "운영 점검 완료",
    )
