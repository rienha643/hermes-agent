import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import kospi_close_briefing as close_briefing
import kospi_market_guard as guard
import kospi_morning_briefing as morning_briefing


@pytest.mark.parametrize(
    "today, expected_reason",
    [
        (date(2026, 6, 3), "지방선거일"),
        (date(2026, 7, 17), "제헌절"),
        (date(2026, 6, 7), "주말"),
        (date(2026, 12, 25), "성탄절"),
    ],
)
def test_market_closed_reason_covers_requested_closed_days(today, expected_reason):
    assert guard.get_market_closed_reason(today) == expected_reason
    assert guard.is_kr_market_closed(today) is True


def test_market_closed_reason_supports_temporary_holiday_override(monkeypatch):
    monkeypatch.setenv("KR_MARKET_CLOSED_DATES", "2026-06-04")
    assert guard.get_market_closed_reason(date(2026, 6, 4)) == "임시공휴일"
    assert guard.is_kr_market_closed(date(2026, 6, 4)) is True


def test_market_open_day_is_not_closed():
    assert guard.get_market_closed_reason(date(2026, 6, 2)) is None
    assert guard.is_kr_market_closed(date(2026, 6, 2)) is False


def test_stale_data_detection_skips_when_naver_date_is_not_today():
    snapshot = guard.MarketSnapshotDates(
        index_date=date(2026, 6, 1),
        stock_dates=(date(2026, 6, 1), date(2026, 6, 1)),
    )
    decision = guard.should_skip_market_briefing(today=date(2026, 6, 2), market_dates=snapshot)
    assert decision.should_skip is True
    assert decision.skip_kind == "stale_data"
    assert "오늘(2026-06-02)" in decision.reason


@pytest.mark.parametrize("module", [morning_briefing, close_briefing])
def test_notify_mode_prints_short_holiday_message_for_closed_day(module, capsys):
    exit_code = module.main(["--simulate-kst", "2026-06-03T10:00:00+09:00"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "휴장" in out
    assert "지방선거일" in out
    assert "KOSPI" in out


@pytest.mark.parametrize("module", [morning_briefing, close_briefing])
def test_silent_mode_emits_no_stdout_for_closed_day(module, capsys):
    exit_code = module.main(
        [
            "--simulate-kst",
            "2026-06-03T10:00:00+09:00",
            "--market-closed-policy-mode",
            "silent",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert out == ""


@pytest.mark.parametrize("module", [morning_briefing, close_briefing])
def test_stale_data_notify_and_silent_modes(module, monkeypatch, capsys):
    gate = guard.MarketBriefingGate(
        should_skip=True,
        reason="네이버 금융 기준일이 오늘(2026-06-02)이 아닙니다: 2026-06-01",
        skip_kind="stale_data",
    )

    def fake_build_briefing(today):
        return [], gate

    monkeypatch.setattr(module, "build_briefing", fake_build_briefing)

    exit_code = module.main(["--simulate-kst", "2026-06-02T10:00:00+09:00"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "생략" in out
    assert "기준일" in out
    assert "2026-06-01" in out

    exit_code = module.main(
        [
            "--simulate-kst",
            "2026-06-02T10:00:00+09:00",
            "--market-closed-policy-mode",
            "silent",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert out == ""
