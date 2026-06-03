#!/usr/bin/env python3
"""Shared guards for the KOSPI morning/close cron scripts.

This module centralizes the market-closed policy and stale-data checks so the
morning and closing briefings can stay aligned.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Sequence

KST = timezone(timedelta(hours=9))
DEFAULT_POLICY_MODE = "notify"
VALID_POLICY_MODES = {"notify", "silent"}

# Explicitly covered dates requested by the task.
SPECIAL_CLOSED_DATES: dict[date, str] = {
    date(2026, 6, 3): "지방선거일",
    date(2026, 7, 17): "제헌절",
}

# Fixed-date holidays and a small set of common Korean market holidays.
# This is intentionally conservative: when in doubt, prefer skipping rather
# than publishing stale data.
FIXED_CLOSED_DATES: dict[tuple[int, int], str] = {
    (1, 1): "신정",
    (3, 1): "삼일절",
    (5, 5): "어린이날",
    (6, 6): "현충일",
    (8, 15): "광복절",
    (10, 3): "개천절",
    (10, 9): "한글날",
    (12, 25): "성탄절",
}

# Environment overrides allow temporary holidays / exchange holidays to be
# added without code changes.
CLOSED_DATES_ENV_VARS = (
    "KR_MARKET_CLOSED_DATES",
    "KRX_CLOSED_DATES",
    "KOSPI_CLOSED_DATES",
)
POLICY_MODE_ENV_VARS = (
    "market_closed_policy.mode",
    "MARKET_CLOSED_POLICY_MODE",
)


@dataclass(frozen=True)
class MarketClosedInfo:
    is_closed: bool
    reason: str | None = None


@dataclass(frozen=True)
class MarketBriefingGate:
    should_skip: bool
    reason: str | None = None
    skip_kind: str | None = None


@dataclass(frozen=True)
class MarketSnapshotDates:
    """Dates parsed from Naver Finance pages."""

    index_date: date | None = None
    stock_dates: tuple[date, ...] = ()

    def all_dates(self) -> tuple[date, ...]:
        items = [d for d in (self.index_date, *self.stock_dates) if d is not None]
        return tuple(items)

    def unique_dates(self) -> tuple[date, ...]:
        return tuple(sorted({d for d in self.all_dates()}))


_DATE_RE = re.compile(r"(?P<year>20\d{2})\.(?P<month>\d{2})\.(?P<day>\d{2})")


def get_kst_now(now: datetime | None = None) -> datetime:
    """Return a timezone-aware KST datetime."""

    if now is None:
        return datetime.now(KST)
    if now.tzinfo is None:
        return now.replace(tzinfo=KST)
    return now.astimezone(KST)


def parse_kst_datetime(value: str) -> datetime:
    """Parse a manual KST override such as 2026-06-02T10:00:00+09:00."""

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def load_market_closed_policy_mode(explicit_mode: str | None = None) -> str:
    """Resolve the market-closed policy mode.

    Priority: explicit argument -> environment override -> default notify.
    """

    raw = explicit_mode
    if raw is None:
        for env_name in POLICY_MODE_ENV_VARS:
            raw = os.getenv(env_name)
            if raw:
                break
    if raw is None or not raw.strip():
        return DEFAULT_POLICY_MODE
    mode = raw.strip().lower()
    if mode not in VALID_POLICY_MODES:
        raise ValueError(
            f"market_closed_policy.mode must be one of {sorted(VALID_POLICY_MODES)}, got {raw!r}"
        )
    return mode


def _load_extra_closed_dates(env_value: str | None) -> set[date]:
    if not env_value:
        return set()
    items: set[date] = set()
    for token in re.split(r"[\s,]+", env_value.strip()):
        if not token:
            continue
        items.add(date.fromisoformat(token))
    return items


def load_closed_date_overrides() -> set[date]:
    items: set[date] = set()
    for env_name in CLOSED_DATES_ENV_VARS:
        items.update(_load_extra_closed_dates(os.getenv(env_name)))
    return items


def get_market_closed_reason(today: date, extra_closed_dates: Iterable[date] | None = None) -> str | None:
    """Return a human-readable close reason for KRX on the given date."""

    if today.weekday() >= 5:
        return "주말"
    if today in SPECIAL_CLOSED_DATES:
        return SPECIAL_CLOSED_DATES[today]
    if (today.month, today.day) in FIXED_CLOSED_DATES:
        return FIXED_CLOSED_DATES[(today.month, today.day)]
    closed_dates = set(extra_closed_dates or ()) | load_closed_date_overrides()
    if today in closed_dates:
        return "임시공휴일"
    return None


def is_kr_market_closed(today: date, extra_closed_dates: Iterable[date] | None = None) -> bool:
    return get_market_closed_reason(today, extra_closed_dates=extra_closed_dates) is not None


def parse_naver_market_date(html: str) -> date | None:
    """Extract the exchange date embedded in Naver Finance HTML.

    Naver usually exposes the most recent market date in the ``#time`` area or
    nearby ``em.date`` markup, e.g. ``2026.06.02 장마감`` or
    ``2026.06.02 <span>기준(KRX 장마감)</span>``.
    """

    if not html:
        return None

    patterns = (
        r'id="time"[^>]*>.*?(?P<date>20\d{2}\.\d{2}\.\d{2})',
        r'<em class="date">\s*(?P<date>20\d{2}\.\d{2}\.\d{2})\s*<',
        r'(?P<date>20\d{2}\.\d{2}\.\d{2})\s*(?:장마감|장중|기준)',
        r'(?P<date>20\d{2}\.\d{2}\.\d{2})',
    )
    for pattern in patterns:
        match = re.search(pattern, html, re.S)
        if match:
            return date.fromisoformat(match.group("date").replace(".", "-"))
    return None


def build_market_snapshot_dates(index_html: str | None, stock_htmls: Sequence[str] | None = None) -> MarketSnapshotDates:
    index_date = parse_naver_market_date(index_html) if index_html else None
    stock_dates = tuple(
        d for d in (parse_naver_market_date(html) for html in (stock_htmls or ())) if d is not None
    )
    return MarketSnapshotDates(index_date=index_date, stock_dates=stock_dates)


def should_skip_market_briefing(
    *,
    today: date,
    market_dates: MarketSnapshotDates | None = None,
    extra_closed_dates: Iterable[date] | None = None,
) -> MarketBriefingGate:
    """Return a skip decision for the briefing.

    Closed days are handled first. If the market is open, then stale or
    missing Naver Finance dates trigger a skip so we never publish yesterday's
    snapshot as if it were live.
    """

    closed_reason = get_market_closed_reason(today, extra_closed_dates=extra_closed_dates)
    if closed_reason:
        return MarketBriefingGate(
            should_skip=True,
            reason=closed_reason,
            skip_kind="market_closed",
        )

    if market_dates is None:
        return MarketBriefingGate(should_skip=False)

    observed_dates = market_dates.all_dates()
    if not observed_dates:
        return MarketBriefingGate(
            should_skip=True,
            reason="네이버 금융 기준일을 확인하지 못했습니다.",
            skip_kind="stale_data",
        )

    unique_dates = market_dates.unique_dates()
    if len(unique_dates) > 1:
        joined = ", ".join(d.isoformat() for d in unique_dates)
        return MarketBriefingGate(
            should_skip=True,
            reason=f"네이버 금융 데이터 기준일이 서로 다릅니다: {joined}",
            skip_kind="stale_data",
        )

    data_date = unique_dates[0]
    if data_date != today:
        return MarketBriefingGate(
            should_skip=True,
            reason=f"네이버 금융 기준일이 오늘({today.isoformat()})이 아닙니다: {data_date.isoformat()}",
            skip_kind="stale_data",
        )

    return MarketBriefingGate(should_skip=False)


def format_market_skip_message(*, title: str, decision: MarketBriefingGate, today: date) -> str:
    reason = decision.reason or "사유 불명"
    if decision.skip_kind == "market_closed":
        return f"{title} {today.isoformat()} 휴장: {reason}"
    return f"{title} {today.isoformat()} 생략: {reason}"


def should_emit_silent(mode: str) -> bool:
    return mode == "silent"


def normalize_policy_mode(value: str | None) -> str:
    return load_market_closed_policy_mode(value)
