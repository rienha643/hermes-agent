#!/usr/bin/env python3
"""오늘의 KOSPI 오전장 브리핑 생성기.

- Firecrawl / web_search / web_extract 비의존
- requests + 정규식 + 표준 라이브러리만 사용
- 기본값에서는 KRX 휴장일(주말 포함)이면 조용히 종료
- FORCE_RUN=1 이면 휴장 가드 없이 즉시 본문 출력

출력에는 Slack 멘션 <@U0B56MYFW07> 이 포함됩니다.
"""

from __future__ import annotations

import argparse
import html as html_lib
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Optional

import requests

from kospi_market_guard import (
    MarketBriefingGate,
    build_market_snapshot_dates,
    format_market_skip_message,
    get_kst_now,
    load_market_closed_policy_mode,
    should_skip_market_briefing,
)

MENTION = "<@U0B56MYFW07>"
TITLE = "오늘의 KOSPI 오전장 브리핑"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) HermesAgent/1.0"
TIMEOUT = 12

NAVER_BASE = "https://finance.naver.com"
INDEX_URL = f"{NAVER_BASE}/sise/sise_index.naver?code=KOSPI"
STOCK_URL = f"{NAVER_BASE}/item/main.naver?code={{code}}"
UP_URL = f"{NAVER_BASE}/sise/sise_upper.naver"
DOWN_URL = f"{NAVER_BASE}/sise/sise_lower.naver"
RISE_URL = f"{NAVER_BASE}/sise/sise_rise.naver"
FALL_URL = f"{NAVER_BASE}/sise/sise_fall.naver"


@dataclass
class MarketQuote:
    name: str
    code: str
    price: str
    change: str
    change_pct: str
    direction: str
    source: str
    tag: str = ""


@dataclass
class IndexQuote:
    value: str
    change: str
    change_pct: str
    direction: str
    source: str


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"})


def _clean(text: str) -> str:
    return html_lib.unescape(re.sub(r"\s+", " ", text)).strip()


def _fetch(url: str, *, allow_jina: bool = True) -> str:
    try:
        resp = SESSION.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except Exception:
        if not allow_jina:
            raise
        jina_url = "https://r.jina.ai/http://" + url.removeprefix("https://").removeprefix("http://")
        resp = SESSION.get(jina_url, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.text


def _fmt_percent(v: str) -> str:
    v = v.strip()
    if not v:
        return v
    if v.endswith("%"):
        return v
    return v + "%"


def _normalize_direction(direction: str) -> str:
    d = direction.strip()
    if d in {"상승", "up", "plus"}:
        return "상승"
    if d in {"하락", "down", "minus"}:
        return "하락"
    if d in {"보합", "same", "unchanged"}:
        return "보합"
    return d or "추정"


def _is_weekend(today: date) -> bool:
    return today.weekday() >= 5


def should_skip_by_default() -> bool:
    return _is_weekend(datetime.now().date())


def parse_index(html: str, source: str) -> Optional[IndexQuote]:
    m = re.search(
        r'<div class="quotient\s+(?P<class>up|dn|same)"[^>]*>.*?'
        r'<em id="now_value">\s*([0-9,\.]+)\s*</em>.*?'
        r'<span class="fluc" id="change_value_and_rate">\s*<span>([0-9,\.]+)</span>\s*'
        r'([+-]?[0-9,\.]+%)\s*<span class="blind">([^<]+)</span>',
        html,
        re.S,
    )
    if not m:
        return None
    cls = m.group("class")
    value = _clean(m.group(2))
    change = _clean(m.group(3))
    pct = _clean(m.group(4))
    direction = _normalize_direction(_clean(m.group(5)))
    if cls == "dn" and direction == "상승":
        direction = "하락"
    if cls == "same":
        direction = "보합"
    return IndexQuote(value=value, change=change, change_pct=_fmt_percent(pct), direction=direction, source=source)


def parse_stock(html: str, code: str, source: str) -> Optional[MarketQuote]:
    dd_texts = re.findall(r"<dd>(.*?)</dd>", html, re.S)
    text_blob = " | ".join(_clean(re.sub(r"<.*?>", " ", dd)) for dd in dd_texts)
    if not text_blob:
        return None

    name = ""
    m_name = re.search(r"종목명\s+([^|]+)", text_blob)
    if m_name:
        name = _clean(m_name.group(1))

    m = re.search(
        r"현재가\s*([0-9,]+)\s*전일대비\s*(상승|하락|보합)?\s*([0-9,]+)?\s*(플러스|마이너스)?\s*([0-9,.]+)?\s*퍼센트",
        text_blob,
    )
    if not m:
        # fallback: current page can still expose the value via the visible markup
        m = re.search(
            r'<p class="no_today">.*?<span class="blind">\s*([0-9,]+)\s*</span>.*?'
            r'/?p>.*?<p class="no_exday">.*?<span class="ico ([^\"]+)">([^<]+)</span>.*?'
            r'<span class="blind">\s*([0-9,\.]+)\s*</span>',
            html,
            re.S,
        )
        if not m:
            return None
        price = _clean(m.group(1))
        direction = _normalize_direction(_clean(m.group(3)))
        change = _clean(m.group(4))
        pct = _clean(m.group(5))
    else:
        price = _clean(m.group(1))
        direction = _normalize_direction(_clean(m.group(2) or ""))
        change = _clean(m.group(3) or "")
        pct = _clean(m.group(5) or "")
        if m.group(4) == "마이너스" and direction == "상승":
            direction = "하락"
        if m.group(4) == "플러스" and direction == "하락":
            direction = "상승"

    return MarketQuote(
        name=name or code,
        code=code,
        price=price,
        change=change,
        change_pct=_fmt_percent(pct) if pct else "",
        direction=direction,
        source=source,
    )


ROW_RE = re.compile(
    r'<a href="/item/main\.naver\?code=(?P<code>\d{6})" class="tltle">(?P<name>[^<]+)</a></td>\s*'
    r'<td class="number">(?P<price>[^<]+)</td>\s*'
    r'<td class="number">\s*(?:<em class="bu_p [^\"]+"><span class="blind">(?P<tag>[^<]+)</span></em>)?'
    r'<span class="tah p11 [^\"]+">\s*(?P<change>[^<]+)\s*</span>\s*</td>\s*'
    r'<td class="number">\s*<span class="tah p11 [^\"]+">\s*(?P<pct>[+-]?[0-9,\.]+%)\s*</span>\s*</td>',
    re.S,
)


def parse_movers(html: str, source: str, *, limit: int = 5) -> List[MarketQuote]:
    items: List[MarketQuote] = []
    for m in ROW_RE.finditer(html):
        tag = _clean(m.group("tag") or "")
        change = _clean(m.group("change"))
        pct = _clean(m.group("pct"))
        if any(tok in tag for tok in ("상한가", "bu_pup")):
            direction = "상승"
        elif any(tok in tag for tok in ("하한가", "bu_pdn")):
            direction = "하락"
        else:
            if change.startswith("+"):
                direction = "상승"
            elif change.startswith("-"):
                direction = "하락"
            elif pct.startswith("-"):
                direction = "하락"
            elif pct.startswith("+"):
                direction = "상승"
            else:
                direction = "보합"
        if direction == "상승" and not change.startswith(("+", "-")):
            change = "+" + change
        elif direction == "하락" and not change.startswith(("+", "-")):
            change = "-" + change
        items.append(
            MarketQuote(
                name=_clean(m.group("name")),
                code=m.group("code"),
                price=_clean(m.group("price")),
                change=change,
                change_pct=pct,
                direction=direction,
                source=source,
                tag=tag,
            )
        )
        if len(items) >= limit:
            break
    return items


def first_or_empty(items: List[MarketQuote]) -> List[MarketQuote]:
    return items[:1] if items else []


def _section_lines(title: str, items: Iterable[str]) -> List[str]:
    rows = [f"{title}"]
    rows.extend(items)
    return rows


def format_index_section(index: IndexQuote) -> List[str]:
    return [
        f"- KOSPI {index.value} ({index.direction} {index.change} / {index.change_pct})",
        f"  출처: {index.source}",
    ]


def format_stock_section(stock: MarketQuote) -> List[str]:
    return [
        f"- {stock.name}({stock.code}) {stock.price} ({stock.direction} {stock.change} / {stock.change_pct})",
        f"  출처: {stock.source}",
    ]


def format_mover(item: MarketQuote, *, inferred: bool = False) -> str:
    suffix = " (추정)" if inferred else ""
    tag = f"[{item.tag}] " if item.tag else ""
    return (
        f"- {tag}{item.name}({item.code}) {item.price} ({item.direction} {item.change} / {item.change_pct}){suffix}"
        f"\n  출처: {item.source}"
    )


def build_briefing(today: date) -> tuple[List[str], MarketBriefingGate | None]:
    lines: List[str] = []
    lines.append(f"{MENTION} {TITLE}")
    lines.append("")

    index_html = _fetch(INDEX_URL)
    index = parse_index(index_html, INDEX_URL)
    summary: List[str] = []
    stock_htmls: dict[str, str] = {}
    if index:
        summary.append(f"- KOSPI는 현재 {index.value}이며 전일 대비 {index.direction} {index.change} ({index.change_pct})입니다.")

    samsung_stock: Optional[MarketQuote] = None
    hynix_stock: Optional[MarketQuote] = None
    stock_cache: dict[str, Optional[MarketQuote]] = {}
    for code in ("005930", "000660"):
        html = _fetch(STOCK_URL.format(code=code))
        stock_htmls[code] = html
        stock_cache[code] = parse_stock(html, code, STOCK_URL.format(code=code))

    freshness_gate = should_skip_market_briefing(
        today=today,
        market_dates=build_market_snapshot_dates(index_html, [stock_htmls.get("005930", ""), stock_htmls.get("000660", "")]),
    )
    if freshness_gate.should_skip:
        return [], freshness_gate

    samsung_stock = stock_cache.get("005930")
    hynix_stock = stock_cache.get("000660")
    if samsung_stock:
        summary.append(
            f"- 삼성전자는 {samsung_stock.price}으로 {samsung_stock.direction} {samsung_stock.change} ({samsung_stock.change_pct})입니다."
        )
    if hynix_stock:
        summary.append(
            f"- SK하이닉스는 {hynix_stock.price}으로 {hynix_stock.direction} {hynix_stock.change} ({hynix_stock.change_pct})입니다."
        )

    if summary:
        lines.append("핵심 요약")
        lines.extend(summary[:3])
        lines.append("")

    if index:
        lines.extend(_section_lines("1) KOSPI 지수", format_index_section(index)))
    else:
        lines.append("1) KOSPI 지수")
        lines.append("- 지수 정보를 파싱하지 못했습니다.")
        lines.append(f"  출처: {INDEX_URL}")

    for section_no, (code, label) in enumerate([("005930", "삼성전자"), ("000660", "SK하이닉스")], start=2):
        stock = stock_cache.get(code)
        lines.append("")
        lines.append(f"{section_no}) {label}")
        if stock:
            lines.extend(format_stock_section(stock))
        else:
            lines.append("- 종목 정보를 파싱하지 못했습니다.")
            lines.append(f"  출처: {STOCK_URL.format(code=code)}")

    lines.append("")
    lines.append("4) KOSPI 급등락 종목")

    upper = parse_movers(_fetch(UP_URL), UP_URL, limit=3)
    lower = parse_movers(_fetch(DOWN_URL), DOWN_URL, limit=3)
    rise = parse_movers(_fetch(RISE_URL), RISE_URL, limit=3)
    fall = parse_movers(_fetch(FALL_URL), FALL_URL, limit=3)

    # limit up/down 우선, 없으면 ±15% 이상 급등/급락으로 대체
    if upper:
        lines.append("- 상한가")
        lines.extend(format_mover(x) for x in first_or_empty(upper))
    else:
        fallback = [x for x in rise if _pct_value(x.change_pct) >= 15.0]
        if fallback:
            lines.append("- 급등 (상한가 미발견, ±15% 이상 대체)")
            lines.extend(format_mover(x, inferred=True) for x in fallback[:3])
        else:
            lines.append("- 급등 (상한가 미발견, ±15% 이상 대체)")
            lines.append(f"- 자료 부족: {RISE_URL}")

    if lower:
        lines.append("- 하한가")
        lines.extend(format_mover(x) for x in first_or_empty(lower))
    else:
        fallback = [x for x in fall if _pct_value(x.change_pct) <= -15.0]
        if fallback:
            lines.append("- 급락 (하한가 미발견, ±15% 이상 대체)")
            lines.extend(format_mover(x, inferred=True) for x in fallback[:3])
        else:
            lines.append("- 급락 (하한가 미발견, ±15% 이상 대체)")
            lines.append(f"- 자료 부족: {FALL_URL}")

    return lines, None


def build_failure_report(message: str) -> List[str]:
    return [
        f"{MENTION} {TITLE}",
        "",
        "[진단] 브리핑 생성 중 일부 데이터 수집에 실패했습니다.",
        f"- 사유: {message}",
        f"- 지수 시도 출처: {INDEX_URL}",
        f"- 삼성전자 시도 출처: {STOCK_URL.format(code='005930')}",
        f"- SK하이닉스 시도 출처: {STOCK_URL.format(code='000660')}",
        "- 급등락 종목 시도 출처: 네이버 금융 KOSPI 상승/하락 페이지",
        "- 참고: 본 job은 Firecrawl를 사용하지 않고 requests 기반 직접 수집만 수행합니다.",
    ]


_def_pct_re = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?")


def _pct_value(pct: str) -> float:
    m = _def_pct_re.search(pct or "")
    return float(m.group(0)) if m else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KOSPI 오전장 브리핑 생성기")
    parser.add_argument("--simulate-kst", help="KST 시각을 강제로 지정합니다. 예: 2026-06-02T10:00:00+09:00")
    parser.add_argument(
        "--market-closed-policy-mode",
        choices=("notify", "silent"),
        default=None,
        help="휴장/stale 감지 시 notify(기본) 또는 silent(무출력)로 동작합니다.",
    )
    args = parser.parse_args(argv)

    now = get_kst_now(datetime.fromisoformat(args.simulate_kst) if args.simulate_kst else None)
    today = now.date()
    policy_mode = load_market_closed_policy_mode(args.market_closed_policy_mode)

    closed_gate = should_skip_market_briefing(today=today)
    if closed_gate.should_skip:
        if policy_mode != "silent":
            sys.stdout.write(format_market_skip_message(title=TITLE, decision=closed_gate, today=today) + "\n")
        return 0

    try:
        lines, freshness_gate = build_briefing(today=today)
    except Exception as exc:
        lines = build_failure_report(str(exc))
        freshness_gate = None

    if freshness_gate and freshness_gate.should_skip:
        if policy_mode != "silent":
            sys.stdout.write(format_market_skip_message(title=TITLE, decision=freshness_gate, today=today) + "\n")
        return 0

    sys.stdout.write("\n".join(lines).rstrip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
