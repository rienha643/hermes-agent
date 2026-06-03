#!/usr/bin/env python3
"""오늘의 KOSPI 마감 브리핑 생성기.

- Firecrawl / web_search / web_extract / browser 비의존
- requests + 정규식 + 표준 라이브러리만 사용
- 기본적으로 주말에는 조용히 종료
- 평일에는 stdout 으로 본문만 출력
- 실패 시 [SILENT] 대신 진단 메시지 출력
"""

from __future__ import annotations

import argparse
import html as html_lib
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Optional, Tuple

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
TITLE = "오늘의 KOSPI 마감 브리핑"
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
SESSION.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }
)


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
    return v if v.endswith("%") else v + "%"


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


def _pct_value(pct: str) -> float:
    m = re.search(r"[+-]?[0-9]+(?:\.[0-9]+)?", pct or "")
    return float(m.group(0)) if m else 0.0


def parse_index(html: str, source: str) -> Optional[IndexQuote]:
    # 우선순위: 정규 페이지 구조 → Jina 미러/대체 마크업
    patterns = [
        re.compile(
            r'<div class="quotient\s+(?P<class>up|dn|same)"[^>]*>.*?'
            r'<em id="now_value">\s*([0-9,\.]+)\s*</em>.*?'
            r'<span class="fluc" id="change_value_and_rate">\s*<span>([0-9,\.]+)</span>\s*'
            r'([+-]?[0-9,\.]+%)\s*<span class="blind">([^<]+)</span>',
            re.S,
        ),
        re.compile(
            r'현재가\s*([0-9,\.]+)\s*전일대비\s*(상승|하락|보합)?\s*([0-9,\.]+)?\s*(?:플러스|마이너스)?\s*([+-]?[0-9,\.]+%)?\s*',
            re.S,
        ),
    ]

    for idx, pat in enumerate(patterns):
        m = pat.search(html)
        if not m:
            continue
        if idx == 0:
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
        value = _clean(m.group(1))
        direction = _normalize_direction(_clean(m.group(2) or ""))
        change = _clean(m.group(3) or "")
        pct = _clean(m.group(4) or "")
        return IndexQuote(value=value, change=change, change_pct=_fmt_percent(pct), direction=direction, source=source)

    # fallback: Jina/미러에 자주 보이는 라벨 기반 추출
    m = re.search(r"KOSPI.*?([0-9,]+(?:\.[0-9]+)?)\s*([+-][0-9,\.]+)\s*([+-]?[0-9,\.]+%)", html, re.S)
    if m:
        value = _clean(m.group(1))
        change = _clean(m.group(2))
        pct = _clean(m.group(3))
        direction = "상승" if change.startswith("+") or pct.startswith("+") else "하락" if change.startswith("-") or pct.startswith("-") else "보합"
        return IndexQuote(value=value, change=change.lstrip("+-"), change_pct=_fmt_percent(pct), direction=direction, source=source)

    return None


def parse_stock(html: str, code: str, source: str) -> Optional[MarketQuote]:
    dd_texts = re.findall(r"<dd>(.*?)</dd>", html, re.S)
    text_blob = " | ".join(_clean(re.sub(r"<.*?>", " ", dd)) for dd in dd_texts)

    # 표준 네이버 금융 마크업
    m = re.search(
        r"현재가\s*([0-9,]+)\s*전일대비\s*(상승|하락|보합)?\s*([0-9,]+)?\s*(플러스|마이너스)?\s*([0-9,.]+)?\s*퍼센트",
        text_blob,
    )
    if m:
        price = _clean(m.group(1))
        direction = _normalize_direction(_clean(m.group(2) or ""))
        change = _clean(m.group(3) or "")
        pct = _clean(m.group(5) or "")
        if m.group(4) == "마이너스" and direction == "상승":
            direction = "하락"
        if m.group(4) == "플러스" and direction == "하락":
            direction = "상승"
        return MarketQuote(
            name=_extract_stock_name(text_blob) or code,
            code=code,
            price=price,
            change=change,
            change_pct=_fmt_percent(pct) if pct else "",
            direction=direction,
            source=source,
        )

    # 마감 페이지/미러 대응
    m_today = re.search(r'<p class="no_today">.*?<span class="blind">\s*([0-9,]+)\s*</span>', html, re.S)
    m_exday = re.search(
        r'<p class="no_exday">.*?<span class="ico\s+([^"]+)">([^<]+)</span>.*?'
        r'<span class="blind">\s*([0-9,\.]+)\s*</span>.*?'
        r'<span class="ico\s+plus">\+?</span>.*?<span class="blind">\s*([0-9,\.]+)\s*</span>',
        html,
        re.S,
    )
    if m_today and m_exday:
        price = _clean(m_today.group(1))
        icon_class = _clean(m_exday.group(1))
        direction = _normalize_direction(_clean(m_exday.group(2)))
        change = _clean(m_exday.group(3))
        pct = _clean(m_exday.group(4))
        if "down" in icon_class or "dn" in icon_class:
            direction = "하락"
        elif "up" in icon_class:
            direction = "상승"
        return MarketQuote(
            name=_extract_stock_name(text_blob) or code,
            code=code,
            price=price,
            change=change,
            change_pct=_fmt_percent(pct) if pct else "",
            direction=direction,
            source=source,
        )

    # 미러/대체 페이지의 간단 라벨 기반
    m = re.search(r"종가\s*([0-9,]+).*?(상승|하락|보합).*?([+-]?[0-9,\.]+%)", html, re.S)
    if m:
        return MarketQuote(
            name=_extract_stock_name(text_blob) or code,
            code=code,
            price=_clean(m.group(1)),
            change="",
            change_pct=_fmt_percent(_clean(m.group(3))),
            direction=_normalize_direction(_clean(m.group(2))),
            source=source,
        )

    return None


def _extract_stock_name(text_blob: str) -> str:
    m = re.search(r"종목명\s+([^|]+)", text_blob)
    return _clean(m.group(1)) if m else ""


ROW_RE = re.compile(
    r'<a href="/item/main\.naver\?code=(?P<code>\d{6})" class="tltle">(?P<name>[^<]+)</a></td>\s*'
    r'<td class="number">(?P<price>[^<]+)</td>\s*'
    r'<td class="number">\s*(?:<em class="bu_p [^"]+"><span class="blind">(?P<tag>[^<]+)</span></em>)?'
    r'<span class="tah p11 [^"]+">\s*(?P<change>[^<]+)\s*</span>\s*</td>\s*'
    r'<td class="number">\s*<span class="tah p11 [^"]+">\s*(?P<pct>[+-]?[0-9,\.]+%)\s*</span>\s*</td>',
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


def format_mover(item: MarketQuote, *, inferred: bool = False) -> str:
    suffix = " (추정)" if inferred else ""
    tag = f"[{item.tag}] " if item.tag else ""
    return (
        f"- {tag}{item.name}({item.code}) {item.price} ({item.direction} {item.change} / {item.change_pct}){suffix}"
        f"\n  출처: {item.source}"
    )


def _section_lines(title: str, items: Iterable[str]) -> List[str]:
    rows = [title]
    rows.extend(items)
    return rows


def _trend_phrase(index: Optional[IndexQuote], samsung: Optional[MarketQuote], hynix: Optional[MarketQuote]) -> str:
    if not index:
        return "내일은 지수 방향을 다시 확인해야 합니다."

    if index.direction == "상승":
        base = "내일은 강세 연장 여부와 장중 차익실현 압력을 함께 보시는 것이 좋겠습니다."
    elif index.direction == "하락":
        base = "내일은 낙폭 과대 여부와 외국인 수급 회복 여부를 확인하시는 것이 좋겠습니다."
    else:
        base = "내일은 박스권 이탈 여부와 대형주 수급 변화를 확인하시는 것이 좋겠습니다."

    if samsung and hynix:
        if samsung.direction == "상승" and hynix.direction == "상승":
            base += " 특히 반도체 대형주의 동반 흐름이 이어지는지 보시는 것이 좋겠습니다."
        elif samsung.direction == "하락" and hynix.direction == "하락":
            base += " 특히 반도체 대형주의 약세가 지수에 재차 부담이 되는지 살펴보시는 것이 좋겠습니다."
        else:
            base += " 특히 삼성전자와 SK하이닉스의 엇갈린 흐름이 지수에 미치는 영향을 보시는 것이 좋겠습니다."

    return base


def _summary_line(index: Optional[IndexQuote], samsung: Optional[MarketQuote], hynix: Optional[MarketQuote]) -> str:
    parts: List[str] = []
    if index:
        parts.append(f"KOSPI {index.value} ({index.direction} {index.change} / {index.change_pct})")
    if samsung:
        parts.append(f"삼성전자 {samsung.price} ({samsung.direction} {samsung.change} / {samsung.change_pct})")
    if hynix:
        parts.append(f"SK하이닉스 {hynix.price} ({hynix.direction} {hynix.change} / {hynix.change_pct})")
    return "- " + " | ".join(parts) if parts else "- 핵심 지표를 파싱하지 못했습니다."


def build_briefing(today: date) -> tuple[List[str], MarketBriefingGate | None]:
    lines: List[str] = [f"{MENTION} {TITLE}", ""]

    errors: List[str] = []
    index_html = ""
    index: Optional[IndexQuote] = None
    samsung: Optional[MarketQuote] = None
    hynix: Optional[MarketQuote] = None
    stock_htmls: dict[str, str] = {}

    try:
        index_html = _fetch(INDEX_URL)
        index = parse_index(index_html, INDEX_URL)
    except Exception as exc:
        errors.append(f"KOSPI 지수 수집 실패: {exc}")

    stock_cache: dict[str, Optional[MarketQuote]] = {}
    for code in ("005930", "000660"):
        try:
            html = _fetch(STOCK_URL.format(code=code))
            stock_htmls[code] = html
            stock_cache[code] = parse_stock(html, code, STOCK_URL.format(code=code))
        except Exception as exc:
            stock_cache[code] = None
            errors.append(f"종목 {code} 수집 실패: {exc}")

    samsung = stock_cache.get("005930")
    hynix = stock_cache.get("000660")

    freshness_gate: MarketBriefingGate | None = None
    if not errors:
        freshness_gate = should_skip_market_briefing(
            today=today,
            market_dates=build_market_snapshot_dates(index_html, [stock_htmls.get("005930", ""), stock_htmls.get("000660", "")]),
        )
        if freshness_gate.should_skip:
            return [], freshness_gate

    lines.append("핵심 요약")
    lines.append(_summary_line(index, samsung, hynix))
    lines.append(f"- { _trend_phrase(index, samsung, hynix) }")
    lines.append("")

    lines.append("1) KOSPI 지수")
    if index:
        lines.extend(
            [
                f"- KOSPI {index.value} ({index.direction} {index.change} / {index.change_pct})",
                f"  출처: {index.source}",
            ]
        )
    else:
        lines.append("- 지수 정보를 파싱하지 못했습니다.")
        lines.append(f"  출처: {INDEX_URL}")

    for section_no, (code, label) in enumerate([("005930", "삼성전자"), ("000660", "SK하이닉스")], start=2):
        stock = stock_cache.get(code)
        lines.append("")
        lines.append(f"{section_no}) {label}")
        if stock:
            lines.append(
                f"- {stock.name}({stock.code}) {stock.price} ({stock.direction} {stock.change} / {stock.change_pct})"
            )
            lines.append(f"  출처: {stock.source}")
        else:
            lines.append("- 종목 정보를 파싱하지 못했습니다.")
            lines.append(f"  출처: {STOCK_URL.format(code=code)}")

    lines.append("")
    lines.append("4) KOSPI 급등락 종목")

    movers: List[Tuple[str, str, List[MarketQuote]]] = []
    for title, url in (("상한가", UP_URL), ("하한가", DOWN_URL), ("상승 상위", RISE_URL), ("하락 상위", FALL_URL)):
        try:
            movers.append((title, url, parse_movers(_fetch(url), url, limit=3)))
        except Exception as exc:
            errors.append(f"{title} 페이지 수집 실패: {exc}")
            movers.append((title, url, []))

    up_items = movers[0][2]
    down_items = movers[1][2]
    rise_items = movers[2][2]
    fall_items = movers[3][2]

    if up_items:
        lines.append("- 상한가")
        lines.extend(format_mover(x) for x in up_items[:3])
    else:
        fallback = [x for x in rise_items if _pct_value(x.change_pct) >= 15.0]
        lines.append("- 급등 (상한가 미발견, 상승 상위로 대체)")
        if fallback:
            lines.extend(format_mover(x, inferred=True) for x in fallback[:3])
        else:
            lines.append(f"- 자료 부족: {RISE_URL}")

    if down_items:
        lines.append("- 하한가")
        lines.extend(format_mover(x) for x in down_items[:3])
    else:
        fallback = [x for x in fall_items if _pct_value(x.change_pct) <= -15.0]
        lines.append("- 급락 (하한가 미발견, 하락 상위로 대체)")
        if fallback:
            lines.extend(format_mover(x, inferred=True) for x in fallback[:3])
        else:
            lines.append(f"- 자료 부족: {FALL_URL}")

    lines.append("")
    lines.append("5) 내일 관전 포인트")
    lines.append(f"- {_trend_phrase(index, samsung, hynix)}")

    if errors:
        lines.append("")
        lines.append("[진단]")
        lines.extend(f"- {msg}" for msg in errors)

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KOSPI 마감 브리핑 생성기")
    parser.add_argument("--simulate-kst", help="KST 시각을 강제로 지정합니다. 예: 2026-06-02T15:40:00+09:00")
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
