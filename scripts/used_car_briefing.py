from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Iterable

from used_car_filters import apply_hard_filters, apply_recommendation_filters, summarize_exclusions
from used_car_normalize import normalize_source_label
from used_car_scoring import explain_score, score_listings
from used_car_sources import SOURCE_CONFIG, fetch_all_sources


def dedupe_listings(listings: Iterable[dict]) -> list[dict]:
    best_by_key: dict[tuple, dict] = {}
    for listing in listings:
        key = (
            listing.get("brand"),
            listing.get("model") or listing.get("title"),
            listing.get("year"),
            listing.get("price_krw"),
        )
        current = best_by_key.get(key)
        if current is None or listing.get("score", 0) > current.get("score", 0):
            best_by_key[key] = listing
    return sorted(best_by_key.values(), key=lambda item: item.get("score", 0), reverse=True)


def render_markdown(
    listings: list[dict],
    collected_count: int,
    filtered_count: int,
    exclusion_summary: dict[str, int],
    recommendation_exclusion_summary: dict[str, int],
    source_errors: dict[str, str],
    source_coverage: list[str] | None = None,
    collection_failed: bool = False,
    fetch_failure_count: int = 0,
) -> str:
    if collection_failed or (not listings and collected_count == 0 and (source_errors or fetch_failure_count > 0)):
        lines = ["[중고차 알림] 수집 실패"]
        for source, reason in source_errors.items():
            lines.append(f"- {source}: {reason}")
        if not source_errors and fetch_failure_count:
            lines.append(f"- 상세 수집 실패 {fetch_failure_count}건")
        return "\n".join(lines)

    if not listings:
        if filtered_count == 0:
            lines = ["[중고차 알림] 조건에 맞는 매물이 없습니다."]
            lines.append(f"요약: 수집 {collected_count}건 / 하드필터 통과 0건")
        else:
            lines = ["[중고차 알림] 조건에 맞는 추천 매물이 없습니다."]
            lines.append(
                f"요약: 수집 {collected_count}건 / 하드필터 통과 {filtered_count}건 / 추천 기준 통과 0건"
            )
        merged = merge_summaries(exclusion_summary, recommendation_exclusion_summary)
        if merged:
            reasons = ", ".join(f"{format_reason(reason)} {count}" for reason, count in sorted(merged.items()))
            lines.append(f"제외 요인: {reasons}")
        if source_errors:
            errors = ", ".join(f"{source}={reason}" for source, reason in source_errors.items())
            lines.append(f"수집 경고: {errors}")
        if source_coverage:
            lines.append(f"소스별 현황: {', '.join(source_coverage)}")
        return "\n".join(lines)

    lines = [f"[중고차 알림] 조건 일치 {len(listings)}건", ""]
    for index, listing in enumerate(listings, start=1):
        lines.extend(render_listing(index, listing))
        lines.append("")
    lines.append(
        f"요약: 수집 {collected_count}건 / 하드필터 통과 {filtered_count}건 / 추천 기준 통과 {len(listings)}건"
    )
    if source_coverage:
        lines.append(f"소스별 현황: {', '.join(source_coverage)}")
    merged = merge_summaries(exclusion_summary, recommendation_exclusion_summary)
    if merged:
        reasons = ", ".join(f"{format_reason(reason)} {count}" for reason, count in sorted(merged.items()))
        lines.append(f"제외 요인: {reasons}")
    if source_errors:
        errors = ", ".join(f"{source}={reason}" for source, reason in source_errors.items())
        lines.append(f"수집 경고: {errors}")
    return "\n".join(lines).strip()


def render_listing(index: int, listing: dict) -> list[str]:
    price = format_krw(listing.get("price_krw"))
    year = listing.get("year")
    month = listing.get("month")
    year_label = f"{year}년" + (f" {month}월" if month else "") if year else "연식 미확인"
    mileage = f"{listing.get('mileage_km', 0):,}km" if listing.get("mileage_km") is not None else "주행거리 미확인"
    fuel = listing.get("fuel") or "연료 미확인"
    drivetrain = listing.get("drivetrain") or "미확인"
    warranty_label = listing.get("warranty_label") or ("있음" if listing.get("warranty_flag") else "없음")
    required = listing.get("required_options_matched", [])
    highlights = listing.get("highlight_options_matched", [])
    accident = listing.get("accident_note") or "미확인"
    flood = listing.get("flood_note") or "미확인"
    option_status = "옵션 미확인" if not listing.get("option_data_present") else f"{len(required)}/4" + (f" ({', '.join(required)})" if required else "")
    return [
        f"{index}) [{format_link_text(listing.get('title'))}]({listing.get('detail_url')})",
        f"- {price} | {year_label} | {mileage} | {fuel} | {drivetrain}",
        f"- 인증: {'O' if listing.get('certified_flag') else 'X'} | 보증: {'O' if listing.get('warranty_flag') else 'X'} ({warranty_label})",
        f"- 필수옵션: {option_status}",
        f"- 강조옵션: {', '.join(highlights) if highlights else '없음'}",
        f"- 사고/침수: {accident} / {flood}",
        f"- 소스: {normalize_source_label(listing.get('source', ''))} | 점수: {listing.get('score', 0)}점 ({listing.get('score_tier', '미분류')})",
        f"- 점수근거: {explain_score(listing)}",
    ]


def format_krw(value: int | None) -> str:
    return f"{value:,}원" if value is not None else "가격 미확인"


def format_link_text(title: str | None) -> str:
    cleaned = (title or "").strip()
    if not cleaned:
        return "제목 미확인"
    if cleaned.startswith("[") and "]" in cleaned:
        return cleaned.split("]", 1)[1].lstrip()
    return cleaned


def merge_summaries(*summaries: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for summary in summaries:
        for reason, count in summary.items():
            merged[reason] = merged.get(reason, 0) + count
    return merged


def format_reason(reason: str) -> str:
    mapping = {
        "body_type": "세단/비SUV",
        "mileage": "주행거리 초과",
        "fuel": "연료 제외",
        "brand": "브랜드 제외",
        "year": "연식 기준 미달",
        "sale_status": "판매상태 제외",
        "missing_source": "source 누락",
        "missing_brand": "brand 누락",
        "missing_body_type": "body_type 누락",
        "missing_year": "year 누락",
        "missing_mileage": "mileage 누락",
        "missing_fuel": "fuel 누락",
        "missing_price": "price 누락",
        "missing_listing_id": "listing_id 누락",
        "missing_title": "title 누락",
        "score_below_threshold": "추천점수 미달",
        "required_options_below_min": "필수옵션 부족",
        "option_unverified": "옵션 미확인",
        "drivetrain_domestic": "국산 2WD 제외",
        "size_class": "경형/소형 제외",
    }
    return mapping.get(reason, reason)


def build_brief(limit_per_source: int = 24) -> tuple[str, dict]:
    raw_listings, source_errors = fetch_all_sources(limit_per_source=limit_per_source)
    source_errors = dict(source_errors)
    fetch_failure_count = sum(1 for item in raw_listings if item.get("fetch_error"))
    successful_raw = [item for item in raw_listings if not item.get("fetch_error")]
    filtered, excluded = apply_hard_filters(successful_raw)
    scored = score_listings(filtered)
    recommended, recommendation_excluded = apply_recommendation_filters(scored)
    deduped = dedupe_listings(recommended)
    source_coverage = build_source_coverage(successful_raw, filtered, deduped)
    collection_failed = not successful_raw and (bool(source_errors) or fetch_failure_count > 0)
    markdown = render_markdown(
        listings=deduped,
        collected_count=len(successful_raw),
        filtered_count=len(filtered),
        exclusion_summary=summarize_exclusions(excluded),
        recommendation_exclusion_summary=summarize_exclusions(recommendation_excluded),
        source_errors=source_errors,
        source_coverage=source_coverage,
        collection_failed=collection_failed,
        fetch_failure_count=fetch_failure_count,
    )
    stats = {
        "raw_collected": len(raw_listings),
        "collected": len(successful_raw),
        "filtered": len(filtered),
        "recommended": len(deduped),
        "excluded": len(excluded),
        "recommendation_excluded": len(recommendation_excluded),
        "fetch_failures": fetch_failure_count,
        "collection_failed": collection_failed,
        "source_errors": source_errors,
        "source_coverage": source_coverage,
    }
    return markdown, stats


def build_source_coverage(raw: list[dict], filtered: list[dict], recommended: list[dict]) -> list[str]:
    raw_counts = _source_counts(raw)
    filtered_counts = _source_counts(filtered)
    recommended_counts = _source_counts(recommended)
    lines = []
    for source in SOURCE_CONFIG:
        label = normalize_source_label(source)
        lines.append(f"{label} {raw_counts.get(source, 0)}→{filtered_counts.get(source, 0)}→{recommended_counts.get(source, 0)}")
    return lines


def _source_counts(listings: list[dict]) -> Counter:
    return Counter(item.get("source") for item in listings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Used car briefing generator (phase 1)")
    parser.add_argument("--limit-per-source", type=int, default=24)
    args = parser.parse_args()
    markdown, _stats = build_brief(limit_per_source=args.limit_per_source)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
