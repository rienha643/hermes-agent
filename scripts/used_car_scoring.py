from __future__ import annotations

from statistics import median
from typing import Iterable

from used_car_normalize import REQUIRED_OPTION_ALIASES

SOURCE_TRUST_SCORES = {
    "hyundai_certified": 20,
    "kia_certified": 20,
    "kcar": 18,
    "encar_certified": 16,
    "usedcar_destroyer": 11,
    "jungcar_tv": 9,
}


def score_listings(listings: list[dict]) -> list[dict]:
    group_prices = _build_group_price_baseline(listings)
    return [score_listing(listing, group_prices) for listing in listings]


def score_listing(listing: dict, group_prices: dict | None = None) -> dict:
    score_breakdown: dict[str, int] = {}

    certified_score = 0
    if listing.get("certified_flag"):
        certified_score += 8
    if listing.get("warranty_flag"):
        certified_score += 5
    if listing.get("certified_flag") and listing.get("warranty_flag"):
        certified_score += 2
    score_breakdown["certified_warranty"] = certified_score

    required_options = listing.get("required_options_matched", []) or []
    option_data_present = bool(listing.get("option_data_present"))
    required_score = min(len(required_options), 4) * 5
    if len(required_options) >= 4:
        required_score += 5
    if not option_data_present:
        required_score = 0
    score_breakdown["required_options"] = min(required_score, 25)

    highlight_options = listing.get("highlight_options_matched", []) or []
    highlight_score = min(len(highlight_options), 5) * 2
    premium_bonus = sum(1 for option in highlight_options if option in {"HUD", "선루프", "어라운드뷰", "원격스마트주차보조"})
    score_breakdown["highlight_options"] = min(highlight_score + premium_bonus, 15)

    accident_score = 0
    accident_note = (listing.get("accident_note") or "").lower()
    flood_note = (listing.get("flood_note") or "").lower()
    lower_body_note = (listing.get("lower_body_note") or "").lower()
    if "무사고" in accident_note:
        accident_score += 8
    if "없음" in flood_note or "무" in flood_note and "침수" in flood_note:
        accident_score += 5
    if any(token in lower_body_note for token in ["이상 없음", "정상"]):
        accident_score += 2
    score_breakdown["accident_history"] = min(accident_score, 15)

    score_breakdown["price_reasonableness"] = _price_score(listing, group_prices or {})
    score_breakdown["mileage"] = _mileage_score(listing.get("mileage_km"))
    score_breakdown["year"] = _year_score(listing.get("year"))
    score_breakdown["source_trust"] = min(SOURCE_TRUST_SCORES.get(listing.get("source"), 0), 20)
    score_breakdown["drivetrain"] = 5 if listing.get("drivetrain") else 0
    score_breakdown["option_data"] = 5 if option_data_present else 0

    total = sum(score_breakdown.values())
    scored = dict(listing)
    scored["score_breakdown"] = score_breakdown
    scored["score"] = total
    if total >= 70:
        scored["score_tier"] = "우선 추천"
    elif total >= 55:
        scored["score_tier"] = "일반 추천"
    else:
        scored["score_tier"] = "검토 필요"
    return scored


def _build_group_price_baseline(listings: Iterable[dict]) -> dict[tuple, float]:
    grouped: dict[tuple, list[int]] = {}
    for listing in listings:
        key = (listing.get("brand"), listing.get("model"), listing.get("fuel"), listing.get("year"))
        price = listing.get("price_krw")
        if key[0] and key[1] and key[2] and key[3] and price:
            grouped.setdefault(key, []).append(int(price))
    return {key: median(values) for key, values in grouped.items() if values}


def _price_score(listing: dict, group_prices: dict[tuple, float]) -> int:
    key = (listing.get("brand"), listing.get("model"), listing.get("fuel"), listing.get("year"))
    price = listing.get("price_krw")
    if not price:
        return 0
    baseline = group_prices.get(key)
    if not baseline:
        return 8
    ratio = price / baseline
    if ratio <= 0.95:
        return 15
    if ratio <= 1.05:
        return 12
    if ratio <= 1.15:
        return 7
    return 0


def _mileage_score(mileage_km: int | None) -> int:
    if mileage_km is None:
        return 0
    if mileage_km < 40000:
        return 10
    if mileage_km < 80000:
        return 7
    if mileage_km < 120000:
        return 4
    return 0


def _year_score(year: int | None) -> int:
    if year is None:
        return 0
    if year >= 2023:
        return 5
    if year >= 2021:
        return 4
    if year >= 2019:
        return 3
    if year >= 2017:
        return 2
    return 1


def explain_score(listing: dict) -> str:
    breakdown = listing.get("score_breakdown") or {}
    trust = breakdown.get("source_trust", 0) + breakdown.get("certified_warranty", 0)
    options = (
        breakdown.get("required_options", 0)
        + breakdown.get("highlight_options", 0)
        + breakdown.get("option_data", 0)
    )
    condition = breakdown.get("accident_history", 0)
    value = breakdown.get("price_reasonableness", 0)
    age_use = breakdown.get("mileage", 0) + breakdown.get("year", 0)
    drivetrain = breakdown.get("drivetrain", 0)
    parts = [
        f"신뢰 {trust}",
        f"옵션 {options}",
        f"상태 {condition}",
        f"가격 {value}",
        f"주행/연식 {age_use}",
    ]
    if drivetrain:
        parts.append(f"구동 {drivetrain}")
    return " / ".join(parts)
