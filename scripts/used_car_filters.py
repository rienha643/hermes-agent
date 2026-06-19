from __future__ import annotations

from collections import Counter
from typing import Iterable

from used_car_normalize import ALLOWED_BRANDS, ALLOWED_FUELS, ALLOWED_SOURCES

REQUIRED_FIELDS = ["source", "brand", "body_type", "year", "mileage_km", "fuel", "price_krw", "listing_id", "title"]
DOMESTIC_BRANDS = {"현대", "기아"}
DOMESTIC_ALLOWED_DRIVETRAINS = {"AWD", "4WD"}
EXCLUDED_BODY_KEYWORDS = {"카니발", "스타리아"}
EXCLUDED_SMALL_MODEL_KEYWORDS = {
    "레이",
    "캐스퍼",
    "코나",
    "베뉴",
    "셀토스",
    "니로",
    "티볼리",
    "트랙스",
    "트레일블레이저",
    "XM3",
    "티록",
    "T-ROC",
    "UX",
    "EX30",
}
MISSING_FIELD_REASONS = {
    "source": "missing_source",
    "brand": "missing_brand",
    "body_type": "missing_body_type",
    "year": "missing_year",
    "mileage_km": "missing_mileage",
    "fuel": "missing_fuel",
    "price_krw": "missing_price",
    "listing_id": "missing_listing_id",
    "title": "missing_title",
}


def apply_hard_filters(listings: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    excluded: list[dict] = []
    for listing in listings:
        reason = _first_exclusion_reason(listing)
        if reason is None:
            kept.append(listing)
        else:
            excluded.append({"listing": listing, "reason": reason})
    return kept, excluded


def _first_exclusion_reason(listing: dict) -> str | None:
    for field in REQUIRED_FIELDS:
        if listing.get(field) in (None, "", []):
            return MISSING_FIELD_REASONS[field]
    if listing.get("source") not in ALLOWED_SOURCES:
        return "source"
    if listing.get("brand") not in ALLOWED_BRANDS:
        return "brand"
    title = str(listing.get("title") or "")
    normalized_title = title.upper()
    if any(keyword in title for keyword in EXCLUDED_BODY_KEYWORDS):
        return "body_type"
    if any(keyword.upper() in normalized_title for keyword in EXCLUDED_SMALL_MODEL_KEYWORDS):
        return "size_class"
    if listing.get("body_type") != "SUV":
        return "body_type"
    year = listing.get("year")
    if year is None or int(year) < 2015:
        return "year"
    mileage_km = listing.get("mileage_km")
    if mileage_km is None or int(mileage_km) >= 150000:
        return "mileage"
    if listing.get("fuel") not in ALLOWED_FUELS:
        return "fuel"
    if listing.get("brand") in DOMESTIC_BRANDS and listing.get("drivetrain") not in DOMESTIC_ALLOWED_DRIVETRAINS:
        return "drivetrain_domestic"
    sale_status = listing.get("sale_status")
    if sale_status and sale_status not in {"판매중", "상담가능", "노출중"}:
        return "sale_status"
    return None


def apply_recommendation_filters(listings: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    excluded: list[dict] = []
    for listing in listings:
        score = listing.get("score", 0)
        option_data_present = bool(listing.get("option_data_present"))
        required_count = len(listing.get("required_options_matched", []) or [])
        if not option_data_present:
            excluded.append({"listing": listing, "reason": "option_unverified"})
            continue
        if required_count < 2:
            excluded.append({"listing": listing, "reason": "required_options_below_min"})
            continue
        if score < 55:
            excluded.append({"listing": listing, "reason": "score_below_threshold"})
            continue
        kept.append(listing)
    return kept, excluded


def summarize_exclusions(excluded: Iterable[dict]) -> dict[str, int]:
    counter = Counter(item["reason"] for item in excluded)
    return dict(counter)
