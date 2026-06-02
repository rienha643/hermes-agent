import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from used_car_filters import apply_hard_filters, apply_recommendation_filters, summarize_exclusions


def _base_listing(**overrides):
    item = {
        "source": "jungcar_tv",
        "brand": "현대",
        "body_type": "SUV",
        "year": 2020,
        "month": 1,
        "mileage_km": 50000,
        "fuel": "가솔린",
        "price_krw": 25000000,
        "listing_id": "1",
        "title": "현대 팰리세이드",
        "sale_status": "판매중",
        "score": 72,
        "option_data_present": True,
        "required_options_matched": ["열선 시트", "후방카메라"],
    }
    item.update(overrides)
    return item


def test_apply_hard_filters_keeps_valid_suv_listing():
    kept, excluded = apply_hard_filters([_base_listing()])
    assert len(kept) == 1
    assert excluded == []


def test_apply_hard_filters_excludes_sedan_lpg_and_high_mileage():
    kept, excluded = apply_hard_filters(
        [
            _base_listing(listing_id="sedan", body_type="세단"),
            _base_listing(listing_id="lpg", fuel="LPG"),
            _base_listing(listing_id="mileage", mileage_km=150000),
        ]
    )
    assert kept == []
    reasons = {entry["listing"]["listing_id"]: entry["reason"] for entry in excluded}
    assert reasons["sedan"] == "body_type"
    assert reasons["lpg"] == "fuel"
    assert reasons["mileage"] == "mileage"


def test_apply_hard_filters_excludes_missing_required_fields():
    kept, excluded = apply_hard_filters([_base_listing(year=None), _base_listing(listing_id="no_brand", brand=None)])
    assert kept == []
    reasons = {entry["listing"]["listing_id"]: entry["reason"] for entry in excluded}
    assert reasons["1"] == "missing_year"
    assert reasons["no_brand"] == "missing_brand"


def test_summarize_exclusions_counts_reasons():
    _, excluded = apply_hard_filters(
        [
            _base_listing(listing_id="a", body_type="세단"),
            _base_listing(listing_id="b", body_type="세단"),
            _base_listing(listing_id="c", fuel="LPG"),
        ]
    )
    summary = summarize_exclusions(excluded)
    assert summary["body_type"] == 2
    assert summary["fuel"] == 1


def test_apply_recommendation_filters_excludes_low_score_and_option_issues():
    kept, excluded = apply_recommendation_filters(
        [
            _base_listing(listing_id="good", score=72, option_data_present=True, required_options_matched=["열선 시트", "후방카메라"]),
            _base_listing(listing_id="low-score", score=30, option_data_present=True, required_options_matched=["열선 시트", "후방카메라"]),
            _base_listing(listing_id="option-miss", score=80, option_data_present=True, required_options_matched=[]),
            _base_listing(listing_id="option-unknown", score=80, option_data_present=False, required_options_matched=[]),
        ]
    )
    assert [item["listing_id"] for item in kept] == ["good"]
    reasons = {entry["listing"]["listing_id"]: entry["reason"] for entry in excluded}
    assert reasons["low-score"] == "score_below_threshold"
    assert reasons["option-miss"] == "required_options_below_min"
    assert reasons["option-unknown"] == "option_unverified"
