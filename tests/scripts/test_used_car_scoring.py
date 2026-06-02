import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from used_car_scoring import score_listing


def _listing(**overrides):
    item = {
        "source": "kcar",
        "certified_flag": True,
        "warranty_flag": True,
        "required_options_matched": ["열선 시트", "후방카메라", "크루즈 컨트롤", "메모리시트"],
        "highlight_options_matched": ["HUD", "선루프", "어라운드뷰"],
        "option_data_present": True,
        "accident_note": "무사고",
        "flood_note": "침수 없음",
        "mileage_km": 30000,
        "year": 2022,
        "drivetrain": "AWD",
        "price_krw": 30000000,
        "brand": "현대",
        "model": "팰리세이드",
        "body_type": "SUV",
        "fuel": "가솔린",
    }
    item.update(overrides)
    return item


def test_score_listing_rewards_certification_options_and_clean_history():
    scored = score_listing(_listing())
    assert scored["score"] >= 70
    assert scored["score_breakdown"]["certified_warranty"] > 0
    assert scored["score_breakdown"]["required_options"] == 25
    assert scored["score_breakdown"]["accident_history"] >= 10


def test_score_listing_penalizes_missing_optional_quality_signals():
    scored = score_listing(
        _listing(
            certified_flag=False,
            warranty_flag=False,
            required_options_matched=[],
            highlight_options_matched=[],
            accident_note=None,
            flood_note=None,
            drivetrain=None,
            mileage_km=140000,
            year=2015,
        )
    )
    assert scored["score"] < 55
    assert scored["score_breakdown"]["required_options"] == 0
    assert scored["score_breakdown"]["drivetrain"] == 0
