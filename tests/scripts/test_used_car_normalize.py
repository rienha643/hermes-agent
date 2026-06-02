import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from used_car_normalize import (
    normalize_brand,
    parse_year_month,
    parse_mileage_km,
    parse_price_krw,
    normalize_fuel,
    infer_body_type,
    normalize_drivetrain,
)


def test_normalize_brand_maps_korean_and_english_variants():
    assert normalize_brand("메르세데스-벤츠") == "벤츠"
    assert normalize_brand("Mercedes-Benz") == "벤츠"
    assert normalize_brand("현대") == "현대"
    assert normalize_brand("포르쉐") == "포르쉐"


def test_parse_year_month_extracts_year_and_month():
    assert parse_year_month("2019.01.17") == (2019, 1)
    assert parse_year_month("2018년 12월") == (2018, 12)
    assert parse_year_month("2015") == (2015, None)


def test_parse_mileage_km_extracts_integer():
    assert parse_mileage_km("221,408km") == 221408
    assert parse_mileage_km("150000 km") == 150000


def test_parse_price_krw_converts_manwon_to_krw():
    assert parse_price_krw("2,020 만원") == 20200000
    assert parse_price_krw("720만원") == 7200000


def test_normalize_fuel_maps_supported_types():
    assert normalize_fuel("휘발유") == "가솔린"
    assert normalize_fuel("가솔린") == "가솔린"
    assert normalize_fuel("경유") == "디젤"
    assert normalize_fuel("디젤") == "디젤"
    assert normalize_fuel("하이브리드") == "하이브리드"
    assert normalize_fuel("LPG") == "LPG"


def test_infer_body_type_detects_suv_keywords():
    assert infer_body_type("벤츠 GLS클래스 X167 GLS 450 4MATIC") == "SUV"
    assert infer_body_type("현대 팰리세이드 디젤 2.2 4WD 익스클루시브") == "SUV"
    assert infer_body_type("현대 아반떼 1.6 모던") == "세단"


def test_normalize_drivetrain_extracts_supported_values():
    assert normalize_drivetrain("4WD 익스클루시브") == "4WD"
    assert normalize_drivetrain("GLS 450 4MATIC") == "AWD"
    assert normalize_drivetrain("2WD T7") == "FWD"
    assert normalize_drivetrain("후륜구동 세단") == "RWD"
