import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from used_car_sources import (
    parse_encar_listing,
    parse_kcar_listing,
    fetch_all_sources,
)


ENCAR_ITEM = {
    "Id": "41414437",
    "Manufacturer": "제네시스",
    "Model": "GV80 쿠페",
    "Badge": "2.5T 가솔린 AWD",
    "Transmission": "오토",
    "FuelType": "가솔린",
    "Year": 202401.0,
    "Mileage": 31807.0,
    "Price": 6800.0,
    "OfficeCityState": "경기",
    "DealerName": "기수헌",
    "Trust": ["ExtendWarranty", "HomeService"],
    "ServiceMark": ["EncarMeetgo", "EncarDiagnosisP1"],
}

ENCAR_HTML = """
<html><body>
GV80 쿠페 2.5T 가솔린 AWD 연식 24/01식 연형정보 주행거리 31,807 km 연료 가솔린 차량번호 177어5708
엔카진단+ 프레임 무사고 확인 내외부 차량 관리 상태 확인 옵션 정보 더보기
선루프 있음 후방카메라 있음 열선시트 있음 통풍시트 있음 스마트크루즈 있음
보증현황
</body></html>
"""

KCAR_ITEM = {
    "carCd": "EC61358223",
    "mnuftrNm": "기아",
    "modelNm": "쏘렌토",
    "grdNm": "2.2 디젤",
    "grdDtlNm": "시그니처 4WD",
    "prdcnYr": "2022",
    "mfgDt": "202203",
    "milg": "13572",
    "fuelNm": "디젤",
    "dcPrc": "3250",
    "cntrNm": "청주직영점",
    "statCd": "CAR_STATUS_020",
}

KCAR_DETAIL = {
    "master": {"whelNm": "전륜", "prstMilg": "13572"},
    "carhistory": {"carForm": "SUV", "fuel": "디젤", "owncarDmgeAcdtCnt": "0", "acdtCnt": "0", "fldgAcdtCnt": "0"},
    "warrantyInfo": [{"available": True, "gurntePrdInfo": "KW6 국산차,준중형차 180일, 10000Km"}],
    "mainOptList": [
        {"optioncdName": "열선시트 : 운전석", "optnNm": "열선시트 : 운전석", "grpParent": "열선시트", "grpChild": "운전석", "optDesc": "열선"},
        {"optioncdName": "후방카메라", "optnNm": "후방카메라", "grpParent": "카메라", "grpChild": "후방", "optDesc": "후카"},
        {"optioncdName": "크루즈컨트롤", "optnNm": "크루즈컨트롤", "grpParent": "크루즈", "grpChild": None, "optDesc": "주행보조"},
    ],
    "optList": [
        {"optioncdName": "메모리시트", "optnNm": "메모리시트", "grpParent": "시트", "grpChild": "메모리", "optDesc": "메모리 시트"},
        {"optioncdName": "HUD", "optnNm": "HUD", "grpParent": None, "grpChild": None, "optDesc": "헤드업 디스플레이"},
    ],
    "carHistoryAccList": [],
}


def test_parse_encar_listing_extracts_certification_options_and_fields():
    listing = parse_encar_listing(ENCAR_ITEM, "https://fem.encar.com/cars/detail/41414437", ENCAR_HTML)
    assert listing["source"] == "encar_certified"
    assert listing["brand"] == "제네시스"
    assert listing["year"] == 2024
    assert listing["month"] == 1
    assert listing["mileage_km"] == 31807
    assert listing["fuel"] == "가솔린"
    assert listing["price_krw"] == 68000000
    assert listing["drivetrain"] == "AWD"
    assert listing["certified_flag"] is True
    assert listing["warranty_flag"] is True
    assert "후방카메라" in listing["required_options_matched"]
    assert "통풍시트" in listing["highlight_options_matched"]


def test_parse_kcar_listing_extracts_detail_json_fields():
    listing = parse_kcar_listing(KCAR_ITEM, "https://www.kcar.com/bc/car-info?i_sCarCd=EC61358223", KCAR_DETAIL)
    assert listing["source"] == "kcar"
    assert listing["brand"] == "기아"
    assert listing["model"] == "쏘렌토"
    assert listing["body_type"] == "SUV"
    assert listing["year"] == 2022
    assert listing["month"] == 3
    assert listing["mileage_km"] == 13572
    assert listing["fuel"] == "디젤"
    assert listing["price_krw"] == 32500000
    assert listing["warranty_flag"] is True
    assert listing["seller_type"] == "직영"
    assert listing["accident_note"] == "무사고"
    assert listing["flood_note"] == "침수 없음"
    assert set(["열선 시트", "후방카메라", "크루즈 컨트롤", "메모리시트"]).issubset(set(listing["required_options_matched"]))


def test_fetch_all_sources_isolates_source_failures(monkeypatch):
    import used_car_sources as src

    def fake_fetch(source, limit=24):
        if source == "usedcar_destroyer":
            return [{"source": source, "listing_id": "1"}], None
        return [], "boom"

    monkeypatch.setattr(src, "fetch_source_listings", fake_fetch)
    listings, errors = fetch_all_sources(["usedcar_destroyer", "encar_certified", "kcar"], limit_per_source=1)
    assert len(listings) == 1
    assert errors == {"엔카 인증매물": "boom", "K Car": "boom"}
