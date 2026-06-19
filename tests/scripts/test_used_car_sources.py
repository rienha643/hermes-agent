import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from used_car_sources import (
    build_kcar_detail_url,
    build_kia_detail_url,
    parse_encar_listing,
    parse_hyundai_certified_listing,
    parse_kia_certified_listing,
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

HYUNDAI_CARD = """
<li class="type02">
  <a href="javascript:common.link.goodsDeatil(&#39;HSX260202024021&#39;);">
    <div class="unit_info">
      <div class="name">2024 코나(SX2) 가솔린 1.6 터보 2WD 인스퍼레이션</div>
      <div class="drive">
        <span>24년 01월</span>
        <span>13,866km</span>
        <span>336소2143</span>
        <span>양산</span>
      </div>
      <div class="price"><span class="txt pay"><em>2,630</em><i>만원</i></span></div>
    </div>
  </a>
</li>
"""

HYUNDAI_DETAIL_HTML = """
<html><body>
상품기본정보 옵션 열선시트 후방카메라 스마트크루즈 통풍시트 오토홀드
무사고 침수 없음 자동변속기 보증
</body></html>
"""

KIA_ITEM = {
    "id": 11775,
    "price": 20970000,
    "drivingDistance": 115268,
    "firstRegisteredOn": "2022-03-22",
    "modelName": "셀토스 1.6 가솔린 시그니처 2WD",
    "modelCodeName": "셀토스",
    "modelEngine": "1.6 가솔린 터보",
    "modelYear": 2023,
    "modelCategory": "SUV",
    "customKeywords": [{"keyword": "보험이력없음"}],
}

KIA_DETAIL = {
    "id": 11775,
    "car": {
        "price": 20970000,
        "modelName": "셀토스 1.6 가솔린 시그니처 2WD",
        "modelCodeName": "셀토스",
        "modelCategory": "SUV",
        "firstRegisteredOn": "2022-03-22",
        "drivingDistance": 115268,
        "engine": "1.6 가솔린 터보",
        "fuelType": "GASOLINE",
        "mission": "A/T",
    },
}

KIA_OPTIONS = {
    "seat": ["1열 열선시트", "1열 통풍시트", "운전석 파워시트"],
    "comport": ["크루즈 컨트롤", "오토홀드", "버튼시동 스마트키 시스템(원격시동 포함)"],
    "multimedia": ["후방 모니터"],
}

KIA_INSURANCE = {
    "myCarDamage": {"accident": "없음, 0 원"},
    "opponentCarDamage": {"accident": "없음, 0 원"},
    "specialAccidentHistory": {"floodInsuranceAccident": "없음"},
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
    listing = parse_kcar_listing(KCAR_ITEM, "https://www.kcar.com/bc/detail/carInfoDtl?carCd=EC61358223", KCAR_DETAIL)
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


def test_parse_hyundai_certified_listing_extracts_card_and_detail_fields():
    listing = parse_hyundai_certified_listing(
        HYUNDAI_CARD,
        "https://certified.hyundai.com/p/goods/goodsDetail.do?goodsNo=HSX260202024021",
        HYUNDAI_DETAIL_HTML,
    )
    assert listing["source"] == "hyundai_certified"
    assert listing["listing_id"] == "HSX260202024021"
    assert listing["brand"] == "현대"
    assert listing["model"] == "코나"
    assert listing["body_type"] == "SUV"
    assert listing["year"] == 2024
    assert listing["month"] == 1
    assert listing["mileage_km"] == 13866
    assert listing["price_krw"] == 26300000
    assert listing["drivetrain"] == "FWD"
    assert listing["certified_flag"] is True
    assert listing["warranty_flag"] is True
    assert listing["accident_note"] == "무사고"
    assert listing["flood_note"] == "침수 없음"
    assert {"열선 시트", "후방카메라", "크루즈 컨트롤"}.issubset(set(listing["required_options_matched"]))


def test_parse_kia_certified_listing_extracts_api_detail_options_and_insurance():
    listing = parse_kia_certified_listing(
        KIA_ITEM,
        "https://cpo.kia.com/product/detail/11775/",
        KIA_DETAIL,
        KIA_OPTIONS,
        KIA_INSURANCE,
    )
    assert listing["source"] == "kia_certified"
    assert listing["listing_id"] == "11775"
    assert listing["brand"] == "기아"
    assert listing["model"] == "셀토스"
    assert listing["body_type"] == "SUV"
    assert listing["year"] == 2022
    assert listing["month"] == 3
    assert listing["mileage_km"] == 115268
    assert listing["fuel"] == "가솔린"
    assert listing["price_krw"] == 20970000
    assert listing["drivetrain"] == "FWD"
    assert listing["certified_flag"] is True
    assert listing["warranty_flag"] is True
    assert listing["accident_note"] == "무사고"
    assert listing["flood_note"] == "침수 없음"
    assert {"열선 시트", "크루즈 컨트롤"}.issubset(set(listing["required_options_matched"]))
    assert "통풍시트" in listing["highlight_options_matched"]


def test_build_kcar_detail_url_uses_browser_detail_route():
    assert build_kcar_detail_url("EC61353662") == "https://www.kcar.com/bc/detail/carInfoDtl?i_sCarCd=EC61353662"
    assert build_kcar_detail_url("EC61358223") == "https://www.kcar.com/bc/detail/carInfoDtl?i_sCarCd=EC61358223"
    assert build_kcar_detail_url("EC61356147") == "https://www.kcar.com/bc/detail/carInfoDtl?i_sCarCd=EC61356147"


def test_build_kia_detail_url_uses_browser_detail_route():
    assert build_kia_detail_url(11775) == "https://cpo.kia.com/product/detail/11775/"


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
