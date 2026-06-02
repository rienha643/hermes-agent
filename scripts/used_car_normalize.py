from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable

ALLOWED_BRANDS = {"현대", "기아", "BMW", "벤츠", "포르쉐"}
ALLOWED_FUELS = {"가솔린", "디젤", "하이브리드"}
ALLOWED_SOURCES = {"usedcar_destroyer", "jungcar_tv", "encar_certified", "kcar"}

REQUIRED_OPTION_ALIASES = {
    "열선 시트": ["열선시트", "열선 시트", "시트열선", "앞좌석 열선", "뒷좌석 열선", "열선"],
    "후방카메라": ["후방카메라", "후방 카메라", "백카메라", "후방캠", "후카"],
    "크루즈 컨트롤": ["크루즈컨트롤", "크루즈 컨트롤", "크루즈", "스마트크루즈", "SCC", "어댑티브크루즈"],
    "메모리시트": ["메모리시트", "메모리 시트"],
}

HIGHLIGHT_OPTION_ALIASES = {
    "통풍시트": ["통풍시트", "통풍 시트"],
    "오토홀드": ["오토홀드", "AUTO HOLD", "AUTOHOLD"],
    "전동트렁크": ["전동트렁크", "전동 트렁크", "파워테일게이트", "파워 테일게이트", "파워트렁크", "파워 트렁크"],
    "원격스마트주차보조": ["원격스마트주차보조", "원격 스마트 주차 보조", "RSPA"],
    "베이지시트": ["베이지시트", "베이지 시트"],
    "HUD": ["HUD", "헤드업디스플레이", "헤드업 디스플레이", "헤드업"],
    "선루프": ["선루프", "썬루프", "파노라마선루프", "파노라마 선루프", "파노라마"],
    "어라운드뷰": ["어라운드뷰", "어라운드 뷰", "서라운드뷰", "서라운드 뷰", "360도 카메라"],
    "원격시동": ["원격시동", "원격 시동", "리모트 스타트"],
}

SUV_KEYWORDS = {
    "현대": ["팰리세이드", "싼타페", "투싼", "코나", "베뉴", "넥쏘"],
    "기아": ["쏘렌토", "스포티지", "셀토스", "모하비", "니로", "EV9"],
    "BMW": ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "XM", "IX"],
    "벤츠": ["GLA", "GLB", "GLC", "GLE", "GLS", "G클래스", "EQB", "EQC", "EQE SUV", "EQS SUV", "G바겐", "지바겐"],
    "포르쉐": ["카이엔", "마칸"],
}

BRAND_PATTERNS = [
    (re.compile(r"메르세데스\s*-?\s*벤츠|mercedes\s*-?\s*benz", re.I), "벤츠"),
    (re.compile(r"^benz$|벤츠", re.I), "벤츠"),
    (re.compile(r"^bmw$|BMW", re.I), "BMW"),
    (re.compile(r"포르쉐|porsche", re.I), "포르쉐"),
    (re.compile(r"현대|hyundai", re.I), "현대"),
    (re.compile(r"기아|kia", re.I), "기아"),
]


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_brand(value: str | None) -> str | None:
    if not value:
        return None
    value = clean_text(value)
    for pattern, brand in BRAND_PATTERNS:
        if pattern.search(value):
            return brand
    return value


def parse_year_month(value: str | None) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    value = clean_text(value)
    compact = re.sub(r"[^0-9]", "", value)
    if len(compact) >= 6:
        year = int(compact[:4])
        month = int(compact[4:6])
        if 1 <= month <= 12:
            return year, month
    m = re.search(r"(20\d{2}|19\d{2})(?:[./년\-\s]+(\d{1,2}))?", value)
    if not m:
        return None, None
    year = int(m.group(1))
    month = int(m.group(2)) if m.group(2) else None
    if month and not 1 <= month <= 12:
        month = None
    return year, month


def parse_mileage_km(value: str | None) -> int | None:
    if not value:
        return None
    digits = re.sub(r"[^0-9]", "", value)
    return int(digits) if digits else None


def parse_price_krw(value: str | None) -> int | None:
    if not value:
        return None
    digits = re.sub(r"[^0-9]", "", value)
    if not digits:
        return None
    return int(digits) * 10000


def normalize_fuel(value: str | None) -> str | None:
    if not value:
        return None
    value = clean_text(value)
    lowered = value.lower()
    if any(token in lowered for token in ["휘발유", "가솔린", "gasoline"]):
        return "가솔린"
    if any(token in lowered for token in ["경유", "디젤", "diesel"]):
        return "디젤"
    if "하이브리드" in value or "hybrid" in lowered:
        return "하이브리드"
    if "lpg" in lowered:
        return "LPG"
    if "전기" in value or "ev" == lowered:
        return "전기"
    return value


def normalize_transmission(value: str | None) -> str | None:
    if not value:
        return None
    value = clean_text(value)
    lowered = value.lower()
    if any(token in lowered for token in ["오토", "자동", "auto"]):
        return "오토"
    if any(token in lowered for token in ["수동", "manual"]):
        return "수동"
    if "dct" in lowered:
        return "DCT"
    if "cvt" in lowered:
        return "CVT"
    return value


def normalize_drivetrain(value: str | None) -> str | None:
    if not value:
        return None
    value = clean_text(value)
    upper = value.upper()
    if "4MATIC" in upper or "XDRIVE" in upper or re.search(r"\bAWD\b", upper):
        return "AWD"
    if re.search(r"\b4WD\b", upper) or "사륜" in value:
        return "4WD"
    if re.search(r"\bFWD\b", upper) or "2WD" in upper or "전륜" in value:
        return "FWD"
    if re.search(r"\bRWD\b", upper) or "후륜" in value:
        return "RWD"
    return None


def infer_body_type(title: str | None) -> str | None:
    if not title:
        return None
    title = clean_text(title)
    brand = normalize_brand(title)
    for known_brand, keywords in SUV_KEYWORDS.items():
        if brand == known_brand or any(k in title.upper() for k in [kw.upper() for kw in keywords]):
            if any(kw.upper() in title.upper() for kw in keywords):
                return "SUV"
    sedan_keywords = ["아반떼", "K5", "K8", "그랜저", "S클래스", "E클래스", "5시리즈", "7시리즈", "파나메라", "세단"]
    if any(keyword.upper() in title.upper() for keyword in sedan_keywords):
        return "세단"
    if any(token in title.upper() for token in ["SUV", "CUV"]):
        return "SUV"
    return None


def extract_option_tokens(text: str | None) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    found: list[str] = []
    for canonical, aliases in {**REQUIRED_OPTION_ALIASES, **HIGHLIGHT_OPTION_ALIASES}.items():
        if any(alias.lower() in text.lower() for alias in aliases):
            found.append(canonical)
    return sorted(set(found))


def detect_option_data_present(text: str | None, option_tokens: Iterable[str] | None = None) -> bool:
    text = clean_text(text)
    tokens = list(option_tokens or [])
    if tokens:
        return True
    if not text:
        return False
    if "차량옵션" in text or "옵션 및 상세설명" in text:
        broad_hints = [
            "시트",
            "카메라",
            "크루즈",
            "HUD",
            "선루프",
            "파노라마",
            "오토홀드",
            "트렁크",
            "어라운드",
            "원격시동",
        ]
        return any(hint.lower() in text.lower() for hint in broad_hints)
    return False


def intersect_options(option_tokens: Iterable[str], candidates: dict[str, list[str]]) -> list[str]:
    tokens = set(option_tokens)
    return [name for name in candidates if name in tokens]


def normalize_source_label(source: str) -> str:
    return {
        "usedcar_destroyer": "중고차파괴자",
        "jungcar_tv": "중카TV",
        "encar_certified": "엔카 인증매물",
        "kcar": "K Car",
    }.get(source, source)
