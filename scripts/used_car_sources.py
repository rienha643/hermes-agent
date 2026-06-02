from __future__ import annotations

import json
import re
from urllib.parse import quote, urljoin

import requests

from used_car_normalize import (
    HIGHLIGHT_OPTION_ALIASES,
    REQUIRED_OPTION_ALIASES,
    clean_text,
    detect_option_data_present,
    extract_option_tokens,
    infer_body_type,
    intersect_options,
    normalize_brand,
    normalize_drivetrain,
    normalize_fuel,
    normalize_source_label,
    normalize_transmission,
    now_iso,
    parse_mileage_km,
    parse_price_krw,
    parse_year_month,
)

USER_AGENT = "Mozilla/5.0"
TIMEOUT = 30
ENCAR_QUERY = "(And.Hidden.N._.(Or.ServiceMark.EncarDiagnosisP0._.ServiceMark.EncarDiagnosisP1._.ServiceMark.EncarDiagnosisP2.))"
SOURCE_CONFIG = {
    "usedcar_destroyer": {
        "base_url": "https://xn--299a7fv36e6lbb3goqn.com",
        "list_url": "https://xn--299a7fv36e6lbb3goqn.com/search/get_search?fn=model&country=all",
        "allow_jina_fallback": False,
    },
    "jungcar_tv": {
        "base_url": "https://xn--tv-9z9j31p.com",
        "list_url": "https://xn--tv-9z9j31p.com/search/get_search?fn=model&country=all",
        "allow_jina_fallback": True,
    },
    "encar_certified": {
        "api_url": "https://api.encar.com/search/car/list/premium",
    },
    "kcar": {
        "list_api": "https://api.kcar.com/bc/stockCar/list",
        "detail_api": "https://api.kcar.com/bc/car-info-detail-of-ng",
    },
}


def fetch_source_listings(source: str, limit: int = 24) -> tuple[list[dict], str | None]:
    if source not in SOURCE_CONFIG:
        return [], f"지원하지 않는 소스: {source}"
    try:
        if source in {"usedcar_destroyer", "jungcar_tv"}:
            return _fetch_site_html_source(source, limit), None
        if source == "encar_certified":
            return _fetch_encar_certified(limit), None
        if source == "kcar":
            return _fetch_kcar(limit), None
        return [], f"지원하지 않는 소스: {source}"
    except Exception as exc:
        return [], str(exc)


def fetch_all_sources(sources: list[str] | None = None, limit_per_source: int = 24) -> tuple[list[dict], dict[str, str]]:
    sources = sources or list(SOURCE_CONFIG.keys())
    all_listings: list[dict] = []
    errors: dict[str, str] = {}
    for source in sources:
        listings, error = fetch_source_listings(source, limit=limit_per_source)
        all_listings.extend(listings)
        if error:
            errors[normalize_source_label(source)] = error
    return all_listings, errors


def _fetch_site_html_source(source: str, limit: int) -> list[dict]:
    cfg = SOURCE_CONFIG[source]
    list_html = fetch_html(cfg["list_url"], allow_jina_fallback=False)
    detail_urls = extract_detail_urls(list_html, cfg["base_url"])[:limit]
    listings: list[dict] = []
    for detail_url in detail_urls:
        try:
            detail_html = fetch_html(detail_url, allow_jina_fallback=cfg["allow_jina_fallback"])
        except Exception as exc:
            listings.append(build_failed_listing(source, detail_url, str(exc)))
            continue
        listings.append(parse_detail_page(source, detail_url, detail_html))
    return listings


def _fetch_encar_certified(limit: int) -> list[dict]:
    cfg = SOURCE_CONFIG["encar_certified"]
    params = {
        "count": "true",
        "q": ENCAR_QUERY,
        "sr": "|ModifiedDate|0|20",
        "inav": "|Metadata|Sort",
        "cursor": "",
    }
    response = requests.get(cfg["api_url"], params=params, headers=_json_headers(), timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    results = data.get("SearchResults", [])[:limit]
    listings: list[dict] = []
    for item in results:
        listing_id = str(item.get("Id"))
        detail_url = f"https://fem.encar.com/cars/detail/{listing_id}"
        try:
            detail_html = fetch_html(detail_url, allow_jina_fallback=False)
        except Exception as exc:
            listings.append(build_failed_listing("encar_certified", detail_url, str(exc)))
            continue
        listings.append(parse_encar_listing(item, detail_url, detail_html))
    return listings


def _fetch_kcar(limit: int) -> list[dict]:
    cfg = SOURCE_CONFIG["kcar"]
    response = requests.get(
        cfg["list_api"],
        params={"page": 1, "perPage": limit},
        headers=_json_headers(),
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    items = response.json().get("data", {}).get("list", [])[:limit]
    listings: list[dict] = []
    for item in items:
        car_cd = item.get("carCd")
        detail_url = build_kcar_detail_url(car_cd)
        try:
            detail_response = requests.get(
                cfg["detail_api"],
                params={"i_sCarCd": car_cd},
                headers=_json_headers(),
                timeout=TIMEOUT,
            )
            detail_response.raise_for_status()
            detail_json = detail_response.json().get("data", {})
        except Exception as exc:
            listings.append(build_failed_listing("kcar", detail_url, str(exc)))
            continue
        listings.append(parse_kcar_listing(item, detail_url, detail_json))
    return listings


def build_kcar_detail_url(car_cd: str | None) -> str:
    return f"https://www.kcar.com/bc/detail/carInfoDtl?i_sCarCd={car_cd}"


def fetch_html(url: str, allow_jina_fallback: bool = False) -> str:
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        response.raise_for_status()
        return response.text
    except Exception:
        if allow_jina_fallback:
            jina_url = build_jina_url(url)
            response = requests.get(jina_url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text
        raise


def build_jina_url(url: str) -> str:
    if url.startswith("http://"):
        return f"https://r.jina.ai/http://{url[len('http://'):] }"
    if url.startswith("https://"):
        return f"https://r.jina.ai/http://{url[len('https://'):] }"
    return f"https://r.jina.ai/http://{url}"


def _json_headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def extract_detail_urls(html: str, base_url: str) -> list[str]:
    urls = []
    for match in re.findall(r'/search/detail/(\d+)', html):
        url = urljoin(base_url, f'/search/detail/{match}')
        if url not in urls:
            urls.append(url)
    return urls


def parse_detail_page(source: str, detail_url: str, html: str) -> dict:
    text = html_to_text(html)
    listing_id = detail_url.rstrip('/').split('/')[-1]
    title = extract_title(html, text)
    brand = extract_brand(title)
    year, month = parse_year_month(extract_field(text, '연식', '최초등록일'))
    mileage_km = parse_mileage_km(extract_field(text, '주행거리', '차량번호'))
    fuel = normalize_fuel(extract_field(text, '연료', '변속기'))
    transmission = normalize_transmission(extract_field(text, '변속기', '색상'))
    price_krw = parse_price_krw(extract_field(text, '판매가', '사고이력조회'))
    drivetrain = normalize_drivetrain(title + ' ' + text)
    option_tokens = extract_option_tokens(text)
    option_data_present = detect_option_data_present(text, option_tokens)
    raw_labels = extract_raw_labels(text)
    accident_note = infer_accident_note(text)
    flood_note = infer_flood_note(text)
    warranty_flag, warranty_label = infer_warranty(text, source)
    seller_name = extract_field(text, '이름', '연락처')
    region = infer_region(text)
    model = extract_model(title, brand)
    return {
        'source': source,
        'listing_id': listing_id,
        'listing_url': detail_url,
        'detail_url': detail_url,
        'title': title,
        'brand': brand,
        'model': model,
        'body_type': infer_body_type(title),
        'year': year,
        'month': month,
        'mileage_km': mileage_km,
        'fuel': fuel,
        'transmission': transmission,
        'price_krw': price_krw,
        'drivetrain': drivetrain,
        'certified_flag': False,
        'warranty_flag': warranty_flag,
        'warranty_label': warranty_label,
        'seller_name': seller_name,
        'seller_type': '상사' if '상사' in text else '플랫폼',
        'region': region,
        'accident_note': accident_note,
        'flood_note': flood_note,
        'option_tokens': option_tokens,
        'option_data_present': option_data_present,
        'required_options_matched': intersect_options(option_tokens, REQUIRED_OPTION_ALIASES),
        'highlight_options_matched': intersect_options(option_tokens, HIGHLIGHT_OPTION_ALIASES),
        'raw_labels': raw_labels,
        'raw_text_excerpt': text[:500],
        'collected_at': now_iso(),
        'sale_status': infer_sale_status(text),
    }


def parse_encar_listing(item: dict, detail_url: str, detail_html: str) -> dict:
    text = html_to_text(detail_html)
    listing_id = str(item.get('Id'))
    brand = normalize_brand(item.get('Manufacturer'))
    model = clean_text(item.get('Model')) or None
    badge = clean_text(item.get('Badge'))
    title = f'[{brand}] {model} {badge}'.strip()
    year, month = parse_year_month(_format_encar_year(item.get('Year')))
    mileage_km = _safe_int(item.get('Mileage'))
    fuel = normalize_fuel(item.get('FuelType'))
    transmission = normalize_transmission(item.get('Transmission'))
    price_krw = _parse_encar_price(item.get('Price'))
    option_tokens = extract_option_tokens(text)
    option_data_present = detect_option_data_present(text, option_tokens)
    raw_labels = list(item.get('Trust') or []) + list(item.get('ServiceMark') or [])
    body_type = infer_body_type(title)
    return {
        'source': 'encar_certified',
        'listing_id': listing_id,
        'listing_url': detail_url,
        'detail_url': detail_url,
        'title': title,
        'brand': brand,
        'model': model,
        'body_type': body_type,
        'year': year,
        'month': month,
        'mileage_km': mileage_km,
        'fuel': fuel,
        'transmission': transmission,
        'price_krw': price_krw,
        'drivetrain': normalize_drivetrain(title + ' ' + text),
        'certified_flag': any(mark.startswith('EncarDiagnosis') for mark in (item.get('ServiceMark') or [])),
        'warranty_flag': 'ExtendWarranty' in (item.get('Trust') or []),
        'warranty_label': '엔카 보증/진단' if 'ExtendWarranty' in (item.get('Trust') or []) else ('엔카 진단' if any(mark.startswith('EncarDiagnosis') for mark in (item.get('ServiceMark') or [])) else None),
        'seller_name': clean_text(item.get('DealerName')) or None,
        'seller_type': '플랫폼',
        'region': clean_text(item.get('OfficeCityState')) or None,
        'accident_note': infer_accident_note(text) or ('무사고' if '프레임 무사고' in text else None),
        'flood_note': infer_flood_note(text),
        'option_tokens': option_tokens,
        'option_data_present': option_data_present,
        'required_options_matched': intersect_options(option_tokens, REQUIRED_OPTION_ALIASES),
        'highlight_options_matched': intersect_options(option_tokens, HIGHLIGHT_OPTION_ALIASES),
        'raw_labels': raw_labels,
        'raw_text_excerpt': text[:500],
        'collected_at': now_iso(),
        'sale_status': '판매중',
    }


def parse_kcar_listing(item: dict, detail_url: str, detail_json: dict) -> dict:
    master = detail_json.get('master', {}) or {}
    carhistory = detail_json.get('carhistory', {}) or {}
    warranty_info = detail_json.get('warrantyInfo', []) or []
    main_opt_list = detail_json.get('mainOptList', []) or []
    opt_list = detail_json.get('optList', []) or []
    title = build_kcar_title(item)
    brand = normalize_brand(item.get('mnuftrNm'))
    model = clean_text(item.get('modelNm')) or None
    option_tokens = extract_option_tokens(' '.join(_collect_kcar_option_texts(main_opt_list, opt_list)))
    option_data_present = bool(main_opt_list or opt_list)
    drivetrain = normalize_drivetrain(' '.join([title, str(master.get('whelNm', '')), str(item.get('grdDtlNm', ''))]))
    accident_note = infer_kcar_accident_note(carhistory, detail_json)
    flood_note = infer_kcar_flood_note(carhistory)
    year, month = parse_year_month(str(item.get('mfgDt') or item.get('prdcnYr') or ''))
    return {
        'source': 'kcar',
        'listing_id': str(item.get('carCd')),
        'listing_url': detail_url,
        'detail_url': detail_url,
        'title': title,
        'brand': brand,
        'model': model,
        'body_type': infer_kcar_body_type(item, carhistory, title),
        'year': year,
        'month': month,
        'mileage_km': parse_mileage_km(str(item.get('milg') or master.get('prstMilg') or '')),
        'fuel': normalize_fuel(item.get('fuelNm') or carhistory.get('fuel')),
        'transmission': normalize_transmission(item.get('trnsmsnNm')),
        'price_krw': parse_price_krw(f"{item.get('dcPrc') or item.get('prc')} 만원"),
        'drivetrain': drivetrain,
        'certified_flag': clean_text(item.get('cntrNm') or '').endswith('직영점') or item.get('gmCertYn') == 'Y',
        'warranty_flag': any(info.get('available') for info in warranty_info),
        'warranty_label': _extract_kcar_warranty_label(warranty_info),
        'seller_name': clean_text(item.get('selerNm') or item.get('cntrNm')) or None,
        'seller_type': '직영' if clean_text(item.get('cntrNm') or '').endswith('직영점') else '플랫폼',
        'region': infer_region(clean_text(item.get('cntrNm') or '')),
        'accident_note': accident_note,
        'flood_note': flood_note,
        'option_tokens': option_tokens,
        'option_data_present': option_data_present,
        'required_options_matched': intersect_options(option_tokens, REQUIRED_OPTION_ALIASES),
        'highlight_options_matched': intersect_options(option_tokens, HIGHLIGHT_OPTION_ALIASES),
        'raw_labels': _extract_kcar_raw_labels(item, master, warranty_info),
        'raw_text_excerpt': clean_text(json.dumps({'item': item, 'master': master, 'carhistory': carhistory}, ensure_ascii=False))[:500],
        'collected_at': now_iso(),
        'sale_status': '판매중' if item.get('statCd') in {'CAR_STATUS010', 'CAR_STATUS_010', '00'} else '판매종료',
    }


def build_failed_listing(source: str, detail_url: str, reason: str) -> dict:
    listing_id = detail_url.rstrip('/').split('/')[-1]
    return {
        'source': source,
        'listing_id': listing_id,
        'listing_url': detail_url,
        'detail_url': detail_url,
        'title': f'수집 실패: {listing_id}',
        'brand': None,
        'model': None,
        'body_type': None,
        'year': None,
        'month': None,
        'mileage_km': None,
        'fuel': None,
        'transmission': None,
        'price_krw': None,
        'drivetrain': None,
        'certified_flag': False,
        'warranty_flag': False,
        'warranty_label': None,
        'seller_name': None,
        'seller_type': None,
        'region': None,
        'accident_note': None,
        'flood_note': None,
        'option_tokens': [],
        'option_data_present': False,
        'required_options_matched': [],
        'highlight_options_matched': [],
        'raw_labels': [],
        'raw_text_excerpt': reason,
        'collected_at': now_iso(),
        'sale_status': None,
        'fetch_error': reason,
    }


def html_to_text(html: str) -> str:
    text = re.sub(r'<script.*?</script>', ' ', html, flags=re.S | re.I)
    text = re.sub(r'<style.*?</style>', ' ', text, flags=re.S | re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    return clean_text(text)


def extract_title(html: str, text: str) -> str:
    title_match = re.search(r'<title>(.*?)</title>', html, flags=re.I | re.S)
    if title_match:
        title_value = clean_text(title_match.group(1))
        if title_value and title_value not in {'중카TV', '60만 유튜버 중고차파괴자 - 수원중고차매매사이트'}:
            return title_value
    m = re.search(r'(\[[^\]]+\]\s*.+?)\s+차량번호', text)
    if m:
        return clean_text(m.group(1))
    m = re.search(r'(\[[^\]]+\]\s*.+?)\s+판매가', text)
    if m:
        return clean_text(m.group(1))
    m = re.search(r'^\[(?P<brand>[^\]]+)\]\s*(?P<title>.+?)\s+if\(', text)
    if m:
        return clean_text(f"[{m.group('brand')}] {m.group('title')}")
    return clean_text(text[:120])


def extract_brand(title: str) -> str | None:
    m = re.match(r'^\[(?P<brand>[^\]]+)\]', title)
    if m:
        return normalize_brand(m.group('brand'))
    return normalize_brand(title)


def extract_model(title: str, brand: str | None) -> str | None:
    title = re.sub(r'^\[[^\]]+\]\s*', '', title)
    if not title:
        return None
    if brand == '현대':
        for keyword in ['팰리세이드', '싼타페', '투싼', '코나', '베뉴', '넥쏘']:
            if keyword in title:
                return keyword
    if brand == '기아':
        for keyword in ['쏘렌토', '스포티지', '셀토스', '모하비', '니로', 'EV9', '레이']:
            if keyword in title:
                return keyword
    if brand == 'BMW':
        for keyword in ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'XM', 'iX']:
            if keyword.upper() in title.upper():
                return keyword
    if brand == '벤츠':
        for keyword in ['GLA', 'GLB', 'GLC', 'GLE', 'GLS', 'G클래스', 'EQB', 'EQC', 'EQE SUV', 'EQS SUV', 'CLA']:
            if keyword.upper() in title.upper() or keyword in title:
                return keyword
    if brand == '포르쉐':
        for keyword in ['카이엔', '마칸']:
            if keyword in title:
                return keyword
    return clean_text(title.split()[0]) if title.split() else None


def extract_field(text: str, start_label: str, end_label: str) -> str | None:
    pattern = re.escape(start_label) + r'\s*(.*?)\s*' + re.escape(end_label)
    m = re.search(pattern, text)
    if m:
        return clean_text(m.group(1))
    return None


def infer_accident_note(text: str) -> str | None:
    if '프레임 무사고' in text or '무사고' in text:
        return '무사고'
    if '단순교환' in text:
        return '단순교환'
    if '사고이력조회' in text:
        return '사고이력 별도 확인 필요'
    return None


def infer_flood_note(text: str) -> str | None:
    if '침수 없음' in text or '침수이력 없음' in text:
        return '침수 없음'
    if '침수' in text:
        return '침수 관련 표기 확인 필요'
    return None


def infer_warranty(text: str, source: str) -> tuple[bool, str | None]:
    if '보증' in text:
        if source in {'usedcar_destroyer', 'jungcar_tv'}:
            return True, '소스 보증 문구 있음'
    return False, None


def infer_region(text: str) -> str | None:
    for region in ['서울', '경기', '수원', '인천', '부산', '대구', '광주', '대전', '울산', '청주']:
        if region in text:
            return region
    return None


def infer_sale_status(text: str) -> str | None:
    if any(token in text for token in ['판매완료', '거래완료', '예약중']):
        return '판매종료'
    if any(token in text for token in ['판매가', '상담', '무료문자상담하기']):
        return '판매중'
    return None


def extract_raw_labels(text: str) -> list[str]:
    labels = []
    for token in ['보증', '무사고', '4WD', 'AWD', 'FWD', 'RWD', '하이브리드', '디젤', '가솔린']:
        if token in text:
            labels.append(token)
    return labels


def _safe_int(value) -> int | None:
    if value in (None, '', []):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _format_encar_year(value) -> str:
    if value is None:
        return ''
    raw = str(value)
    digits = re.sub(r'[^0-9]', '', raw)
    if len(digits) >= 6:
        return f'{digits[:4]}.{digits[4:6]}'
    return digits


def _parse_encar_price(value) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value) * 10000)
    except Exception:
        return None


def _collect_kcar_option_texts(main_opt_list: list[dict], opt_list: list[dict]) -> list[str]:
    texts = []
    for row in list(main_opt_list) + list(opt_list):
        texts.extend([
            clean_text(str(row.get('optioncdName') or '')),
            clean_text(str(row.get('optnNm') or '')),
            clean_text(str(row.get('grpParent') or '')),
            clean_text(str(row.get('grpChild') or '')),
            clean_text(str(row.get('optDesc') or '')),
        ])
    return [text for text in texts if text]


def _extract_kcar_warranty_label(warranty_info: list[dict]) -> str | None:
    available = [info for info in warranty_info if info.get('available')]
    if not available:
        return None
    first = available[0]
    return clean_text(str(first.get('gurntePrdInfo') or first.get('wrntyPrdNm') or '')) or 'K Car 보증 가능'


def _extract_kcar_raw_labels(item: dict, master: dict, warranty_info: list[dict]) -> list[str]:
    labels = []
    if clean_text(item.get('cntrNm') or '').endswith('직영점'):
        labels.append('직영')
    if any(info.get('available') for info in warranty_info):
        labels.append('보증')
    wheel_name = clean_text(master.get('whelNm') or '')
    if wheel_name:
        labels.append(wheel_name)
    return labels


def build_kcar_title(item: dict) -> str:
    brand = normalize_brand(item.get('mnuftrNm')) or clean_text(item.get('mnuftrNm'))
    model = clean_text(item.get('modelNm') or '')
    grade = clean_text(item.get('grdNm') or '')
    grade_detail = clean_text(item.get('grdDtlNm') or '')
    parts = [f'[{brand}]', model, grade, grade_detail]
    return clean_text(' '.join(part for part in parts if part))


def infer_kcar_body_type(item: dict, carhistory: dict, title: str) -> str | None:
    car_form = clean_text(carhistory.get('carForm') or '')
    if any(token in car_form for token in ['SUV', 'RV']):
        return 'SUV'
    return infer_body_type(title)


def infer_kcar_accident_note(carhistory: dict, detail_json: dict) -> str | None:
    own_damage = _safe_int(carhistory.get('owncarDmgeAcdtCnt')) or 0
    total_accident = _safe_int(carhistory.get('acdtCnt')) or 0
    if own_damage == 0 and total_accident == 0:
        return '무사고'
    acc_list = detail_json.get('carHistoryAccList') or []
    if acc_list:
        return '사고이력 있음'
    if total_accident > 0:
        return '사고이력 있음'
    return None


def infer_kcar_flood_note(carhistory: dict) -> str | None:
    flood_count = _safe_int(carhistory.get('fldgAcdtCnt'))
    if flood_count is None:
        return None
    if flood_count == 0:
        return '침수 없음'
    return '침수 이력 있음'
