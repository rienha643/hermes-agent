from __future__ import annotations

import json
import re
from html import unescape
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
    "hyundai_certified": {
        "list_url": "https://certified.hyundai.com/p/search/vehicle/list",
        "detail_url": "https://certified.hyundai.com/p/goods/goodsDetail.do?goodsNo={goods_no}",
    },
    "kia_certified": {
        "search_api": "https://cpo.kia.com/api/search/",
        "detail_api": "https://cpo.kia.com/api/product/detail/{id}/",
        "options_api": "https://cpo.kia.com/api/product/options/{id}/",
        "insurance_api": "https://cpo.kia.com/api/product/insurance-history/{id}/",
    },
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
        if source == "hyundai_certified":
            return _fetch_hyundai_certified(limit), None
        if source == "kia_certified":
            return _fetch_kia_certified(limit), None
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


def _fetch_hyundai_certified(limit: int) -> list[dict]:
    cfg = SOURCE_CONFIG["hyundai_certified"]
    params = {
        "ntcSeq": "",
        "type": "PLP",
        "pageIdx": 1,
        "rowsPerPage": limit,
        "startNo": 0,
        "listCnt": limit,
        "sortType": "popularity",
        "srchType": "srchWord",
        "searchWord": "",
        "sdStatCd": "",
        "selectedCodeList": "",
        "lowPrice": "",
        "highPrice": "",
        "lowMileage": "",
        "highMileage": "",
        "lowModelYear": "",
        "highModelYear": "",
        "carGubunList": "",
        "fuelList": "",
        "exteriorColorList": "",
        "optionList": "",
        "keywordList": "",
        "filtered": "false",
        "posNoList": "",
    }
    response = requests.post(
        cfg["list_url"],
        data=params,
        headers={**_json_headers(), "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8", "X-Requested-With": "XMLHttpRequest"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    cards = extract_hyundai_cards(response.text)[:limit]
    listings: list[dict] = []
    for card in cards:
        goods_no = extract_hyundai_goods_no(card)
        detail_url = cfg["detail_url"].format(goods_no=goods_no)
        try:
            detail_html = fetch_html(detail_url, allow_jina_fallback=False)
        except Exception as exc:
            listings.append(build_failed_listing("hyundai_certified", detail_url, str(exc)))
            continue
        listings.append(parse_hyundai_certified_listing(card, detail_url, detail_html))
    return listings


def _fetch_kia_certified(limit: int) -> list[dict]:
    cfg = SOURCE_CONFIG["kia_certified"]
    response = requests.get(
        cfg["search_api"],
        params={"size": limit, "sort": "DISPLAYED_AT_DESC"},
        headers=_json_headers(),
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    items = (data.get("results") or data.get("content") or [])[:limit]
    listings: list[dict] = []
    for item in items:
        item_id = item.get("id")
        detail_url = build_kia_detail_url(item_id)
        try:
            detail_response = requests.get(cfg["detail_api"].format(id=item_id), headers=_json_headers(), timeout=TIMEOUT)
            detail_response.raise_for_status()
            options_response = requests.get(cfg["options_api"].format(id=item_id), headers=_json_headers(), timeout=TIMEOUT)
            options_response.raise_for_status()
            insurance_response = requests.get(cfg["insurance_api"].format(id=item_id), headers=_json_headers(), timeout=TIMEOUT)
            insurance_response.raise_for_status()
        except Exception as exc:
            listings.append(build_failed_listing("kia_certified", detail_url, str(exc)))
            continue
        listings.append(parse_kia_certified_listing(item, detail_url, detail_response.json(), options_response.json(), insurance_response.json()))
    return listings


def build_kcar_detail_url(car_cd: str | None) -> str:
    return f"https://www.kcar.com/bc/detail/carInfoDtl?i_sCarCd={car_cd}"


def build_kia_detail_url(item_id: str | int | None) -> str:
    return f"https://cpo.kia.com/product/detail/{item_id}/"


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


def parse_hyundai_certified_listing(card_html: str, detail_url: str, detail_html: str) -> dict:
    text = html_to_text(detail_html)
    listing_id = extract_hyundai_goods_no(card_html)
    title = extract_hyundai_card_title(card_html)
    brand = infer_hyundai_brand(title)
    drive_spans = extract_hyundai_drive_spans(card_html)
    year, month = parse_year_month(expand_short_korean_year(drive_spans[0] if drive_spans else ""))
    mileage_km = parse_mileage_km(drive_spans[1] if len(drive_spans) > 1 else None)
    region = drive_spans[3] if len(drive_spans) > 3 else infer_region(text)
    price_krw = parse_price_krw(extract_hyundai_card_price(card_html))
    option_tokens = extract_option_tokens(text)
    option_data_present = detect_option_data_present(text, option_tokens)
    return {
        "source": "hyundai_certified",
        "listing_id": listing_id,
        "listing_url": detail_url,
        "detail_url": detail_url,
        "title": title,
        "brand": brand,
        "model": extract_model(title, brand),
        "body_type": infer_body_type(title),
        "year": year,
        "month": month,
        "mileage_km": mileage_km,
        "fuel": normalize_fuel(title + " " + text),
        "transmission": normalize_transmission(text),
        "price_krw": price_krw,
        "drivetrain": normalize_drivetrain(title + " " + text),
        "certified_flag": True,
        "warranty_flag": True,
        "warranty_label": "현대/제네시스 인증중고차",
        "seller_name": "현대/제네시스 인증중고차",
        "seller_type": "제조사 인증",
        "region": region,
        "accident_note": infer_accident_note(text),
        "flood_note": infer_flood_note(text),
        "option_tokens": option_tokens,
        "option_data_present": option_data_present,
        "required_options_matched": intersect_options(option_tokens, REQUIRED_OPTION_ALIASES),
        "highlight_options_matched": intersect_options(option_tokens, HIGHLIGHT_OPTION_ALIASES),
        "raw_labels": extract_raw_labels(title + " " + text) + ["제조사 인증"],
        "raw_text_excerpt": text[:500],
        "collected_at": now_iso(),
        "sale_status": infer_sale_status(text) or "판매중",
    }


def parse_kia_certified_listing(item: dict, detail_url: str, detail_json: dict, options_json: dict, insurance_json: dict) -> dict:
    car = detail_json.get("car", {}) or item
    listing_id = str(detail_json.get("id") or item.get("id"))
    title = clean_text(car.get("modelName") or item.get("modelName"))
    option_text = " ".join(flatten_strings(options_json))
    option_tokens = extract_option_tokens(option_text)
    option_data_present = detect_option_data_present(option_text, option_tokens)
    year, month = parse_year_month(car.get("firstRegisteredOn") or item.get("firstRegisteredOn") or str(car.get("modelYear") or item.get("modelYear") or ""))
    accident_note = infer_kia_accident_note(insurance_json)
    flood_note = infer_kia_flood_note(insurance_json)
    return {
        "source": "kia_certified",
        "listing_id": listing_id,
        "listing_url": detail_url,
        "detail_url": detail_url,
        "title": f"[기아] {title}",
        "brand": "기아",
        "model": clean_text(car.get("modelCodeName") or item.get("modelCodeName")) or extract_model(title, "기아"),
        "body_type": clean_text(car.get("modelCategory") or item.get("modelCategory")) or infer_body_type(title),
        "year": year,
        "month": month,
        "mileage_km": _safe_int(car.get("drivingDistance") or item.get("drivingDistance")),
        "fuel": normalize_fuel(car.get("fuelType") or car.get("engine") or item.get("modelEngine") or title),
        "transmission": normalize_transmission(car.get("mission")),
        "price_krw": _safe_int(car.get("price") or item.get("price")),
        "drivetrain": normalize_drivetrain(title + " " + clean_text(car.get("engine") or "")),
        "certified_flag": True,
        "warranty_flag": True,
        "warranty_label": "기아 인증중고차",
        "seller_name": "기아 인증중고차",
        "seller_type": "제조사 인증",
        "region": None,
        "accident_note": accident_note,
        "flood_note": flood_note,
        "option_tokens": option_tokens,
        "option_data_present": option_data_present,
        "required_options_matched": intersect_options(option_tokens, REQUIRED_OPTION_ALIASES),
        "highlight_options_matched": intersect_options(option_tokens, HIGHLIGHT_OPTION_ALIASES),
        "raw_labels": extract_raw_labels(title + " " + option_text) + flatten_kia_keywords(item),
        "raw_text_excerpt": clean_text(json.dumps({"item": item, "insurance": insurance_json}, ensure_ascii=False))[:500],
        "collected_at": now_iso(),
        "sale_status": "판매중",
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
    if brand == '제네시스':
        for keyword in ['GV60', 'GV70', 'GV80']:
            if keyword.upper() in title.upper():
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
    if brand == '볼보':
        for keyword in ['XC40', 'XC60', 'XC90', 'EX30', 'EX40', 'EX90']:
            if keyword.upper() in title.upper():
                return keyword
    if brand == '테슬라':
        for keyword in ['Model Y', 'Model X', '모델Y', '모델X']:
            if keyword.upper() in title.upper():
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
    for region in ['서울', '경기', '수원', '인천', '부산', '대구', '광주', '대전', '울산', '청주', '양산']:
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


def extract_hyundai_cards(html: str) -> list[str]:
    parts = re.split(r'<li\s+class="type02">', html)
    cards = []
    for part in parts[1:]:
        card = '<li class="type02">' + part
        if extract_hyundai_goods_no(card):
            cards.append(card)
    return cards


def extract_hyundai_goods_no(card_html: str) -> str:
    m = re.search(r"goodsDeatil\(&#39;([^&]+)&#39;\)", card_html)
    if m:
        return clean_text(m.group(1))
    m = re.search(r"goodsNo=([A-Za-z0-9]+)", card_html)
    return clean_text(m.group(1)) if m else ""


def extract_hyundai_card_title(card_html: str) -> str:
    m = re.search(r'<div class="name">(.*?)</div>', card_html, flags=re.S)
    if m:
        return clean_text(unescape(html_to_text(m.group(1))))
    m = re.search(r'alt="([^"]+)"', card_html)
    return clean_text(unescape(m.group(1))) if m else "제목 미확인"


def extract_hyundai_drive_spans(card_html: str) -> list[str]:
    m = re.search(r'<div class="drive">(.*?)</div>', card_html, flags=re.S)
    if not m:
        return []
    return [clean_text(unescape(html_to_text(span))) for span in re.findall(r'<span>(.*?)</span>', m.group(1), flags=re.S)]


def extract_hyundai_card_price(card_html: str) -> str | None:
    m = re.search(r'<span class="txt pay">\s*<em>(.*?)</em>\s*<i>만원</i>', card_html, flags=re.S)
    if not m:
        return None
    return f"{clean_text(unescape(m.group(1)))}만원"


def expand_short_korean_year(value: str) -> str:
    m = re.search(r'(\d{2})년\s*(\d{1,2})월', value)
    if not m:
        return value
    year = int(m.group(1))
    prefix = 2000 if year < 80 else 1900
    return f"{prefix + year}.{int(m.group(2)):02d}"


def infer_hyundai_brand(title: str) -> str | None:
    if any(keyword in title.upper() for keyword in ["G70", "G80", "G90", "GV60", "GV70", "GV80", "제네시스"]):
        return "제네시스"
    if extract_model(title, "현대"):
        return "현대"
    return normalize_brand(title)


def flatten_strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for child in value.values():
            strings.extend(flatten_strings(child))
        return strings
    if isinstance(value, list):
        strings: list[str] = []
        for child in value:
            strings.extend(flatten_strings(child))
        return strings
    return [str(value)]


def flatten_kia_keywords(item: dict) -> list[str]:
    keywords = []
    for row in item.get("customKeywords") or []:
        keyword = clean_text(row.get("keyword") if isinstance(row, dict) else str(row))
        if keyword:
            keywords.append(keyword)
    return keywords


def infer_kia_accident_note(insurance_json: dict) -> str | None:
    own = clean_text(((insurance_json.get("myCarDamage") or {}).get("accident") or ""))
    opponent = clean_text(((insurance_json.get("opponentCarDamage") or {}).get("accident") or ""))
    if own.startswith("없음") and opponent.startswith("없음"):
        return "무사고"
    if own or opponent:
        return "보험이력 확인 필요"
    return None


def infer_kia_flood_note(insurance_json: dict) -> str | None:
    flood = clean_text(((insurance_json.get("specialAccidentHistory") or {}).get("floodInsuranceAccident") or ""))
    if flood == "없음":
        return "침수 없음"
    if flood:
        return "침수 관련 표기 확인 필요"
    return None
