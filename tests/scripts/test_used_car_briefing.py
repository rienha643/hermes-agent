import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from used_car_briefing import build_brief, dedupe_listings, render_markdown


def _listing(**overrides):
    item = {
        "source": "jungcar_tv",
        "listing_id": "1",
        "detail_url": "https://example.com/1",
        "title": "[현대] 팰리세이드",
        "brand": "현대",
        "model": "팰리세이드",
        "body_type": "SUV",
        "year": 2020,
        "month": 1,
        "mileage_km": 50000,
        "fuel": "가솔린",
        "price_krw": 25000000,
        "drivetrain": "AWD",
        "certified_flag": True,
        "warranty_flag": True,
        "warranty_label": "1년 2만km",
        "required_options_matched": ["열선 시트", "후방카메라"],
        "highlight_options_matched": ["HUD"],
        "option_data_present": True,
        "accident_note": "무사고",
        "flood_note": "침수 없음",
        "score": 72,
        "score_tier": "우선 추천",
    }
    item.update(overrides)
    return item


def test_dedupe_listings_keeps_higher_scored_duplicate():
    listings = [
        _listing(listing_id="1", title="현대 팰리세이드 AWD", score=60),
        _listing(listing_id="2", title="현대 팰리세이드 AWD", score=75),
    ]
    deduped = dedupe_listings(listings)
    assert len(deduped) == 1
    assert deduped[0]["listing_id"] == "2"


def test_render_markdown_success_includes_stats_and_vehicle_lines():
    markdown = render_markdown(
        listings=[_listing()],
        collected_count=10,
        filtered_count=1,
        exclusion_summary={"fuel": 2, "body_type": 3},
        recommendation_exclusion_summary={},
        source_errors={},
    )
    assert "[중고차 알림] 조건 일치 1건" in markdown
    assert "25,000,000원" in markdown
    assert "AWD" in markdown
    assert "요약: 수집 10건 / 하드필터 통과 1건 / 추천 기준 통과 1건" in markdown
    assert "[[현대]" not in markdown
    assert "[팰리세이드](https://example.com/1)" in markdown


def test_render_markdown_distinguishes_no_matches_from_fetch_failures():
    no_matches = render_markdown(
        listings=[],
        collected_count=8,
        filtered_count=0,
        exclusion_summary={"body_type": 8},
        recommendation_exclusion_summary={"score_below_threshold": 3},
        source_errors={},
    )
    assert "조건에 맞는 매물이 없습니다" in no_matches
    assert "하드필터 통과 0건" in no_matches

    recommendation_shortfall = render_markdown(
        listings=[],
        collected_count=8,
        filtered_count=3,
        exclusion_summary={"body_type": 2},
        recommendation_exclusion_summary={"score_below_threshold": 3},
        source_errors={},
    )
    assert "조건에 맞는 추천 매물이 없습니다" in recommendation_shortfall
    assert "추천점수 미달 3" in recommendation_shortfall

    failures = render_markdown(
        listings=[],
        collected_count=0,
        filtered_count=0,
        exclusion_summary={},
        recommendation_exclusion_summary={},
        source_errors={"중카TV": "403"},
    )
    assert "수집 실패" in failures
    assert "중카TV: 403" in failures

    detail_failures = render_markdown(
        listings=[],
        collected_count=0,
        filtered_count=0,
        exclusion_summary={},
        recommendation_exclusion_summary={},
        source_errors={},
        collection_failed=True,
        fetch_failure_count=4,
    )
    assert "수집 실패" in detail_failures
    assert "상세 수집 실패 4건" in detail_failures


def test_build_brief_reports_collection_failure_when_all_fetches_fail(monkeypatch):
    import used_car_briefing as briefing

    def fake_fetch_all_sources(limit_per_source=24):
        return [
            {
                "source": "jungcar_tv",
                "listing_id": "f1",
                "fetch_error": "timeout",
            }
        ], {}

    monkeypatch.setattr(briefing, "fetch_all_sources", fake_fetch_all_sources)
    markdown, stats = build_brief(limit_per_source=8)
    assert "[중고차 알림] 수집 실패" in markdown
    assert "상세 수집 실패 1건" in markdown
    assert stats["collection_failed"] is True
    assert stats["fetch_failures"] == 1


def test_build_brief_reports_no_matches_when_hard_filters_remove_all(monkeypatch):
    import used_car_briefing as briefing

    def fake_fetch_all_sources(limit_per_source=24):
        return [
            {
                "source": "jungcar_tv",
                "listing_id": "1",
                "detail_url": "https://example.com/1",
                "title": "[현대] 그랜저",
                "brand": "현대",
                "model": "그랜저",
                "body_type": "세단",
                "year": 2020,
                "mileage_km": 50000,
                "fuel": "가솔린",
                "price_krw": 25000000,
                "score": 80,
                "option_data_present": True,
                "required_options_matched": ["열선 시트", "후방카메라"],
            }
        ], {}

    monkeypatch.setattr(briefing, "fetch_all_sources", fake_fetch_all_sources)
    markdown, stats = build_brief(limit_per_source=8)
    assert "[중고차 알림] 조건에 맞는 매물이 없습니다." in markdown
    assert "하드필터 통과 0건" in markdown
    assert stats["filtered"] == 0


def test_build_brief_reports_recommendation_shortfall_when_scores_fail(monkeypatch):
    import used_car_briefing as briefing

    def fake_fetch_all_sources(limit_per_source=24):
        return [
            {
                "source": "jungcar_tv",
                "listing_id": "1",
                "detail_url": "https://example.com/1",
                "title": "[현대] 팰리세이드",
                "brand": "현대",
                "model": "팰리세이드",
                "body_type": "SUV",
                "year": 2019,
                "mileage_km": 140000,
                "fuel": "가솔린",
                "price_krw": 25000000,
                "drivetrain": "AWD",
                "certified_flag": False,
                "warranty_flag": False,
                "option_data_present": True,
                "required_options_matched": ["열선 시트", "후방카메라"],
                "highlight_options_matched": [],
                "accident_note": None,
                "flood_note": None,
                "sale_status": "판매중",
            }
        ], {}

    monkeypatch.setattr(briefing, "fetch_all_sources", fake_fetch_all_sources)
    markdown, stats = build_brief(limit_per_source=8)
    assert "[중고차 알림] 조건에 맞는 추천 매물이 없습니다." in markdown
    assert "추천 기준 통과 0건" in markdown
    assert stats["filtered"] == 1
    assert stats["recommended"] == 0
