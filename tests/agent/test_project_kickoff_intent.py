from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.conversation_loop import (
    _bootstrap_games_project_kickoff,
    _extract_project_name_from_kickoff_message,
    _has_project_kickoff_intent,
    run_conversation,
)


def test_positive_kickoff_intent_detects_game_start_request():
    message = "2D 서브컬쳐 턴제 RPG 게임 제작을 시작합니다"

    assert _has_project_kickoff_intent(message) is True


def test_positive_project_name_extraction_from_game_kickoff_phrase():
    message = "2D 서브컬쳐 턴제 RPG 게임 제작을 시작합니다"

    assert _extract_project_name_from_kickoff_message(message) == "2D 서브컬쳐 턴제 RPG"


def test_positive_project_name_extraction_from_marked_project_name_phrase():
    message = "프로젝트명 망각구역으로 게임 개발을 시작합니다"

    assert _extract_project_name_from_kickoff_message(message) == "망각구역"


def test_positive_project_name_extraction_from_explicit_project_name_sentence():
    message = "프로젝트 이름은 Project Alpha입니다. 새 게임 프로젝트를 생성해주세요"

    assert _extract_project_name_from_kickoff_message(message) == "Project Alpha"


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            '[Replying to: "서브컬쳐 게임 프로젝트"]\n\n프로젝트명 별빛전선으로 게임 개발을 시작합니다.',
            "별빛전선",
        ),
        (
            '[Replying to: "서브컬쳐 게임 프로젝트"]\n\n프로젝트 이름은 Project Alpha입니다. 새 게임 프로젝트를 생성해주세요.',
            "Project Alpha",
        ),
        (
            '프로젝트명 은하성역으로 게임 개발을 시작합니다',
            "은하성역",
        ),
        (
            '프로젝트명 이세계전선으로 게임 개발을 시작합니다',
            "이세계전선",
        ),
        (
            '프로젝트명 가온으로 게임 개발을 시작합니다',
            "가온",
        ),
        (
            '프로젝트명 망각구역으로 게임 개발을 시작합니다',
            "망각구역",
        ),
        (
            '[Replying to: "서브컬쳐 게임 프로젝트"]\n\n프로젝트명 은하성역으로 게임 개발을 시작합니다',
            "은하성역",
        ),
    ],
)
def test_explicit_project_name_markers_override_reply_context_quotes(message, expected):
    assert _extract_project_name_from_kickoff_message(message) == expected


def test_reply_context_without_explicit_project_name_keeps_existing_policy():
    message = '[Replying to: "서브컬쳐 게임 프로젝트"] 신규 게임 프로젝트를 시작합니다'

    assert _extract_project_name_from_kickoff_message(message) is None


def test_negative_kickoff_intent_rejects_document_workflow():
    message = "2D 서브컬쳐 턴제 RPG 기획서를 작성해 주세요"

    assert _has_project_kickoff_intent(message) is False
    assert _extract_project_name_from_kickoff_message(message) is None


def test_boundary_case_requires_explicit_project_name_even_with_game_start_intent():
    message = "신규 게임 프로젝트를 시작합니다"

    assert _has_project_kickoff_intent(message) is True
    assert _extract_project_name_from_kickoff_message(message) is None
    assert _bootstrap_games_project_kickoff(message) is None


def test_bootstrap_is_blocked_when_project_name_is_missing(monkeypatch):
    called = {"registry": 0, "scaffold": 0}

    def fake_get_or_create_project_record(*args, **kwargs):
        called["registry"] += 1
        return SimpleNamespace(project_name=args[0], project_id="should-not-happen")

    def fake_create_games_project_tree(*args, **kwargs):
        called["scaffold"] += 1
        return "should-not-happen"

    monkeypatch.setattr("gateway.project_registry.get_or_create_project_record", fake_get_or_create_project_record)
    monkeypatch.setattr("gateway.project_registry.create_games_project_tree", fake_create_games_project_tree)

    assert _bootstrap_games_project_kickoff("신규 게임 프로젝트를 시작합니다") is None
    assert called == {"registry": 0, "scaffold": 0}


def test_bootstrap_explicitly_creates_registry_and_games_scaffold(monkeypatch):
    observed = {}
    record = SimpleNamespace(
        project_name="2D 서브컬쳐 턴제 RPG",
        normalized_name="2D_서브컬쳐_턴제_RPG",
        project_id="260601_2D_서브컬쳐_턴제_RPG",
        created_on="2026-06-01",
    )

    def fake_get_or_create_project_record(project_name, created_on, *, registry_path=None):
        observed["registry"] = {
            "project_name": project_name,
            "created_on": str(created_on),
            "registry_path": registry_path,
        }
        return record

    def fake_create_games_project_tree(project_name, created_on, *, work_root=None, registry_path=None, project_record=None):
        observed["scaffold"] = {
            "project_name": project_name,
            "created_on": str(created_on),
            "work_root": work_root,
            "registry_path": registry_path,
            "project_record": project_record,
        }
        return f"/tmp/Games/{record.project_id}"

    monkeypatch.setattr("gateway.project_registry.get_or_create_project_record", fake_get_or_create_project_record)
    monkeypatch.setattr("gateway.project_registry.create_games_project_tree", fake_create_games_project_tree)

    result = _bootstrap_games_project_kickoff("2D 서브컬쳐 턴제 RPG 게임 제작을 시작합니다")

    assert result == (record, f"/tmp/Games/{record.project_id}")
    assert observed["registry"]["project_name"] == "2D 서브컬쳐 턴제 RPG"
    assert observed["scaffold"]["project_record"] is record
    assert observed["scaffold"]["project_name"] == "2D 서브컬쳐 턴제 RPG"


def test_bootstrap_supports_project_name_marker_phrase(monkeypatch):
    observed = {}
    record = SimpleNamespace(
        project_name="망각구역",
        normalized_name="망각구역",
        project_id="260601_망각구역",
        created_on="2026-06-01",
    )

    def fake_get_or_create_project_record(project_name, created_on, *, registry_path=None):
        observed["registry"] = {
            "project_name": project_name,
            "created_on": str(created_on),
            "registry_path": registry_path,
        }
        return record

    def fake_create_games_project_tree(project_name, created_on, *, work_root=None, registry_path=None, project_record=None):
        observed["scaffold"] = {
            "project_name": project_name,
            "created_on": str(created_on),
            "work_root": work_root,
            "registry_path": registry_path,
            "project_record": project_record,
        }
        return f"/tmp/Games/{record.project_id}"

    monkeypatch.setattr("gateway.project_registry.get_or_create_project_record", fake_get_or_create_project_record)
    monkeypatch.setattr("gateway.project_registry.create_games_project_tree", fake_create_games_project_tree)

    result = _bootstrap_games_project_kickoff("프로젝트명 망각구역으로 게임 개발을 시작합니다")

    assert result == (record, f"/tmp/Games/{record.project_id}")
    assert observed["registry"]["project_name"] == "망각구역"
    assert observed["scaffold"]["project_record"] is record


def test_run_conversation_bootstraps_before_routing(monkeypatch):
    class Bootstrapped(Exception):
        pass

    agent = SimpleNamespace(
        _ensure_db_session=lambda: None,
        _restore_primary_runtime=lambda: None,
        _memory_write_origin="assistant_tool",
        session_id="session-123",
        provider="openai",
        model="gpt-4.1",
    )

    seen = {}

    def fake_bootstrap(user_message, **kwargs):
        seen["user_message"] = user_message
        seen["kwargs"] = kwargs
        raise Bootstrapped()

    monkeypatch.setattr("agent.conversation_loop._install_safe_stdio", lambda: None)
    monkeypatch.setattr("agent.conversation_loop.set_session_context", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.conversation_loop.set_current_write_origin", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.conversation_loop._bootstrap_games_project_kickoff", fake_bootstrap)

    with pytest.raises(Bootstrapped):
        run_conversation(agent, "2D 서브컬쳐 턴제 RPG 게임 제작을 시작합니다", conversation_history=[])

    assert seen["user_message"] == "2D 서브컬쳐 턴제 RPG 게임 제작을 시작합니다"


def test_general_document_request_does_not_create_project(monkeypatch):
    called = {"registry": 0, "scaffold": 0}

    def fake_get_or_create_project_record(*args, **kwargs):
        called["registry"] += 1
        return SimpleNamespace(project_name=args[0], project_id="unexpected")

    def fake_create_games_project_tree(*args, **kwargs):
        called["scaffold"] += 1
        return "unexpected"

    monkeypatch.setattr("gateway.project_registry.get_or_create_project_record", fake_get_or_create_project_record)
    monkeypatch.setattr("gateway.project_registry.create_games_project_tree", fake_create_games_project_tree)

    assert _bootstrap_games_project_kickoff("게임 기획서 초안을 작성해 주세요") is None
    assert called == {"registry": 0, "scaffold": 0}


@pytest.mark.parametrize(
    "message",
    [
        "망각구역 게임 기획서를 다듬어주세요",
        "망각구역 프로젝트 문서를 수정해주세요",
        "신규 게임 기획서를 작성해주세요",
        "게임 프로젝트 기획서를 작성해주세요",
    ],
)
def test_negative_document_requests_do_not_bootstrap_project(message):
    assert _bootstrap_games_project_kickoff(message) is None
