from __future__ import annotations

from pathlib import Path
from datetime import date

import pytest

from gateway import project_registry
from gateway.artifact_delete import (
    APPROVAL_PAPERWORK,
    ArtifactDeleteOrchestrator,
    build_delete_dry_run,
    execute_approved_local_delete,
)


@pytest.fixture()
def hermeswork_root(tmp_path, monkeypatch):
    local_root = tmp_path / "HermesWork"
    monkeypatch.setenv("HERMESWORK_ROOT", str(local_root))
    monkeypatch.setenv("HERMESWORK_NAS_ROOT", r"\\test-nas\\Hermes")
    return local_root


def _make_category_file(root: Path, category: str, relative: str) -> Path:
    path = root / category.capitalize() / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("sample", encoding="utf-8")
    return path


def test_documents_dry_run_has_user_friendly_ux_and_approval_metadata(hermeswork_root):
    path = _make_category_file(hermeswork_root, "documents", "note.txt")
    result = build_delete_dry_run(path, category="documents")
    assert result is not None
    assert result["delete_mode"] == "dry-run"
    assert result["deletion_executed"] is False
    assert result["category"] == "documents"
    assert result["local_path"] == str(path.resolve())
    assert result["nas_path"] == r"\\test-nas\\Hermes\Documents\note.txt"
    assert result["will_delete_local"] is True
    assert result["will_delete_nas"] is True
    assert result["local_delete_planned"] is True
    assert result["nas_delete_planned"] is True
    assert result["requires_approval"] is True
    assert result["approval_purpose"].strip()
    assert result["approval_work"].strip()
    assert result["approval_metadata"]["purpose"].strip()
    assert result["approval_metadata"]["work"].strip()
    assert result["approval_metadata"]["required"] is True
    assert result["registry_cleanup_attempted"] is False
    assert result["registry_cleanup_executed"] is False
    assert result["registry_cleanup_status"] == "not_attempted"
    assert result["registry_cleanup_project_id"] is None
    assert result["registry_cleanup_error"] is None
    assert "삭제 모드: dry-run" in result["user_message"]
    assert "실제 삭제 실행: 아니오" in result["user_message"]
    assert "승인 필요: 예" in result["user_message"]


def test_story_dry_run_has_user_friendly_ux_and_approval_metadata(hermeswork_root):
    path = _make_category_file(hermeswork_root, "story", "chapter-1.md")
    result = build_delete_dry_run(path, category="story")
    assert result is not None
    assert result["delete_mode"] == "dry-run"
    assert result["deletion_executed"] is False
    assert result["category"] == "story"
    assert result["will_delete_local"] is True
    assert result["will_delete_nas"] is True
    assert result["requires_approval"] is True
    assert result["approval_metadata"]["required"] is True
    assert result["registry_cleanup_attempted"] is False
    assert result["registry_cleanup_executed"] is False
    assert result["registry_cleanup_status"] == "not_attempted"
    assert result["registry_cleanup_project_id"] is None
    assert result["registry_cleanup_error"] is None
    assert "삭제 모드: dry-run" in result["user_message"]
    assert "승인 필요: 예" in result["user_message"]


def test_image_dry_run_has_user_friendly_ux_and_approval_metadata(hermeswork_root):
    path = _make_category_file(hermeswork_root, "image", "cover.png")
    result = build_delete_dry_run(path, category="image")
    assert result is not None
    assert result["delete_mode"] == "dry-run"
    assert result["deletion_executed"] is False
    assert result["category"] == "image"
    assert result["will_delete_local"] is True
    assert result["will_delete_nas"] is True
    assert result["requires_approval"] is True
    assert result["approval_metadata"]["required"] is True
    assert result["registry_cleanup_attempted"] is False
    assert result["registry_cleanup_executed"] is False
    assert result["registry_cleanup_status"] == "not_attempted"
    assert result["registry_cleanup_project_id"] is None
    assert result["registry_cleanup_error"] is None
    assert "삭제 모드: dry-run" in result["user_message"]
    assert "승인 필요: 예" in result["user_message"]


def test_games_dry_run_has_user_friendly_ux_and_approval_metadata(hermeswork_root):
    path = _make_category_file(hermeswork_root, "games", "save.dat")
    result = build_delete_dry_run(path, category="games")
    assert result is not None
    assert result["delete_mode"] == "dry-run"
    assert result["deletion_executed"] is False
    assert result["category"] == "games"
    assert result["will_delete_local"] is True
    assert result["will_delete_nas"] is True
    assert result["requires_approval"] is True
    assert result["approval_metadata"]["required"] is True
    assert result["registry_cleanup_attempted"] is False
    assert result["registry_cleanup_executed"] is False
    assert result["registry_cleanup_status"] == "not_attempted"
    assert result["registry_cleanup_project_id"] is None
    assert result["registry_cleanup_error"] is None
    assert "삭제 모드: dry-run" in result["user_message"]
    assert "승인 필요: 예" in result["user_message"]


@pytest.mark.parametrize(
    "relative",
    [
        ".gitkeep",
        "ai-image-gallery/index.md",
        "SOUL.md",
        "config.yaml",
        "project_registry.json",
    ],
)
def test_operational_assets_are_blocked(hermeswork_root, relative):
    protected = _make_category_file(hermeswork_root, "documents", relative)
    result = build_delete_dry_run(protected, category="documents")
    assert result is not None
    assert result["delete_mode"] == "blocked"
    assert result["deletion_executed"] is False
    assert result["will_delete_local"] is False
    assert result["will_delete_nas"] is False
    assert result["local_delete_planned"] is False
    assert result["nas_delete_planned"] is False
    assert result["requires_approval"] is False
    assert result["approval_metadata"]["required"] is False
    assert result["warnings"]
    assert result["blocked_reasons"]
    assert any("protected asset blocked" in warning for warning in result["warnings"])
    assert "운영 자산으로 판단되어 삭제 계획을 생성하지 않았습니다." in result["user_message"]
    assert "차단 사유" in result["user_message"]


def test_approved_project_root_folder_delete_cleans_registry_entry_and_reports_cleanup(hermeswork_root, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(hermeswork_root.parent / "profiles" / "coder"))
    record = project_registry.register_project("Project Alpha", date(2026, 6, 1))
    project_root = hermeswork_root / "Games" / record.project_id
    (project_root / "UnityProject" / "Assets").mkdir(parents=True, exist_ok=True)
    (project_root / "UnityProject" / "Assets" / "readme.txt").write_text("hello", encoding="utf-8")

    result = execute_approved_local_delete(
        project_root,
        category="games",
        approved=True,
        approval_proof={"user_approved": True, "approval_id": "approval-1"},
        local_root=hermeswork_root,
    )

    assert result is not None
    assert result["delete_mode"] == "actual_delete"
    assert result["deletion_executed"] is True
    assert result["local_delete_executed"] is True
    assert result["local_delete_verified"] is True
    assert result["registry_cleanup_attempted"] is True
    assert result["registry_cleanup_executed"] is True
    assert result["registry_cleanup_status"] == "removed"
    assert result["registry_cleanup_project_id"] == record.project_id
    assert result["registry_cleanup_error"] is None
    assert not project_root.exists()
    assert project_registry.load_project_registry() == {}


def test_approved_project_root_folder_delete_reports_not_found_without_registry_entry(hermeswork_root, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(hermeswork_root.parent / "profiles" / "coder"))
    project_root = hermeswork_root / "Documents" / "260601_은월전선"
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "notes.txt").write_text("hello", encoding="utf-8")

    result = execute_approved_local_delete(
        project_root,
        category="documents",
        approved=True,
        approval_proof={"user_approved": True, "approval_id": "approval-1"},
        local_root=hermeswork_root,
    )

    assert result is not None
    assert result["registry_cleanup_attempted"] is True
    assert result["registry_cleanup_executed"] is True
    assert result["registry_cleanup_status"] == "not_found"
    assert result["registry_cleanup_project_id"] == "260601_은월전선"
    assert result["registry_cleanup_error"] is None
    assert not project_root.exists()
    assert not project_registry.project_registry_path().exists()


def test_approved_file_delete_does_not_cleanup_registry(hermeswork_root, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(hermeswork_root.parent / "profiles" / "coder"))
    record = project_registry.register_project("Project Alpha", date(2026, 6, 1))
    project_file = hermeswork_root / "Documents" / record.project_id / "note.txt"
    project_file.parent.mkdir(parents=True, exist_ok=True)
    project_file.write_text("hello", encoding="utf-8")

    result = execute_approved_local_delete(
        project_file,
        category="documents",
        approved=True,
        approval_proof={"user_approved": True, "approval_id": "approval-1"},
        local_root=hermeswork_root,
    )

    assert result is not None
    assert result["deletion_executed"] is True
    assert result["local_delete_verified"] is True
    assert result["registry_cleanup_attempted"] is False
    assert result["registry_cleanup_executed"] is False
    assert result["registry_cleanup_status"] == "skipped_not_project_root"
    assert result["registry_cleanup_project_id"] is None
    assert project_registry.load_project_registry()["project_alpha"].project_id == record.project_id
    assert not project_file.exists()


def test_local_delete_failure_skips_registry_cleanup(hermeswork_root, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(hermeswork_root.parent / "profiles" / "coder"))
    record = project_registry.register_project("Project Alpha", date(2026, 6, 1))
    project_root = hermeswork_root / "Games" / record.project_id
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "keep.txt").write_text("hello", encoding="utf-8")

    from gateway import artifact_delete as artifact_delete_module

    monkeypatch.setattr(artifact_delete_module, "_delete_local_target", lambda target: (_ for _ in ()).throw(RuntimeError("boom")))

    result = execute_approved_local_delete(
        project_root,
        category="games",
        approved=True,
        approval_proof={"user_approved": True, "approval_id": "approval-1"},
        local_root=hermeswork_root,
    )

    assert result is not None
    assert result["deletion_executed"] is False
    assert result["local_delete_executed"] is False
    assert result["local_delete_verified"] is False
    assert result["registry_cleanup_attempted"] is False
    assert result["registry_cleanup_executed"] is False
    assert result["registry_cleanup_status"] == "skipped_local_delete_failed"
    assert project_root.exists()
    assert project_registry.load_project_registry()["project_alpha"].project_id == record.project_id


def test_outside_hermeswork_path_is_not_artifact_delete_target(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("sample", encoding="utf-8")

    result = build_delete_dry_run(outside)
    assert result is None
    assert outside.exists()


def test_approval_metadata_is_never_blank(hermeswork_root, monkeypatch):
    path = _make_category_file(hermeswork_root, "documents", "draft.txt")
    monkeypatch.setitem(APPROVAL_PAPERWORK, "approval_purpose", "")
    with pytest.raises(ValueError, match="approval_purpose must not be blank"):
        build_delete_dry_run(path, category="documents")

    monkeypatch.setitem(APPROVAL_PAPERWORK, "approval_purpose", "Hermes 아티팩트 삭제")
    monkeypatch.setitem(APPROVAL_PAPERWORK, "approval_work", "")
    with pytest.raises(ValueError, match="approval_work must not be blank"):
        build_delete_dry_run(path, category="documents")


def test_orchestrator_rejects_non_hermeswork_category_mismatch(hermeswork_root):
    path = _make_category_file(hermeswork_root, "documents", "draft.txt")
    orchestrator = ArtifactDeleteOrchestrator(local_root=hermeswork_root, nas_root=r"\\test-nas\\Hermes")
    assert orchestrator.build_plan(path, category="games") is None
