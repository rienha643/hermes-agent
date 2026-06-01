from __future__ import annotations

from pathlib import Path

import pytest

from gateway.artifact_delete import APPROVAL_PAPERWORK, ArtifactDeleteOrchestrator, build_delete_dry_run


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
