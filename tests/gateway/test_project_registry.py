from __future__ import annotations

from datetime import date
from pathlib import Path

import gateway.project_registry as project_registry


def test_normalize_project_name_collapses_separators_to_underscores():
    assert project_registry.normalize_project_name("망각 구역") == "망각_구역"
    assert project_registry.normalize_project_name("  My  Project-Name!  ") == "My_Project_Name"


def test_project_registry_path_uses_shared_root_across_profiles(monkeypatch, tmp_path):
    shared_root = tmp_path / "shared-root"

    monkeypatch.setenv("HERMES_HOME", str(shared_root / "profiles" / "coder"))
    coder_path = project_registry.project_registry_path()

    monkeypatch.setenv("HERMES_HOME", str(shared_root / "profiles" / "designer"))
    designer_path = project_registry.project_registry_path()

    assert coder_path == designer_path == shared_root / "state" / "project_registry.json"


def test_register_project_reuses_existing_project_id_across_dates(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "coder"))

    record_1 = project_registry.register_project("망각 구역", date(2026, 6, 1))
    record_2 = project_registry.register_project("망각 구역", date(2026, 6, 12))

    assert record_1.project_id == "260601_망각_구역"
    assert record_2.project_id == record_1.project_id
    assert record_2.created_on == record_1.created_on == "2026-06-01"

    registry_path = tmp_path / "state" / "project_registry.json"
    assert registry_path.exists()
    saved = registry_path.read_text(encoding="utf-8")
    assert "망각_구역" in saved
    assert "260601_망각_구역" in saved


def test_register_project_reuses_same_project_id_across_profiles(monkeypatch, tmp_path):
    shared_root = tmp_path / "shared-root"

    monkeypatch.setenv("HERMES_HOME", str(shared_root / "profiles" / "coder"))
    coder_record = project_registry.register_project("Project Alpha", date(2026, 6, 1))

    monkeypatch.setenv("HERMES_HOME", str(shared_root / "profiles" / "designer"))
    designer_record = project_registry.register_project("Project Alpha", date(2026, 7, 4))

    assert coder_record.project_id == designer_record.project_id == "260601_Project_Alpha"
    assert designer_record.created_on == coder_record.created_on == "2026-06-01"

    registry = project_registry.load_project_registry()
    assert registry["project_alpha"].project_id == "260601_Project_Alpha"
    assert (shared_root / "state" / "project_registry.json").exists()


def test_create_games_project_tree_builds_expected_unity_folder_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "designer"))
    monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))

    project_root = project_registry.create_games_project_tree("망각 구역", "2026-06-01")

    expected_dirs = [
        project_root,
        project_root / "UnityProject",
        project_root / "Builds",
        project_root / "External",
        project_root / "UnityProject" / "Assets",
        project_root / "UnityProject" / "Packages",
        project_root / "UnityProject" / "ProjectSettings",
        project_root / "External" / "Spine",
        project_root / "External" / "Live2D",
        project_root / "External" / "Audio",
        project_root / "External" / "Import",
        project_root / "External" / "VFX",
    ]

    assert project_root == tmp_path / "HermesWork" / "Games" / "260601_망각_구역"
    for expected_dir in expected_dirs:
        assert expected_dir.exists()
        assert expected_dir.is_dir()

    assert not any(path.is_file() for path in project_root.rglob("*"))
    assert not (project_root / "UnityProject" / "ProjectVersion.txt").exists()
    assert not list(project_root.rglob("*.sln"))
    assert not list(project_root.rglob("*.csproj"))
    assert not list(project_root.rglob("*.unity"))


def test_create_games_project_tree_reuses_registry_entry_for_later_dates(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "qa"))
    monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))

    first_root = project_registry.create_games_project_tree("Project Alpha", "2026-06-01")
    second_root = project_registry.create_games_project_tree("Project Alpha", "2026-07-04")

    assert first_root == second_root
    assert first_root.name == "260601_Project_Alpha"

    registry = project_registry.load_project_registry()
    assert registry["project_alpha"].project_id == "260601_Project_Alpha"


def test_next_versioned_child_path_advances_from_existing_and_versioned_sources(tmp_path):
    directory = tmp_path / "out"
    directory.mkdir()
    (directory / "brief_v1.docx").write_text("one", encoding="utf-8")
    (directory / "brief_v2.docx").write_text("two", encoding="utf-8")

    assert project_registry.next_versioned_child_path(directory, "brief.docx") == directory / "brief_v3.docx"
    assert project_registry.next_versioned_child_path(directory, "brief_v3.docx") == directory / "brief_v3.docx"
