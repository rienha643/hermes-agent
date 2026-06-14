from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import gateway.document_artifacts as document_artifacts
import gateway.project_registry as project_registry


_MINIMAL_DOCX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

_MINIMAL_DOCX_DOCUMENT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:r>
        <w:t>hello</w:t>
      </w:r>
    </w:p>
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


@pytest.fixture()
def fake_hermes_work(monkeypatch, tmp_path):
    work_root = tmp_path / "HermesWork"

    def fake_get_hermes_work_dir(*parts):
        path = work_root
        for part in parts:
            path = path / part
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fake_resolve_project_artifact_dir(category, project_name, created_on=None, *, work_root=None, registry_path=None):
        project_id = f"260601_{project_registry.normalize_project_name(project_name)}"
        return None, fake_get_hermes_work_dir(category) / project_id

    hook_calls: list[dict] = []

    def fake_queue_nas_sync_hook(**kwargs):
        hook_calls.append(kwargs)

    monkeypatch.setattr(document_artifacts, "get_hermes_work_dir", fake_get_hermes_work_dir)
    monkeypatch.setattr(document_artifacts, "resolve_project_artifact_dir", fake_resolve_project_artifact_dir)
    monkeypatch.setattr(document_artifacts, "queue_nas_sync_hook", fake_queue_nas_sync_hook)
    return work_root, hook_calls


class TestPublishDocumentArtifact:
    def test_copies_docx_into_documents_and_modernizes_package(self, fake_hermes_work, tmp_path):
        work_root, hook_calls = fake_hermes_work
        source = tmp_path / "lovecomedy-world-setting.docx"
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", _MINIMAL_DOCX_CONTENT_TYPES)
            zf.writestr("word/document.xml", _MINIMAL_DOCX_DOCUMENT)

        published = document_artifacts.publish_document_artifact(source, folder_name="reports")

        expected = work_root / "Documents" / "260601_reports" / "lovecomedy-world-setting_v1.docx"
        assert published == expected
        assert published.exists()
        assert hook_calls
        assert hook_calls[0]["category"] == "documents"
        assert hook_calls[0]["artifact_path"] == published
        assert hook_calls[0]["scope"] == "260601_reports"
        assert hook_calls[0]["source_root"] == work_root / "Documents" / "260601_reports"

        with zipfile.ZipFile(published, "r") as zf:
            names = set(zf.namelist())
        assert "word/settings.xml" in names
        assert "word/theme/theme1.xml" in names
        assert "word/fontTable.xml" in names
        assert "word/numbering.xml" in names
        with zipfile.ZipFile(published, "r") as zf:
            rels = zf.read("word/_rels/document.xml.rels").decode("utf-8")
            assert "relationships/settings" in rels
            assert "relationships/theme" in rels
            assert "relationships/fontTable" in rels
            assert "relationships/numbering" in rels

    def test_leaves_already_published_documents_in_place(self, fake_hermes_work):
        work_root, hook_calls = fake_hermes_work
        existing = work_root / "Documents" / "docs" / "report.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("# report\n", encoding="utf-8")

        published = document_artifacts.publish_document_artifact(existing)

        assert published == existing
        assert hook_calls == []
        assert document_artifacts.is_document_artifact_intermediate_path(existing)

    def test_intermediate_report_files_are_not_published(self, fake_hermes_work):
        work_root, hook_calls = fake_hermes_work
        intermediate = work_root / "logs" / "curator" / "20260614-123456" / "run.json"
        intermediate.parent.mkdir(parents=True, exist_ok=True)
        intermediate.write_text("{}", encoding="utf-8")

        published = document_artifacts.publish_document_artifact(intermediate)

        assert published == intermediate
        assert hook_calls == []
        assert document_artifacts.is_document_artifact_intermediate_path(intermediate)

    def test_routes_story_like_documents_to_story_and_ignores_ai_agent_scope(self, fake_hermes_work, tmp_path):
        work_root, hook_calls = fake_hermes_work
        source = tmp_path / "worldbuilding-lore.docx"
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", _MINIMAL_DOCX_CONTENT_TYPES)
            zf.writestr("word/document.xml", _MINIMAL_DOCX_DOCUMENT)

        published = document_artifacts.publish_document_artifact(source, folder_name="worldbuilding")

        expected = work_root / "Story" / "260601_worldbuilding" / "worldbuilding-lore_v1.docx"
        assert published == expected
        assert published.exists()
        assert hook_calls
        assert hook_calls[0]["category"] == "story"
        assert hook_calls[0]["artifact_path"] == published
        assert hook_calls[0]["scope"] == "260601_worldbuilding"
        assert hook_calls[0]["source_root"] == work_root / "Story" / "260601_worldbuilding"

    def test_routes_story_title_from_internal_folder_to_story_root(self, fake_hermes_work, tmp_path):
        work_root, hook_calls = fake_hermes_work
        source_dir = tmp_path / "hermes-agent"
        source_dir.mkdir(parents=True, exist_ok=True)
        source = source_dir / "간단한_스릴러_세계관_설정.docx"
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", _MINIMAL_DOCX_CONTENT_TYPES)
            zf.writestr("word/document.xml", _MINIMAL_DOCX_DOCUMENT)

        published = document_artifacts.publish_document_artifact(source, folder_name="hermes-agent")

        expected = work_root / "Story" / "260601_misc" / f"{source.stem}_v1.docx"
        assert published == expected
        assert hook_calls[0]["category"] == "story"
        assert hook_calls[0]["scope"] == "260601_misc"
        assert hook_calls[0]["source_root"] == work_root / "Story" / "260601_misc"
        assert not (work_root / "Documents" / "hermes-agent").exists()

    def test_normalizes_internal_document_scope_to_misc(self, fake_hermes_work, tmp_path):
        work_root, hook_calls = fake_hermes_work
        source_dir = tmp_path / "speedy"
        source_dir.mkdir(parents=True, exist_ok=True)
        source = source_dir / "게임_기획서.docx"
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", _MINIMAL_DOCX_CONTENT_TYPES)
            zf.writestr("word/document.xml", _MINIMAL_DOCX_DOCUMENT)

        published = document_artifacts.publish_document_artifact(source, folder_name="speedy")

        expected = work_root / "Documents" / "260601_misc" / f"{source.stem}_v1.docx"
        assert published == expected
        assert hook_calls[0]["category"] == "documents"
        assert hook_calls[0]["scope"] == "260601_misc"
        assert hook_calls[0]["source_root"] == work_root / "Documents" / "260601_misc"

    def test_routes_tyr_and_lyra_hints_to_story_root(self, fake_hermes_work, tmp_path):
        work_root, hook_calls = fake_hermes_work
        for folder_name, filename in [("tyr", "연대기_초안.docx"), ("lyra", "설정집_초안.docx")]:
            source = tmp_path / f"{folder_name}_{filename}"
            with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("[Content_Types].xml", _MINIMAL_DOCX_CONTENT_TYPES)
                zf.writestr("word/document.xml", _MINIMAL_DOCX_DOCUMENT)

            published = document_artifacts.publish_document_artifact(source, folder_name=folder_name)
            expected = work_root / "Story" / f"260601_{folder_name}" / f"{source.stem}_v1.docx"
            assert published == expected
            assert hook_calls[-1]["category"] == "story"
            assert hook_calls[-1]["scope"] == f"260601_{folder_name}"
            assert hook_calls[-1]["source_root"] == work_root / "Story" / f"260601_{folder_name}"

    def test_keeps_story_artifacts_without_ai_agent_subfolder(self, fake_hermes_work):
        work_root, hook_calls = fake_hermes_work
        existing = work_root / "Story" / "ai_agent" / "간단한_로맨스_세계관_설정.docx"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("placeholder", encoding="utf-8")

        published = document_artifacts.publish_document_artifact(existing)

        expected = work_root / "Story" / "간단한_로맨스_세계관_설정.docx"
        assert published == expected
        assert published.exists()
        assert published.read_text(encoding="utf-8") == "placeholder"
        assert hook_calls
        assert hook_calls[0]["category"] == "story"
        assert hook_calls[0]["artifact_path"] == expected
        assert hook_calls[0]["source_root"] == work_root / "Story"
        assert hook_calls[0]["scope"] == ""

    def test_repairs_story_duplicate_tree_and_normalizes_source_root(self, fake_hermes_work):
        work_root, hook_calls = fake_hermes_work
        nested = work_root / "Story" / "ai_agent" / "Archive" / "Documents" / "Games" / "Image" / "Story" / "간단한_호러_세계관_설정.docx"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("deep-tree", encoding="utf-8")

        published = document_artifacts.publish_document_artifact(nested)

        expected = work_root / "Story" / "간단한_호러_세계관_설정.docx"
        assert published == expected
        assert published.exists()
        assert published.read_text(encoding="utf-8") == "deep-tree"
        assert not (work_root / "Story" / "ai_agent").exists()
        assert hook_calls
        assert hook_calls[0]["category"] == "story"
        assert hook_calls[0]["artifact_path"] == expected
        assert hook_calls[0]["source_root"] == work_root / "Story"
        assert hook_calls[0]["scope"] == ""


@pytest.mark.parametrize(
    ("published_path", "expected_block"),
    [
        (
            Path("/tmp/HermesWork/Documents/planning/brief.docx"),
            "- 산출물\n  - 형식: DOCX\n  - 저장 위치: `HermesWork/Documents/planning/brief.docx`\n  - 전달 방식: Slack 첨부\n  - NAS 상태: hook state 생성",
        ),
        (
            Path("/tmp/HermesWork/Story/worldbuilding/report.docx"),
            "- 산출물\n  - 형식: DOCX\n  - 저장 위치: `HermesWork/Story/worldbuilding/report.docx`\n  - 전달 방식: Slack 첨부\n  - NAS 상태: hook state 생성",
        ),
    ],
)
def test_format_document_artifact_block_prefers_hermeswork_relative_paths(monkeypatch, tmp_path, published_path, expected_block):
    monkeypatch.setattr(document_artifacts, "get_hermes_work_dir", lambda *parts: tmp_path / "HermesWork" / Path(*parts))
    artifact = tmp_path / "HermesWork" / published_path.relative_to("/tmp/HermesWork")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("placeholder", encoding="utf-8")

    assert document_artifacts.format_document_artifact_block(artifact) == expected_block
