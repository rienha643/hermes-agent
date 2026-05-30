from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import gateway.document_artifacts as document_artifacts


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

    hook_calls: list[dict] = []

    def fake_queue_nas_sync_hook(**kwargs):
        hook_calls.append(kwargs)

    monkeypatch.setattr(document_artifacts, "get_hermes_work_dir", fake_get_hermes_work_dir)
    monkeypatch.setattr(document_artifacts, "queue_nas_sync_hook", fake_queue_nas_sync_hook)
    return work_root, hook_calls


class TestPublishDocumentArtifact:
    def test_copies_docx_into_documents_and_modernizes_package(self, fake_hermes_work, tmp_path):
        work_root, hook_calls = fake_hermes_work
        source = tmp_path / "lovecomedy-world-setting.docx"
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", _MINIMAL_DOCX_CONTENT_TYPES)
            zf.writestr("word/document.xml", _MINIMAL_DOCX_DOCUMENT)

        published = document_artifacts.publish_document_artifact(source, folder_name="worldbuilding")

        expected = work_root / "Documents" / "worldbuilding" / source.name
        assert published == expected
        assert published.exists()
        assert hook_calls
        assert hook_calls[0]["category"] == "documents"
        assert hook_calls[0]["artifact_path"] == published
        assert hook_calls[0]["scope"] == "worldbuilding"
        assert hook_calls[0]["source_root"] == work_root / "Documents" / "worldbuilding"

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
        assert hook_calls
        assert hook_calls[0]["artifact_path"] == existing
        assert hook_calls[0]["source_root"] == existing.parent
        assert hook_calls[0]["scope"] == "docs"
