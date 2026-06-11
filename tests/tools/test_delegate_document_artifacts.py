import json
from pathlib import Path

import tools.delegate_tool as delegate_tool


def test_delegate_result_prioritizes_final_docx_over_sidecars(tmp_path):
    final_a = tmp_path / "final_a.docx"
    final_b = tmp_path / "final_b.pdf"
    sidecar_dir = tmp_path / "_sidecars"
    sidecar_dir.mkdir()
    ocr = sidecar_dir / "ocr_all.txt"
    image = sidecar_dir / "page_1.png"
    for path in (final_a, final_b, ocr, image):
        path.write_bytes(b"x")

    result = delegate_tool._prepare_delegate_document_artifacts(
        [str(ocr), str(final_b), str(image), str(final_a)]
    )

    assert result["artifact_files"] == [str(final_b), str(final_a)]
    assert result["artifacts"] == [str(final_b), str(final_a)]
    assert result["sidecar_files"] == [str(ocr), str(image)]


def test_delegate_moves_intermediates_into_sidecars_when_final_doc_exists(tmp_path):
    final_doc = tmp_path / "final.docx"
    ocr = tmp_path / "ocr_all.txt"
    page = tmp_path / "page_1.png"
    for path in (final_doc, ocr, page):
        path.write_bytes(b"x")

    result = delegate_tool._prepare_delegate_document_artifacts([str(ocr), str(final_doc), str(page)])

    sidecar_dir = tmp_path / "_sidecars"
    assert result["artifact_files"] == [str(final_doc)]
    assert result["sidecar_files"] == [str(sidecar_dir / "ocr_all.txt"), str(sidecar_dir / "page_1.png")]
    assert (sidecar_dir / "ocr_all.txt").exists()
    assert (sidecar_dir / "page_1.png").exists()
    assert not ocr.exists()
    assert not page.exists()


def test_delegate_max_iterations_returns_partial_status():
    assert delegate_tool._delegate_status_from_exit(raw_summary="usable", interrupted=False, completed=False) == "partial"


def test_designer_document_tasks_get_90_iteration_budget(monkeypatch):
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"max_iterations": 50, "max_iterations_by_profile": {}})

    task = {
        "goal": "PDF 이력서를 DOCX 템플릿으로 작성해줘",
        "context": "최종 산출물은 docx 문서입니다.",
        "profile": "designer",
    }

    assert delegate_tool._effective_delegate_max_iterations([task], requested_default=50) == 90


def test_non_designer_tasks_keep_configured_iteration_budget(monkeypatch):
    task = {"goal": "review code", "context": "", "profile": "coder"}

    assert delegate_tool._effective_delegate_max_iterations([task], requested_default=50) == 50
