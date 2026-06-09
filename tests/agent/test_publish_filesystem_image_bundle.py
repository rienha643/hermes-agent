from __future__ import annotations

import json
from pathlib import Path

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de00000010494441547801635c0e000000feff03000006000557bfabd400"
    "00000049454e44ae426082"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_publish_filesystem_image_bundle_creates_versioned_bundle_and_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))

    from agent import image_gen_provider as provider_mod
    from gateway.project_registry import resolve_project_artifact_dir

    source = tmp_path / "output" / "angelica_smoke_00001_.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(PNG_1PX)

    hook_calls = []
    monkeypatch.setattr(
        provider_mod,
        "queue_nas_sync_hook",
        lambda **kwargs: hook_calls.append(kwargs) or True,
    )

    bundle = provider_mod.publish_filesystem_image_bundle(
        source,
        prefix="angelica_smoke",
        project_name="angelica_smoke_test",
        artifact_name="angelica_smoke",
        category="smoke_test",
        workflow_json={"8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "angelica_smoke"}}},
        prompt_payload={
            "prompt": "simple cute anime girl",
            "negative_prompt": "blurry",
            "width": 512,
            "height": 512,
            "batch_size": 1,
            "seed": 123,
            "sampler": "euler",
            "steps": 12,
            "cfg": 7,
            "denoise": 1,
            "raw_prompt_payload": {"prompt": {"8": {"class_type": "SaveImage"}}},
        },
        metadata={
            "provider": "comfy-local",
            "prompt_id": "pid-123",
            "api_base_url": "http://172.22.224.1:8188",
            "checkpoint": "AOM3A1_orangemixs.safetensors",
            "vae": "animevae.pt",
            "loras": [],
            "controlnet_used": False,
            "seed": 123,
            "sampler": "euler",
            "steps": 12,
            "cfg": 7,
            "denoise": 1,
            "created_at": "2026-06-05T00:00:00Z",
            "category": "smoke_test",
            "output_source_path": str(source),
        },
    )

    _, expected_dir = resolve_project_artifact_dir("Image", "angelica_smoke_test")
    assert bundle["published_dir"] == expected_dir
    assert bundle["primary_image_path"] == expected_dir / "angelica_smoke_v1.png"
    assert bundle["primary_image_path"].exists()
    assert bundle["workflow_path"] == expected_dir / "_sidecars" / "angelica_smoke_v1.workflow.json"
    assert bundle["prompt_path"] == expected_dir / "_sidecars" / "angelica_smoke_v1.prompt.json"
    assert bundle["metadata_path"] == expected_dir / "_sidecars" / "angelica_smoke_v1.metadata.json"
    assert bundle["manifest_path"] == expected_dir / "_sidecars" / "manifest.json"
    assert bundle["sidecar_dir"] == expected_dir / "_sidecars"
    assert bundle["storage_verification"]["logical_path"] == str(expected_dir.parent)
    assert bundle["storage_verification"]["realpath"] == str((expected_dir.parent).resolve())
    assert bundle["storage_verification"]["is_ssd_root"] in (True, False)
    assert bundle["storage_verification"]["is_symlink"] == expected_dir.parent.is_symlink()
    assert bundle["nas_hook_requested"] is True

    workflow = _read_json(bundle["workflow_path"])
    prompt_payload = _read_json(bundle["prompt_path"])
    metadata = _read_json(bundle["metadata_path"])
    manifest = _read_json(bundle["manifest_path"])

    assert workflow["8"]["class_type"] == "SaveImage"
    assert prompt_payload["seed"] == 123
    assert metadata["published_primary_path"] == str(bundle["primary_image_path"])
    assert metadata["published_dir"] == str(expected_dir)
    assert metadata["published_sidecar_dir"] == str(expected_dir / "_sidecars")
    assert metadata["storage_verification"]["logical_path"] == str(expected_dir.parent)
    assert metadata["storage_verification"]["is_ssd_root"] in (True, False)
    assert metadata["nas_hook_requested"] is True
    assert manifest["primary_image"] == "angelica_smoke_v1.png"
    assert any(file_path.endswith("angelica_smoke_v1.workflow.json") for file_path in manifest["files"])
    assert any(file_path.endswith("angelica_smoke_v1.prompt.json") for file_path in manifest["files"])
    assert any(file_path.endswith("angelica_smoke_v1.metadata.json") for file_path in manifest["files"])
    assert any(file_path.endswith("manifest.json") for file_path in manifest["files"])
    assert manifest["sidecars"]["metadata"] == "angelica_smoke_v1.metadata.json"
    assert manifest["sidecars"]["manifest"] == "manifest.json"

    assert hook_calls == [
        {
            "category": "image",
            "scope": expected_dir.name,
            "artifact_path": bundle["primary_image_path"],
            "source_root": expected_dir,
        }
    ]


def test_publish_filesystem_image_bundle_stores_sidecars_under_ssd_or_reports_verification(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    source = tmp_path / "output" / "smoke_00001_.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(PNG_1PX)

    mock_ssd_root = tmp_path / "mnt_ssd" / "HermesWork" / "Image"
    mock_ssd_root.mkdir(parents=True, exist_ok=True)
    mock_logical_root = tmp_path / "mock_HermesWork"
    mock_logical_root.symlink_to(mock_ssd_root.parent, target_is_directory=True)
    monkeypatch.setenv("HERMES_WORK_ROOT", str(mock_logical_root))

    hook_calls = []

    from agent import image_gen_provider as provider_mod
    monkeypatch.setattr(
        provider_mod,
        "queue_nas_sync_hook",
        lambda **kwargs: hook_calls.append(kwargs) or True,
    )
    monkeypatch.setattr(provider_mod, "_SSD_IMAGE_ROOT", mock_ssd_root)

    bundle = provider_mod.publish_filesystem_image_bundle(
        source,
        prefix="smoke",
        project_name="smoke_ssd_test",
        artifact_name="smoke",
        category="smoke_test",
        workflow_json={"8": {}},
        prompt_payload={
            "prompt": "a",
            "negative_prompt": "",
            "width": 512,
            "height": 512,
            "batch_size": 1,
            "seed": 123,
            "sampler": "euler",
            "steps": 1,
            "cfg": 1,
            "denoise": 1,
            "raw_prompt_payload": {},
        },
        metadata={
            "provider": "comfy-local",
            "prompt_id": "pid-1",
            "api_base_url": "http://172.22.224.1:8188",
            "checkpoint": "AOM3A1_orangemixs.safetensors",
            "vae": "animevae.pt",
            "loras": [],
            "controlnet_used": False,
            "seed": 123,
            "sampler": "euler",
            "steps": 1,
            "cfg": 1,
            "denoise": 1,
            "created_at": "2026-06-05T00:00:00Z",
            "category": "smoke_test",
            "output_source_path": str(source),
        },
    )

    assert bundle["storage_verification"]["logical_path"] == str((mock_logical_root / "Image"))
    assert bundle["storage_verification"]["realpath"] == str(mock_ssd_root)
    assert bundle["storage_verification"]["is_ssd_root"] is True
    assert bundle["storage_verification"]["is_symlink"] is True
    assert bundle["sidecar_dir"] == bundle["published_dir"] / "_sidecars"
    assert hook_calls == [
        {
            "category": "image",
            "scope": bundle["published_dir"].name,
            "artifact_path": bundle["primary_image_path"],
            "source_root": bundle["published_dir"],
        }
    ]



def test_publish_filesystem_image_bundle_uses_next_version_when_name_repeats(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))

    from agent import image_gen_provider as provider_mod

    source = tmp_path / "output" / "angelica_smoke_00001_.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(PNG_1PX)

    monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

    first = provider_mod.publish_filesystem_image_bundle(
        source,
        prefix="angelica_smoke",
        project_name="angelica_smoke_test",
        artifact_name="angelica_smoke",
        category="smoke_test",
        workflow_json={},
        prompt_payload={
            "prompt": "a",
            "negative_prompt": "b",
            "width": 512,
            "height": 512,
            "batch_size": 1,
            "seed": 1,
            "sampler": "euler",
            "steps": 1,
            "cfg": 1,
            "denoise": 1,
            "raw_prompt_payload": {},
        },
        metadata={
            "provider": "comfy-local",
            "prompt_id": "pid-1",
            "api_base_url": "http://172.22.224.1:8188",
            "checkpoint": "AOM3A1_orangemixs.safetensors",
            "vae": "animevae.pt",
            "loras": [],
            "controlnet_used": False,
            "seed": 1,
            "sampler": "euler",
            "steps": 1,
            "cfg": 1,
            "denoise": 1,
            "created_at": "2026-06-05T00:00:00Z",
            "category": "smoke_test",
            "output_source_path": str(source),
        },
    )
    second = provider_mod.publish_filesystem_image_bundle(
        source,
        prefix="angelica_smoke",
        project_name="angelica_smoke_test",
        artifact_name="angelica_smoke",
        category="smoke_test",
        workflow_json={},
        prompt_payload={
            "prompt": "a",
            "negative_prompt": "b",
            "width": 512,
            "height": 512,
            "batch_size": 1,
            "seed": 1,
            "sampler": "euler",
            "steps": 1,
            "cfg": 1,
            "denoise": 1,
            "raw_prompt_payload": {},
        },
        metadata={
            "provider": "comfy-local",
            "prompt_id": "pid-2",
            "api_base_url": "http://172.22.224.1:8188",
            "checkpoint": "AOM3A1_orangemixs.safetensors",
            "vae": "animevae.pt",
            "loras": [],
            "controlnet_used": False,
            "seed": 1,
            "sampler": "euler",
            "steps": 1,
            "cfg": 1,
            "denoise": 1,
            "created_at": "2026-06-05T00:00:00Z",
            "category": "smoke_test",
            "output_source_path": str(source),
        },
    )

    assert first["primary_image_path"].name == "angelica_smoke_v1.png"
    assert second["primary_image_path"].name == "angelica_smoke_v2.png"
    assert second["workflow_path"].name == "angelica_smoke_v2.workflow.json"


def test_publish_filesystem_image_bundle_groups_multiple_artifacts_under_same_project(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))

    from agent import image_gen_provider as provider_mod

    source = tmp_path / "output" / "grouped_00001_.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(PNG_1PX)

    monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

    common_prompt_payload = {
        "prompt": "a",
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "batch_size": 1,
        "seed": 1,
        "sampler": "euler",
        "steps": 1,
        "cfg": 1,
        "denoise": 1,
        "raw_prompt_payload": {},
    }
    common_metadata = {
        "provider": "comfy-local",
        "api_base_url": "http://172.22.224.1:8188",
        "checkpoint": "AOM3A1_orangemixs.safetensors",
        "vae": "animevae.pt",
        "loras": [],
        "controlnet_used": False,
        "sampler": "euler",
        "steps": 1,
        "cfg": 1,
        "denoise": 1,
        "created_at": "2026-06-05T00:00:00Z",
    }

    bundle_a = provider_mod.publish_filesystem_image_bundle(
        source,
        prefix="qualification",
        project_name="ANG_TXT_001_qualification",
        artifact_name="ang_txt_001_por_01",
        category="portrait",
        workflow_json={},
        prompt_payload=dict(common_prompt_payload, seed=1),
        metadata=dict(common_metadata, prompt_id="pid-a", seed=1),
    )
    bundle_b = provider_mod.publish_filesystem_image_bundle(
        source,
        prefix="qualification",
        project_name="ANG_TXT_001_qualification",
        artifact_name="ang_txt_001_fb_01",
        category="full_body",
        workflow_json={},
        prompt_payload=dict(common_prompt_payload, seed=2),
        metadata=dict(common_metadata, prompt_id="pid-b", seed=2),
    )
    bundle_c = provider_mod.publish_filesystem_image_bundle(
        source,
        prefix="qualification",
        project_name="ANG_TXT_001_qualification",
        artifact_name="ang_txt_001_env_01",
        category="environment",
        workflow_json={},
        prompt_payload=dict(common_prompt_payload, seed=3),
        metadata=dict(common_metadata, prompt_id="pid-c", seed=3),
    )

    assert bundle_a["published_dir"] == bundle_b["published_dir"] == bundle_c["published_dir"]
    entries = sorted(p.name for p in bundle_a["published_dir"].iterdir())
    assert "ang_txt_001_por_01_v1.png" in entries
    assert "ang_txt_001_fb_01_v1.png" in entries
    assert "ang_txt_001_env_01_v1.png" in entries
    assert "_sidecars" in entries
    sidecar_dir = bundle_a["published_dir"] / "_sidecars"
    assert sidecar_dir.is_dir()
    assert any(path.suffix == ".json" for path in sidecar_dir.iterdir())
    # All sidecar JSONs are contained under _sidecars, not mixed into root scope
    assert bundle_a["workflow_path"].parent == sidecar_dir


def test_run_manifest_and_qualification_report_can_be_written_for_grouped_publish(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))

    from agent import image_gen_provider as provider_mod

    published_dir = tmp_path / "HermesWork" / "Image" / "260605_ANG_TXT_001_qualification"
    published_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [
        {
            "run_id": "ANG-TXT-001-POR-01",
            "artifact_name": "ang_txt_001_por_01",
            "primary_image": "ang_txt_001_por_01_v1.png",
            "workflow_json": "ang_txt_001_por_01_v1.workflow.json",
            "prompt_json": "ang_txt_001_por_01_v1.prompt.json",
            "metadata_json": "ang_txt_001_por_01_v1.metadata.json",
            "category": "portrait",
            "seed": 21001,
            "status": {"technical_result": "Pass", "publish_status": "Pass", "nas_hook_status": "Pass", "slack_status": "Pass"},
        },
        {
            "run_id": "ANG-TXT-001-FB-01",
            "artifact_name": "ang_txt_001_fb_01",
            "primary_image": "ang_txt_001_fb_01_v1.png",
            "workflow_json": "ang_txt_001_fb_01_v1.workflow.json",
            "prompt_json": "ang_txt_001_fb_01_v1.prompt.json",
            "metadata_json": "ang_txt_001_fb_01_v1.metadata.json",
            "category": "full_body",
            "seed": 22001,
            "status": {"technical_result": "Pass", "publish_status": "Pass", "nas_hook_status": "Pass", "slack_status": "Pass"},
        },
    ]

    run_manifest_path = provider_mod.write_run_manifest(
        published_dir,
        workflow_code="ANG-TXT-001",
        workflow_name="TXT2IMG Basic",
        run_kind="qualification",
        project_name="ANG_TXT_001_qualification",
        project_id="260605_ANG_TXT_001_qualification",
        artifacts=artifacts,
        summary={"total_runs": 2, "artifact_count": 2},
    )
    report_path = provider_mod.write_qualification_report(
        published_dir,
        {
            "workflow_code": "ANG-TXT-001",
            "workflow_name": "TXT2IMG Basic",
            "test_name": "ANG-TXT-001 Production Qualification Test",
            "run_date": "2026-06-05",
            "production_gate_result": "Hold",
            "summary": {"total_runs": 2, "technical_pass_count": 2, "visual_pass_count": 1, "visual_warning_count": 1, "visual_fail_count": 0, "full_body_face_eye_fail_count": 0, "publish_success_count": 2, "nas_hook_success_count": 2, "slack_success_count": 2},
            "runs": [{"run_id": "ANG-TXT-001-POR-01", "category": "portrait", "prompt_summary": "heroine close-up portrait", "seed": 21001, "image_path": "ang_txt_001_por_01_v1.png", "technical_status": "Pass", "visual_qc": "Pass", "full_body_face_eye_qc": "NA", "final_decision": "Accept", "reviewer_note": "ok"}],
            "lifecycle_after_proposed": {"core_pipeline_status": "Production", "use_case_status": {"portrait": "Production Candidate", "full_body": "MVP", "environment": "MVP"}},
            "user_feedback": [],
            "known_risks": ["full body character outputs may suffer degraded eye/pupil alignment due to small face scale"],
            "next_actions": ["tighten QC"],
        },
    )

    run_manifest = _read_json(run_manifest_path)
    report = _read_json(report_path)

    assert run_manifest["manifest_type"] == "run_summary"
    assert run_manifest["workflow_code"] == "ANG-TXT-001"
    assert len(run_manifest["artifacts"]) == 2
    assert run_manifest["artifacts"][0]["artifact_name"] == "ang_txt_001_por_01"
    assert report["workflow_code"] == "ANG-TXT-001"
    assert report["lifecycle_after_proposed"]["core_pipeline_status"] == "Production"
    assert report["lifecycle_after_proposed"]["use_case_status"]["full_body"] == "MVP"


def test_finalize_run_manifest_status_recomputes_total_runs_from_artifacts_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))

    from agent import image_gen_provider as provider_mod

    published_dir = tmp_path / "HermesWork" / "Image" / "260605_ANG_TXT_001_qualification"
    published_dir.mkdir(parents=True, exist_ok=True)

    provider_mod.write_run_manifest(
        published_dir,
        workflow_code="ANG-TXT-001",
        workflow_name="TXT2IMG Basic",
        run_kind="qualification",
        project_name="ANG_TXT_001_qualification",
        project_id="260605_ANG_TXT_001_qualification",
        artifacts=[
            {"run_id": "ANG-TXT-001-POR-01", "artifact_name": "ang_txt_001_por_01", "primary_image": "a.png", "workflow_json": "a.workflow.json", "prompt_json": "a.prompt.json", "metadata_json": "a.metadata.json", "category": "portrait", "seed": 1, "status": {"technical_result": "Pass", "publish_status": "Pass", "nas_hook_status": "Pass", "slack_status": "Pending"}},
            {"run_id": "ANG-TXT-001-FB-01", "artifact_name": "ang_txt_001_fb_01", "primary_image": "b.png", "workflow_json": "b.workflow.json", "prompt_json": "b.prompt.json", "metadata_json": "b.metadata.json", "category": "full_body", "seed": 2, "status": {"technical_result": "Pass", "publish_status": "Pass", "nas_hook_status": "Pass", "slack_status": "Pending"}},
            {"run_id": "ANG-TXT-001-ENV-01", "artifact_name": "ang_txt_001_env_01", "primary_image": "c.png", "workflow_json": "c.workflow.json", "prompt_json": "c.prompt.json", "metadata_json": "c.metadata.json", "category": "environment", "seed": 3, "status": {"technical_result": "Pass", "publish_status": "Pass", "nas_hook_status": "Pass", "slack_status": "Pending"}},
        ],
        summary={"artifact_count": 3, "total_runs": 1},
    )

    manifest_path = provider_mod.finalize_run_manifest_status(published_dir)
    manifest = _read_json(manifest_path)

    assert manifest["summary"]["artifact_count"] == 3
    assert manifest["summary"]["total_runs"] == 3
    assert manifest["summary"]["completed_run_count"] == 3
    assert manifest["summary"]["failed_run_count"] == 0


def test_finalize_run_manifest_status_prefers_planned_run_count(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))

    from agent import image_gen_provider as provider_mod

    published_dir = tmp_path / "HermesWork" / "Image" / "260605_ANG_TXT_001_qualification"
    published_dir.mkdir(parents=True, exist_ok=True)

    provider_mod.write_run_manifest(
        published_dir,
        workflow_code="ANG-TXT-001",
        workflow_name="TXT2IMG Basic",
        run_kind="qualification",
        project_name="ANG_TXT_001_qualification",
        project_id="260605_ANG_TXT_001_qualification",
        artifacts=[
            {"run_id": "ANG-TXT-001-POR-01", "artifact_name": "ang_txt_001_por_01", "primary_image": "a.png", "workflow_json": "a.workflow.json", "prompt_json": "a.prompt.json", "metadata_json": "a.metadata.json", "category": "portrait", "seed": 1, "status": {"technical_result": "Pass", "publish_status": "Pass", "nas_hook_status": "Pass", "slack_status": "Pending"}},
            {"run_id": "ANG-TXT-001-FB-01", "artifact_name": "ang_txt_001_fb_01", "primary_image": "b.png", "workflow_json": "b.workflow.json", "prompt_json": "b.prompt.json", "metadata_json": "b.metadata.json", "category": "full_body", "seed": 2, "status": {"technical_result": "Pass", "publish_status": "Pass", "nas_hook_status": "Pass", "slack_status": "Pending"}},
            {"run_id": "ANG-TXT-001-ENV-01", "artifact_name": "ang_txt_001_env_01", "primary_image": "c.png", "workflow_json": "c.workflow.json", "prompt_json": "c.prompt.json", "metadata_json": "c.metadata.json", "category": "environment", "seed": 3, "status": {"technical_result": "Pass", "publish_status": "Pass", "nas_hook_status": "Pass", "slack_status": "Pending"}},
        ],
        summary={"planned_run_count": 10, "artifact_count": 3, "total_runs": 1},
    )

    manifest_path = provider_mod.finalize_run_manifest_status(published_dir)
    manifest = _read_json(manifest_path)

    assert manifest["summary"]["planned_run_count"] == 10
    assert manifest["summary"]["artifact_count"] == 3
    assert manifest["summary"]["total_runs"] == 10


def test_update_run_delivery_status_and_finalize_qualification_report(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))

    from agent import image_gen_provider as provider_mod

    published_dir = tmp_path / "HermesWork" / "Image" / "260605_ANG_TXT_001_qualification"
    published_dir.mkdir(parents=True, exist_ok=True)

    provider_mod.write_run_manifest(
        published_dir,
        workflow_code="ANG-TXT-001",
        workflow_name="TXT2IMG Basic",
        run_kind="qualification",
        project_name="ANG_TXT_001_qualification",
        project_id="260605_ANG_TXT_001_qualification",
        artifacts=[
            {"run_id": "ANG-TXT-001-POR-01", "artifact_name": "ang_txt_001_por_01", "primary_image": "a.png", "workflow_json": "a.workflow.json", "prompt_json": "a.prompt.json", "metadata_json": "a.metadata.json", "category": "portrait", "seed": 1, "status": {"technical_result": "Pass", "publish_status": "Pass", "nas_hook_status": "Pass", "slack_status": "Pending"}},
            {"run_id": "ANG-TXT-001-FB-01", "artifact_name": "ang_txt_001_fb_01", "primary_image": "b.png", "workflow_json": "b.workflow.json", "prompt_json": "b.prompt.json", "metadata_json": "b.metadata.json", "category": "full_body", "seed": 2, "status": {"technical_result": "Pass", "publish_status": "Pass", "nas_hook_status": "Pass", "slack_status": "Pending"}},
            {"run_id": "ANG-TXT-001-ENV-01", "artifact_name": "ang_txt_001_env_01", "primary_image": "c.png", "workflow_json": "c.workflow.json", "prompt_json": "c.prompt.json", "metadata_json": "c.metadata.json", "category": "environment", "seed": 3, "status": {"technical_result": "Pass", "publish_status": "Pass", "nas_hook_status": "Pass", "slack_status": "Pending"}},
        ],
        summary={"planned_run_count": 3, "artifact_count": 3, "total_runs": 3},
    )
    provider_mod.write_qualification_report(
        published_dir,
        {
            "workflow_code": "ANG-TXT-001",
            "workflow_name": "TXT2IMG Basic",
            "test_name": "ANG-TXT-001 Production Qualification Test",
            "run_date": "2026-06-05",
            "production_gate_result": "Hold",
            "summary": {"total_runs": 3, "technical_pass_count": 3, "visual_pass_count": 0, "visual_warning_count": 0, "visual_fail_count": 0, "full_body_face_eye_fail_count": 0, "publish_success_count": 3, "nas_hook_success_count": 3, "slack_success_count": 0},
            "runs": [
                {"run_id": "ANG-TXT-001-POR-01", "category": "portrait", "prompt_summary": "a", "seed": 1, "image_path": "a.png", "technical_status": "Pass", "visual_qc": "Unchecked", "full_body_face_eye_qc": "NA", "final_decision": "Recorded", "reviewer_note": ""},
                {"run_id": "ANG-TXT-001-FB-01", "category": "full_body", "prompt_summary": "b", "seed": 2, "image_path": "b.png", "technical_status": "Pass", "visual_qc": "Unchecked", "full_body_face_eye_qc": "Unchecked", "final_decision": "Recorded", "reviewer_note": ""},
                {"run_id": "ANG-TXT-001-ENV-01", "category": "environment", "prompt_summary": "c", "seed": 3, "image_path": "c.png", "technical_status": "Pass", "visual_qc": "Unchecked", "full_body_face_eye_qc": "NA", "final_decision": "Recorded", "reviewer_note": ""},
            ],
            "lifecycle_after_proposed": {"core_pipeline_status": "Production", "use_case_status": {"portrait": "Production Candidate", "full_body": "MVP", "environment": "MVP"}},
            "user_feedback": [],
            "known_risks": [],
            "next_actions": [],
        },
    )

    provider_mod.update_run_delivery_status(
        published_dir,
        delivery_result={
            "ANG-TXT-001-POR-01": "Pass",
            "ANG-TXT-001-FB-01": "Fail",
            "ANG-TXT-001-ENV-01": "Skipped",
        },
    )
    report_path = provider_mod.finalize_qualification_report_status(
        published_dir,
        delivery_result={
            "ANG-TXT-001-POR-01": "Pass",
            "ANG-TXT-001-FB-01": "Fail",
            "ANG-TXT-001-ENV-01": "Skipped",
        },
    )

    manifest = _read_json(published_dir / "run_manifest.json")
    report = _read_json(report_path)

    assert [entry["status"]["slack_status"] for entry in manifest["artifacts"]] == ["Pass", "Fail", "Skipped"]
    assert report["summary"]["slack_success_count"] == 1
    assert report["summary"]["slack_fail_count"] == 1
    assert report["summary"]["slack_pending_count"] == 0
    assert report["summary"]["slack_skipped_count"] == 1
