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
    assert bundle["workflow_path"].name == "angelica_smoke_v1.workflow.json"
    assert bundle["prompt_path"].name == "angelica_smoke_v1.prompt.json"
    assert bundle["metadata_path"].name == "angelica_smoke_v1.metadata.json"
    assert bundle["manifest_path"].name == "manifest.json"
    assert bundle["nas_hook_requested"] is True

    workflow = _read_json(bundle["workflow_path"])
    prompt_payload = _read_json(bundle["prompt_path"])
    metadata = _read_json(bundle["metadata_path"])
    manifest = _read_json(bundle["manifest_path"])

    assert workflow["8"]["class_type"] == "SaveImage"
    assert prompt_payload["seed"] == 123
    assert metadata["published_primary_path"] == str(bundle["primary_image_path"])
    assert metadata["published_dir"] == str(expected_dir)
    assert metadata["nas_hook_requested"] is True
    assert manifest["primary_image"] == "angelica_smoke_v1.png"
    assert "angelica_smoke_v1.workflow.json" in manifest["files"]
    assert manifest["sidecars"]["metadata"] == "angelica_smoke_v1.metadata.json"

    assert hook_calls == [
        {
            "category": "image",
            "scope": expected_dir.name,
            "artifact_path": bundle["primary_image_path"],
            "source_root": expected_dir,
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
