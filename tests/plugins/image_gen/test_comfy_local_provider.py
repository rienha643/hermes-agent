#!/usr/bin/env python3
"""Tests for the local ComfyUI image generation plugin."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

COMFY_MOD = importlib.import_module("plugins.image_gen.comfy-local")
ComfyLocalImageGenProvider = COMFY_MOD.ComfyLocalImageGenProvider
resolve_checkpoint = COMFY_MOD.resolve_checkpoint

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de00000010494441547801635c0e000000feff03000006000557bfabd400"
    "00000049454e44ae426082"
)


class TestComfyLocalImageGenProviderSurface:
    def test_name(self):
        assert ComfyLocalImageGenProvider().name == "comfy-local"

    def test_display_name(self):
        assert ComfyLocalImageGenProvider().display_name == "Comfy Local"


class TestComfyLocalCheckpointResolution:
    CHECKPOINTS = [
        "animagine-xl-4.0-opt.safetensors",
        "catCarrier_v90.safetensors",
        "hakushiMix_v141.safetensors",
        "hdaRainbowIllusMixV1_v13.safetensors",
        "waiIllustriousSDXL_v170.safetensors",
    ]

    def test_exact_match_uses_remote_checkpoint_filename(self):
        result = resolve_checkpoint("catCarrier_v90.safetensors", self.CHECKPOINTS)

        assert result["ok"] is True
        assert result["resolved_checkpoint"] == "catCarrier_v90.safetensors"
        assert result["resolution_mode"] == "exact"
        assert result["candidate_count"] == 1
        assert result["candidates"] == ["catCarrier_v90.safetensors"]

    def test_extension_insensitive_match_accepts_name_without_suffix(self):
        result = resolve_checkpoint("animagine-xl-4.0-opt", self.CHECKPOINTS)

        assert result["ok"] is True
        assert result["resolved_checkpoint"] == "animagine-xl-4.0-opt.safetensors"
        assert result["resolution_mode"] == "extension_insensitive"
        assert result["candidate_count"] == 1

    def test_case_and_separator_normalized_match(self):
        result = resolve_checkpoint("CAT CARRIER V90", self.CHECKPOINTS)

        assert result["ok"] is True
        assert result["resolved_checkpoint"] == "catCarrier_v90.safetensors"
        assert result["resolution_mode"] == "normalized"
        assert result["candidate_count"] == 1

    def test_unique_partial_alias_match_for_animagine(self):
        result = resolve_checkpoint("animagine", self.CHECKPOINTS)

        assert result["ok"] is True
        assert result["resolved_checkpoint"] == "animagine-xl-4.0-opt.safetensors"
        assert result["resolution_mode"] == "unique_partial"
        assert result["candidate_count"] == 1

    def test_unique_partial_alias_match_for_wai(self):
        result = resolve_checkpoint("wai", self.CHECKPOINTS)

        assert result["ok"] is True
        assert result["resolved_checkpoint"] == "waiIllustriousSDXL_v170.safetensors"
        assert result["resolution_mode"] == "unique_partial"
        assert result["candidate_count"] == 1

    def test_unique_partial_alias_match_for_hakushi(self):
        result = resolve_checkpoint("hakushi", self.CHECKPOINTS)

        assert result["ok"] is True
        assert result["resolved_checkpoint"] == "hakushiMix_v141.safetensors"
        assert result["resolution_mode"] == "unique_partial"
        assert result["candidate_count"] == 1

    def test_ambiguous_partial_match_fails_with_candidates(self):
        result = resolve_checkpoint("illustrious", self.CHECKPOINTS)

        assert result["ok"] is False
        assert result["error_type"] == "checkpoint_ambiguous"
        assert result["resolution_mode"] == "ambiguous"
        assert result["candidate_count"] == 2
        assert result["candidates"] == ["hdaRainbowIllusMixV1_v13.safetensors", "waiIllustriousSDXL_v170.safetensors"]
        assert result["resolved_checkpoint"] is None

    def test_missing_checkpoint_resolves_to_smoke_e2e_default_without_blacklist(self):
        result = resolve_checkpoint(
            "gpt-image-2-medium",
            self.CHECKPOINTS,
            task_context="WINDOWS REMOTE COMFYUI fresh E2E smoke",
        )

        assert result["ok"] is True
        assert result["requested_checkpoint"] == "gpt-image-2-medium"
        assert result["resolved_checkpoint"] == "animagine-xl-4.0-opt.safetensors"
        assert result["resolution_mode"] == "default_for_smoke"
        assert result.get("source_model_rejected") is None
        assert result["candidate_count"] == 0
        assert result["candidates"] == []
        assert result["available_checkpoints_count"] == 5

    def test_missing_checkpoint_fails_for_general_user_task_with_candidates(self):
        result = resolve_checkpoint(
            "gpt-image-2-medium",
            self.CHECKPOINTS,
            task_context="custom user image generation",
        )

        assert result["ok"] is False
        assert result["error_type"] == "checkpoint_not_found"
        assert result["requested_checkpoint"] == "gpt-image-2-medium"
        assert result["resolved_checkpoint"] is None
        assert result["resolution_mode"] == "not_found"
        assert result["candidate_count"] == len(self.CHECKPOINTS)
        assert result["candidates"] == self.CHECKPOINTS

    def test_existing_checkpoint_is_used_even_if_name_looks_like_external_model(self):
        result = resolve_checkpoint(
            "gpt-image-2-medium.safetensors",
            ["animagine-xl-4.0-opt.safetensors", "gpt-image-2-medium.safetensors"],
            task_context="WINDOWS REMOTE COMFYUI fresh E2E smoke",
        )

        assert result["ok"] is True
        assert result["requested_checkpoint"] == "gpt-image-2-medium.safetensors"
        assert result["resolved_checkpoint"] == "gpt-image-2-medium.safetensors"
        assert result["resolution_mode"] == "exact"
        assert result.get("source_model_rejected") is None


class TestComfyLocalImageGenProviderGenerate:

    def test_generate_runs_history_gated_publish_and_writes_sidecars(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "angelica_smoke_00001_.png"
        output_file.write_bytes(PNG_1PX)

        calls = {"post": [], "get": []}

        class Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            calls["get"].append(url)
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["AOM3A1_orangemixs.safetensors", "Nullstyle_v20.safetensors"])
            if url.endswith("/models/vae"):
                return Response(["animevae.pt"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-123": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "8": {
                                    "images": [
                                        {"filename": "angelica_smoke_00001_.png", "subfolder": "", "type": "output"}
                                    ]
                                }
                            },
                        }
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, timeout=None):
            calls["post"].append((url, json, timeout))
            return Response({"prompt_id": "pid-123", "number": 0, "node_errors": {}})

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "simple cute anime girl, clean face, sharp eyes, game illustration style",
            aspect_ratio="square",
            project_name="angelica_smoke_test",
            artifact_name="angelica_smoke",
            vae="animevae.pt",
            negative_prompt="low quality, blurry, bad anatomy, text, watermark",
            width=512,
            height=512,
            seed=123456789,
            steps=12,
            cfg_scale=7,
            sampler_name="euler",
            denoise=1,
        )

        assert calls["post"][0][0] == "http://172.22.224.1:8188/prompt"
        workflow = calls["post"][0][1]["prompt"]
        assert workflow["1"]["inputs"]["ckpt_name"] == "AOM3A1_orangemixs.safetensors"
        assert workflow["6"]["inputs"]["vae_name"] == "animevae.pt"
        assert workflow["8"]["inputs"]["filename_prefix"] == "angelica_smoke"
        assert workflow["2"]["inputs"]["width"] == 512
        assert workflow["2"]["inputs"]["height"] == 512
        assert workflow["7"]["inputs"]["vae"] == ["6", 0]
        assert result["success"] is True
        assert result["provider"] == "comfy-local"
        assert result["model"] == "AOM3A1_orangemixs.safetensors"
        assert result["local_status"] == "생성 완료"
        assert result["publish_status"] == "HermesWork publish 완료"
        assert result["nas_status"] == "동기화 요청됨"
        assert result["slack_status"] == "primary image 준비됨"
        assert result["image"].startswith(str(tmp_path / "HermesWork" / "Image"))
        assert result["artifact_path"] == result["image"]
        assert Path(result["image"]).exists()
        assert Path(result["workflow_path"]).exists()
        assert Path(result["prompt_path"]).exists()
        assert Path(result["metadata_path"]).exists()
        assert Path(result["manifest_path"]).exists()
        assert Path(result["integrity_path"]).exists()
        assert Path(result["workflow_path"]).parent.name == "sidecar"
        assert not list(Path(result["image"]).parent.glob("*.json"))
        assert Path(result["prompt_path"]).parent == Path(result["manifest_path"]).parent
        assert Path(result["metadata_path"]).parent == Path(result["manifest_path"]).parent
        assert Path(result["integrity_path"]).parent == Path(result["manifest_path"]).parent
        assert result["primary_image"] == Path(result["image"]).name
        assert Path(result["workflow_path"]).name == result["sidecars"]["workflow"]
        assert result["media_files"] == [str(result["image"])]
        assert result["artifact_files"][0] == str(result["image"])
        assert result["artifact_files"][1] == str(result["workflow_path"])
        assert str(result["prompt_path"]) in result["artifact_files"]
        assert str(result["metadata_path"]) in result["artifact_files"]
        assert str(result["manifest_path"]) in result["artifact_files"]
        assert str(result["integrity_path"]) in result["artifact_files"]
        assert result["file_sha256"] == json.loads(Path(result["integrity_path"]).read_text())["primary_image_sha256"]
        assert json.loads(Path(result["manifest_path"]).read_text())["integrity"]["status"] == "Pass"
        assert result["nas_hook_requested"] is True
        assert result["nas_evidence"]["hook_requested"] is True
        assert result["nas_evidence"]["mirror_verified"] is False
        assert result["nas_evidence"]["mirror_path"] is None
        assert result["slack_upload_evidence"] is False
        assert "tmp/windows_remote_smoke" not in result["artifact_path"]

    def test_generate_resolves_missing_smoke_e2e_checkpoint_to_remote_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "fresh_e2e_00001_.png").write_bytes(PNG_1PX)
        calls = {"post": []}

        class Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["animagine-xl-4.0-opt.safetensors", "other.safetensors"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-123": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {"8": {"images": [{"filename": "fresh_e2e_00001_.png", "subfolder": "", "type": "output"}]}},
                        }
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, timeout=None):
            calls["post"].append((url, json, timeout))
            return Response({"prompt_id": "pid-123", "number": 0, "node_errors": {}})

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "fresh e2e smoke image",
            aspect_ratio="square",
            model="gpt-image-2-medium",
            project_name="windows_remote_comfyui_fresh_e2e",
            artifact_name="fresh_e2e",
            task_context="WINDOWS REMOTE COMFYUI fresh E2E smoke",
        )

        assert result["success"] is True
        assert calls["post"][0][1]["prompt"]["1"]["inputs"]["ckpt_name"] == "animagine-xl-4.0-opt.safetensors"
        assert result["requested_checkpoint"] == "gpt-image-2-medium"
        assert result["resolved_checkpoint"] == "animagine-xl-4.0-opt.safetensors"
        assert result["resolution_mode"] == "default_for_smoke"
        assert result["resolution_reason"] == "default_for_smoke"
        assert result.get("source_model_rejected") is None
        assert result["candidate_count"] == 0
        assert result["candidates"] == []
        metadata = json.loads(Path(result["metadata_path"]).read_text())
        assert metadata["requested_checkpoint"] == "gpt-image-2-medium"
        assert metadata["resolved_checkpoint"] == "animagine-xl-4.0-opt.safetensors"
        assert metadata["resolution_mode"] == "default_for_smoke"
        assert metadata.get("source_model_rejected") is None
        assert metadata["candidate_count"] == 0
        assert metadata["candidates"] == []

    def test_generate_fails_missing_general_checkpoint_without_blacklist(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        calls = {"post": []}

        class Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["animagine-xl-4.0-opt.safetensors", "other.safetensors"])
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, timeout=None):
            calls["post"].append((url, json, timeout))
            raise AssertionError("Prompt submit must not happen when general checkpoint is missing")

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)

        result = ComfyLocalImageGenProvider().generate(
            "custom user image",
            aspect_ratio="square",
            model="gpt-image-2-medium",
            project_name="custom_user_project",
            artifact_name="custom_user_image",
        )

        assert result["success"] is False
        assert result["error_type"] == "checkpoint_not_found"
        assert result["requested_checkpoint"] == "gpt-image-2-medium"
        assert result["resolved_checkpoint"] is None
        assert calls["post"] == []

    def test_generate_uses_checkpoint_vae_fallback_when_vae_not_configured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "smoke_test_no_vae_00001_.png"
        output_file.write_bytes(PNG_1PX)

        calls = {"post": [], "get": []}

        class Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            calls["get"].append(url)
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["AOM3A1_orangemixs.safetensors", "Nullstyle_v20.safetensors"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-123": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "8": {
                                    "images": [
                                        {
                                            "filename": "smoke_test_no_vae_00001_.png",
                                            "subfolder": "",
                                            "type": "output",
                                        }
                                    ]
                                }
                            },
                        }
                    }
                )
            if url.endswith("/models/vae"):
                raise AssertionError("VAE endpoint should not be queried when VAE is not configured")
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, timeout=None):
            calls["post"].append((url, json, timeout))
            return Response({"prompt_id": "pid-123", "number": 0, "node_errors": {}})

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "simple cute anime girl, clean face, sharp eyes, game illustration style",
            aspect_ratio="square",
            project_name="angelica_smoke_test",
            artifact_name="smoke_test_no_vae",
            negative_prompt="low quality, blurry",
            width=512,
            height=512,
        )

        assert calls["post"][0][0] == "http://172.22.224.1:8188/prompt"
        workflow = calls["post"][0][1]["prompt"]
        assert workflow["1"]["inputs"]["ckpt_name"] == "AOM3A1_orangemixs.safetensors"
        assert "6" not in workflow
        assert workflow["7"]["inputs"]["vae"] == ["1", 2]
        assert not any("/models/vae" in url for url in calls["get"])
        assert result["success"] is True
        assert result["prompt_id"] == "pid-123"

    def test_generate_fails_when_configured_vae_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "smoke_test_missing_vae_00001_.png"
        output_file.write_bytes(PNG_1PX)

        class Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["AOM3A1_orangemixs.safetensors", "Nullstyle_v20.safetensors"])
            if url.endswith("/models/vae"):
                return Response(["animevae.pt"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-123": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "8": {
                                    "images": [
                                        {
                                            "filename": "smoke_test_missing_vae_00001_.png",
                                            "subfolder": "",
                                            "type": "output",
                                        }
                                    ]
                                }
                            },
                        }
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, timeout=None):
            return Response({"prompt_id": "pid-123", "number": 0, "node_errors": {}})

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)

        result = ComfyLocalImageGenProvider().generate(
            "simple cute anime girl, clean face, sharp eyes, game illustration style",
            vae="missing_vae.pt",
            aspect_ratio="square",
            project_name="angelica_smoke_test",
            artifact_name="smoke_test_missing_vae",
        )

        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert result["error"] == "Requested VAE not found in ComfyUI: missing_vae.pt"

    def test_generate_finds_output_via_fallback_output_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        # Simulate stale config: configured dir has no image, real output is elsewhere.
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "configured-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        configured_output = Path(tmp_path / "configured-output")
        fallback_output = Path(tmp_path / "real-output")
        configured_output.mkdir(parents=True, exist_ok=True)
        fallback_output.mkdir(parents=True, exist_ok=True)
        output_file = fallback_output / "smoke_test_retry_00001_.png"
        output_file.write_bytes(PNG_1PX)

        calls = []
        published_dir = tmp_path / "HermesWork" / "Image" / "angelica_smoke_test"
        published_dir.mkdir(parents=True, exist_ok=True)
        primary = published_dir / "angelica_smoke_v1.png"

        class Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["AOM3A1_orangemixs.safetensors", "Nullstyle_v20.safetensors"])
            if url.endswith("/models/vae"):
                return Response(["animevae.pt"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-123": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {"8": {"images": [{"filename": "smoke_test_retry_00001_.png", "subfolder": "", "type": "output"}]}}},
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, timeout=None):
            return Response({"prompt_id": "pid-123", "number": 0, "node_errors": {}})

        def fake_publish(source_path, **kwargs):
            calls.append(source_path)
            workflow = published_dir / "angelica_smoke_v1.workflow.json"
            prompt = published_dir / "angelica_smoke_v1.prompt.json"
            metadata = published_dir / "angelica_smoke_v1.metadata.json"
            manifest = published_dir / "manifest.json"
            integrity = published_dir / "integrity.json"
            for p in [workflow, prompt, metadata, manifest, integrity, primary]:
                if p.name == primary.name:
                    primary.write_bytes(PNG_1PX)
                else:
                    p.write_text("{}", encoding="utf-8")
            return {
                "project_id": "angelica_smoke_test",
                "published_dir": published_dir,
                "primary_image_path": primary,
                "workflow_path": workflow,
                "prompt_path": prompt,
                "metadata_path": metadata,
                "manifest_path": manifest,
                "integrity_path": integrity,
                "primary_image": primary.name,
                "sidecars": {"workflow": workflow.name, "prompt": prompt.name, "metadata": metadata.name, "manifest": manifest.name, "integrity": integrity.name},
                "file_sha256": "test-sha256",
                "nas_hook_requested": True,
            }

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            comfy_mod,
            "_candidate_output_dirs",
            lambda: [configured_output, fallback_output],
        )
        monkeypatch.setattr(comfy_mod, "publish_filesystem_image_bundle", fake_publish)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "simple cute anime girl, clean face, sharp eyes, game illustration style",
            aspect_ratio="square",
            project_name="angelica_smoke_test",
            artifact_name="angelica_smoke",
        )

        assert result["success"] is True
        assert calls == [output_file]

    def test_generate_warns_and_errors_when_publish_bundle_fails(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "angelica_smoke_00001_.png"
        output_file.write_bytes(PNG_1PX)

        class Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["AOM3A1_orangemixs.safetensors"])
            if url.endswith("/models/vae"):
                return Response(["animevae.pt"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-123": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "8": {"images": [{"filename": "angelica_smoke_00001_.png", "subfolder": "", "type": "output"}]}
                            },
                        }
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(
            comfy_mod.requests,
            "post",
            lambda *a, **k: Response({"prompt_id": "pid-123", "number": 0, "node_errors": {}}),
        )
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            comfy_mod,
            "publish_filesystem_image_bundle",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NAS unavailable")),
        )

        with caplog.at_level(logging.WARNING):
            result = ComfyLocalImageGenProvider().generate(
                "simple cute anime girl",
                aspect_ratio="square",
                project_name="angelica_smoke_test",
                artifact_name="angelica_smoke",
            )

        assert result["success"] is False
        assert result["error_type"] == "io_error"
        assert any("Could not publish ComfyUI image bundle to HermesWork" in rec.getMessage() for rec in caplog.records)

    def test_generate_downloads_remote_view_when_history_output_file_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "missing-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        get_urls = []

        class Response:
            def __init__(self, payload=None, content=b""):
                self._payload = payload
                self.content = content

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            get_urls.append(url)
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["AOM3A1_orangemixs.safetensors"])
            if url.endswith("/models/vae"):
                return Response(["animevae.pt"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-123": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "8": {
                                    "images": [
                                        {"filename": "angelica_smoke_00001_.png", "subfolder": "", "type": "output"}
                                    ]
                                }
                            },
                        }
                    }
                )
            if "/view?" in url:
                return Response(content=PNG_1PX)
            raise AssertionError(f"unexpected GET {url}")

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(
            comfy_mod.requests,
            "post",
            lambda *a, **k: Response({"prompt_id": "pid-123", "number": 0, "node_errors": {}}),
        )
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "simple cute anime girl",
            project_name="remote_view_test",
            artifact_name="remote_view_artifact",
        )

        assert result["success"] is True
        assert any(url == "http://172.22.224.1:8188/view?filename=angelica_smoke_00001_.png&type=output&subfolder=" for url in get_urls)
        assert result["output_source_origin"] == "remote_view_download"
        assert result["image"].startswith(str(tmp_path / "HermesWork" / "Image"))
        assert "/tmp/windows_remote_smoke" not in result["artifact_path"]
        assert result["media_files"] == [result["image"]]
        assert Path(result["integrity_path"]).exists()
        assert json.loads(Path(result["manifest_path"]).read_text())["integrity"]["primary_image_sha256"] == result["file_sha256"]
        assert result["nas_hook_requested"] is True

    def test_generate_writes_run_manifest_and_qualification_report_when_requested(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "ang_txt_001_por_01_00001_.png"
        output_file.write_bytes(PNG_1PX)

        class Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["AOM3A1_orangemixs.safetensors"])
            if url.endswith("/models/vae"):
                return Response(["animevae.pt"])
            if "/history/" in url:
                return Response({"pid-123": {"status": {"completed": True, "status_str": "success"}, "outputs": {"8": {"images": [{"filename": "ang_txt_001_por_01_00001_.png", "subfolder": "", "type": "output"}]}}}})
            raise AssertionError(f"unexpected GET {url}")

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", lambda *a, **k: Response({"prompt_id": "pid-123", "number": 0, "node_errors": {}}))
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "anime portrait of a young heroine",
            aspect_ratio="square",
            project_name="ANG_TXT_001_qualification",
            artifact_name="ang_txt_001_por_01",
            negative_prompt="blurry",
            qualification_context={
                "workflow_code": "ANG-TXT-001",
                "workflow_name": "TXT2IMG Basic",
                "run_kind": "qualification",
                "summary": {"total_runs": 1, "artifact_count": 1},
                "report": {
                    "workflow_code": "ANG-TXT-001",
                    "workflow_name": "TXT2IMG Basic",
                    "test_name": "ANG-TXT-001 Production Qualification Test",
                    "run_date": "2026-06-05",
                    "production_gate_result": "Hold",
                    "summary": {"total_runs": 1, "technical_pass_count": 1, "visual_pass_count": 1, "visual_warning_count": 0, "visual_fail_count": 0, "full_body_face_eye_fail_count": 0, "publish_success_count": 1, "nas_hook_success_count": 1, "slack_success_count": 1},
                    "runs": [],
                    "lifecycle_after_proposed": {"core_pipeline_status": "Production", "use_case_status": {"portrait": "Production Candidate", "full_body": "MVP", "environment": "MVP"}},
                    "user_feedback": [],
                    "known_risks": [],
                    "next_actions": [],
                },
            },
        )

        assert Path(result["run_manifest_path"]).exists()
        assert Path(result["qualification_report_path"]).exists()


class TestComfyLocalCharacterProductionPreset:
    def _run_generate(
        self,
        monkeypatch,
        tmp_path,
        *,
        prompt,
        negative_prompt=None,
        subject_dominance=None,
        model="AOM3A1_orangemixs.safetensors",
        task_context="character production",
        output_filename="heroine_00001_.png",
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / output_filename
        output_file.write_bytes(PNG_1PX)

        captured = {}

        class Response:
            def __init__(self, payload=None, content=None):
                self._payload = payload
                self.content = content or b""

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["AOM3A1_orangemixs.safetensors", "animagine-xl-4.0-opt.safetensors"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-123": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "8": {
                                    "images": [
                                        {"filename": output_filename, "subfolder": "", "type": "output"}
                                    ]
                                }
                            },
                        }
                    }
                )
            if "/view?" in url:
                return Response(content=PNG_1PX)
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, timeout=None):
            assert url == "http://172.22.224.1:8188/prompt"
            return Response({"prompt_id": "pid-123", "number": 0, "node_errors": {}})

        def fake_publish(source, *, prefix, project_name, artifact_name, category, workflow_json, prompt_payload, metadata):
            captured.update(
                {
                    "source": source,
                    "prefix": prefix,
                    "project_name": project_name,
                    "artifact_name": artifact_name,
                    "category": category,
                    "workflow_json": workflow_json,
                    "prompt_payload": prompt_payload,
                    "metadata": metadata,
                }
            )
            published_dir = tmp_path / "HermesWork" / "Image" / "heroine_project"
            return {
                "project_id": "project-1",
                "published_dir": published_dir,
                "primary_image_path": source,
                "workflow_path": tmp_path / "workflow.json",
                "prompt_path": tmp_path / "prompt.json",
                "metadata_path": tmp_path / "metadata.json",
                "manifest_path": tmp_path / "manifest.json",
                "integrity_path": tmp_path / "integrity.json",
                "primary_image": source.name,
                "sidecars": {
                    "workflow": "workflow.json",
                    "prompt": "prompt.json",
                    "metadata": "metadata.json",
                    "manifest": "manifest.json",
                    "integrity": "integrity.json",
                    "dir": str(tmp_path / "sidecar"),
                },
                "file_sha256": "sha256",
                "storage_verification": {"ok": True},
                "sidecar_dir": tmp_path / "sidecar",
                "nas_hook_requested": True,
                "nas_evidence": {
                    "hook_requested": True,
                    "mirror_verified": False,
                    "mirror_path": None,
                    "hook_log_path": str(tmp_path / "nas.log"),
                },
            }

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(comfy_mod, "publish_filesystem_image_bundle", fake_publish)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = comfy_mod.ComfyLocalImageGenProvider().generate(
            prompt,
            aspect_ratio="portrait",
            model=model,
            project_name="heroine_project",
            artifact_name="heroine_art",
            negative_prompt=negative_prompt,
            subject_dominance=subject_dominance,
            task_context=task_context,
        )
        return result, captured

    def test_generate_applies_character_production_runtime_preset(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt="캐릭터 미소녀 전신 RPG 수집형 heroine gacha full body",
            negative_prompt="blurry, watermark",
            subject_dominance=80,
        )

        assert result["success"] is True
        assert captured["prompt_payload"]["runtime_preset"] == "character_production"
        assert captured["prompt_payload"]["steps"] == 28
        assert captured["prompt_payload"]["cfg"] == 5.0
        assert captured["prompt_payload"]["sampler"] == "dpmpp_2m"
        assert captured["prompt_payload"]["scheduler"] == "karras"
        assert captured["prompt_payload"]["subject_dominance"] == 80.0
        assert "single character" in captured["prompt_payload"]["subject_dominance_rule"]
        assert "character" in captured["workflow_json"]["3"]["inputs"]["text"].casefold()
        assert "beautiful girl" in captured["workflow_json"]["3"]["inputs"]["text"].casefold()
        assert "full body" in captured["workflow_json"]["3"]["inputs"]["text"].casefold()
        prompt_text = captured["workflow_json"]["3"]["inputs"]["text"]
        for term in ("1girl", "solo", "looking at viewer", "detailed eyes", "RPG protagonist", "safe", "masterpiece"):
            assert term in prompt_text
        negative_text = captured["workflow_json"]["4"]["inputs"]["text"]
        for term in ("low quality", "bad anatomy", "unreadable face", "background focus", "covered face", "blurry", "watermark"):
            assert term in negative_text
        assert result["preset"] == "character_production"
        assert result["prompt_translation_policy"] == "character-skeleton + keyword-translate + subject-dominance guidance"
        assert result["steps"] == 28
        assert result["cfg_scale"] == 5.0
        assert result["sampler_name"] == "dpmpp_2m"
        assert result["scheduler"] == "karras"

    def test_generate_injects_negative_baseline_when_character_negative_prompt_missing(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt="캐릭터 전신 heroine gacha",
            subject_dominance=80,
        )

        assert result["success"] is True
        assert captured["prompt_payload"]["negative_prompt"] == COMFY_MOD.CHARACTER_PRODUCTION_NEGATIVE_BASELINE
        assert captured["prompt_payload"]["negative_prompt"] == captured["metadata"]["negative_prompt"]
        assert captured["workflow_json"]["4"]["inputs"]["text"] == captured["prompt_payload"]["negative_prompt"]
        assert result["negative_prompt"] == captured["prompt_payload"]["negative_prompt"]

    def test_generate_defaults_character_subject_dominance_to_80_when_missing(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt="캐릭터 전신 heroine gacha",
            negative_prompt="blurry",
        )

        assert result["success"] is True
        assert captured["prompt_payload"]["subject_dominance"] == 80.0
        assert captured["metadata"]["subject_dominance"] == 80.0
        assert "single character" in captured["prompt_payload"]["subject_dominance_rule"]
        assert result["subject_dominance"] == 80.0
        assert result["subject_dominance_rule"] == captured["prompt_payload"]["subject_dominance_rule"]

    def test_generate_keeps_smoke_e2e_default_separate_from_character_production(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt="캐릭터 전신 heroine gacha",
            negative_prompt="blurry",
            subject_dominance=80,
            task_context="fresh E2E smoke verification",
        )

        assert result["success"] is True
        assert captured["prompt_payload"]["resolution_mode"] == "exact"
        assert captured["prompt_payload"]["resolved_checkpoint"] == "AOM3A1_orangemixs.safetensors"
        assert captured["metadata"]["preset"] == "character_production"
        assert result["resolution_mode"] == "exact"
        assert result["resolved_checkpoint"] == "AOM3A1_orangemixs.safetensors"


class TestComfyLocalPluginRegistration:
    def test_register_wires_provider_into_registry(self):
        from agent import image_gen_registry

        image_gen_registry._reset_for_tests()
        ctx = MagicMock()
        COMFY_MOD.register(ctx)
        ctx.register_image_gen_provider.assert_called_once()
        (registered,), _ = ctx.register_image_gen_provider.call_args
        assert isinstance(registered, ComfyLocalImageGenProvider)
        assert image_gen_registry.get_provider("comfy-local") is None
