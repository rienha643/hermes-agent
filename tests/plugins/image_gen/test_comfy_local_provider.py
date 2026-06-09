#!/usr/bin/env python3
"""Tests for the local ComfyUI image generation plugin."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from unittest.mock import MagicMock

COMFY_MOD = importlib.import_module("plugins.image_gen.comfy-local")
ComfyLocalImageGenProvider = COMFY_MOD.ComfyLocalImageGenProvider

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
        assert Path(result["workflow_path"]).parent.name == "_sidecars"
        assert Path(result["prompt_path"]).parent == Path(result["manifest_path"]).parent
        assert Path(result["metadata_path"]).parent == Path(result["manifest_path"]).parent
        assert result["primary_image"] == Path(result["image"]).name
        assert Path(result["workflow_path"]).name == result["sidecars"]["workflow"]
        assert result["media_files"] == [str(result["image"])]
        assert result["artifact_files"][0] == str(result["image"])
        assert result["artifact_files"][1] == str(result["workflow_path"])
        assert str(result["prompt_path"]) in result["artifact_files"]
        assert str(result["metadata_path"]) in result["artifact_files"]
        assert str(result["manifest_path"]) in result["artifact_files"]

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
            for p in [workflow, prompt, metadata, manifest, primary]:
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
                "primary_image": primary.name,
                "sidecars": {"workflow": workflow.name, "prompt": prompt.name, "metadata": metadata.name},
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

    def test_generate_fails_when_history_success_but_output_file_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "missing-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")

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

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(
            comfy_mod.requests,
            "post",
            lambda *a, **k: Response({"prompt_id": "pid-123", "number": 0, "node_errors": {}}),
        )
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)

        result = ComfyLocalImageGenProvider().generate("simple cute anime girl")
        assert result["success"] is False
        assert result["error_type"] == "io_error"

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
