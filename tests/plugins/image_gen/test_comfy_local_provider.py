#!/usr/bin/env python3
"""Tests for the local ComfyUI image generation plugin."""

from __future__ import annotations

import importlib
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
        assert result["success"] is True
        assert result["provider"] == "comfy-local"
        assert result["model"] == "AOM3A1_orangemixs.safetensors"
        assert result["local_status"] == "생성 완료"
        assert result["publish_status"] == "HermesWork publish 완료"
        assert result["nas_status"] == "동기화 요청됨"
        assert result["slack_status"] == "primary image 준비됨"
        assert result["image"].startswith(str(tmp_path / "HermesWork" / "Image"))
        assert Path(result["image"]).exists()
        assert Path(result["workflow_path"]).exists()
        assert Path(result["prompt_path"]).exists()
        assert Path(result["metadata_path"]).exists()
        assert Path(result["manifest_path"]).exists()
        assert result["primary_image"] == Path(result["image"]).name
        assert result["sidecars"]["workflow"] == Path(result["workflow_path"]).name

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
