#!/usr/bin/env python3
"""Tests for the local Forge / A1111 image generation plugin."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock

FORGE_MOD = importlib.import_module("plugins.image_gen.forge-local")
ForgeLocalImageGenProvider = FORGE_MOD.ForgeLocalImageGenProvider

# 1×1 transparent PNG — valid bytes for save_b64_image().
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBAp+L"
    "8QAAAABJRU5ErkJggg=="
)


class TestForgeLocalImageGenProviderSurface:
    def test_name(self):
        assert ForgeLocalImageGenProvider().name == "forge-local"

    def test_display_name(self):
        assert ForgeLocalImageGenProvider().display_name == "Forge Local"

    def test_setup_schema(self):
        schema = ForgeLocalImageGenProvider().get_setup_schema()
        assert schema["name"] == "Forge Local"
        assert schema["badge"] == "local"
        assert schema["env_vars"] == []


class TestForgeLocalImageGenProviderAvailability:
    def test_is_available_true_when_sd_models_respond(self, monkeypatch):
        forge_mod = importlib.import_module("plugins.image_gen.forge-local")

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return [{"title": "Nullstyle_v20.safetensors", "model_name": "Nullstyle_v20"}]

        monkeypatch.setattr(forge_mod.requests, "get", lambda *a, **k: Response())
        assert ForgeLocalImageGenProvider().is_available() is True

    def test_is_available_false_on_error(self, monkeypatch):
        forge_mod = importlib.import_module("plugins.image_gen.forge-local")

        def boom(*args, **kwargs):
            raise RuntimeError("offline")

        monkeypatch.setattr(forge_mod.requests, "get", boom)
        assert ForgeLocalImageGenProvider().is_available() is False


class TestForgeLocalImageGenProviderGenerate:
    def test_generate_calls_txt2img_and_saves_artifact(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        forge_mod = importlib.import_module("plugins.image_gen.forge-local")
        from hermes_constants import get_hermes_work_dir

        captured = {}

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "images": [TINY_PNG_B64],
                    "parameters": captured.get("payload", {}),
                    "info": "ok",
                }

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            captured["timeout"] = timeout
            return Response()

        monkeypatch.setattr(forge_mod.requests, "post", fake_post)
        monkeypatch.setattr(forge_mod, "_resolve_base_url", lambda: "http://172.22.224.1:7860")
        monkeypatch.setattr(forge_mod, "_resolve_model", lambda: "Nullstyle_v20")

        result = ForgeLocalImageGenProvider().generate("a cute robot on a desk", aspect_ratio="square")

        assert captured["url"] == "http://172.22.224.1:7860/sdapi/v1/txt2img"
        assert captured["timeout"] == 300
        assert captured["payload"]["prompt"] == "a cute robot on a desk"
        assert captured["payload"]["width"] == 1024
        assert captured["payload"]["height"] == 1024
        assert captured["payload"]["override_settings"]["sd_model_checkpoint"] == "Nullstyle_v20"
        assert result["success"] is True
        assert result["provider"] == "forge-local"
        assert result["model"] == "Nullstyle_v20"
        assert result["local_path"] == result["image"]
        assert result["nas_status"] == "동기화 요청됨"
        assert result["slack_status"] == "완료"
        assert result["message"] == (
            f"로컬 저장: {result['image']}\n"
            "NAS 반영: 동기화 요청됨\n"
            "Slack 첨부: 완료"
        )
        assert result["image"].startswith(str(get_hermes_work_dir("Image")))
        assert Path(result["image"]).exists()

    def test_generate_publishes_into_provider_prefixed_image_tree_when_metadata_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        forge_mod = importlib.import_module("plugins.image_gen.forge-local")

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"images": [TINY_PNG_B64], "parameters": {}, "info": "ok"}

        monkeypatch.setattr(forge_mod.requests, "post", lambda *a, **k: Response())
        monkeypatch.setattr(forge_mod, "_resolve_base_url", lambda: "http://172.22.224.1:7860")
        monkeypatch.setattr(forge_mod, "_resolve_model", lambda: "Nullstyle_v20")

        result = ForgeLocalImageGenProvider().generate(
            "a cute robot on a desk",
            aspect_ratio="square",
        )

        assert result["success"] is True
        assert result["provider"] == "forge-local"
        assert result["model"] == "Nullstyle_v20"
        assert result["local_path"] == result["image"]
        assert result["nas_status"] == "동기화 요청됨"
        assert result["slack_status"] == "완료"
        assert result["image"].startswith(str(tmp_path / "HermesWork" / "Image"))
        assert Path(result["image"]).parent.name.endswith("_forge_test")
        assert Path(result["image"]).name == "forge_test_v1.png"

    def test_generate_publishes_into_project_scoped_image_tree(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        forge_mod = importlib.import_module("plugins.image_gen.forge-local")

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"images": [TINY_PNG_B64], "parameters": {}, "info": "ok"}

        monkeypatch.setattr(forge_mod.requests, "post", lambda *a, **k: Response())
        monkeypatch.setattr(forge_mod, "_resolve_base_url", lambda: "http://172.22.224.1:7860")
        monkeypatch.setattr(forge_mod, "_resolve_model", lambda: "Nullstyle_v20")

        result = ForgeLocalImageGenProvider().generate(
            "a cute robot on a desk",
            aspect_ratio="square",
            project_name="망각구역",
            artifact_name="scene",
        )

        assert result["success"] is True
        assert result["provider"] == "forge-local"
        assert result["model"] == "Nullstyle_v20"
        assert result["local_path"] == result["image"]
        assert result["nas_status"] == "동기화 요청됨"
        assert result["slack_status"] == "완료"
        assert result["image"].startswith(str(tmp_path / "HermesWork" / "Image"))
        assert Path(result["image"]).parent.name.endswith("_망각구역")
        assert Path(result["image"]).name == "scene_v1.png"

    def test_generate_invalid_prompt(self):
        result = ForgeLocalImageGenProvider().generate("   ")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_generate_handles_api_error(self, monkeypatch):
        forge_mod = importlib.import_module("plugins.image_gen.forge-local")

        class ErrResp:
            status_code = 500
            text = "boom"

            def json(self):
                return {"detail": "bad"}

        def fake_post(*args, **kwargs):
            exc = forge_mod.requests.HTTPError("bad")
            exc.response = ErrResp()
            raise exc

        monkeypatch.setattr(forge_mod.requests, "post", fake_post)
        monkeypatch.setattr(forge_mod, "_resolve_base_url", lambda: "http://172.22.224.1:7860")
        monkeypatch.setattr(forge_mod, "_resolve_model", lambda: "Nullstyle_v20")

        result = ForgeLocalImageGenProvider().generate("a test prompt")
        assert result["success"] is False
        assert result["error_type"] == "api_error"


class TestForgeLocalImageGenPluginRegistration:
    def test_register_wires_provider_into_registry(self):
        from agent import image_gen_registry

        image_gen_registry._reset_for_tests()
        ctx = MagicMock()
        FORGE_MOD.register(ctx)
        ctx.register_image_gen_provider.assert_called_once()
        (registered,), _ = ctx.register_image_gen_provider.call_args
        assert isinstance(registered, ForgeLocalImageGenProvider)
        assert image_gen_registry.get_provider("forge-local") is None

