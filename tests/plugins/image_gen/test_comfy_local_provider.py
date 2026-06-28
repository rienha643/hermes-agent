#!/usr/bin/env python3
"""Tests for the local ComfyUI image generation plugin."""

from __future__ import annotations

import importlib
import json
import logging
import unicodedata
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


class TestComfyLocalSourceImageResolution:
    def test_resolves_korean_artifact_typo_by_unique_date_version(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        actual_dir = tmp_path / "HermesWork" / "Image" / "260624_이미지"
        actual_dir.mkdir(parents=True)
        actual = actual_dir / "이미지_v19.png"
        actual.write_bytes(PNG_1PX)

        requested = tmp_path / "HermesWork" / "Image" / "260624_이피치" / "이피치_v19.png"
        resolved, evidence = COMFY_MOD._resolve_existing_source_image_path(requested)

        assert resolved == actual
        assert evidence["source_path_resolution"] == "unique_date_version_match"
        assert evidence["requested_source_image_path"] == str(requested)
        assert evidence["resolved_source_image_path"] == str(actual)

    def test_resolves_unicode_decomposed_filename_variant(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        actual_dir = tmp_path / "HermesWork" / "Image" / "260624_이미지"
        actual_dir.mkdir(parents=True)
        actual = actual_dir / "이미지_v21.png"
        actual.write_bytes(PNG_1PX)

        requested = actual_dir / "이미지_v21.png"
        resolved, evidence = COMFY_MOD._resolve_existing_source_image_path(requested)

        assert resolved is not None
        assert resolved.is_file()
        assert unicodedata.normalize("NFC", resolved.name) == "이미지_v21.png"
        assert evidence["source_path_resolution"] in {
            "exact",
            "unicode_normalized",
            "sibling_unicode_normalized",
            "unique_date_version_match",
        }

    def test_resolves_worker_inserted_underscore_filename_variant(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        actual_dir = tmp_path / "HermesWork" / "Image" / "260628_selected_candidate3_source"
        actual_dir.mkdir(parents=True)
        actual = actual_dir / "candidate3_source.png"
        actual.write_bytes(PNG_1PX)

        requested = actual_dir / "candidate_3_source.png"
        resolved, evidence = COMFY_MOD._resolve_existing_source_image_path(requested)

        assert resolved == actual
        assert evidence["requested_source_image_path"] == str(requested)
        assert evidence["resolved_source_image_path"] == str(actual)
        assert evidence["source_path_resolution"] == "sibling_loose_filename_match"


class TestComfyLocalSourceTaskPromptGuard:
    def test_rejects_source_image_task_text_as_plain_txt2img_prompt(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")

        def fail_get(*_args, **_kwargs):
            raise AssertionError("source-image prompt-only guard should run before ComfyUI GET")

        def fail_post(*_args, **_kwargs):
            raise AssertionError("source-image prompt-only guard should run before ComfyUI POST")

        monkeypatch.setattr(comfy_mod.requests, "get", fail_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fail_post)

        result = ComfyLocalImageGenProvider().generate(
            'source_image_path: /Volumes/SSD_Hermes/HermesWork/Image/260624_이피치/이피치_v19.png, '
            'operation: "source_preserving_postprocess", title: "restore v19"',
            aspect_ratio="portrait",
        )

        assert result["success"] is False
        assert result["error_type"] == "source_image_task_prompt_only"
        assert result["detected_source_image_task_prompt_only"] is True
        assert result["completion_report_forbidden"] is True


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


class TestComfyLocalLoraPresetResolution:
    def test_character_production_uses_stable_lora_by_default(self):
        stack = COMFY_MOD._resolve_lora_stack({}, runtime_preset={"preset": "character_production"})

        assert stack == [
            {
                "preset": "stable",
                "name": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA,
                "weight": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA_WEIGHT,
                "use_case": "default user-approved subculture character illustration",
                "clip_weight": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA_WEIGHT,
            }
        ]

    def test_portrait_production_uses_user_selected_f_stack_by_default(self):
        stack = COMFY_MOD._resolve_lora_stack({}, runtime_preset={"preset": "portrait_production"})

        assert stack == [
            {
                "preset": "portrait_primary",
                "name": COMFY_MOD.DEFAULT_VIDEO_SOURCE_STYLE_LORA,
                "weight": COMFY_MOD.DEFAULT_VIDEO_SOURCE_STYLE_LORA_WEIGHT,
                "use_case": "user-selected top portrait/close-up style baseline from 2026-06-22 F candidate",
                "clip_weight": COMFY_MOD.DEFAULT_VIDEO_SOURCE_STYLE_LORA_WEIGHT,
            },
            {
                "preset": "portrait_primary_detail",
                "name": COMFY_MOD.DEFAULT_ADD_MICRO_DETAILS_LORA,
                "weight": COMFY_MOD.DEFAULT_ADD_MICRO_DETAILS_LORA_WEIGHT,
                "use_case": "portrait/close-up micro-detail companion from WAI cross-format confirmation",
                "clip_weight": COMFY_MOD.DEFAULT_ADD_MICRO_DETAILS_LORA_WEIGHT,
            },
        ]

    def test_key_art_style_preset_uses_pornmaster_key_visual_weight(self):
        stack = COMFY_MOD._resolve_lora_stack({"style_preset": "key_art"}, runtime_preset=None)

        assert stack == [
            {
                "preset": "key_art",
                "name": COMFY_MOD.DEFAULT_KEY_ART_LORA,
                "weight": COMFY_MOD.DEFAULT_KEY_ART_LORA_WEIGHT,
                "use_case": "key visual or intentional image distortion effect",
                "clip_weight": COMFY_MOD.DEFAULT_KEY_ART_LORA_WEIGHT,
            }
        ]

    def test_matte_skin_style_preset_is_retained_as_deprecated_review_lora(self):
        stack = COMFY_MOD._resolve_lora_stack({"style_preset": "matte_skin"}, runtime_preset=None)

        assert stack == [
            {
                "preset": "stable",
                "name": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA,
                "weight": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA_WEIGHT,
                "use_case": "default user-approved subculture character illustration",
                "clip_weight": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA_WEIGHT,
            },
            {
                "preset": "matte_skin",
                "name": COMFY_MOD.DEFAULT_MATTE_PRODUCTION_LORA,
                "weight": COMFY_MOD.DEFAULT_MATTE_PRODUCTION_LORA_WEIGHT,
                "use_case": "deprecated/review: matte lighting may remove too much illumination",
                "clip_weight": COMFY_MOD.DEFAULT_MATTE_PRODUCTION_LORA_WEIGHT,
            }
        ]

    def test_selected_lora_style_presets_resolve_user_verdict_candidates(self):
        cases = [
            (
                "eye_detail",
                "eye_detail",
                COMFY_MOD.DEFAULT_EYE_DETAIL_LORA,
                COMFY_MOD.DEFAULT_EYE_DETAIL_LORA_WEIGHT,
                "review/no-op alone in template check; prefer eye_gloss when eye improvement is desired",
            ),
            (
                "detail_smooth",
                "detail_smooth",
                COMFY_MOD.DEFAULT_DETAIL_SMOOTH_LORA,
                COMFY_MOD.DEFAULT_DETAIL_SMOOTH_LORA_WEIGHT,
                "optional outfit/material detail; do not default because face/character identity may drift",
            ),
            (
                "detail_enhancer",
                "detail_enhancer",
                COMFY_MOD.DEFAULT_DETAIL_ENHANCER_LORA,
                COMFY_MOD.DEFAULT_DETAIL_ENHANCER_LORA_WEIGHT,
                "review/rejected for default: worst in second character recheck",
            ),
            (
                "glossy_skin",
                "glossy_skin",
                COMFY_MOD.DEFAULT_GLOSSY_SKIN_LORA,
                COMFY_MOD.DEFAULT_GLOSSY_SKIN_LORA_WEIGHT,
                "best skin/outfit gloss candidate",
            ),
            (
                "video_source",
                "video_source",
                COMFY_MOD.DEFAULT_VIDEO_SOURCE_STYLE_LORA,
                COMFY_MOD.DEFAULT_VIDEO_SOURCE_STYLE_LORA_WEIGHT,
                "best hand/lighting candidate; stable style candidate for image-to-video source material",
            ),
            (
                "k_nai",
                "video_source",
                COMFY_MOD.DEFAULT_VIDEO_SOURCE_STYLE_LORA,
                COMFY_MOD.DEFAULT_VIDEO_SOURCE_STYLE_LORA_WEIGHT,
                "best hand/lighting candidate; stable style candidate for image-to-video source material",
            ),
            (
                "dnf_anima_experimental",
                "dnf_anima_experimental",
                COMFY_MOD.DEFAULT_DNF_ANIMA_EXPERIMENTAL_LORA,
                COMFY_MOD.DEFAULT_DNF_ANIMA_EXPERIMENTAL_LORA_WEIGHT,
                "experimental alternate taste; not a default preset",
            ),
        ]

        for requested, preset, name, weight, use_case in cases:
            stack = COMFY_MOD._resolve_lora_stack({"style_preset": requested}, runtime_preset=None)
            expected = [
                {
                    "preset": "stable",
                    "name": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA,
                    "weight": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA_WEIGHT,
                    "use_case": "default user-approved subculture character illustration",
                    "clip_weight": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA_WEIGHT,
                },
                {
                    "preset": preset,
                    "name": name,
                    "weight": weight,
                    "use_case": use_case,
                    "clip_weight": weight,
                },
            ]
            if requested in {"video_source", "k_nai", "dnf_anima_experimental"}:
                expected = [expected[1]]

            assert stack == expected

    def test_eye_gloss_style_preset_matches_selected_composite_stack(self):
        stack = COMFY_MOD._resolve_lora_stack({"style_preset": "eye_gloss"}, runtime_preset=None)

        assert stack == [
            {
                "preset": "stable",
                "name": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA,
                "weight": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA_WEIGHT,
                "use_case": "default user-approved subculture character illustration",
                "clip_weight": COMFY_MOD.DEFAULT_STABLE_STYLE_LORA_WEIGHT,
            },
            {
                "preset": "eye_detail",
                "name": COMFY_MOD.DEFAULT_EYE_DETAIL_LORA,
                "weight": COMFY_MOD.DEFAULT_EYE_DETAIL_LORA_WEIGHT,
                "use_case": "eye detail component for eye_gloss composite",
                "clip_weight": COMFY_MOD.DEFAULT_EYE_DETAIL_LORA_WEIGHT,
            },
            {
                "preset": "glossy_skin",
                "name": COMFY_MOD.DEFAULT_GLOSSY_SKIN_LORA,
                "weight": COMFY_MOD.DEFAULT_GLOSSY_SKIN_LORA_WEIGHT,
                "use_case": "gloss/skin component for eye_gloss composite",
                "clip_weight": COMFY_MOD.DEFAULT_GLOSSY_SKIN_LORA_WEIGHT,
            },
        ]

    def test_explicit_lora_name_overrides_style_preset(self):
        stack = COMFY_MOD._resolve_lora_stack(
            {
                "style_preset": "key_art",
                "lora_name": "custom.safetensors",
                "lora_weight": 0.25,
                "lora_clip_weight": 0.2,
            },
            runtime_preset={"preset": "character_production"},
        )

        assert stack == [
            {
                "preset": "key_art",
                "name": "custom.safetensors",
                "weight": 0.25,
                "clip_weight": 0.2,
                "use_case": "explicit lora",
            }
        ]


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

    def test_generate_source_preserving_face_hand_postprocess_uses_loadimage_workflow(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        source_image = tmp_path / "source.png"
        source_image.write_bytes(PNG_1PX)
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "postprocess_00001_.png"
        output_file.write_bytes(PNG_1PX)

        calls = {"post": [], "get": []}

        class Response:
            def __init__(self, payload, content=b""):
                self._payload = payload
                self.content = content

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            calls["get"].append(url)
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["pornmasterAnime_ilV5.safetensors"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-post": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "99": {
                                    "images": [
                                        {"filename": "postprocess_00001_.png", "subfolder": "", "type": "output"}
                                    ]
                                }
                            },
                        }
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, files=None, data=None, timeout=None):
            calls["post"].append((url, json, files, data, timeout))
            if url.endswith("/upload/image"):
                assert files and "image" in files
                assert data == {"overwrite": "true", "type": "input"}
                return Response({"name": "source.png", "subfolder": "", "type": "input"})
            if url.endswith("/prompt"):
                return Response({"prompt_id": "pid-post", "number": 0, "node_errors": {}})
            raise AssertionError(f"unexpected POST {url}")

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "preserve original anime character, source-preserving face and hand postprocess",
            aspect_ratio="portrait",
            model="pornmasterAnime_ilV5.safetensors",
            project_name="angelica_postprocess_test",
            artifact_name="postprocess",
            operation="source_preserving_postprocess",
            source_image_path=str(source_image),
            postprocess_preset="face8m_d035_hand9c_d025",
        )

        prompt_calls = [call for call in calls["post"] if call[0].endswith("/prompt")]
        assert len(prompt_calls) == 1
        workflow = prompt_calls[0][1]["prompt"]
        assert workflow["2"]["class_type"] == "LoadImage"
        assert workflow["2"]["inputs"]["image"] == "source.png"
        assert workflow["5"]["class_type"] == "UltralyticsDetectorProvider"
        assert workflow["5"]["inputs"]["model_name"] == "bbox/face_yolov8m.pt"
        assert workflow["6"]["class_type"] == "FaceDetailer"
        assert workflow["6"]["inputs"]["denoise"] == 0.35
        assert workflow["9"]["inputs"]["model_name"] == "bbox/hand_yolov9c.pt"
        assert workflow["11"]["class_type"] == "DetailerForEach"
        assert workflow["11"]["inputs"]["denoise"] == 0.25
        assert result["success"] is True
        assert result["workflow_key"] == "source_preserving_face8m_hand9c_v1"
        assert result["postprocess_preset"] == "face8m_d035_hand9c_d025"
        assert result["evidence"]["source_image"] == str(source_image)
        assert result["evidence"]["face_detailer"]["model"] == "bbox/face_yolov8m.pt"
        assert result["evidence"]["hand_detailer"]["model"] == "bbox/hand_yolov9c.pt"
        assert result["actual_width"] == 1
        assert result["actual_height"] == 1
        assert result["output_resolution"] == "1x1"
        assert result["evidence"]["output_resolution"] == "1x1"
        assert result["report_evidence"] == {
            "operation": "source_preserving_postprocess",
            "canonical_operation": "source_preserving_postprocess",
            "postprocess_preset": "face8m_d035_hand9c_d025",
            "workflow_key": "source_preserving_face8m_hand9c_v1",
            "workflow_path": str(result["workflow_path"]),
            "prompt_id": "pid-post",
            "source_image_path": str(source_image),
            "requested_source_image_path": str(source_image),
            "resolved_source_image_path": str(source_image),
            "source_path_resolution": "exact",
            "output_image": result["primary_image"],
            "artifact_path": result["artifact_path"],
            "output_resolution": "1x1",
            "actual_width": 1,
            "actual_height": 1,
            "face_detailer": {"model": "bbox/face_yolov8m.pt", "denoise": 0.35, "steps": 16, "cfg": 5.5},
            "hand_detailer": {"model": "bbox/hand_yolov9c.pt", "denoise": 0.25, "steps": 14, "cfg": 5.5},
        }
        metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
        assert metadata["report_evidence"] == result["report_evidence"]
        assert metadata["face_detailer"] == result["report_evidence"]["face_detailer"]
        assert metadata["hand_detailer"] == result["report_evidence"]["hand_detailer"]
        assert Path(result["image"]).exists()

    def test_generate_source_preserving_depth_canny_postprocess_uses_promoted_workflow(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        source_image = tmp_path / "source.png"
        source_image.write_bytes(PNG_1PX)
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "depth_canny_postprocess_00001_.png"
        output_file.write_bytes(PNG_1PX)

        calls = {"post": [], "get": []}

        class Response:
            def __init__(self, payload, content=b""):
                self._payload = payload
                self.content = content

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            calls["get"].append(url)
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["waiIllustriousSDXL_v170.safetensors"])
            if url.endswith("/models/vae"):
                return Response(["Anime SDXL VAE DPipe Prototype.safetensors"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-depth-canny-post": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "99": {
                                    "images": [
                                        {
                                            "filename": "depth_canny_postprocess_00001_.png",
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

        def fake_post(url, json=None, files=None, data=None, timeout=None):
            calls["post"].append((url, json, files, data, timeout))
            if url.endswith("/upload/image"):
                assert files and "image" in files
                assert data == {"overwrite": "true", "type": "input"}
                return Response({"name": "source.png", "subfolder": "", "type": "input"})
            if url.endswith("/prompt"):
                return Response({"prompt_id": "pid-depth-canny-post", "number": 0, "node_errors": {}})
            raise AssertionError(f"unexpected POST {url}")

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "preserve pose and silhouette, convert to elegant black dress key visual",
            aspect_ratio="portrait",
            model="pornmasterAnime_ilV5.safetensors",
            project_name="angelica_depth_canny_postprocess_test",
            artifact_name="depth_canny_postprocess",
            operation="source_preserving_postprocess",
            source_image_path=str(source_image),
            postprocess_preset="depth50_canny100_face8m_hand9c_v1",
        )

        prompt_calls = [call for call in calls["post"] if call[0].endswith("/prompt")]
        assert len(prompt_calls) == 1
        workflow = prompt_calls[0][1]["prompt"]
        assert workflow["2"]["class_type"] == "VAELoader"
        assert workflow["2"]["inputs"]["vae_name"] == "Anime SDXL VAE DPipe Prototype.safetensors"
        assert workflow["3"]["class_type"] == "LoraLoader"
        assert workflow["4"]["class_type"] == "LoraLoader"
        assert workflow["21"]["class_type"] == "Zoe_DepthAnythingPreprocessor"
        assert workflow["22"]["class_type"] == "CannyEdgePreprocessor"
        assert workflow["25"]["class_type"] == "ControlNetLoader"
        assert workflow["26"]["class_type"] == "ControlNetApplyAdvanced"
        assert workflow["27"]["class_type"] == "ControlNetLoader"
        assert workflow["28"]["class_type"] == "ControlNetApplyAdvanced"
        assert workflow["31"]["class_type"] == "FaceDetailer"
        assert workflow["34"]["class_type"] == "DetailerForEach"
        assert result["success"] is True
        assert result["model"] == "waiIllustriousSDXL_v170.safetensors"
        assert result["workflow_key"] == "source_preserving_depth50_canny100_face8m_hand9c_v1"
        assert result["postprocess_preset"] == "depth50_canny100_face8m_hand9c_v1"
        assert result["vae"] == "Anime SDXL VAE DPipe Prototype.safetensors"
        assert result["controlnet_used"] is True
        assert len(result["controlnet_stack"]) == 2
        assert result["evidence"]["vae"] == "Anime SDXL VAE DPipe Prototype.safetensors"
        assert result["evidence"]["controlnet_used"] is True
        assert result["evidence"]["controlnet_stack"][0]["name"] == "controlnet_zoe_depth_sdxl_1_0.safetensors"
        assert result["evidence"]["controlnet_stack"][1]["name"] == "illustriousXLCanny_v10.safetensors"
        assert result["report_evidence"]["postprocess_preset"] == "depth50_canny100_face8m_hand9c_v1"
        metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
        assert metadata["resolved_checkpoint"] == "waiIllustriousSDXL_v170.safetensors"
        assert metadata["vae"] == "Anime SDXL VAE DPipe Prototype.safetensors"
        assert metadata["controlnet_used"] is True
        assert Path(result["image"]).exists()

    def test_generate_source_preserving_detailer_normalizes_operation_typo(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        source_image = tmp_path / "source.png"
        source_image.write_bytes(PNG_1PX)
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "local_retouch_00001_.png").write_bytes(PNG_1PX)

        calls = {"post": [], "get": []}

        class Response:
            def __init__(self, payload, content=b""):
                self._payload = payload
                self.content = content

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            calls["get"].append(url)
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["pornmasterAnime_ilV5.safetensors"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-local-retouch": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "99": {
                                    "images": [
                                        {"filename": "local_retouch_00001_.png", "subfolder": "", "type": "output"}
                                    ]
                                }
                            },
                        }
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, files=None, data=None, timeout=None):
            calls["post"].append((url, json, files, data, timeout))
            if url.endswith("/upload/image"):
                assert files and "image" in files
                assert data == {"overwrite": "true", "type": "input"}
                return Response({"name": "source.png", "subfolder": "", "type": "input"})
            if url.endswith("/prompt"):
                return Response({"prompt_id": "pid-local-retouch", "number": 0, "node_errors": {}})
            raise AssertionError(f"unexpected POST {url}")

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "source-preserving local hand retouch",
            aspect_ratio="portrait",
            model="pornmasterAnime_ilV5.safetensors",
            project_name="angelica_local_retouch_test",
            artifact_name="local_retouch",
            operation="local_retress",
            source_image_path=str(source_image),
        )

        prompt_calls = [call for call in calls["post"] if call[0].endswith("/prompt")]
        workflow = prompt_calls[0][1]["prompt"]
        assert workflow["11"]["class_type"] == "DetailerForEach"
        assert result["success"] is True
        assert result["workflow_key"] == "source_preserving_face8m_hand9c_v1"
        assert result["requested_operation"] == "local_retouch"
        assert result["raw_requested_operation"] == "local_retress"
        assert result["canonical_operation"] == "source_preserving_postprocess"
        assert result["report_evidence"]["operation"] == "local_retouch"
        assert result["report_evidence"]["canonical_operation"] == "source_preserving_postprocess"
        metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
        assert metadata["requested_operation"] == "local_retouch"
        assert metadata["raw_requested_operation"] == "local_retress"
        assert metadata["report_evidence"] == result["report_evidence"]

    def test_generate_source_image_upscale_uses_upscale_model_workflow(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        source_image = tmp_path / "source.png"
        source_image.write_bytes(PNG_1PX)
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "upscale_00001_.png").write_bytes(PNG_1PX)

        calls = {"post": [], "get": []}

        class Response:
            def __init__(self, payload):
                self._payload = payload
                self.content = PNG_1PX

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            calls["get"].append(url)
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["pornmasterAnime_ilV5.safetensors"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-upscale": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "99": {
                                    "images": [
                                        {"filename": "upscale_00001_.png", "subfolder": "", "type": "output"}
                                    ]
                                }
                            },
                        }
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, files=None, data=None, timeout=None):
            calls["post"].append((url, json, files, data, timeout))
            if url.endswith("/upload/image"):
                assert files and "image" in files
                assert data == {"overwrite": "true", "type": "input"}
                return Response({"name": "source.png", "subfolder": "", "type": "input"})
            if url.endswith("/prompt"):
                return Response({"prompt_id": "pid-upscale", "number": 0, "node_errors": {}})
            raise AssertionError(f"unexpected POST {url}")

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "source image upscale only",
            aspect_ratio="portrait",
            model="pornmasterAnime_ilV5.safetensors",
            project_name="angelica_upscale_test",
            artifact_name="upscale",
            operation="upscale",
            source_image_path=str(source_image),
            upscale_model="4x-UltraSharp.pth",
        )

        prompt_calls = [call for call in calls["post"] if call[0].endswith("/prompt")]
        assert len(prompt_calls) == 1
        workflow = prompt_calls[0][1]["prompt"]
        assert workflow["1"]["class_type"] == "LoadImage"
        assert workflow["2"]["class_type"] == "UpscaleModelLoader"
        assert workflow["2"]["inputs"]["model_name"] == "4x-UltraSharp.pth"
        assert workflow["3"]["class_type"] == "ImageUpscaleWithModel"
        assert result["success"] is True
        assert result["workflow_key"] == "source_image_4x_ultrasharp_v1"
        assert result["upscale_model"] == "4x-UltraSharp.pth"
        assert result["evidence"]["source_image"] == str(source_image)
        assert result["requested_operation"] == "upscale"
        assert result["canonical_operation"] == "upscale"
        assert result["output_resolution"] == "1x1"
        assert result["report_evidence"] == {
            "operation": "upscale",
            "canonical_operation": "upscale",
            "upscale_model": "4x-UltraSharp.pth",
            "workflow_key": "source_image_4x_ultrasharp_v1",
            "workflow_path": str(result["workflow_path"]),
            "prompt_id": "pid-upscale",
            "source_image_path": str(source_image),
            "output_image": result["primary_image"],
            "artifact_path": result["artifact_path"],
            "output_resolution": "1x1",
            "actual_width": 1,
            "actual_height": 1,
        }
        metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
        assert metadata["requested_operation"] == "upscale"
        assert metadata["canonical_operation"] == "upscale"
        assert metadata["output_resolution"] == "1x1"
        assert metadata["report_evidence"] == result["report_evidence"]
        assert Path(result["image"]).exists()

    def test_generate_masked_inpaint_uses_source_and_mask_uploads(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        source_image = tmp_path / "source.png"
        source_image.write_bytes(PNG_1PX)
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "masked_00001_.png").write_bytes(PNG_1PX)

        calls = {"post": [], "get": []}
        upload_names = iter(("source.png", "mask.png"))

        class Response:
            def __init__(self, payload):
                self._payload = payload
                self.content = PNG_1PX

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            calls["get"].append(url)
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["pornmasterAnime_ilV5.safetensors"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-mask": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "99": {
                                    "images": [
                                        {"filename": "masked_00001_.png", "subfolder": "", "type": "output"}
                                    ]
                                }
                            },
                        }
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, files=None, data=None, timeout=None):
            calls["post"].append((url, json, files, data, timeout))
            if url.endswith("/upload/image"):
                assert files and "image" in files
                assert data == {"overwrite": "true", "type": "input"}
                return Response({"name": next(upload_names), "subfolder": "", "type": "input"})
            if url.endswith("/prompt"):
                return Response({"prompt_id": "pid-mask", "number": 0, "node_errors": {}})
            raise AssertionError(f"unexpected POST {url}")

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "preserve original character, fix only the left hand to have exactly five fingers",
            aspect_ratio="portrait",
            model="pornmasterAnime_ilV5.safetensors",
            project_name="angelica_masked_inpaint_test",
            artifact_name="masked",
            operation="masked_inpaint",
            source_image_path=str(source_image),
            mask_target="left_hand",
            mask_box={"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            mask_feather_px=0,
            grow_mask_by=4,
            seed=20260625,
            steps=18,
            cfg_scale=6.0,
            denoise=0.3,
        )

        upload_calls = [call for call in calls["post"] if call[0].endswith("/upload/image")]
        assert len(upload_calls) == 2
        prompt_calls = [call for call in calls["post"] if call[0].endswith("/prompt")]
        assert len(prompt_calls) == 1
        workflow = prompt_calls[0][1]["prompt"]
        assert workflow["1"]["class_type"] == "LoadImage"
        assert workflow["1"]["inputs"]["image"] == "source.png"
        assert workflow["2"]["class_type"] == "LoadImageMask"
        assert workflow["2"]["inputs"]["image"] == "mask.png"
        assert workflow["12"]["class_type"] == "VAEEncodeForInpaint"
        assert workflow["12"]["inputs"]["grow_mask_by"] == 4
        assert workflow["3"]["inputs"]["denoise"] == 0.3
        assert result["success"] is True
        assert result["workflow_key"] == "source_masked_inpaint_v1"
        assert result["mask_target"] == "left_hand"
        assert result["mask_box"] == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
        assert result["mask_box_px"] == {"left": 0, "top": 0, "right": 1, "bottom": 1}
        assert result["mask_shape"] == "rectangle"
        assert result["mask_coverage_ratio"] == 1.0
        assert result["masked_inpaint_requires_visual_review"] is True
        assert result["masked_inpaint_safety"]["localized_target"] is True
        assert result["masked_inpaint_safety"]["requires_visual_review"] is True
        assert result["actual_width"] == 1
        assert result["actual_height"] == 1
        assert result["output_resolution"] == "1x1"
        assert result["report_evidence"] == {
            "operation": "masked_inpaint",
            "workflow_key": "source_masked_inpaint_v1",
            "workflow_path": str(result["workflow_path"]),
            "prompt_id": "pid-mask",
            "source_image_path": str(source_image),
            "mask_target": "left_hand",
            "mask_box": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            "mask_box_px": {"left": 0, "top": 0, "right": 1, "bottom": 1},
            "mask_shape": "rectangle",
            "mask_coverage_ratio": 1.0,
            "masked_inpaint_requires_visual_review": True,
            "output_image": result["primary_image"],
            "artifact_path": result["artifact_path"],
            "output_resolution": "1x1",
            "actual_width": 1,
            "actual_height": 1,
        }
        metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
        assert metadata["report_evidence"] == result["report_evidence"]
        assert Path(result["image"]).exists()

    def test_generate_masked_inpaint_local_hand_uses_safer_defaults(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        source_image = tmp_path / "source.png"
        source_image.write_bytes(PNG_1PX)
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "masked_safe_00001_.png").write_bytes(PNG_1PX)

        calls = {"post": [], "get": []}
        upload_names = iter(("source.png", "mask.png"))

        class Response:
            def __init__(self, payload):
                self._payload = payload
                self.content = PNG_1PX

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            calls["get"].append(url)
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["pornmasterAnime_ilV5.safetensors"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-mask-safe": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "99": {
                                    "images": [
                                        {"filename": "masked_safe_00001_.png", "subfolder": "", "type": "output"}
                                    ]
                                }
                            },
                        }
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, files=None, data=None, timeout=None):
            calls["post"].append((url, json, files, data, timeout))
            if url.endswith("/upload/image"):
                assert files and "image" in files
                assert data == {"overwrite": "true", "type": "input"}
                return Response({"name": next(upload_names), "subfolder": "", "type": "input"})
            if url.endswith("/prompt"):
                return Response({"prompt_id": "pid-mask-safe", "number": 0, "node_errors": {}})
            raise AssertionError(f"unexpected POST {url}")

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "preserve original character, fix only the visible left hand",
            aspect_ratio="portrait",
            model="pornmasterAnime_ilV5.safetensors",
            project_name="angelica_masked_inpaint_safe_test",
            artifact_name="masked_safe",
            operation="masked_inpaint",
            source_image_path=str(source_image),
            mask_target="left_hand",
            mask_box={"x": 0.15, "y": 0.78, "w": 0.10, "h": 0.16},
        )

        prompt_calls = [call for call in calls["post"] if call[0].endswith("/prompt")]
        workflow = prompt_calls[0][1]["prompt"]
        assert workflow["12"]["inputs"]["grow_mask_by"] == 0
        assert workflow["3"]["inputs"]["denoise"] == 0.55
        assert result["mask_feather_px"] == 6
        assert result["grow_mask_by"] == 0
        assert result["masked_inpaint_safety"]["adjustments"] == [
            "localized_target_default_grow_mask_by_0",
            "localized_target_default_mask_feather_px_6",
            "localized_target_default_denoise_max_0_55",
        ]
        assert result["masked_inpaint_requires_visual_review"] is True

    def test_generate_masked_inpaint_can_use_detailer_bbox_mask_source(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork"))
        monkeypatch.setenv("COMFY_LOCAL_IMAGE_BASE_URL", "http://172.22.224.1:8188")
        monkeypatch.setenv("COMFY_LOCAL_OUTPUT_DIR", str(tmp_path / "comfy-output"))

        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        source_image = tmp_path / "source.png"
        source_image.write_bytes(PNG_1PX)
        output_dir = Path(tmp_path / "comfy-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "masked_detailer_00001_.png").write_bytes(PNG_1PX)

        calls = {"post": [], "get": []}

        class Response:
            def __init__(self, payload):
                self._payload = payload
                self.content = PNG_1PX

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None):
            calls["get"].append(url)
            if url.endswith("/system_stats"):
                return Response({"system": {"os": "win32"}, "devices": [{"name": "GPU"}]})
            if url.endswith("/models/checkpoints"):
                return Response(["pornmasterAnime_ilV5.safetensors"])
            if "/history/" in url:
                return Response(
                    {
                        "pid-mask-detailer": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "99": {
                                    "images": [
                                        {"filename": "masked_detailer_00001_.png", "subfolder": "", "type": "output"}
                                    ]
                                }
                            },
                        }
                    }
                )
            raise AssertionError(f"unexpected GET {url}")

        def fake_post(url, json=None, files=None, data=None, timeout=None):
            calls["post"].append((url, json, files, data, timeout))
            if url.endswith("/upload/image"):
                assert files and "image" in files
                assert data == {"overwrite": "true", "type": "input"}
                return Response({"name": "source.png", "subfolder": "", "type": "input"})
            if url.endswith("/prompt"):
                return Response({"prompt_id": "pid-mask-detailer", "number": 0, "node_errors": {}})
            raise AssertionError(f"unexpected POST {url}")

        monkeypatch.setattr(comfy_mod.requests, "get", fake_get)
        monkeypatch.setattr(comfy_mod.requests, "post", fake_post)
        monkeypatch.setattr(comfy_mod.time, "sleep", lambda *_args, **_kwargs: None)
        import agent.image_gen_provider as provider_mod
        monkeypatch.setattr(provider_mod, "queue_nas_sync_hook", lambda **kwargs: True)

        result = ComfyLocalImageGenProvider().generate(
            "preserve original character, repair only the left hand",
            aspect_ratio="portrait",
            model="pornmasterAnime_ilV5.safetensors",
            project_name="angelica_masked_inpaint_detailer_test",
            artifact_name="masked_detailer",
            operation="masked_inpaint",
            source_image_path=str(source_image),
            mask_source="detailer_bbox",
            mask_target="left_hand",
        )

        upload_calls = [call for call in calls["post"] if call[0].endswith("/upload/image")]
        assert len(upload_calls) == 1
        prompt_calls = [call for call in calls["post"] if call[0].endswith("/prompt")]
        workflow = prompt_calls[0][1]["prompt"]
        assert workflow["5"]["class_type"] == "UltralyticsDetectorProvider"
        assert workflow["5"]["inputs"]["model_name"] == "bbox/hand_yolov9c.pt"
        assert workflow["9"]["class_type"] == "BboxDetectorSEGS"
        assert workflow["10"]["class_type"] == "SegsToCombinedMask"
        assert workflow["11"]["class_type"] == "GrowMask"
        assert workflow["13"]["class_type"] == "FeatherMask"
        assert workflow["12"]["class_type"] == "VAEEncodeForInpaint"
        assert workflow["12"]["inputs"]["mask"] == ["13", 0]
        assert workflow["12"]["inputs"]["grow_mask_by"] == 0
        assert result["success"] is True
        assert result["workflow_key"] == "source_detailer_bbox_masked_inpaint_v1"
        assert result["mask_shape"] == "detailer_bbox_segs"
        assert result["mask_box"] is None
        assert result["masked_inpaint_safety"]["mask_source"] == "detailer_bbox"
        assert result["masked_inpaint_safety"]["detailer_detector"]["model_name"] == "bbox/hand_yolov9c.pt"
        assert "experimental_detailer_bbox_mask_source_requires_visual_review" in result["masked_inpaint_safety"]["warnings"]

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
                return Response(["AOM3A1_orangemixs.safetensors", "waiIllustriousSDXL_v170.safetensors"])
            if url.endswith("/models/vae"):
                return Response(["animevae.pt", "Anime SDXL VAE DPipe Prototype.safetensors"])
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
                return Response(["AOM3A1_orangemixs.safetensors", "waiIllustriousSDXL_v170.safetensors"])
            if url.endswith("/models/vae"):
                return Response(["animevae.pt", "Anime SDXL VAE DPipe Prototype.safetensors"])
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
        seed=None,
        model="AOM3A1_orangemixs.safetensors",
        task_context="character production",
        output_filename="heroine_00001_.png",
        **extra_kwargs,
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
                return Response([
                    "AOM3A1_orangemixs.safetensors",
                    "animagine-xl-4.0-opt.safetensors",
                    "pornmasterAnime_ilV5.safetensors",
                    "waiIllustriousSDXL_v170.safetensors",
                ])
            if url.endswith("/models/vae"):
                return Response(["Anime SDXL VAE DPipe Prototype.safetensors"])
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
            seed=seed,
            task_context=task_context,
            **extra_kwargs,
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
        assert captured["prompt_payload"]["workflow_key"] == "character_key_visual_txt2img_v1"
        assert captured["metadata"]["workflow_key"] == "character_key_visual_txt2img_v1"
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
        assert result["workflow_key"] == "character_key_visual_txt2img_v1"
        assert result["evidence"]["workflow_key"] == "character_key_visual_txt2img_v1"
        assert result["prompt_translation_policy"] == "character-skeleton + keyword-translate + subject-dominance guidance"
        assert result["steps"] == 28
        assert result["cfg_scale"] == 5.0
        assert result["sampler_name"] == "dpmpp_2m"
        assert result["scheduler"] == "karras"

    def test_generate_routes_key_visual_workflow_to_v8_style_anchors_only(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt=(
                "masterpiece, polished mobile game promotional poster, "
                "three stacked cinematic zones divided by luminous diagonal light bands, "
                "top zone emotional close-up, middle ornate fantasy duo, bottom dynamic solo"
            ),
            negative_prompt=(
                "Brown Dust II, Neowiz, text, logo, "
                "single full-body centered portrait, one-person-only poster"
            ),
            workflow_key="character_key_visual_txt2img_v1",
            output_type="key_visual",
            cfg_scale=5.0,
            steps=12,
        )

        assert result["success"] is True
        assert captured["prompt_payload"]["runtime_preset"] == "key_visual_subculture_v1"
        assert captured["prompt_payload"]["workflow_key"] == "character_key_visual_txt2img_v1"
        assert captured["metadata"]["workflow_key"] == "character_key_visual_txt2img_v1"
        assert captured["prompt_payload"]["steps"] == 32
        assert captured["prompt_payload"]["cfg"] == 6.5
        assert captured["prompt_payload"]["sampler"] == "dpmpp_2m"
        assert captured["prompt_payload"]["scheduler"] == "karras"
        assert captured["metadata"]["prompt_translation_policy"] == (
            "key-visual-subculture-v1 + sfw-sanitize + v8-style-anchors-only + no subject/composition rewrite"
        )
        prompt_text = captured["workflow_json"]["3"]["inputs"]["text"]
        assert "subculture anime game illustration" in prompt_text
        assert "light novel cover art" in prompt_text
        assert "anime key visual" in prompt_text
        assert "three stacked cinematic zones" in prompt_text
        assert "1girl, solo, full body" not in prompt_text
        assert "athletic anime heroine" not in prompt_text
        assert "gym" not in prompt_text
        assert "medium full shot" not in prompt_text
        assert "camera at chest height" not in prompt_text
        assert "single character focus" not in prompt_text
        negative_text = captured["workflow_json"]["4"]["inputs"]["text"]
        assert "single full-body centered portrait" in negative_text
        assert "one-person-only poster" in negative_text
        assert "Brown Dust II" in negative_text
        assert result["preset"] == "key_visual_subculture_v1"
        assert result["workflow_key"] == "character_key_visual_txt2img_v1"
        assert result["cfg_scale"] == 6.5

    def test_reference_image_path_without_explicit_experiment_is_blocked(self, monkeypatch, tmp_path):
        reference_image = tmp_path / "reference.png"
        reference_image.write_bytes(PNG_1PX)

        result, _captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt="SFW key visual, same heroine, cinematic academy library",
            negative_prompt="text, logo, watermark",
            output_type="key_visual",
            reference_image_path=str(reference_image),
        )

        assert result["success"] is False
        assert result["error_type"] == "reference_identity_requires_explicit_experiment"
        assert result["reference_identity_status"] == "blocked_to_prevent_default_route_contamination"

    def test_reference_identity_experiment_accepts_explicit_operation_workflow_and_reference_path(
        self, monkeypatch, tmp_path
    ):
        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        reference_image = tmp_path / "reference.png"
        reference_image.write_bytes(PNG_1PX)
        uploaded = {
            "name": "reference.png",
            "subfolder": "",
            "type": "input",
            "source_path": str(reference_image),
        }
        monkeypatch.setattr(comfy_mod, "_upload_comfy_input_file", lambda *_args, **_kwargs: uploaded)

        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt="SFW key visual, same heroine identity as reference, academy poster",
            negative_prompt="text, logo, watermark, different character",
            output_type="key_visual",
            operation="reference_identity_txt2img",
            workflow_key="character_reference_key_visual_experimental_v1",
            reference_image_path=str(reference_image),
        )

        assert result["success"] is True
        assert result["preset"] == "reference_identity_experimental_v1"
        assert result["workflow_key"] == "character_reference_key_visual_experimental_v1"
        evidence = captured["metadata"]["reference_identity_evidence"]
        assert evidence["reference_identity_status"] == "experimental_only"
        assert evidence["explicit_route_guard"] is True
        assert evidence["experimental_reference_identity"] is False
        assert evidence["optional_audit_flag_present"] is False
        assert captured["metadata"]["loras"] == [
            {
                "preset": "stable",
                "name": "00_illustrious_style_candidates\\pornmaster-Aesthetics-v2-lora.safetensors",
                "weight": 0.15,
                "use_case": "default user-approved subculture character illustration",
                "clip_weight": 0.15,
            }
        ]
        assert captured["workflow_json"]["53"]["class_type"] == "IPAdapterAdvanced"

    def test_reference_identity_experiment_uses_ipadapter_only_with_full_guard(self, monkeypatch, tmp_path):
        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        reference_image = tmp_path / "reference.png"
        reference_image.write_bytes(PNG_1PX)
        uploaded = {
            "name": "reference.png",
            "subfolder": "",
            "type": "input",
            "source_path": str(reference_image),
        }
        monkeypatch.setattr(comfy_mod, "_upload_comfy_input_file", lambda *_args, **_kwargs: uploaded)

        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt=(
                "SFW promotional key visual, same character identity as reference, "
                "magical academy library, elegant heroine, cinematic lighting"
            ),
            negative_prompt="text, logo, watermark, different hair color, different outfit",
            output_type="key_visual",
            operation="reference_identity_txt2img",
            workflow_key="character_reference_key_visual_experimental_v1",
            experimental_reference_identity=True,
            reference_image_path=str(reference_image),
        )

        assert result["success"] is True
        assert result["preset"] == "reference_identity_experimental_v1"
        assert result["workflow_key"] == "character_reference_key_visual_experimental_v1"
        assert captured["metadata"]["workflow_key"] == "character_reference_key_visual_experimental_v1"
        assert captured["metadata"]["category"] == "experimental_reference_identity"
        assert captured["metadata"]["loras"] == [
            {
                "preset": "stable",
                "name": "00_illustrious_style_candidates\\pornmaster-Aesthetics-v2-lora.safetensors",
                "weight": 0.15,
                "use_case": "default user-approved subculture character illustration",
                "clip_weight": 0.15,
            }
        ]
        assert captured["metadata"]["reference_identity_evidence"]["reference_identity_status"] == "experimental_only"
        assert captured["metadata"]["reference_identity_evidence"]["default_route_contamination_guard"] is True
        assert captured["metadata"]["reference_identity_evidence"]["uploaded_reference"] == uploaded
        assert captured["workflow_json"]["50"]["class_type"] == "IPAdapterModelLoader"
        assert captured["workflow_json"]["51"]["class_type"] == "CLIPVisionLoader"
        assert captured["workflow_json"]["52"]["class_type"] == "LoadImage"
        assert captured["workflow_json"]["52"]["inputs"]["image"] == "reference.png"
        assert captured["workflow_json"]["53"]["class_type"] == "IPAdapterAdvanced"
        assert captured["workflow_json"]["53"]["inputs"]["model"] == ["21", 0]
        assert captured["workflow_json"]["53"]["inputs"]["ipadapter"] == ["50", 0]
        assert captured["workflow_json"]["53"]["inputs"]["clip_vision"] == ["51", 0]
        assert captured["workflow_json"]["5"]["inputs"]["model"] == ["53", 0]
        assert captured["prompt_payload"]["runtime_preset"] == "reference_identity_experimental_v1"
        assert captured["prompt_payload"]["reference_identity_evidence"]["required_explicit_operation"] == "reference_identity_txt2img"
        assert result["evidence"]["reference_identity_evidence"]["required_explicit_workflow_key"] == "character_reference_key_visual_experimental_v1"
        assert result["report_evidence"]["reference_identity_evidence"]["experimental_reference_identity"] is True

    def test_reference_identity_blocks_fullbody_reference_to_key_visual_without_override(self, monkeypatch, tmp_path):
        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        reference_dir = tmp_path / "fullbody_reference"
        reference_dir.mkdir()
        reference_image = reference_dir / "reference.png"
        reference_image.write_bytes(PNG_1PX)
        sidecar_dir = reference_dir / "sidecar"
        sidecar_dir.mkdir()
        (sidecar_dir / "metadata.json").write_text(
            json.dumps({"workflow_key": "fullbody_v8_scene_txt2img_v2"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            comfy_mod,
            "_upload_comfy_input_file",
            lambda *_args, **_kwargs: {
                "name": "reference.png",
                "subfolder": "",
                "type": "input",
                "source_path": str(reference_image),
            },
        )

        result, _captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt="SFW key visual, same heroine identity as reference, academy poster",
            negative_prompt="text, logo, watermark, different character",
            output_type="key_visual",
            operation="reference_identity_txt2img",
            workflow_key="character_reference_key_visual_experimental_v1",
            experimental_reference_identity=True,
            reference_image_path=str(reference_image),
        )

        assert result["success"] is False
        assert result["error_type"] == "reference_identity_workflow_family_mismatch"
        assert result["reference_identity_status"] == "blocked_workflow_family_mismatch"
        assert result["reference_source_workflow_key"] == "fullbody_v8_scene_txt2img_v2"
        assert result["reference_source_family"] == "fullbody"
        assert result["requested_reference_family"] == "key_visual"

    def test_reference_identity_fullbody_route_preserves_fullbody_v8_settings(self, monkeypatch, tmp_path):
        comfy_mod = importlib.import_module("plugins.image_gen.comfy-local")
        reference_dir = tmp_path / "fullbody_reference"
        reference_dir.mkdir()
        reference_image = reference_dir / "reference.png"
        reference_image.write_bytes(PNG_1PX)
        sidecar_dir = reference_dir / "sidecar"
        sidecar_dir.mkdir()
        (sidecar_dir / "metadata.json").write_text(
            json.dumps({"workflow_key": "fullbody_v8_scene_txt2img_v2"}),
            encoding="utf-8",
        )
        uploaded = {
            "name": "reference.png",
            "subfolder": "",
            "type": "input",
            "source_path": str(reference_image),
        }
        monkeypatch.setattr(comfy_mod, "_upload_comfy_input_file", lambda *_args, **_kwargs: uploaded)

        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt=(
                "SFW fullbody anime game character illustration. "
                "Use the reference image only as temporary identity guidance for this experiment: "
                "same original heroine identity, same short pink bob hair silhouette, same teal blue eyes, "
                "same navy academy uniform with red tie and sailor collar. "
                "Preserve the fullbody_v8 image type: head-to-toe full body visible. "
                "New scene: magical academy hall, visible shoes"
            ),
            negative_prompt="text, logo, watermark, different character",
            output_type="fullbody",
            operation="reference_identity_txt2img",
            workflow_key="fullbody_v8_reference_identity_experimental_v1",
            experimental_reference_identity=True,
            reference_image_path=str(reference_image),
        )

        assert result["success"] is True
        assert result["preset"] == "reference_identity_fullbody_experimental_v1"
        assert result["workflow_key"] == "fullbody_v8_reference_identity_experimental_v1"
        assert captured["metadata"]["workflow_key"] == "fullbody_v8_reference_identity_experimental_v1"
        assert captured["metadata"]["reference_identity_evidence"]["reference_source_family"] == "fullbody"
        assert captured["metadata"]["reference_identity_evidence"]["requested_reference_family"] == "fullbody"
        assert captured["workflow_json"]["2"]["inputs"]["width"] == 1024
        assert captured["workflow_json"]["2"]["inputs"]["height"] == 1536
        assert captured["workflow_json"]["5"]["inputs"]["steps"] == 28
        assert captured["workflow_json"]["5"]["inputs"]["cfg"] == 6.0
        assert captured["workflow_json"]["5"]["inputs"]["sampler_name"] == "euler"
        assert captured["workflow_json"]["5"]["inputs"]["scheduler"] == "normal"
        assert captured["workflow_json"]["50"]["inputs"]["ipadapter_file"] == "ip-adapter-plus-face_sdxl_vit-h.safetensors"
        assert captured["workflow_json"]["53"]["inputs"]["weight"] == 0.42
        assert captured["workflow_json"]["53"]["inputs"]["weight_type"] == "linear"
        assert captured["workflow_json"]["53"]["inputs"]["end_at"] == 0.55
        assert captured["workflow_json"]["53"]["inputs"]["embeds_scaling"] == "V only"
        assert captured["workflow_json"]["53"]["inputs"]["model"] == ["21", 0]
        positive_prompt = captured["workflow_json"]["3"]["inputs"]["text"]
        assert "Use the reference image" not in positive_prompt
        assert "temporary identity guidance" not in positive_prompt
        assert "this experiment" not in positive_prompt
        assert "short pink bob hair silhouette" in positive_prompt
        assert "teal blue eyes" in positive_prompt
        assert "navy academy uniform" in positive_prompt
        assert ", ," not in positive_prompt
        assert "copied reference background" in captured["workflow_json"]["4"]["inputs"]["text"]
        assert "background person" in captured["workflow_json"]["4"]["inputs"]["text"]
        assert captured["metadata"]["loras"] == [
            {
                "preset": "stable",
                "name": "00_illustrious_style_candidates\\pornmaster-Aesthetics-v2-lora.safetensors",
                "weight": 0.15,
                "use_case": "default user-approved subculture character illustration",
                "clip_weight": 0.15,
            }
        ]

    def test_generate_routes_portrait_request_to_portrait_preset(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt="미소녀 얼굴 초상 portrait, ornate costume, clean subculture illustration",
            negative_prompt="blurry, watermark",
        )

        assert result["success"] is True
        assert captured["prompt_payload"]["runtime_preset"] == "portrait_production"
        assert captured["prompt_payload"]["workflow_key"] == "portrait_round_v1_txt2img_v1"
        assert captured["metadata"]["workflow_key"] == "portrait_round_v1_txt2img_v1"
        assert captured["prompt_payload"]["width"] == 1024
        assert captured["prompt_payload"]["height"] == 1216
        assert captured["prompt_payload"]["cfg"] == 6.0
        assert captured["prompt_payload"]["sampler"] == "euler"
        assert captured["prompt_payload"]["scheduler"] == "normal"
        assert captured["prompt_payload"]["seed"] == 12345
        assert captured["metadata"]["checkpoint"] == "waiIllustriousSDXL_v170.safetensors"
        assert captured["metadata"]["vae"] == "Anime SDXL VAE DPipe Prototype.safetensors"
        prompt_text = captured["workflow_json"]["3"]["inputs"]["text"]
        assert "upper body portrait" in prompt_text
        assert "light novel cover art" in prompt_text
        assert "anime key visual" in prompt_text
        assert "ornate costume" in prompt_text
        assert "subculture illustration" in prompt_text
        assert "premium game character portrait" not in prompt_text
        assert "patterned ornamental backdrop" not in prompt_text
        assert "full-body character art" not in prompt_text
        assert "standing full body" not in prompt_text
        assert [item["preset"] for item in captured["metadata"]["loras"]] == [
            "portrait_primary",
            "portrait_primary_detail",
        ]
        assert captured["metadata"]["loras"][0]["name"] == r"00_illustrious_style_candidates\K NAI Style.safetensors"
        assert captured["metadata"]["loras"][0]["weight"] == 0.65
        assert captured["metadata"]["loras"][1]["name"] == r"03_utility_detail_enhancer\AddMicroDetails_Illustrious_v6.safetensors"
        assert captured["metadata"]["loras"][1]["weight"] == 0.20
        assert captured["workflow_json"]["6"]["inputs"]["vae_name"] == "Anime SDXL VAE DPipe Prototype.safetensors"
        assert captured["workflow_json"]["7"]["inputs"]["vae"] == ["6", 0]
        negative_text = captured["workflow_json"]["4"]["inputs"]["text"]
        assert "bad hands" in negative_text
        assert "extra fingers" in negative_text
        assert "picture frame" not in negative_text
        assert result["preset"] == "portrait_production"
        assert result["workflow_key"] == "portrait_round_v1_txt2img_v1"
        assert result["evidence"]["workflow_key"] == "portrait_round_v1_txt2img_v1"
        assert result["prompt_translation_policy"] == "portrait-round-v1-skeleton + keyword-translate + sfw-sanitize + portrait-primary-wai-knai-addmicro"

    def test_generate_routes_fullbody_to_vertical_fullbody_v8_workflow(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt=(
                "SFW T32 fullbody, full body visible from head to toe, "
                "fantasy academy heroine, visible shoes, small ornate book in one hand"
            ),
            negative_prompt="portrait, close-up, cropped feet",
            output_type="fullbody",
        )

        assert result["success"] is True
        assert captured["prompt_payload"]["runtime_preset"] == "fullbody_production"
        assert captured["prompt_payload"]["workflow_key"] == "fullbody_v8_scene_txt2img_v2"
        assert captured["metadata"]["workflow_key"] == "fullbody_v8_scene_txt2img_v2"
        assert captured["metadata"]["output_type"] == "fullbody"
        assert captured["prompt_payload"]["width"] == 1024
        assert captured["prompt_payload"]["height"] == 1536
        assert captured["workflow_json"]["2"]["inputs"]["width"] == 1024
        assert captured["workflow_json"]["2"]["inputs"]["height"] == 1536
        assert captured["metadata"]["checkpoint"] == "pornmasterAnime_ilV5.safetensors"
        assert captured["metadata"]["vae"] == "Anime SDXL VAE DPipe Prototype.safetensors"
        assert captured["metadata"]["loras"][0]["preset"] == "stable"
        assert captured["metadata"]["loras"][0]["name"] == r"00_illustrious_style_candidates\pornmaster-Aesthetics-v2-lora.safetensors"
        assert captured["metadata"]["loras"][0]["weight"] == 0.15
        prompt_text = captured["workflow_json"]["3"]["inputs"]["text"]
        assert "full body character art" in prompt_text
        assert "head-to-toe visible" in prompt_text
        assert "full feet visible" in prompt_text
        assert "fantasy academy heroine" in prompt_text
        assert "upper body portrait" not in prompt_text
        negative_text = captured["workflow_json"]["4"]["inputs"]["text"]
        assert "portrait" in negative_text
        assert "close-up" in negative_text
        assert "cropped feet" in negative_text
        assert result["preset"] == "fullbody_production"
        assert result["workflow_key"] == "fullbody_v8_scene_txt2img_v2"
        assert result["evidence"]["workflow_key"] == "fullbody_v8_scene_txt2img_v2"
        assert result["prompt_translation_policy"] == (
            "fullbody-v8-scene-v2 + sfw-sanitize + solo/head-to-toe guard + "
            "finished-color anti-sketch guard + scene/full-feet anti-wallpaper guard"
        )

    def test_portrait_skeleton_does_not_inject_fixed_identity_traits(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt=(
                "portrait, short pink hair, teal eyes, simple navy uniform, "
                "warm magical library background, subculture illustration"
            ),
            negative_prompt="blurry, watermark",
            output_type="portrait",
        )

        assert result["success"] is True
        prompt_text = captured["workflow_json"]["3"]["inputs"]["text"]
        assert "short pink hair" in prompt_text
        assert "teal eyes" in prompt_text
        assert "simple navy uniform" in prompt_text
        assert "golden eyes" not in prompt_text
        assert "long hair" not in prompt_text
        assert "ornate costume" not in prompt_text

    def test_generate_respects_explicit_sampler_settings_with_portrait_preset(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt="미소녀 얼굴 초상 portrait, ornate costume, clean subculture illustration",
            negative_prompt="blurry, watermark",
            output_type="portrait",
            steps=28,
            cfg_scale=5.0,
            sampler_name="dpmpp_2m",
            scheduler="karras",
        )

        assert result["success"] is True
        assert captured["prompt_payload"]["runtime_preset"] == "portrait_production"
        assert captured["prompt_payload"]["workflow_key"] == "portrait_round_v1_txt2img_v1"
        assert captured["prompt_payload"]["steps"] == 28
        assert captured["prompt_payload"]["cfg"] == 5.0
        assert captured["prompt_payload"]["sampler"] == "dpmpp_2m"
        assert captured["prompt_payload"]["scheduler"] == "karras"
        assert captured["workflow_json"]["5"]["inputs"]["steps"] == 28
        assert captured["workflow_json"]["5"]["inputs"]["cfg"] == 5.0
        assert captured["workflow_json"]["5"]["inputs"]["sampler_name"] == "dpmpp_2m"
        assert captured["workflow_json"]["5"]["inputs"]["scheduler"] == "karras"
        assert result["steps"] == 28
        assert result["cfg_scale"] == 5.0
        assert result["sampler_name"] == "dpmpp_2m"
        assert result["scheduler"] == "karras"

    def test_generate_routes_standing_sprite_to_fullbody_v8_workflow(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt=(
                "standing sprite, full body visible from head to toe, "
                "fantasy guild receptionist heroine, visible shoes, staff in left hand"
            ),
            negative_prompt="portrait, close-up, cropped feet",
            output_type="standing_sprite",
        )

        assert result["success"] is True
        assert captured["prompt_payload"]["runtime_preset"] == "standing_sprite_production"
        assert captured["prompt_payload"]["workflow_key"] == "fullbody_v8_scene_txt2img_v2"
        assert captured["metadata"]["workflow_key"] == "fullbody_v8_scene_txt2img_v2"
        assert captured["metadata"]["output_type"] == "standing_sprite"
        assert captured["prompt_payload"]["width"] == 1024
        assert captured["prompt_payload"]["height"] == 1536
        assert captured["workflow_json"]["2"]["inputs"]["width"] == 1024
        assert captured["workflow_json"]["2"]["inputs"]["height"] == 1536
        prompt_text = captured["workflow_json"]["3"]["inputs"]["text"]
        assert "game standing sprite" in prompt_text
        assert "full body standing character art" in prompt_text
        assert "readable full silhouette" in prompt_text
        assert "fantasy guild receptionist heroine" in prompt_text
        assert "upper body portrait" not in prompt_text
        negative_text = captured["workflow_json"]["4"]["inputs"]["text"]
        assert "portrait" in negative_text
        assert "close-up" in negative_text
        assert "cropped feet" in negative_text
        assert captured["metadata"]["loras"][0]["preset"] == "stable"
        assert result["preset"] == "standing_sprite_production"
        assert result["workflow_key"] == "fullbody_v8_scene_txt2img_v2"
        assert result["evidence"]["workflow_key"] == "fullbody_v8_scene_txt2img_v2"
        assert result["prompt_translation_policy"] == (
            "standing-sprite-v1 + fullbody-v8-workflow + sfw-sanitize + "
            "production sprite skeleton + no portrait rewrite"
        )

    def test_generate_routes_v8_style_request_to_portrait_workflow_without_portrait_rewrite(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt=(
                "v8_style_workflow, 이미지_v8 workflow, SFW athletic anime heroine, "
                "head-to-knees framing, gym background, barbell, clear facial features"
            ),
            negative_prompt="blurry, watermark, close-up",
            seed=20260625,
        )

        assert result["success"] is True
        assert captured["prompt_payload"]["runtime_preset"] == "v8_style_workflow"
        assert captured["prompt_payload"]["workflow_key"] == "portrait_round_v1_txt2img_v1"
        assert captured["metadata"]["workflow_key"] == "portrait_round_v1_txt2img_v1"
        assert captured["prompt_payload"]["width"] == 1024
        assert captured["prompt_payload"]["height"] == 1536
        assert captured["prompt_payload"]["cfg"] == 6.0
        assert captured["prompt_payload"]["sampler"] == "euler"
        assert captured["prompt_payload"]["scheduler"] == "normal"
        assert captured["prompt_payload"]["seed"] == 20260625
        prompt_text = captured["workflow_json"]["3"]["inputs"]["text"]
        assert "v8_style_workflow" in prompt_text
        assert "head-to-knees framing" in prompt_text
        assert "1girl" in prompt_text
        assert "anime girl" in prompt_text
        assert "visible human face" in prompt_text
        assert "subculture anime game illustration" in prompt_text
        assert "medium full shot" in prompt_text
        assert "camera at chest height" in prompt_text
        assert "no first-person perspective" in prompt_text
        assert "upper body portrait" not in prompt_text
        negative_text = captured["workflow_json"]["4"]["inputs"]["text"]
        assert "extreme close-up" in negative_text
        assert "first-person view" in negative_text
        assert "knees foreground" in negative_text
        assert "close-up" in negative_text
        assert "camera head" in negative_text
        assert "faceless" in negative_text
        assert "muscular man" in negative_text
        assert captured["metadata"]["loras"] == []
        assert captured["metadata"]["vae"] is None
        assert "6" not in captured["workflow_json"]
        assert captured["workflow_json"]["7"]["inputs"]["vae"] == ["1", 2]
        assert result["preset"] == "v8_style_workflow"
        assert result["workflow_key"] == "portrait_round_v1_txt2img_v1"
        assert result["evidence"]["workflow_key"] == "portrait_round_v1_txt2img_v1"
        assert result["prompt_translation_policy"] == "v8-style-workflow + sfw-sanitize + composition-guard + no portrait rewrite"

    def test_v8_style_preserves_explicit_vae_and_normalizes_lora_paths(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt=(
                "v8_style_workflow, 1girl, solo, anime girl, "
                "subculture anime game illustration, medium full shot"
            ),
            vae="Anime SDXL VAE DPipe Prototype.safetensors",
            loras=[
                {
                    "name": r"00_illustrious_style_candidates\\K NAI Style.safetensors",
                    "weight": 0.7,
                    "clip_weight": 0.7,
                },
                {
                    "name": r"03_utility_detail_enhancer\\AddMicroDetails_Illustrious_v6.safetensors",
                    "weight": 0.2,
                    "clip_weight": 0.2,
                },
            ],
        )

        assert result["success"] is True
        assert captured["metadata"]["workflow_key"] == "portrait_round_v1_txt2img_v1"
        assert captured["metadata"]["vae"] == "Anime SDXL VAE DPipe Prototype.safetensors"
        assert captured["workflow_json"]["6"]["class_type"] == "VAELoader"
        assert captured["workflow_json"]["7"]["inputs"]["vae"] == ["6", 0]
        assert captured["workflow_json"]["21"]["inputs"]["lora_name"] == r"00_illustrious_style_candidates\K NAI Style.safetensors"
        assert captured["workflow_json"]["22"]["inputs"]["lora_name"] == r"03_utility_detail_enhancer\AddMicroDetails_Illustrious_v6.safetensors"
        assert captured["metadata"]["loras"][0]["name"] == r"00_illustrious_style_candidates\K NAI Style.safetensors"

    def test_v8_style_composition_only_prompt_keeps_subject_baseline(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt=(
                "v8_style_workflow, medium full shot, cowboy shot, "
                "from head to above knees, camera at chest height, balanced perspective"
            ),
            seed=20260625,
        )

        assert result["success"] is True
        assert captured["prompt_payload"]["runtime_preset"] == "v8_style_workflow"
        prompt_text = captured["workflow_json"]["3"]["inputs"]["text"]
        assert "1girl" in prompt_text
        assert "anime girl" in prompt_text
        assert "athletic anime heroine" in prompt_text
        assert "visible human face" in prompt_text
        assert "polished eyes" in prompt_text
        assert "subculture anime game illustration" in prompt_text
        assert "medium full shot" in prompt_text
        negative_text = captured["workflow_json"]["4"]["inputs"]["text"]
        assert "camera head" in negative_text
        assert "faceless" in negative_text
        assert "monster body" in negative_text

    def test_generate_sanitizes_weighted_nsfw_tag_from_sfw_portrait_prompt(self, monkeypatch, tmp_path):
        result, captured = self._run_generate(
            monkeypatch,
            tmp_path,
            prompt="미소녀 얼굴 초상, silver hair, blue dress, (NSFW:1.3), no NSFW, no full body",
            negative_prompt="blurry",
        )

        assert result["success"] is True
        prompt_text = captured["workflow_json"]["3"]["inputs"]["text"]
        assert "NSFW" not in prompt_text
        assert "no upper body portrait" not in prompt_text
        assert "NSFW" not in captured["prompt_payload"]["source_prompt"]
        assert "NSFW" not in captured["metadata"]["source_prompt"]
        assert "NSFW" not in captured["metadata"]["translated_prompt"]

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
        assert captured["prompt_payload"]["resolved_checkpoint"] == "pornmasterAnime_ilV5.safetensors"
        assert captured["metadata"]["preset"] == "character_production"
        assert captured["metadata"]["workflow_key"] == "character_key_visual_txt2img_v1"
        assert result["resolution_mode"] == "exact"
        assert result["resolved_checkpoint"] == "pornmasterAnime_ilV5.safetensors"


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
