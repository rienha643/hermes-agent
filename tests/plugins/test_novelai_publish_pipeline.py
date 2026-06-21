from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import yaml


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid 1x1 RGBA PNG.
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a"
            "0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c6360000002000100ffff03000006000557bfab"
            "0000000049454e44ae426082"
        )
    )


def _write_raw_response_sidecars(sidecar: Path, png: Path) -> None:
    raw_zip = sidecar / "response.zip"
    with zipfile.ZipFile(raw_zip, "w") as zf:
        zf.write(png, "image_0.png")
    (sidecar / "response.bin").write_bytes(raw_zip.read_bytes())


def _write_common_sidecars(sidecar: Path, source_png: Path, seed: int = 123) -> None:
    sidecar.mkdir(parents=True, exist_ok=True)
    (sidecar / "request.json").write_text(json.dumps({"input": "safe prompt"}), encoding="utf-8")
    (sidecar / "response.json").write_text(json.dumps({"status": 200}), encoding="utf-8")
    (sidecar / "metadata.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
    (sidecar / "manifest.json").write_text(json.dumps({"files": {"png": str(source_png)}}), encoding="utf-8")
    (sidecar / "integrity.json").write_text(json.dumps({"sha256": "old"}), encoding="utf-8")
    _write_raw_response_sidecars(sidecar, source_png)


def test_novelai_default_request_builder_uses_policy_defaults() -> None:
    import plugins.image_gen.novelai as novelai

    payload = novelai.build_novelai_request_payload(prompt="safe prompt")
    params = payload["parameters"]

    assert payload["input"].startswith("best quality")
    assert "subculture illustration" in payload["input"]
    assert "anime illustration" in payload["input"]
    assert payload["input"].endswith("safe prompt")
    assert payload["model"] == novelai.DEFAULT_MODEL
    assert payload["action"] == "generate"
    assert params["sampler"] == novelai.NAI_SAMPLER
    assert params["sampler_label"] == "DPM++ SDE"
    assert "dpmpp" in params["sampler"]
    assert "sde" in params["sampler"]
    assert params["sm"] is True
    assert params["sm_dyn"] is True
    assert params["qualityToggle"] is False
    assert params["add_quality_tags"] is False
    assert params["ucPreset"] == 0
    assert params["undesired_content_preset"] == "none"
    assert params["negative_prompt"] == novelai.NAI_DEFAULT_NEGATIVE_PROMPT
    assert params["uc"] == novelai.NAI_DEFAULT_NEGATIVE_PROMPT
    assert params["v4_prompt"]["caption"]["base_caption"] == payload["input"]
    assert params["v4_negative_prompt"]["caption"]["base_caption"] == novelai.NAI_DEFAULT_NEGATIVE_PROMPT
    for term in [
        "normal quality",
        "bad anatomy",
        "bad hands",
        "malformed fingers",
        "extra digits",
        "bad feet",
        "JPEG artifacts",
        "chromatic aberration",
        "scan artifacts",
    ]:
        assert term in params["negative_prompt"]
    assert payload["policy"]["safe_range"] == "SAFE_1024_RANGE"
    assert payload["policy"]["high_res_policy"] == "HIGH_RES_REQUIRES_APPROVAL"


def test_novelai_custom_negative_prompt_preserves_policy_baseline() -> None:
    import plugins.image_gen.novelai as novelai

    payload = novelai.build_novelai_request_payload(
        prompt="safe prompt, best quality",
        negative_prompt="flat lighting, bad hands",
    )
    params = payload["parameters"]

    assert payload["input"].count("best quality") == 1
    assert "safe prompt" in payload["input"]
    assert params["negative_prompt"].startswith("flat lighting")
    assert "bad hands" in params["negative_prompt"]
    assert params["negative_prompt"].count("bad hands") == 1
    assert "scan artifacts" in params["negative_prompt"]
    assert params["uc"] == params["negative_prompt"]
    assert params["v4_negative_prompt"]["caption"]["base_caption"] == params["negative_prompt"]


def test_novelai_style_preset_env_merges_sfw_subculture_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    import plugins.image_gen.novelai as novelai

    monkeypatch.setenv(novelai.NAI_STYLE_PRESET_ENV, "game_default_subculture")

    payload = novelai.build_novelai_request_payload(
        prompt="safe prompt, cinematic lighting",
        negative_prompt="flat lighting, bad hands",
    )
    params = payload["parameters"]

    assert payload["policy"]["style_preset"] == "game_default_subculture"
    assert "polished cel shading" in payload["input"]
    assert "fantasy game character art" in payload["input"]
    assert payload["input"].count("cinematic lighting") == 1
    assert payload["input"].endswith("safe prompt")
    assert params["negative_prompt"].startswith("flat lighting")
    assert "photorealistic" in params["negative_prompt"]
    assert "realistic skin texture" in params["negative_prompt"]
    assert params["negative_prompt"].count("flat lighting") == 1
    assert params["v4_prompt"]["caption"]["base_caption"] == payload["input"]
    assert params["v4_negative_prompt"]["caption"]["base_caption"] == params["negative_prompt"]


def test_novelai_unknown_style_preset_env_does_not_change_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    import plugins.image_gen.novelai as novelai

    monkeypatch.setenv(novelai.NAI_STYLE_PRESET_ENV, "missing-preset")

    payload = novelai.build_novelai_request_payload(prompt="safe prompt")

    assert payload["policy"]["style_preset"] is None
    assert "polished cel shading" not in payload["input"]
    assert "photorealistic" not in payload["parameters"]["negative_prompt"]


@pytest.mark.parametrize("width,height", [(1024, 1024), (832, 1216), (768, 1344)])
def test_novelai_safe_1024_range_payloads_are_allowed(width: int, height: int) -> None:
    import plugins.image_gen.novelai as novelai

    payload = novelai.build_novelai_request_payload(prompt="safe prompt", width=width, height=height)

    assert payload["parameters"]["width"] == width
    assert payload["parameters"]["height"] == height
    assert novelai.is_safe_1024_range(width, height) is True


@pytest.mark.parametrize("width,height", [(1536, 1536), (2048, 2048)])
def test_novelai_high_resolution_payloads_require_explicit_approval(width: int, height: int) -> None:
    import plugins.image_gen.novelai as novelai

    with pytest.raises(novelai.NovelAIResolutionApprovalRequired) as excinfo:
        novelai.build_novelai_request_payload(prompt="safe prompt", width=width, height=height)

    assert excinfo.value.policy == "HIGH_RES_REQUIRES_APPROVAL"
    assert excinfo.value.width == width
    assert excinfo.value.height == height


def test_novelai_upscale_and_high_resolution_flags_require_approval() -> None:
    import plugins.image_gen.novelai as novelai

    with pytest.raises(novelai.NovelAIResolutionApprovalRequired) as upscale_exc:
        novelai.build_novelai_request_payload(prompt="safe prompt", upscale=True)
    assert upscale_exc.value.reason == "upscale"

    with pytest.raises(novelai.NovelAIResolutionApprovalRequired) as high_res_exc:
        novelai.build_novelai_request_payload(prompt="safe prompt", high_resolution=True)
    assert high_res_exc.value.reason == "high_resolution"


def test_novelai_high_resolution_approval_allows_payload_build_only() -> None:
    import plugins.image_gen.novelai as novelai

    payload = novelai.build_novelai_request_payload(
        prompt="safe prompt",
        width=1536,
        height=1536,
        high_res_approved=True,
    )

    assert payload["parameters"]["width"] == 1536
    assert payload["parameters"]["height"] == 1536
    assert payload["policy"]["high_res_approved"] is True


def test_novelai_dry_run_generate_returns_payload_without_live_generation() -> None:
    import plugins.image_gen.novelai as novelai

    provider = novelai.NovelAIImageGenProvider()
    result = provider.generate("safe prompt", dry_run_request=True, width=1024, height=1024)

    assert result["success"] is True
    assert result["provider"] == "novelai"
    assert result["dry_run_request"] is True
    assert result["request_payload"]["parameters"]["width"] == 1024
    assert result["request_payload"]["parameters"]["sampler"] == novelai.NAI_SAMPLER


def test_novelai_live_generation_requires_explicit_operator_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    import plugins.image_gen.novelai as novelai

    called = False

    def fake_post(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("NovelAI live endpoint must not be called without explicit approval")

    monkeypatch.setenv("NOVELAI_API_KEY", "test-key-not-printed")
    monkeypatch.setattr(novelai, "_post_novelai_generation", fake_post)

    result = novelai.NovelAIImageGenProvider().generate("safe prompt", width=1024, height=1024)

    assert result["success"] is False
    assert result["error_type"] == "approval_required"
    assert novelai.LIVE_GENERATION_REQUIRES_APPROVAL in result["error"]
    assert called is False


def test_novelai_policy_docs_match_code_defaults() -> None:
    import plugins.image_gen.novelai as novelai

    docs = [
        Path("website/docs/user-guide/features/novelai-generation-policy.md"),
        Path("website/docs/user-guide/features/hermes-image-generation-standard.md"),
    ]
    for doc_path in docs:
        text = doc_path.read_text(encoding="utf-8")
        assert "Add Quality Tags | `OFF`" in text
        assert "Undesired Content Preset | `NONE`" in text
        assert "Sampler | `DPM++ SDE`" in text
        assert "SMEA | `ON`" in text
        assert "DYN | `ON`" in text
        assert "SAFE_1024_RANGE" in text
        assert "HIGH_RES_REQUIRES_APPROVAL" in text
        for term in novelai.NAI_DEFAULT_NEGATIVE_PROMPT.split(",\n"):
            assert term in text

    assert novelai.NAI_ADD_QUALITY_TAGS is False
    assert novelai.NAI_UNDESIRED_CONTENT_PRESET == "none"
    assert novelai.NAI_SAMPLER_LABEL == "DPM++ SDE"
    assert novelai.NAI_SMEA is True
    assert novelai.NAI_DYN is True
    assert novelai.SAFE_1024_RANGE_MAX_PIXELS == 1024 * 1024
    assert novelai.HIGH_RES_REQUIRES_APPROVAL == "HIGH_RES_REQUIRES_APPROVAL"


def test_novelai_publish_existing_generation_creates_hermeswork_bundle_and_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_root = tmp_path / "profile" / "generated" / "nai_safe_smoke" / "run-001"
    source_png = generated_root / "nai_safe_smoke.png"
    _write_png(source_png)
    sidecar = generated_root / "sidecar"
    _write_common_sidecars(sidecar, source_png)

    work_root = tmp_path / "HermesWork" / "Image"
    stale_sidecar = work_root / "NAI" / "run-001" / "sidecar"
    stale_sidecar.mkdir(parents=True)
    (stale_sidecar / "response.bin").write_bytes(b"stale raw")
    (stale_sidecar / "response.zip").write_bytes(b"stale raw")
    monkeypatch.setenv("HERMES_WORK_ROOT", str(work_root))
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(work_root))

    calls = []

    import plugins.image_gen.novelai as novelai

    def fake_queue_nas_sync_hook(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(novelai, "queue_nas_sync_hook", fake_queue_nas_sync_hook)

    result = novelai.publish_existing_generation(
        source_png=source_png,
        source_sidecar_dir=sidecar,
        run_id="run-001",
        prompt="safe prompt",
        model="nai-diffusion-4-5-curated",
        aspect_ratio="square",
    )

    published = work_root / "NAI" / "run-001" / "image_000.png"
    assert result["success"] is True
    assert result["provider"] == "novelai"
    assert result["image"] == str(published)
    assert result["media_files"] == [str(published)]
    assert result["nas_hook_requested"] is True
    assert published.read_bytes() == source_png.read_bytes()

    published_sidecar = published.parent / "sidecar"
    for name in ["request.json", "response.json", "metadata.json", "manifest.json", "integrity.json"]:
        assert (published_sidecar / name).is_file()

    manifest = json.loads((published_sidecar / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_path"] == str(source_png.resolve())
    assert manifest["published_path"] == str(published)
    assert manifest["published_resolved_path"] == str(published.resolve())
    assert manifest["files"]["png"] == str(published)
    assert manifest["checks"]["slack_upload"] == "READY"
    assert manifest["checks"]["nas_hook"] == "REQUESTED"
    assert manifest["raw_response_saved"] is False
    assert "response.bin" not in manifest["files"].get("sidecar", {})
    assert "response.zip" not in manifest["files"].get("sidecar", {})

    response = json.loads((published_sidecar / "response.json").read_text(encoding="utf-8"))
    assert response["raw_response_saved"] is False
    assert response["response_shape"] == "zip"
    assert response["response_bytes"] > 0
    assert response["response_sha256"]
    assert response["normalized_artifacts"]["count"] == 1
    assert not (published_sidecar / "response.bin").exists()
    assert not (published_sidecar / "response.zip").exists()

    assert calls
    assert calls[0]["category"] == "image"
    assert calls[0]["scope"] == "NAI/run-001"
    assert calls[0]["artifact_path"] == published
    assert calls[0]["source_root"] == published.parent

    from tools.send_message_tool import _validate_slack_media_candidates

    allowed, blocked = _validate_slack_media_candidates([(str(published), False)], expected_count=1)
    assert allowed == [(str(published.resolve()), False)]
    assert blocked == []

    generated_allowed, generated_blocked = _validate_slack_media_candidates([(str(source_png), False)])
    assert generated_allowed == []
    assert generated_blocked[0]["reason"] == "outside_allowed_publish_roots"


def test_novelai_publish_debug_flag_preserves_raw_response_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_root = tmp_path / "profile" / "generated" / "nai_safe_smoke" / "run-debug"
    source_png = generated_root / "nai_safe_smoke.png"
    _write_png(source_png)
    sidecar = generated_root / "sidecar"
    _write_common_sidecars(sidecar, source_png, seed=456)

    work_root = tmp_path / "HermesWork" / "Image"
    monkeypatch.setenv("HERMES_WORK_ROOT", str(work_root))
    monkeypatch.setenv("HERMES_NAI_SAVE_RAW_RESPONSE", "1")

    import plugins.image_gen.novelai as novelai

    monkeypatch.setattr(novelai, "queue_nas_sync_hook", lambda **kwargs: False)

    result = novelai.publish_existing_generation(
        source_png=source_png,
        source_sidecar_dir=sidecar,
        run_id="run-debug",
        prompt="safe prompt",
    )

    published_sidecar = Path(result["sidecar_dir"])
    assert (published_sidecar / "response.bin").is_file()
    assert (published_sidecar / "response.zip").is_file()
    manifest = json.loads((published_sidecar / "manifest.json").read_text(encoding="utf-8"))
    response = json.loads((published_sidecar / "response.json").read_text(encoding="utf-8"))
    assert manifest["raw_response_saved"] is True
    assert response["raw_response_saved"] is True
    assert "response.bin" in manifest["files"]["sidecar"]
    assert "response.zip" in manifest["files"]["sidecar"]


@pytest.mark.asyncio
async def test_novelai_published_media_enters_slack_upload_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_root = tmp_path / "HermesWork" / "Image"
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(work_root))
    published = work_root / "NAI" / "run-002" / "image_000.png"
    _write_png(published)

    import tools.send_message_tool as smt
    from gateway.config import Platform

    seen = {}

    async def fake_send_slack_via_adapter(pconfig, chat_id, message, media_files=None, thread_id=None, force_document=False):
        seen["chat_id"] = chat_id
        seen["thread_id"] = thread_id
        seen["media_files"] = media_files
        seen["force_document"] = force_document
        return {"success": True, "platform": "slack", "chat_id": chat_id, "has_files": True}

    monkeypatch.setattr(smt, "_send_slack_via_adapter", fake_send_slack_via_adapter)

    result = await smt._send_to_platform(
        Platform.SLACK,
        SimpleNamespace(token="xoxb-test"),
        "C123",
        "published NAI smoke",
        thread_id="1781642857.850649",
        media_files=[(str(published), False)],
    )

    assert result["success"] is True
    assert result["has_files"] is True
    assert seen == {
        "chat_id": "C123",
        "thread_id": "1781642857.850649",
        "media_files": [(str(published), False)],
        "force_document": False,
    }


def test_seir_tool_profile_is_generation_capable_but_still_minimal() -> None:
    from toolsets import resolve_toolset

    tools = set(resolve_toolset("seir-tool-profile"))

    assert "image_generate" in tools
    assert "terminal" not in tools
    assert "execute_code" not in tools
    assert "patch" not in tools
    assert "delegate_task" not in tools
    assert "cronjob" not in tools


def test_artist_grok_routes_image_generation_to_novelai_without_changing_main_llm() -> None:
    config = yaml.safe_load(Path("/Users/hermes/.hermes/profiles/artist_grok/config.yaml").read_text(encoding="utf-8"))

    assert config["image_gen"]["provider"] == "novelai"
    assert config["model"]["provider"] == "custom:gemma4-local"
    assert config["slack"]["require_mention"] is True


def test_novelai_live_generation_uses_mocked_endpoint_and_publishes_zip_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.image_gen.novelai as novelai

    response_png = tmp_path / "response.png"
    _write_png(response_png)
    response_zip = tmp_path / "response.zip"
    with zipfile.ZipFile(response_zip, "w") as zf:
        zf.write(response_png, "image_0.png")

    seen = {}

    def fake_post(payload, *, api_key, endpoint, timeout):
        seen["payload"] = payload
        seen["api_key_present"] = bool(api_key)
        seen["endpoint"] = endpoint
        seen["timeout"] = timeout
        return novelai.NovelAIHTTPResponse(
            status=200,
            headers={"content-type": "application/zip"},
            body=response_zip.read_bytes(),
        )

    monkeypatch.setenv("NOVELAI_API_KEY", "test-key-not-printed")
    work_root = tmp_path / "HermesWork" / "Image"
    monkeypatch.setenv("HERMES_WORK_ROOT", str(work_root))
    monkeypatch.setattr(novelai, "_post_novelai_generation", fake_post)
    monkeypatch.setattr(novelai, "queue_nas_sync_hook", lambda **kwargs: False)

    provider = novelai.NovelAIImageGenProvider()
    result = provider.generate(
        "safe prompt",
        aspect_ratio="square",
        run_id="live-001",
        width=1024,
        height=1024,
        live_generation_approved=True,
    )

    published = work_root / "NAI" / "live-001" / "image_000.png"
    sidecar = published.parent / "sidecar"
    assert result["success"] is True
    assert result["image"] == str(published)
    assert result["media_files"] == [str(published)]
    assert published.read_bytes() == response_png.read_bytes()
    assert seen["api_key_present"] is True
    assert seen["payload"]["parameters"]["sampler"] == novelai.NAI_SAMPLER
    assert seen["payload"]["parameters"]["qualityToggle"] is False

    for name in ["request.json", "response.json", "metadata.json", "manifest.json", "integrity.json"]:
        assert (sidecar / name).is_file()
    assert not (sidecar / "response.bin").exists()
    assert not (sidecar / "response.zip").exists()

    response = json.loads((sidecar / "response.json").read_text(encoding="utf-8"))
    assert response["status"] == 200
    assert response["response_shape"] == "zip"
    assert response["response_bytes"] == len(response_zip.read_bytes())
    assert response["raw_response_saved"] is False
    assert response["normalized_artifacts"]["count"] == 1

    manifest = json.loads((sidecar / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checks"]["png"] == "PASS"
    assert manifest["checks"]["publish"] == "PASS"
    assert manifest["checks"]["slack_upload"] == "READY"


def test_novelai_live_generation_normalizes_direct_png_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.image_gen.novelai as novelai

    response_png = tmp_path / "response.png"
    _write_png(response_png)

    def fake_post(payload, *, api_key, endpoint, timeout):
        return novelai.NovelAIHTTPResponse(
            status=200,
            headers={"content-type": "image/png"},
            body=response_png.read_bytes(),
        )

    monkeypatch.setenv("NOVELAI_API_KEY", "test-key-not-printed")
    monkeypatch.setenv("HERMES_WORK_ROOT", str(tmp_path / "HermesWork" / "Image"))
    monkeypatch.setattr(novelai, "_post_novelai_generation", fake_post)
    monkeypatch.setattr(novelai, "queue_nas_sync_hook", lambda **kwargs: False)

    result = novelai.NovelAIImageGenProvider().generate(
        "safe prompt",
        run_id="live-png",
        live_generation_approved=True,
    )

    published = Path(result["image"])
    assert result["success"] is True
    assert published.read_bytes() == response_png.read_bytes()
    response = json.loads((published.parent / "sidecar" / "response.json").read_text(encoding="utf-8"))
    assert response["response_shape"] == "png"
