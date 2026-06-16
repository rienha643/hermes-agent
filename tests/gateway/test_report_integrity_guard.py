import hashlib
from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.run import (
    _compact_gateway_final_response,
    _sanitize_gateway_final_response,
    _validate_gateway_report_integrity,
)


def test_placeholder_hash_in_angelica_result_is_invalid():
    text = """[WORKER RESULT: Angelica]
[NSFW STABILITY ROUND RESULTS]
checkpoint: hdaRainbowIllusMixV1_v13.safetensors
output_path: /Users/hermes/HermesWork/Image/nsfw_stability_round_v1/hdaRainbowIllusMixV1_v13.safetensors/image.png
prompt_hash: 4e3a82f1... (calculated)
workflow_hash: 8f2c91a3... (calculated)
file_sha256: a1b2c3d4... (calculated)
Slack Upload: PASS
NAS Hook: PASS
"""

    result = _validate_gateway_report_integrity(text, attachment_count=1)

    assert result.valid is False
    assert "INVALID_RESULT" in result.text
    assert "a1b2c3d4" not in result.text
    assert "(calculated)" not in result.text


def test_calculated_marker_in_result_is_invalid():
    text = """[NSFW STABILITY ROUND RESULTS]
prompt_hash: 1234... (calculated)
file_sha256: abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890
"""

    result = _validate_gateway_report_integrity(text, attachment_count=1)

    assert result.valid is False
    assert "placeholder_or_uncomputed_hash" in result.reasons


def test_missing_output_path_is_invalid(tmp_path: Path):
    missing = tmp_path / "missing.png"
    text = f"""[NSFW STABILITY ROUND RESULTS]
output_path: {missing}
file_sha256: abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890
"""

    result = _validate_gateway_report_integrity(text, attachment_count=1)

    assert result.valid is False
    assert "output_path_missing" in result.reasons


def test_slack_pass_with_zero_attachments_is_invalid(tmp_path: Path):
    image = tmp_path / "result.png"
    image.write_bytes(b"png-bytes")
    text = f"""[NSFW STABILITY ROUND RESULTS]
output_path: {image}
Slack Upload: PASS
file_sha256: {"0" * 64}
"""

    result = _validate_gateway_report_integrity(text, attachment_count=0)

    assert result.valid is False
    assert "slack_pass_without_attachment" in result.reasons


def test_nas_pass_with_missing_mirror_is_invalid(tmp_path: Path):
    image = tmp_path / "result.png"
    image.write_bytes(b"png-bytes")
    text = f"""[NSFW STABILITY ROUND RESULTS]
output_path: {image}
NAS Hook: PASS
file_sha256: {"0" * 64}
"""

    result = _validate_gateway_report_integrity(text, attachment_count=1)

    assert result.valid is False
    assert "nas_pass_without_mirror" in result.reasons


def test_single_png_safe_verify_allows_missing_sidecar_when_png_completed(tmp_path: Path):
    image = tmp_path / "single.png"
    image.write_bytes(b"real image bytes")
    text = f"""[SINGLE PNG SAFE VERIFY]
ComfyUI Submit: YES
prompt_id: 6f724fbb-2160-4846-bd5a-ebd6e95724ad
History Status: COMPLETED
PNG Created: YES
output_path: {image}
Final Verdict: PNG_GENERATION_PASS
"""

    result = _validate_gateway_report_integrity(text, attachment_count=0)

    assert result.valid is True
    assert result.text == text
    assert not any(reason.startswith("sidecar_missing:") for reason in result.reasons)


@pytest.mark.parametrize(
    "history_line",
    [
        "history_status: success",
        "status_str: success",
        "history status: success",
        "history_status: completed",
        "history status: completed",
        "completed: true",
        "history completed: true",
    ],
)
def test_single_png_safe_verify_accepts_completed_equivalent_history_statuses(
    tmp_path: Path,
    history_line: str,
):
    image = tmp_path / "single.png"
    image.write_bytes(b"real image bytes")
    text = f"""[SINGLE PNG SAFE VERIFY]
ComfyUI Submit: YES
prompt_id: 6f724fbb-2160-4846-bd5a-ebd6e95724ad
{history_line}
PNG Created: YES
output_path: {image}
Final Verdict: PNG_GENERATION_PASS
"""

    result = _validate_gateway_report_integrity(text, attachment_count=0)

    assert result.valid is True
    assert result.reasons == []


@pytest.mark.parametrize(
    "history_line",
    [
        "history_status: error",
        "history_status: interrupted",
        "history_status: running",
    ],
)
def test_single_png_safe_verify_rejects_incomplete_history_statuses(tmp_path: Path, history_line: str):
    image = tmp_path / "single.png"
    image.write_bytes(b"real image bytes")
    text = f"""[SINGLE PNG SAFE VERIFY]
ComfyUI Submit: YES
prompt_id: 6f724fbb-2160-4846-bd5a-ebd6e95724ad
{history_line}
PNG Created: YES
output_path: {image}
Final Verdict: PNG_GENERATION_PASS
"""

    result = _validate_gateway_report_integrity(text, attachment_count=0)

    assert result.valid is False
    assert "history_not_completed" in result.reasons


def test_single_png_safe_verify_rejects_fake_prompt_id(tmp_path: Path):
    image = tmp_path / "single.png"
    image.write_bytes(b"real image bytes")
    text = f"""[SINGLE PNG SAFE VERIFY]
ComfyUI Submit: YES
prompt_id: fake-prompt-id
History Status: COMPLETED
PNG Created: YES
output_path: {image}
Final Verdict: PNG_GENERATION_PASS
"""

    result = _validate_gateway_report_integrity(text, attachment_count=0)

    assert result.valid is False
    assert "prompt_id_invalid" in result.reasons


def test_single_png_safe_verify_still_rejects_missing_output_path(tmp_path: Path):
    missing = tmp_path / "missing.png"
    text = f"""[SINGLE PNG SAFE VERIFY]
ComfyUI Submit: YES
prompt_id: 6f724fbb-2160-4846-bd5a-ebd6e95724ad
History Status: COMPLETED
PNG Created: YES
output_path: {missing}
Final Verdict: PNG_GENERATION_PASS
"""

    result = _validate_gateway_report_integrity(text, attachment_count=0)

    assert result.valid is False
    assert "output_path_missing" in result.reasons


def test_nsfw_result_still_requires_sidecar(tmp_path: Path):
    image = tmp_path / "result.png"
    image.write_bytes(b"real image bytes")
    text = f"""[NSFW STABILITY ROUND RESULTS]
prompt_id: 6f724fbb-2160-4846-bd5a-ebd6e95724ad
History Status: COMPLETED
output_path: {image}
file_sha256: {"0" * 64}
Slack Upload: FAIL
NAS Hook: FAIL
"""

    result = _validate_gateway_report_integrity(text, attachment_count=0)

    assert result.valid is False
    assert "sidecar_missing:prompt.json" in result.reasons


def test_nsfw_result_report_is_not_compacted():
    text = "[NSFW STABILITY ROUND RESULTS]\n" + "\n".join(
        f"checkpoint line {idx:02d}: NAS Hook: PASS Slack Upload: PASS provenance detail"
        for idx in range(1, 31)
    )

    compacted, applied = _compact_gateway_final_response(text)

    assert applied is False
    assert compacted == text
    assert "lines omitted" not in compacted
    assert "checkpoint line 30" in compacted


def test_gemma_channel_artifact_sanitizer_still_removes_thought_prefix():
    text = "<|channel>thought\ninternal\n<channel|>[ANGELICA FINAL RESPONSE]\nVISIBLE"

    sanitized = _sanitize_gateway_final_response(Platform.SLACK, text)

    assert sanitized == "[ANGELICA FINAL RESPONSE]\nVISIBLE"
    assert "internal" not in sanitized
    assert "<|channel" not in sanitized
    assert "<channel|>" not in sanitized


def test_valid_result_is_preserved(tmp_path: Path):
    image = tmp_path / "result.png"
    image.write_bytes(b"real image bytes")
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    for name in ("prompt.json", "metadata.json", "workflow.json", "manifest.json"):
        (sidecar / name).write_text("{}")

    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    text = f"""[NSFW STABILITY ROUND RESULTS]
checkpoint: valid.safetensors
output_path: {image}
file_sha256: {digest}
Slack Upload: PASS
NAS Hook: FAIL
"""

    result = _validate_gateway_report_integrity(text, attachment_count=1)

    assert result.valid is True
    assert result.text == text
    assert result.reasons == []


def test_background_process_summary_with_hash_labels_is_not_a_generation_report():
    text = """확인했습니다.

이미 완료된 실제 산출물:
- 8개 체크포인트 PNG 생성 완료
- prompt_id / history_status / file_sha256 / prompt_hash / workflow_hash 실계산 완료
- Slack 첨부 업로드 완료

즉, 백그라운드 프로세스 실패가 아니라 개별 실행으로 완주한 상태입니다.
"""

    result = _validate_gateway_report_integrity(text, attachment_count=0)

    assert result.valid is True
    assert result.reasons == []
    assert result.text == text


@pytest.mark.parametrize(
    "header",
    [
        "[FULL BODY CHALLENGER ROUND V1 RESULTS]",
        "[KEY VISUAL CHALLENGER ROUND V1 RESULTS]",
    ],
)
def test_multi_checkpoint_generation_reports_are_not_delivery_compacted(header: str):
    text = header + "\n" + "\n".join(
        f"checkpoint line {idx:02d}: NAS Hook: PASS Slack Upload: PASS file_sha256: {'a' * 64}"
        for idx in range(1, 40)
    )

    compacted, applied = _compact_gateway_final_response(text)

    assert applied is False
    assert compacted == text
    assert "lines omitted" not in compacted
    assert "checkpoint line 39" in compacted


def test_key_visual_challenger_report_is_not_post_upload_compacted():
    from gateway.platforms.base import _compact_post_upload_reporting

    checkpoints = [
        "hdaRainbowIllusMixV1_v13.safetensors",
        "cottonillustrious_v20.safetensors",
        "catCarrier_v90.safetensors",
        "pornmasterAnime_ilV5.safetensors",
        "hakushiMix_v141.safetensors",
        "illumiyumeXL_v35VPred.safetensors",
        "waiIllustriousSDXL_v170.safetensors",
        "animagine-xl-4.0-opt.safetensors",
    ]
    blocks = []
    for idx, checkpoint in enumerate(checkpoints, start=1):
        blocks.append(
            f"""checkpoint:
{checkpoint}
prompt_id:
6f724fbb-2160-4846-bd5a-ebd6e95724ad
history_status:
success
output_path:
/Users/hermes/HermesWork/Image/key_visual_challenger_round_v1/{checkpoint.removesuffix('.safetensors')}/{checkpoint.removesuffix('.safetensors')}_00001_.png
prompt_hash:
{'a' * 64}
workflow_hash:
{'b' * 64}
file_sha256:
{idx:064x}
artifact_index:
{idx}
sidecar:
PASS
Slack Upload:
PASS
NAS Hook:
FAIL"""
        )
    text = "[KEY VISUAL CHALLENGER ROUND V1 RESULTS]\n\n" + "\n\n".join(blocks)

    compacted = _compact_post_upload_reporting(text)

    assert compacted == text
    assert "lines omitted" not in compacted
    assert "animagine-xl-4.0-opt.safetensors" in compacted
    assert len(compacted.splitlines()) >= 170


def test_full_body_report_with_valid_hashes_but_missing_nas_mirror_is_invalid(tmp_path: Path):
    image_dir = tmp_path / "fullbody_challenger_round_v1" / "valid"
    image_dir.mkdir(parents=True)
    image = image_dir / "valid_00001_.png"
    image.write_bytes(b"real image bytes")
    sidecar = image_dir / "sidecar"
    sidecar.mkdir()
    for name in ("prompt.json", "metadata.json", "workflow.json", "manifest.json"):
        (sidecar / name).write_text("{}")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    text = f"""[FULL BODY CHALLENGER ROUND V1 RESULTS]
checkpoint:
valid.safetensors
prompt_id:
6f724fbb-2160-4846-bd5a-ebd6e95724ad
history_status:
success
output_path:
{image}
file_sha256:
{digest}
Slack Upload:
PASS
NAS Hook:
PASS
"""

    result = _validate_gateway_report_integrity(text, attachment_count=1)

    assert result.valid is False
    assert "nas_pass_without_mirror" in result.reasons
    assert "file_sha256_unparseable" not in result.reasons


def test_full_body_report_still_rejects_missing_sidecar(tmp_path: Path):
    image = tmp_path / "result.png"
    image.write_bytes(b"real image bytes")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    text = f"""[FULL BODY CHALLENGER ROUND V1 RESULTS]
checkpoint: valid.safetensors
prompt_id: 6f724fbb-2160-4846-bd5a-ebd6e95724ad
history_status: success
output_path: {image}
file_sha256: {digest}
Slack Upload: FAIL
NAS Hook: FAIL
"""

    result = _validate_gateway_report_integrity(text, attachment_count=0)

    assert result.valid is False
    assert "sidecar_missing:prompt.json" in result.reasons
