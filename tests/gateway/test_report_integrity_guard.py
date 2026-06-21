import hashlib
from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.run import (
    _apply_post_delivery_evidence_overlay,
    _collect_gateway_nas_mirror_evidence,
    _compact_gateway_final_response,
    _evaluate_gateway_delivery_governance,
    _evaluate_gateway_user_report_governance,
    _gateway_report_language,
    _gateway_report_omitted_present,
    _guard_seir_unverified_generation_claim,
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


def test_gateway_report_language_detects_english_only_completion_report():
    text = """Status: COMPLETE
Summary: The requested task is done.
Next action: Continue with cron stabilization.
"""

    assert _gateway_report_language(text) == "en"
    ping = _evaluate_gateway_user_report_governance(text)

    assert "GOVERNANCE PING" in ping
    assert "report_language_mismatch" in ping
    assert "blocked_state: REPORT_FAIL" in ping


def test_gateway_report_language_allows_korean_with_fixed_english_identifiers():
    text = """상태: 완료
endpoint: http://127.0.0.1:8080
model: gemma-4-12b-it-uncensored-Q4_K_M
다음 조치: cron 안정화로 진행.
"""

    assert _gateway_report_language(text) == "ko"
    assert _evaluate_gateway_user_report_governance(text) == ""


def test_gateway_user_report_governance_pings_on_omitted_protected_report():
    text = """상태: 점검 완료
... (42 lines omitted)
다음 조치: 누락된 원문 보고를 확인.
"""

    assert _gateway_report_omitted_present(text) is True
    ping = _evaluate_gateway_user_report_governance(text)

    assert "GOVERNANCE PING" in ping
    assert "protected_report_omitted" in ping


def test_seir_no_artifact_guard_blocks_generation_progress_without_tool_call():
    text = "[WORKER_RESULT: Seir]\n바로 생성을 시작합니다.\n\n1번 이미지 생성 중..."

    guarded, applied = _guard_seir_unverified_generation_claim(
        text,
        active_profile="artist_grok",
        turn_tool_names=[],
    )

    assert applied is True
    assert "BLOCKED_UNVERIFIED_GENERATION" in guarded
    assert "생성 중" not in guarded


def test_seir_no_artifact_guard_allows_real_image_generate_tool_call():
    text = "[WORKER_RESULT: Seir]\n1번 이미지 생성 중..."

    guarded, applied = _guard_seir_unverified_generation_claim(
        text,
        active_profile="artist_grok",
        turn_tool_names=["image_generate"],
    )

    assert applied is False
    assert guarded == text


def test_seir_no_artifact_guard_allows_blocker_report_without_tool_call():
    text = "[WORKER_RESULT: Seir]\napproval_required: live generation approval is unclear."

    guarded, applied = _guard_seir_unverified_generation_claim(
        text,
        active_profile="artist_grok",
        turn_tool_names=[],
    )

    assert applied is False
    assert guarded == text


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


@pytest.mark.parametrize(
    "header",
    [
        "[COMFYUI RESTART CHECKPOINT VERIFY]",
        "[E2E VALIDATION RESULT]",
        "[E2E VALIDATION ROUND V1]",
        "[CHECKPOINT MIGRATION FINALIZATION]",
        "[CHECKPOINT SELECTION FINALIZATION V2]",
    ],
)
def test_validation_and_migration_reports_are_not_post_upload_compacted(header: str):
    from gateway.platforms.base import _compact_post_upload_reporting

    text = header + "\n" + "\n".join(
        f"validation line {idx:02d}: Slack Upload: PASS provenance checkpoint detail"
        for idx in range(1, 56)
    )

    compacted = _compact_post_upload_reporting(text)

    assert compacted == text
    assert "lines omitted" not in compacted
    assert "validation line 55" in compacted


@pytest.mark.parametrize(
    "header",
    [
        "[COMFYUI RESTART CHECKPOINT VERIFY]",
        "[E2E VALIDATION RESULT]",
        "[E2E VALIDATION ROUND V1]",
        "[CHECKPOINT MIGRATION FINALIZATION]",
        "[CHECKPOINT SELECTION FINALIZATION V2]",
    ],
)
def test_validation_and_migration_reports_are_not_gateway_final_compacted(header: str):
    text = header + "\n" + "\n".join(
        f"validation line {idx:02d}: delivery complete artifact summary detail"
        for idx in range(1, 56)
    )

    compacted, applied = _compact_gateway_final_response(text)

    assert applied is False
    assert compacted == text
    assert "lines omitted" not in compacted
    assert "validation line 55" in compacted


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


def test_governance_delivery_ping_overrides_worker_ping_no_when_slack_evidence_missing(tmp_path: Path):
    image = tmp_path / "260617_HermesWork_Image" / "8f461990-77d3-4b9c-a0e8-8f19362db3ef" / "windows-remote-comfyui-e2e-512_v1.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"real image bytes")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    text = f"""[ANGELICA WINDOWS REMOTE COMFYUI GOVERNANCE FRESH E2E V6]
prompt_id: 8f461990-77d3-4b9c-a0e8-8f19362db3ef
output_path: {image}
file_sha256: {digest}
Slack:
FAIL
Slack Evidence:
message_id: UNKNOWN
thread_ts: UNKNOWN
NAS:
FAIL
nas_sha256: UNKNOWN
Governance Ping:
NO
Final State:
FAIL
"""

    result = _evaluate_gateway_delivery_governance(text, attachment_count=0)

    assert result.decision.final_state == "GOVERNANCE_PING"
    assert "file_upload_evidence" in result.decision.missing_evidence
    assert "mirror_hash_evidence" in result.decision.missing_evidence
    assert result.decision.auto_blocking is False
    assert "GOVERNANCE PING" in result.ping_text


def test_nas_hook_requested_without_mirror_hash_remains_unknown_and_pings(tmp_path: Path):
    image = tmp_path / "scope" / "result.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"real image bytes")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    text = f"""[ANGELICA WINDOWS REMOTE COMFYUI GOVERNANCE FRESH E2E V6]
output_path: {image}
file_sha256: {digest}
Slack:
PASS
NAS:
FAIL
nas_hook_requested: True
nas_sha256: UNKNOWN
Governance Ping:
NO
"""

    result = _evaluate_gateway_delivery_governance(text, attachment_count=1)

    assert result.nas_status == "UNKNOWN"
    assert "mirror_hash_evidence" in result.decision.missing_evidence
    assert result.decision.final_state == "GOVERNANCE_PING"


def test_nas_mirror_hash_match_builds_pass_evidence(tmp_path: Path):
    source_root = tmp_path / "source"
    mirror_root = tmp_path / "mirror" / "Image"
    image = source_root / "260617_HermesWork_Image" / "8f461990-77d3-4b9c-a0e8-8f19362db3ef" / "windows-remote-comfyui-e2e-512_v1.png"
    mirror = mirror_root / image.parent.name / image.name
    image.parent.mkdir(parents=True)
    mirror.parent.mkdir(parents=True)
    image.write_bytes(b"real image bytes")
    mirror.write_bytes(b"real image bytes")

    evidence = _collect_gateway_nas_mirror_evidence(image, mirror_roots=[mirror_root])

    assert evidence.mirror_exists is True
    assert evidence.hash_match is True
    assert evidence.source_sha256 == evidence.mirror_sha256
    assert evidence.mirror_path == mirror


def test_post_delivery_overlay_uses_persisted_slack_and_nas_evidence_over_worker_fail(tmp_path: Path):
    source_root = tmp_path / "source"
    mirror_root = tmp_path / "mirror" / "Image"
    image = source_root / "260617_HermesWork_Image" / "865d11b7-562e-4d9a-bc54-86c259fada37" / "windows-remote-comfyui-e2e-v7_v1.png"
    mirror = mirror_root / image.parent.name / image.name
    sidecar = image.parent / "sidecar"
    sidecar.mkdir(parents=True)
    mirror.parent.mkdir(parents=True)
    image.write_bytes(b"v7 real image bytes")
    mirror.write_bytes(b"v7 real image bytes")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    (sidecar / "slack_evidence.json").write_text(
        """
{
  "message_id": "1781768830.546689",
  "thread_ts": "1781722451.461429",
  "filename": "windows-remote-comfyui-e2e-v7_v1.png",
  "source_path": "SOURCE_PATH",
  "local_sha256": "DIGEST",
  "files_count": 1
}
""".replace("SOURCE_PATH", str(image)).replace("DIGEST", digest),
        encoding="utf-8",
    )
    worker_text = f"""[ANGELICA WINDOWS REMOTE COMFYUI GOVERNANCE FRESH E2E V7]
prompt_id: 865d11b7-562e-4d9a-bc54-86c259fada37
output_path: {image}
file_sha256: {digest}
Slack:
FAIL
Slack Evidence:
message_id: UNKNOWN
thread_ts: UNKNOWN
filename: UNKNOWN
NAS:
FAIL
nas_sha256: UNKNOWN
Governance Ping:
NO
Final State:
FAIL
"""

    overlay = _apply_post_delivery_evidence_overlay(worker_text, mirror_roots=[mirror_root])

    assert overlay.overlay_applied is True
    assert overlay.slack_overlay is True
    assert overlay.nas_overlay is True
    assert overlay.governance_overlay is True
    assert overlay.final_state == "COMPLETE"
    assert "Slack: PASS" in overlay.text
    assert "NAS: PASS" in overlay.text
    assert "Governance Ping: PASS" in overlay.text
    assert "message_id: 1781768830.546689" in overlay.text
    assert str(mirror) in overlay.text
    assert digest in overlay.text
