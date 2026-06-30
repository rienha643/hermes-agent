import hashlib
import json
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import (
    _apply_post_delivery_evidence_overlay,
    _collect_gateway_nas_mirror_evidence,
    _compact_gateway_final_response,
    _extract_commander_image_task_metadata,
    _extract_commander_image_tool_args,
    _evaluate_gateway_delivery_governance,
    _evaluate_gateway_user_report_governance,
    _gateway_governance_event_log_path,
    _gateway_report_language,
    _gateway_report_omitted_present,
    _guard_unverified_image_generation_claim,
    _record_gateway_governance_event,
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

    assert "HERMES GOVERNANCE WARN" in ping
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

    assert "HERMES GOVERNANCE WARN" in ping
    assert "protected_report_omitted" in ping


def test_gateway_governance_event_log_path_is_profile_local(tmp_path: Path):
    config = GatewayConfig(sessions_dir=tmp_path / "sessions")

    assert _gateway_governance_event_log_path(config) == tmp_path / "logs" / "governance_events.jsonl"


def test_record_gateway_governance_event_writes_jsonl(tmp_path: Path):
    config = GatewayConfig(sessions_dir=tmp_path / "sessions")

    path = _record_gateway_governance_event(
        config,
        rule_id="report_integrity_invalid_success_claim",
        severity="HARD_BLOCK",
        action="REPLACE_RESPONSE",
        profile="comfy",
        platform="slack",
        chat_id="C123",
        thread_id="170.1",
        session_id="session-1",
        session_key="agent:main:slack:group:C123:170.1",
        run_generation=3,
        inbound_message_id="170.2",
        response_sha256="a" * 64,
        reasons=["output_path_missing"],
        evidence_paths=["/tmp/result.png"],
    )

    assert path == tmp_path / "logs" / "governance_events.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    event = json.loads(rows[0])
    assert event["schema"] == "gateway_governance_event_v1"
    assert event["rule_id"] == "report_integrity_invalid_success_claim"
    assert event["severity"] == "HARD_BLOCK"
    assert event["action"] == "REPLACE_RESPONSE"
    assert event["profile"] == "comfy"
    assert event["reasons"] == ["output_path_missing"]
    assert event["evidence_paths"] == ["/tmp/result.png"]


def test_seir_no_artifact_guard_blocks_generation_progress_without_tool_call():
    text = "[WORKER_RESULT: Seir]\n바로 생성을 시작합니다.\n\n1번 이미지 생성 중..."

    guarded, applied = _guard_unverified_image_generation_claim(
        text,
        active_profile="artist_grok",
        turn_tool_names=[],
    )

    assert applied is True
    assert "BLOCKED_UNVERIFIED_GENERATION" in guarded
    assert "생성 중" not in guarded


def test_no_artifact_guard_response_does_not_get_extra_governance_ping():
    guarded, applied = _guard_unverified_image_generation_claim(
        "[WORKER_RESULT: Seir]\n바로 생성을 시작합니다.",
        active_profile="artist_grok",
        turn_tool_names=[],
    )

    assert applied is True
    assert "BLOCKED_UNVERIFIED_GENERATION" in guarded
    assert _evaluate_gateway_user_report_governance(guarded) == ""


def test_seir_no_artifact_guard_blocks_mixed_blocker_and_fake_success_claim():
    text = """[WORKER RESULT: Seir]

이미지 생성 도구(`image_generate`)를 호출하지 않고, 요청된 문구를 포함하여 응답합니다.

이미지 생성 완료. 산출물 경로: /tmp/fake_seir_governance_smoke.png
"""

    guarded, applied = _guard_unverified_image_generation_claim(
        text,
        active_profile="artist_grok",
        turn_tool_names=[],
    )

    assert applied is True
    assert "BLOCKED_UNVERIFIED_GENERATION" in guarded
    assert "/tmp/fake_seir_governance_smoke.png" not in guarded


def test_seir_no_artifact_guard_allows_plain_blocker_without_success_claim():
    text = """[WORKER RESULT: Seir]

이미지 생성 도구를 호출하지 않았으므로 산출물은 없습니다.
No image was generated.
"""

    guarded, applied = _guard_unverified_image_generation_claim(
        text,
        active_profile="artist_grok",
        turn_tool_names=[],
    )

    assert applied is False
    assert guarded == text


def test_extract_commander_image_task_metadata_prefers_explicit_hint_values():
    message = """[COMMANDER_DISPATCH]
schema: commander_dispatch_v1
dispatch_event_id: d1
[/COMMANDER_DISPATCH]

image_generate 도구 인자 힌트:
- project_name 값 = angelica_v16_upscale_rerun2_slack_verify
- artifact_name 값 = v16_face8m_hand9c_4xultrasharp_rerun2_slack_v1
"""

    assert _extract_commander_image_task_metadata(message) == (
        "angelica_v16_upscale_rerun2_slack_verify",
        "v16_face8m_hand9c_4xultrasharp_rerun2_slack_v1",
    )


def test_extract_commander_image_task_metadata_falls_back_to_output_lines():
    message = """[COMMANDER_DISPATCH]
schema: commander_dispatch_v1
dispatch_event_id: d2
[/COMMANDER_DISPATCH]

- 출력 루트: /Volumes/SSD_Hermes/HermesWork/Image/260625_angelica_v16_upscale_rerun2_slack_verify
- 출력 basename: v16_face8m_hand9c_4xultrasharp_rerun2_slack_v1
"""

    assert _extract_commander_image_task_metadata(message) == (
        "angelica_v16_upscale_rerun2_slack_verify",
        "v16_face8m_hand9c_4xultrasharp_rerun2_slack_v1",
    )


def test_extract_commander_image_task_metadata_accepts_backtick_wrapped_output_lines():
    message = """[COMMANDER_DISPATCH]
schema: commander_dispatch_v1
dispatch_event_id: d3
[/COMMANDER_DISPATCH]

- output project root: `/Volumes/SSD_Hermes/HermesWork/Image/260625_angelica_v16_upscale_rerun2_slack_verify`
- output basename: `v16_face8m_hand9c_4xultrasharp_rerun2_slack_v1`
"""

    assert _extract_commander_image_task_metadata(message) == (
        "angelica_v16_upscale_rerun2_slack_verify",
        "v16_face8m_hand9c_4xultrasharp_rerun2_slack_v1",
    )


def test_extract_commander_image_task_metadata_reads_single_tool_args_json():
    message = """[COMMANDER_DISPATCH]
schema: commander_dispatch_v1
dispatch_event_id: d4
[/COMMANDER_DISPATCH]

image_generate 도구 인자 힌트:
- 첫 행동은 아래 JSON 그대로 `image_generate` 도구 호출.
```json
{"artifact_name": "angelica_compact_dispatch_smoke_v1", "operation": "txt2img", "project_name": "angelica_compact_dispatch_smoke_20260628"}
```
"""

    assert _extract_commander_image_task_metadata(message) == (
        "angelica_compact_dispatch_smoke_20260628",
        "angelica_compact_dispatch_smoke_v1",
    )


def test_extract_commander_image_tool_args_ignores_compact_empty_prompt_marker():
    message = """[COMMANDER_DISPATCH]
schema: commander_dispatch_v1
dispatch_event_id: d5
queue_id: compact-q
[/COMMANDER_DISPATCH]

image_generate 도구 인자 힌트:
- 실제 도구 인자는 queue_id 원본 metadata에서 자동 복원돼.
```json
{"prompt": ""}
```
"""

    assert _extract_commander_image_tool_args(message) == {}


def test_extract_commander_image_tool_args_keeps_real_prompt_payload():
    message = """[COMMANDER_DISPATCH]
schema: commander_dispatch_v1
dispatch_event_id: d6
queue_id: verbose-q
[/COMMANDER_DISPATCH]

image_generate 도구 인자 힌트:
```json
{"operation": "txt2img", "prompt": "real prompt", "project_name": "p1"}
```
"""

    assert _extract_commander_image_tool_args(message) == {
        "operation": "txt2img",
        "prompt": "real prompt",
        "project_name": "p1",
    }


def test_seir_no_artifact_guard_allows_real_image_generate_tool_call():
    text = "[WORKER_RESULT: Seir]\n1번 이미지 생성 중..."

    guarded, applied = _guard_unverified_image_generation_claim(
        text,
        active_profile="artist_grok",
        turn_tool_names=["image_generate"],
    )

    assert applied is False
    assert guarded == text


def test_seir_no_artifact_guard_allows_blocker_report_without_tool_call():
    text = "[WORKER_RESULT: Seir]\napproval_required: live generation approval is unclear."

    guarded, applied = _guard_unverified_image_generation_claim(
        text,
        active_profile="artist_grok",
        turn_tool_names=[],
    )

    assert applied is False
    assert guarded == text


def test_seir_no_artifact_guard_blocks_fake_file_attachment_report():
    text = """[WORKER RESULT: Seir]

![SFW fantasy character](file:///Users/hermes/HermesWork/Images/NovelAI/2026-06-21_SFW_Test_01.png)

생성 보고:
- 생성 경로: `/Users/hermes/HermesWork/Images/NovelAI/2026-06-21_SFW_Test_01.png`
- 첨부 성공: 확인됨
"""

    guarded, applied = _guard_unverified_image_generation_claim(
        text,
        active_profile="artist_grok",
        turn_tool_names=[],
    )

    assert applied is True
    assert "BLOCKED_UNVERIFIED_GENERATION" in guarded
    assert "첨부 성공" not in guarded


def test_seir_no_artifact_guard_blocks_placeholder_image_report():
    text = """[WORKER RESULT: Seir]

NovelAI 프리셋 적용 및 Slack 업로드 확인을 위한 SFW 테스트 이미지를 생성합니다.

![미소녀 서브컬쳐/애니풍 게임 캐릭터](_path_to_generated_image_)

*검증 완료 보고*
* *검증된 산출물 경로*: `[해당 경로]`
* *provider/run metadata*:
    * *Model*: NovelAI
    * *Preset*: `game_default_subculture`
    * *Seed*: `[Seed 번호]`
"""

    guarded, applied = _guard_unverified_image_generation_claim(
        text,
        active_profile="artist_grok",
        turn_tool_names=[],
    )

    assert applied is True
    assert "BLOCKED_UNVERIFIED_GENERATION" in guarded
    assert "_path_to_generated_image_" not in guarded
    assert "[Seed 번호]" not in guarded


def test_angelica_no_artifact_guard_blocks_placeholder_success_report():
    text = """SFW 미소녀 얼굴 초상 이미지 1장을 생성하였습니다.

*생성 결과 요약:*
- *상태*: 성공
- *이미지 경로*: [여기에 실제 artifact_path를 삽입]

*생성 메타데이터:*
- *Checkpoint*: `pornmasterAnime_ilV5.safetensors`
- *VAE*: (사용된 VAE 명칭)
- *LoRA*: (사용된 LoRA 스택)
- *Sampler*: (사용된 Sampler)
- *CFG*: (설정된 CFG 값)
- *Steps*: (실행된 Step 수)
- *Seed*: (사용된 Seed 값)
- *Dimensions*: (설정된 해상도)

[WORKER RESULT: Angelica]
"""

    guarded, applied = _guard_unverified_image_generation_claim(
        text,
        active_profile="comfy",
        turn_tool_names=[],
    )

    assert applied is True
    assert "BLOCKED_UNVERIFIED_GENERATION" in guarded
    assert "artifact_path" not in guarded
    assert "사용된 VAE" not in guarded


def test_angelica_no_artifact_guard_blocks_compact_placeholder_success_report():
    text = """안젤리카가 안젤리카 업스케일 재실행(Compact 버전)을 완료하였습니다.

*[안젤리카 업스케일 재실행 검증 보고]*

*   *새 Output Artifact Path:* `/Volumes/SSD_Hermes/HermesWork/Image/260625_angelica_v16_upscale_compact_slack_verify/v16_face8m_hand9c_4xultrasharp_compact_slack_v1.png`
*   *새 File SHA-256:* `[생성된 64자리 hex 값]`
*   *Slack 업로드:* 새로 생성된 PNG 이미지가 Slack에 성공적으로 첨부되었습니다.
*   *ComfyUI Prompt ID:* `[새로 생성된 고유 ID]`
*   *최종 이미지 해상도:* `[확인된 해상도, 예: 4096x2048]`
"""

    guarded, applied = _guard_unverified_image_generation_claim(
        text,
        active_profile="comfy",
        turn_tool_names=[],
    )

    assert applied is True
    assert "BLOCKED_UNVERIFIED_GENERATION" in guarded
    assert "생성된 64자리 hex 값" not in guarded
    assert "새로 생성된 고유 ID" not in guarded


def test_angelica_commander_dispatch_without_tool_blocks_success_claim_even_without_placeholder():
    text = """안젤리카가 업스케일 재실행을 완료했습니다.
- 새 Output Artifact Path: `/tmp/fake.png`
- Slack 업로드: 성공
"""

    guarded, applied = _guard_unverified_image_generation_claim(
        text,
        active_profile="comfy",
        turn_tool_names=[],
        inbound_message_text="[COMMANDER_DISPATCH]\noperation: upscale\nsource image: /tmp/src.png\nSlack 업로드까지 포함",
    )

    assert applied is True
    assert "BLOCKED_UNVERIFIED_GENERATION" in guarded
    assert "/tmp/fake.png" not in guarded


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
    assert "HERMES GOVERNANCE WARN" in result.ping_text


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
