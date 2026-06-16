from pathlib import Path

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
    import hashlib

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
