import pytest

from gateway.platforms.base import _compact_post_upload_reporting
from gateway.config import Platform
from gateway.run import (
    _compact_gateway_final_response,
    _looks_like_gateway_delivery_summary,
    _sanitize_gateway_final_response,
    _split_gateway_evaluation_report_parts,
)


class TestGatewayTruncationRegression:
    def test_compact_gateway_final_response_dedupes_delivery_summaries(self):
        repeated_path = "/tmp/" + ("portrait_round_v1/" * 18) + "artifact.json"
        block = (
            "[WORKER RESULT: Eclipse]\n"
            "\nportrait_round_v1 Slack 전달 완료: 이미지 4개 첨부\n"
            f"artifact summary: {repeated_path}\n"
            "NAS Hook: PASS\n"
            "provenance: PASS"
        )
        text = f"{block}\n\n{block}\n\n{block}"

        compacted, applied = _compact_gateway_final_response(text)

        assert applied is True
        assert _looks_like_gateway_delivery_summary(compacted) is True
        lines = [line for line in compacted.splitlines() if line.strip()]
        assert "lines omitted" not in compacted
        assert "[truncated]" not in compacted
        assert len(lines) == len([line for line in block.splitlines() if line.strip()])
        assert compacted.count("NAS Hook: PASS") <= 1
        assert compacted.count("provenance: PASS") <= 1
        assert repeated_path in compacted

    def test_compact_gateway_final_response_leaves_short_messages_alone(self):
        text = "짧은 완료 메시지"
        compacted, applied = _compact_gateway_final_response(text)
        assert compacted == text
        assert applied is False

    def test_long_general_final_response_is_not_compacted(self):
        text = "\n".join(f"general final line {idx:02d}" for idx in range(1, 31))

        compacted, applied = _compact_gateway_final_response(text)

        assert applied is False
        assert compacted == text
        assert "lines omitted" not in compacted
        assert "general final line 30" in compacted

    def test_angelica_style_trace_response_preserves_25_lines(self):
        text = "\n".join(
            ["[PROGRESS_STREAM_TRACE]"]
            + [f"line {idx:02d}" for idx in range(1, 26)]
            + ["[/PROGRESS_STREAM_TRACE]"]
        )

        compacted, applied = _compact_gateway_final_response(text)

        assert applied is False
        assert compacted == text
        assert "lines omitted" not in compacted
        assert "line 25" in compacted

    def test_compact_gateway_final_response_preserves_worker_body_without_delivery_markers(self):
        text = "[WORKER RESULT: Angelica]\n" + "\n".join(
            f"validation line {idx:02d}" for idx in range(1, 31)
        )

        compacted, applied = _compact_gateway_final_response(text)

        assert applied is False
        assert compacted == text
        assert "lines omitted" not in compacted
        assert "validation line 30" in compacted

    def test_long_portrait_review_is_not_compacted(self):
        sections = [
            "Portrait Review",
            "Checkpoint Review",
            "NSFW-lite Review",
            "Full Body Review",
            "Key Visual Review",
            "Evaluation Report",
            "Scoring Report",
            "Main / Reserve Review",
        ]
        text = "\n".join(
            sections
            + [f"평가 항목 {idx}: 점수 근거와 관찰 내용을 생략 없이 기록합니다." for idx in range(1, 45)]
        )

        compacted, applied = _compact_gateway_final_response(text)

        assert applied is False
        assert compacted == text
        assert "lines omitted" not in compacted
        assert "[truncated]" not in compacted
        assert "평가 항목 44" in compacted

    def test_long_evaluation_report_splits_without_deleting_content(self):
        body_lines = [
            "Evaluation Report",
            "Portrait Review",
        ] + [f"원문 평가 라인 {idx}: 보존 대상 내용입니다." for idx in range(1, 90)]
        text = "\n".join(body_lines)

        parts = _split_gateway_evaluation_report_parts(text, max_chars=900)

        assert len(parts) >= 2
        assert parts[0].startswith("[PART 1/")
        assert parts[1].startswith("[PART 2/")
        assert all("lines omitted" not in part for part in parts)
        assert all("[truncated]" not in part for part in parts)
        assert all(len(part) <= 900 for part in parts)
        reconstructed = "".join(part.split("\n", 1)[1] for part in parts)
        assert reconstructed == text
        assert "원문 평가 라인 89" in reconstructed

    def test_post_upload_reporting_still_compacts_non_evaluation_summaries(self):
        block = "Slack 전달 완료\nartifact summary: /tmp/file.png\nNAS Hook: PASS\nprovenance: PASS"
        compacted = _compact_post_upload_reporting((block + "\n\n") * 12)

        assert compacted == block
        assert "lines omitted" not in compacted
        assert "[truncated]" not in compacted

    def test_post_upload_reporting_does_not_compact_evaluation_reports(self):
        text = "Evaluation Report\n" + "\n".join(
            f"점수 근거 {idx}: 평가 보고서 본문" for idx in range(1, 40)
        )

        compacted = _compact_post_upload_reporting(text)

        assert compacted == text
        assert "lines omitted" not in compacted
        assert "점수 근거 39" in compacted

    def test_gemma4_channel_marker_prefix_is_removed_from_final_response(self):
        text = "<channel|>PONG"

        sanitized = _sanitize_gateway_final_response(Platform.SLACK, text)

        assert sanitized == "PONG"

    def test_gemma4_thought_channel_wrapper_keeps_final_answer_only(self):
        text = "<|channel>thought\nThe user wants PONG.\n<channel|>ANSWER"

        sanitized = _sanitize_gateway_final_response(Platform.SLACK, text)

        assert sanitized == "ANSWER"
        assert "The user wants" not in sanitized
        assert "<|channel" not in sanitized
        assert "<channel|>" not in sanitized

    def test_gemma4_pipe_channel_thought_artifact_is_removed(self):
        text = "<|channel|>thought\ninternal reasoning\n<channel|>VISIBLE"

        sanitized = _sanitize_gateway_final_response(Platform.SLACK, text)

        assert sanitized == "VISIBLE"
        assert "internal reasoning" not in sanitized
        assert "channel" not in sanitized
