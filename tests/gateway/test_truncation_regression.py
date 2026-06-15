import pytest

from gateway.run import _compact_gateway_final_response, _looks_like_gateway_delivery_summary


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
        assert len(lines) <= 8
        assert compacted.count("NAS Hook: PASS") <= 1
        assert compacted.count("provenance: PASS") <= 1
        assert repeated_path * 2 not in compacted

    def test_compact_gateway_final_response_leaves_short_messages_alone(self):
        text = "짧은 완료 메시지"
        compacted, applied = _compact_gateway_final_response(text)
        assert compacted == text
        assert applied is False
