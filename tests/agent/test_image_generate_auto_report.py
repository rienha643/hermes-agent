from __future__ import annotations

from agent.conversation_loop import _build_image_generate_auto_completion_report


def test_auto_completion_report_includes_requested_and_resolved_checkpoint(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png")

    report = _build_image_generate_auto_completion_report(
        {
            "report_evidence": {
                "artifact_path": str(image_path),
                "operation": "txt2img",
                "preset": "portrait",
                "output_type": "portrait",
                "workflow_key": "portrait_round_v1_txt2img_v1",
                "checkpoint": "pornmasterAnime_ilV5.safetensors",
                "requested_checkpoint": "pornmasterAnime_ilV5.safetensors",
                "resolved_checkpoint": "pornmasterAnime_ilV5.safetensors",
            }
        },
        user_message="[COMMANDER_DISPATCH]\n작업 요청",
    )

    assert report is not None
    assert "- 요청 Checkpoint: `pornmasterAnime_ilV5.safetensors`" in report
    assert "- 해결 Checkpoint: `pornmasterAnime_ilV5.safetensors`" in report
