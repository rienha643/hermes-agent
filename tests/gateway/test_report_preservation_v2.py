from gateway.run import _compact_gateway_final_response
from gateway.platforms.base import _compact_post_upload_reporting
from gateway.report_preservation import (
    looks_like_operational_delivery_summary,
    looks_like_structured_report,
    looks_like_worker_result_report,
    should_preserve_report_body,
)
from tools.delegate_tool import _format_specialist_result_frame


def _assert_preserved_everywhere(text: str) -> None:
    assert should_preserve_report_body(text) is True
    gateway_text, applied = _compact_gateway_final_response(text)
    assert applied is False
    assert gateway_text == text
    assert "lines omitted" not in gateway_text

    post_text = _compact_post_upload_reporting(text)
    assert post_text == text
    assert "lines omitted" not in post_text


def test_nai_feasibility_study_preserved_despite_delivery_terms():
    text = "\n".join(
        [
            "[WORKER RESULT: Seir]",
            "[NAI FEASIBILITY STUDY V1]",
            "Goal: evaluate whether NAI can be used safely for the target workflow.",
            "API: NovelAI API compatibility and request format review.",
            "Cost: expected usage cost and free/paid tradeoffs.",
            "Commercial Use: license and account-risk notes.",
            "Automation: batch feasibility and operator approval gates.",
            "Risk: credential, policy, and delivery risks.",
            "Evidence: text-only study; no NAI live call was made.",
            "Recommended Pipeline: manual approval before any production call.",
            "Slack Upload: N/A — report text only.",
            "NAS Hook: N/A — no artifact generated.",
            "provenance: study report, not upload summary.",
            "Final Status: FEASIBILITY_REVIEW_COMPLETE",
        ]
        + [f"Finding {idx}: detailed feasibility evidence must remain visible." for idx in range(1, 25)]
    )

    assert looks_like_worker_result_report(text) is True
    assert looks_like_structured_report(text) is True
    _assert_preserved_everywhere(text)


def test_generic_worker_result_structured_report_preserved():
    text = "\n".join(
        [
            "[WORKER RESULT: Tyr]",
            "[WORLD STATE REPORT]",
            "Goal: summarize world-state changes.",
            "Scope: faction, location, and timeline data.",
            "Evidence: session notes and supplied outline.",
            "Risk: continuity drift.",
            "Recommendation: update the scenario bible.",
            "Final Status: READY",
        ]
        + [f"Section {idx}: structured report content." for idx in range(1, 20)]
    )

    assert looks_like_worker_result_report(text) is True
    _assert_preserved_everywhere(text)


def test_character_db_preserved():
    lines = ["[CHARACTER DB V1]", "Status: ACTIVE", "Evidence: curated character records"]
    for idx in range(1, 8):
        lines.extend(
            [
                f"Character ID: CH-{idx:03d}",
                f"Name: Character {idx}",
                "Role: supporting cast",
                "Traits: loyal, conflicted, observant",
                "Status: VERIFIED",
            ]
        )
    text = "\n".join(lines)

    assert looks_like_structured_report(text) is True
    _assert_preserved_everywhere(text)


def test_scenario_db_preserved():
    lines = ["[SCENARIO DB V1]", "Status: ACTIVE", "Evidence: scenario records"]
    for idx in range(1, 8):
        lines.extend(
            [
                f"Chapter: {idx}",
                f"Scene: branching scene {idx}",
                "Choice: player selects a route",
                "Condition: trust >= threshold",
                "Result: unlocks variant outcome",
            ]
        )
    text = "\n".join(lines)

    assert looks_like_structured_report(text) is True
    _assert_preserved_everywhere(text)


def test_system_design_preserved():
    text = "\n".join(
        [
            "System Design V2",
            "Goal: preserve report bodies without marker churn.",
            "Architecture: shared classifier used by delivery layers.",
            "Risk: false positives for operational summaries.",
            "Migration: keep old markers as compatibility fallback.",
            "Evidence: regression tests cover gateway and post-upload paths.",
            "Final Status: DESIGN_READY",
        ]
        + [f"Design Detail {idx}: implementation rationale." for idx in range(1, 24)]
    )

    assert looks_like_structured_report(text) is True
    _assert_preserved_everywhere(text)


def test_game_design_preserved():
    text = "\n".join(
        [
            "Game Design Report",
            "Rules: combat resolves in simultaneous turns.",
            "Balance: high-risk moves require visible tradeoffs.",
            "Systems: progression, inventory, and faction reputation.",
            "Risks: runaway economy and dominant strategy loops.",
            "Status: REVIEW_READY",
            "Recommendation: prototype with constrained resource caps.",
        ]
        + [f"Mechanic {idx}: design detail and tuning note." for idx in range(1, 24)]
    )

    assert looks_like_structured_report(text) is True
    _assert_preserved_everywhere(text)


def test_production_report_preserved_but_not_trusted():
    text = "\n".join(
        [
            "[PRODUCTION REPORT V1]",
            "Goal: summarize generated artifact status without deleting report body.",
            "Evidence: paths and hashes are listed for separate integrity guard validation.",
            "output_path: /tmp/nonexistent-production-artifact.png",
            "prompt_id: 00000000-0000-4000-8000-000000000000",
            "file_sha256: " + "a" * 64,
            "sidecar: prompt.json metadata.json workflow.json manifest.json",
            "Slack Upload: PASS",
            "NAS Hook: PASS",
            "Risk: integrity guard must still verify claims separately.",
            "Final Status: REPORT_BODY_ONLY",
        ]
        + [f"Artifact Detail {idx}: delivery text must remain intact." for idx in range(1, 20)]
    )

    assert should_preserve_report_body(text) is True
    _assert_preserved_everywhere(text)


def test_operational_summary_compact_still_allowed():
    text = "\n".join(
        f"Slack Upload: PASS artifact summary: /tmp/file_{idx}.png NAS Hook: PASS provenance: PASS"
        for idx in range(1, 40)
    )

    assert looks_like_operational_delivery_summary(text) is True
    assert should_preserve_report_body(text) is False

    gateway_text, applied = _compact_gateway_final_response(text)
    assert applied is True
    assert "lines omitted" in gateway_text or len(gateway_text.splitlines()) <= 20

    post_text = _compact_post_upload_reporting(text)
    assert "lines omitted" in post_text or len(post_text.splitlines()) <= 12


def test_delegate_worker_result_frame_preserves_structured_report_body():
    summary = "\n".join(
        [
            "[NAI FEASIBILITY STUDY V1]",
            "Goal: delegate result body must be preserved.",
            "API: compatibility review.",
            "Cost: estimated usage.",
            "Commercial Use: license note.",
            "Automation: approval-gated.",
            "Risk: no live call performed.",
            "Recommended Pipeline: staged test.",
            "Final Status: READY",
        ]
        + [f"Delegate Finding {idx}: full detail." for idx in range(1, 25)]
    )

    frame = _format_specialist_result_frame("Seir", status="completed", task_type="report", summary=summary)

    assert "lines omitted" not in frame
    assert "Delegate Finding 24" in frame
    assert should_preserve_report_body(frame) is True
