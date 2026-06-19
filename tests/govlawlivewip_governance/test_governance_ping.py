from __future__ import annotations

from govlawlivewip_governance.taskrun import (
    DeliveryEvidence,
    ExecutionEvidence,
    ReportEvidence,
    RoutingEvidence,
    TaskIntent,
    evaluate_task_run,
    format_governance_ping,
)


def test_live_ping_validation_emits_ping_without_auto_blocking():
    intent = TaskIntent(task_id="ping-1", user_request_summary="file delivery", task_mode="delivery_verify", required_evidence=["delivery"])
    decision = evaluate_task_run(
        intent,
        RoutingEvidence(requested_role="worker", delegated_role="coder", actual_executor="coder"),
        ExecutionEvidence(boundary_reached=True, tool_called=True, execution_id="exec-1"),
        artifacts=None,
        delivery=DeliveryEvidence(text_message_sent=True),
        sync=None,
        runtime=None,
        tests=None,
        report=ReportEvidence(report_language="ko", required_fields_present=True, report_matches_evidence=True),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.governance_phase == "LIVE_PING_VALIDATION"
    assert decision.auto_blocking is False


def test_enforced_blocking_phase_blocks_and_keeps_ping_metadata():
    intent = TaskIntent(
        task_id="block-1",
        user_request_summary="file delivery",
        task_mode="delivery_verify",
        required_evidence=["delivery"],
        governance_phase="ENFORCED_BLOCKING",
    )
    decision = evaluate_task_run(
        intent,
        RoutingEvidence(requested_role="worker", delegated_role="coder", actual_executor="coder"),
        ExecutionEvidence(boundary_reached=True, tool_called=True, execution_id="exec-1"),
        artifacts=None,
        delivery=DeliveryEvidence(text_message_sent=True),
        sync=None,
        runtime=None,
        tests=None,
        report=ReportEvidence(report_language="ko", required_fields_present=True, report_matches_evidence=True),
    )

    assert decision.final_state == "BLOCKED"
    assert decision.blocked_state == "BLOCKED"
    assert decision.governance_phase == "ENFORCED_BLOCKING"
    assert decision.auto_blocking is True


def test_completion_promotion_guard_prevents_commit_ready_without_e2e():
    intent = TaskIntent(
        task_id="commit-1",
        user_request_summary="commit gate",
        task_mode="commit",
        required_evidence=["fresh_e2e"],
        commit_allowed=True,
    )
    decision = evaluate_task_run(
        intent,
        RoutingEvidence(requested_role="worker", delegated_role="coder", actual_executor="coder"),
        ExecutionEvidence(boundary_reached=True, tool_called=True, execution_id="exec-1", fresh_e2e_done=False),
        artifacts=None,
        delivery=None,
        sync=None,
        runtime=None,
        tests=None,
        report=ReportEvidence(report_language="ko", required_fields_present=True, report_matches_evidence=True),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert "fresh_e2e_evidence" in decision.missing_evidence


def test_governance_ping_formatter_contains_required_slack_fields():
    intent = TaskIntent(task_id="ping-2", user_request_summary="sync", task_mode="sync_verify", required_evidence=["sync"])
    decision = evaluate_task_run(
        intent,
        RoutingEvidence(requested_role="worker", delegated_role="coder", actual_executor="coder"),
        ExecutionEvidence(boundary_reached=True, tool_called=True, execution_id="exec-2"),
        artifacts=None,
        delivery=None,
        sync=None,
        runtime=None,
        tests=None,
        report=ReportEvidence(report_language="ko", required_fields_present=True, report_matches_evidence=True),
    )

    message = format_governance_ping(intent, decision)

    assert "🚨🚨🚨 GOVERNANCE PING 🚨🚨🚨" in message
    assert "⚠️ 거버넌스 법 위반 또는 증거 부족 상태가 감지되었습니다." in message
    assert ":rotating_light:" not in message
    assert ":warning:" not in message
    assert "```" in message
    for field in [
        "task_id: ping-2",
        "task_mode: sync_verify",
        "violation_type:",
        "current_state: GOVERNANCE_PING",
        "blocked_state: BLOCKED",
        "missing_evidence:",
        "next_required_action:",
        "auto_blocking: false",
        "governance_phase: LIVE_PING_VALIDATION",
    ]:
        assert field in message
