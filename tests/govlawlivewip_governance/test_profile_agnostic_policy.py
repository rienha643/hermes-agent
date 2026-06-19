from __future__ import annotations


import pytest

from govlawlivewip_governance.taskrun import DeliveryEvidence, ExecutionEvidence, ReportEvidence, RoutingEvidence, SyncEvidence, TaskIntent, evaluate_task_run


@pytest.mark.parametrize("requested_role,actual_executor", [("Palette", "artist"), ("Eclipse", "coder"), ("Rafina", "qa"), ("unknown-role", "local-process")])
def test_policy_does_not_depend_on_specific_profile_name(requested_role, actual_executor):
    decision = evaluate_task_run(
        TaskIntent(task_id="agnostic-1", user_request_summary="forbidden upload", task_mode="patch", upload_allowed=False),
        RoutingEvidence(requested_role=requested_role, delegated_role="worker", actual_executor=actual_executor),
        ExecutionEvidence(boundary_reached=True, tool_called=True, execution_id="exec-1"),
        artifacts=None,
        delivery=DeliveryEvidence(file_upload_attempted=True),
        sync=SyncEvidence(),
        runtime=None,
        tests=None,
        report=ReportEvidence(report_language="ko", required_fields_present=True, report_matches_evidence=True),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "upload_not_allowed_but_attempted" in decision.blocking_reasons


@pytest.mark.parametrize(
    ("task_mode", "summary"),
    [
        ("fresh_e2e", "image workflow"),
        ("patch", "auth task"),
        ("runtime_apply", "runtime task"),
        ("commit", "git task"),
        ("report_only", "report task"),
    ],
)
def test_policy_applies_across_task_types(task_mode, summary):
    decision = evaluate_task_run(
        TaskIntent(task_id="agnostic-2", user_request_summary=summary, task_mode=task_mode, fresh_required=True),
        RoutingEvidence(requested_role="any", delegated_role="any", actual_executor="any"),
        ExecutionEvidence(boundary_reached=True, tool_called=True, execution_id="exec-2", fresh_output_created=True, reused_artifact=True),
        artifacts=None,
        delivery=DeliveryEvidence(),
        sync=SyncEvidence(),
        runtime=None,
        tests=None,
        report=ReportEvidence(report_language="ko", required_fields_present=True, report_matches_evidence=True),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "fresh_required_but_reused_artifact" in decision.blocking_reasons


def test_role_name_change_does_not_change_rule_result():
    common = dict(
        intent=TaskIntent(task_id="agnostic-3", user_request_summary="sync blocked", task_mode="sync_verify", sync_allowed=False),
        execution=ExecutionEvidence(boundary_reached=True, tool_called=True, execution_id="exec-3"),
        artifacts=None,
        delivery=DeliveryEvidence(),
        sync=SyncEvidence(sync_requested=True),
        runtime=None,
        tests=None,
        report=ReportEvidence(report_language="ko", required_fields_present=True, report_matches_evidence=True),
    )

    first = evaluate_task_run(routing=RoutingEvidence(requested_role="image-worker", delegated_role="a", actual_executor="b"), **common)
    second = evaluate_task_run(routing=RoutingEvidence(requested_role="auth-worker", delegated_role="x", actual_executor="y"), **common)

    assert first.final_state == second.final_state == "GOVERNANCE_PING"
    assert first.blocked_state == second.blocked_state == "BLOCKED"
    assert first.blocking_reasons == second.blocking_reasons
