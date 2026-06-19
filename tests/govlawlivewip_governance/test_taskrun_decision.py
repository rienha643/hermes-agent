from __future__ import annotations


from govlawlivewip_governance.taskrun import (
    DeliveryEvidence,
    ExecutionEvidence,
    ReportEvidence,
    RoutingEvidence,
    RuntimeEvidence,
    SyncEvidence,
    TaskIntent,
    TestEvidence,
    evaluate_task_run,
)


def _base_intent(**overrides):
    data = dict(
        task_id="task-1",
        user_request_summary="governed task",
        task_mode="patch",
        required_evidence=[],
        forbidden_actions=[],
        language_required="ko",
    )
    data.update(overrides)
    return TaskIntent(**data)


def _base_routing(**overrides):
    data = dict(requested_role="worker", delegated_role="coder", actual_executor="coder")
    data.update(overrides)
    return RoutingEvidence(**data)


def _base_execution(**overrides):
    data = dict(boundary_reached=True, tool_called=True, execution_id="exec-1")
    data.update(overrides)
    return ExecutionEvidence(**data)


def _decision(intent=None, routing=None, execution=None, delivery=None, sync=None, runtime=None, tests=None, report=None):
    return evaluate_task_run(
        intent or _base_intent(),
        routing or _base_routing(),
        execution or _base_execution(),
        artifacts=None,
        delivery=delivery or DeliveryEvidence(),
        sync=sync or SyncEvidence(),
        runtime=runtime or RuntimeEvidence(),
        tests=tests or TestEvidence(),
        report=report or ReportEvidence(report_language="ko", required_fields_present=True, report_matches_evidence=True),
    )


def test_forbidden_action_blocks_task():
    decision = _decision(intent=_base_intent(forbidden_actions=["commit"]), execution=_base_execution(commit_performed=True))

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "forbidden_action_performed:commit" in decision.blocking_reasons


def test_missing_file_evidence_prevents_delivery_pass():
    decision = _decision(
        intent=_base_intent(task_mode="delivery_verify", required_evidence=["delivery"]),
        delivery=DeliveryEvidence(text_message_sent=True, file_upload_attempted=False, file_upload_succeeded=False, file_count=0),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "file_upload_evidence" in decision.missing_evidence
    assert "file_delivery_pass_without_file_evidence" in decision.stop_the_line_conditions


def test_missing_mirror_evidence_prevents_sync_pass():
    decision = _decision(
        intent=_base_intent(task_mode="sync_verify", required_evidence=["sync"]),
        sync=SyncEvidence(sync_requested=True, sync_finished=True, mirror_exists=False, hash_match=False),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "mirror_hash_evidence" in decision.missing_evidence
    assert "sync_pass_without_mirror_evidence" in decision.stop_the_line_conditions


def test_runtime_not_applied_prevents_complete():
    decision = _decision(
        intent=_base_intent(task_mode="patch", required_evidence=["runtime"]),
        runtime=RuntimeEvidence(restart_required=True, restart_done=False, runtime_matches_repo=False),
        tests=TestEvidence(tests_run=True, tests_passed=True),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "runtime_apply_required" in decision.missing_evidence
    assert "runtime_unapplied_completion_claim" in decision.stop_the_line_conditions


def test_fresh_required_but_reused_artifact_blocks_task():
    decision = _decision(
        intent=_base_intent(fresh_required=True),
        execution=_base_execution(fresh_output_created=True, reused_artifact=True),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "fresh_required_but_reused_artifact" in decision.blocking_reasons


def test_e2e_missing_prevents_commit_ready():
    decision = _decision(
        intent=_base_intent(task_mode="commit", required_evidence=["fresh_e2e"], commit_allowed=True),
        execution=_base_execution(fresh_output_created=True, fresh_e2e_done=False),
        tests=TestEvidence(tests_run=True, tests_passed=True),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "fresh_e2e_evidence" in decision.missing_evidence
    assert "e2e_missing_for_commit_ready" in decision.stop_the_line_conditions


def test_english_final_report_when_korean_required_is_report_fail():
    decision = _decision(
        intent=_base_intent(task_mode="report_only", required_evidence=["report"], language_required="ko"),
        report=ReportEvidence(report_language="en", required_fields_present=True, report_matches_evidence=True),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "REPORT_FAIL"
    assert decision.auto_blocking is False
    assert "report_language_mismatch" in decision.blocking_reasons


def test_unknown_forbidden_action_fails_closed_when_action_is_present_in_parameters():
    decision = _decision(
        intent=_base_intent(forbidden_actions=["delete_artifact"]),
        execution=_base_execution(actual_parameters={"delete_artifact": True}),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "forbidden_action_performed:delete_artifact" in decision.blocking_reasons


def test_routing_mismatch_blocks_unless_allowed():
    decision = _decision(
        routing=RoutingEvidence(
            requested_role="worker",
            delegated_role="coder",
            actual_executor="artist",
            mismatch_exists=True,
            mismatch_allowed=False,
        )
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "routing_mismatch_not_allowed" in decision.blocking_reasons


def test_patch_with_failed_tests_is_blocked():
    decision = _decision(
        intent=_base_intent(task_mode="patch", required_evidence=["tests"]),
        tests=TestEvidence(tests_run=True, tests_passed=False, tests_failed=1),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "test_evidence" in decision.missing_evidence


def test_forbidden_action_with_falsy_parameter_value_still_blocks():
    decision = _decision(
        intent=_base_intent(forbidden_actions=["delete_artifact"]),
        execution=_base_execution(actual_parameters={"delete_artifact": {}}),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "forbidden_action_performed:delete_artifact" in decision.blocking_reasons


def test_delegated_actual_executor_mismatch_is_derived_when_flag_omitted():
    decision = _decision(
        routing=RoutingEvidence(
            requested_role="worker",
            delegated_role="coder",
            actual_executor="artist",
        )
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "routing_mismatch_not_allowed" in decision.blocking_reasons


def test_unknown_task_mode_fails_closed():
    decision = _decision(intent=_base_intent(task_mode="unsupported"))

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "unknown_task_mode" in decision.blocking_reasons
