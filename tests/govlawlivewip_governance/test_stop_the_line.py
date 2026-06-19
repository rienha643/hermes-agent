from __future__ import annotations


from govlawlivewip_governance.taskrun import DeliveryEvidence, ExecutionEvidence, ReportEvidence, RoutingEvidence, SyncEvidence, TaskIntent, evaluate_task_run


def _evaluate(intent=None, routing=None, execution=None, delivery=None, sync=None, report=None):
    return evaluate_task_run(
        intent or TaskIntent(task_id="stop-1", user_request_summary="stop line", task_mode="report_only"),
        routing or RoutingEvidence(requested_role="worker", delegated_role="coder", actual_executor="coder"),
        execution or ExecutionEvidence(boundary_reached=True, tool_called=True, execution_id="exec-1"),
        artifacts=None,
        delivery=delivery or DeliveryEvidence(),
        sync=sync or SyncEvidence(),
        runtime=None,
        tests=None,
        report=report or ReportEvidence(report_language="ko", required_fields_present=True, report_matches_evidence=True),
    )


def test_omitted_protected_report_blocks():
    decision = _evaluate(report=ReportEvidence(report_language="ko", protected_report=True, omitted_present=True))

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "protected_report_omitted" in decision.stop_the_line_conditions


def test_worker_pass_without_evidence_blocks():
    decision = _evaluate(
        report=ReportEvidence(
            report_language="ko",
            protected_report=True,
            required_fields_present=True,
            report_matches_evidence=False,
            unsupported_pass_claims=["COMPLETE"],
        )
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "worker_pass_without_evidence" in decision.stop_the_line_conditions


def test_text_sent_but_file_evidence_absent_blocks():
    decision = _evaluate(
        intent=TaskIntent(task_id="delivery-1", user_request_summary="deliver file", task_mode="delivery_verify", required_evidence=["delivery"]),
        delivery=DeliveryEvidence(text_message_sent=True, file_upload_attempted=False, file_upload_succeeded=False, file_count=0),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "file_delivery_pass_without_file_evidence" in decision.stop_the_line_conditions


def test_sync_requested_but_mirror_absent_blocks():
    decision = _evaluate(
        intent=TaskIntent(task_id="sync-1", user_request_summary="sync artifact", task_mode="sync_verify", required_evidence=["sync"]),
        sync=SyncEvidence(sync_requested=True, sync_started=True, sync_finished=False, mirror_exists=False, hash_match=False),
    )

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.auto_blocking is False
    assert "sync_pass_without_mirror_evidence" in decision.stop_the_line_conditions
