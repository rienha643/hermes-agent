from __future__ import annotations


from govlawlivewip_governance.taskrun import (
    ArtifactEvidence,
    DeliveryEvidence,
    ExecutionEvidence,
    ReleaseDecision,
    ReportEvidence,
    RoutingEvidence,
    SyncEvidence,
    TaskIntent,
)


def test_task_intent_schema():
    intent = TaskIntent(task_id="t1", user_request_summary="implement governance", task_mode="patch")

    assert intent.task_id == "t1"
    assert intent.task_mode == "patch"
    assert intent.required_outputs == []
    assert intent.required_evidence == []
    assert intent.forbidden_actions == []
    assert intent.allowed_actions == []
    assert intent.fresh_required is False
    assert intent.reuse_allowed is True
    assert intent.upload_allowed is False
    assert intent.sync_allowed is False
    assert intent.restart_allowed is False
    assert intent.commit_allowed is False
    assert intent.push_allowed is False
    assert intent.language_required == "ko"


def test_routing_evidence_schema():
    routing = RoutingEvidence(requested_role="worker", delegated_role="coder", actual_executor="coder")

    assert routing.requested_role == "worker"
    assert routing.delegated_role == "coder"
    assert routing.actual_executor == "coder"
    assert routing.execution_context == ""
    assert routing.routing_reason == ""
    assert routing.mismatch_exists is False
    assert routing.mismatch_allowed is False
    assert routing.mismatch_reason == ""


def test_execution_evidence_schema():
    execution = ExecutionEvidence(boundary_reached=True, tool_called=True, execution_id="run-1")

    assert execution.boundary_reached is True
    assert execution.tool_called is True
    assert execution.provider_called is False
    assert execution.external_service_called is False
    assert execution.execution_id == "run-1"
    assert execution.execution_started_at is None
    assert execution.execution_finished_at is None
    assert execution.fresh_output_created is False
    assert execution.reused_artifact is False
    assert execution.output_paths == []
    assert execution.actual_parameters == {}
    assert execution.requested_parameters == {}
    assert execution.parameter_mismatch is False
    assert execution.commit_performed is False
    assert execution.push_performed is False
    assert execution.fresh_e2e_done is False


def test_artifact_evidence_schema():
    artifacts = ArtifactEvidence(artifacts_exist=True, artifact_paths=["/tmp/a.txt"])

    assert artifacts.artifacts_exist is True
    assert artifacts.canonical_primary is None
    assert artifacts.artifact_paths == ["/tmp/a.txt"]
    assert artifacts.additional_artifacts == []
    assert artifacts.sidecar_paths == []
    assert artifacts.manifest_path is None
    assert artifacts.integrity_path is None
    assert artifacts.hash_values == {}
    assert artifacts.schema_valid is False
    assert artifacts.root_allowed is False


def test_delivery_evidence_schema():
    delivery = DeliveryEvidence(text_message_sent=True, thread_id="thread-1")

    assert delivery.text_message_sent is True
    assert delivery.file_upload_attempted is False
    assert delivery.file_upload_succeeded is False
    assert delivery.file_count == 0
    assert delivery.message_id is None
    assert delivery.thread_id == "thread-1"
    assert delivery.uploaded_filenames == []
    assert delivery.source_paths == []
    assert delivery.delivery_evidence_path is None


def test_sync_evidence_schema():
    sync = SyncEvidence(sync_requested=True)

    assert sync.sync_requested is True
    assert sync.sync_started is False
    assert sync.sync_finished is False
    assert sync.mirror_exists is False
    assert sync.mirror_paths == []
    assert sync.mirror_hashes == {}
    assert sync.source_hashes == {}
    assert sync.hash_match is False
    assert sync.sync_evidence_path is None


def test_report_evidence_schema():
    report = ReportEvidence(report_language="ko", required_fields_present=True)

    assert report.report_language == "ko"
    assert report.omitted_present is False
    assert report.protected_report is True
    assert report.required_fields_present is True
    assert report.unsupported_pass_claims == []
    assert report.parse_errors == []
    assert report.report_matches_evidence is False


def test_release_decision_schema():
    decision = ReleaseDecision(final_state="GOVERNANCE_PING", blocked_state="BLOCKED", next_required_action="collect evidence")

    assert decision.final_state == "GOVERNANCE_PING"
    assert decision.blocked_state == "BLOCKED"
    assert decision.governance_phase == "LIVE_PING_VALIDATION"
    assert decision.auto_blocking is False
    assert decision.blocking_reasons == []
    assert decision.missing_evidence == []
    assert decision.stop_the_line_conditions == []
    assert decision.next_required_action == "collect evidence"
