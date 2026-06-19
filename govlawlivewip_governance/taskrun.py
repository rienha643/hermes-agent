from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

TaskMode = Literal[
    "rca",
    "patch",
    "runtime_apply",
    "fresh_e2e",
    "delivery_verify",
    "sync_verify",
    "report_only",
    "commit",
    "push",
    "planning",
]
VALID_TASK_MODES = {
    "rca",
    "patch",
    "runtime_apply",
    "fresh_e2e",
    "delivery_verify",
    "sync_verify",
    "report_only",
    "commit",
    "push",
    "planning",
}

GovernancePhase = Literal["LIVE_PING_VALIDATION", "ENFORCED_BLOCKING"]

FinalState = Literal[
    "BLOCKED",
    "GOVERNANCE_PING",
    "REPORT_FAIL",
    "RCA_COMPLETE",
    "PATCH_APPLIED",
    "TEST_PASS",
    "RUNTIME_READY",
    "READY_FOR_E2E",
    "FRESH_E2E_PASS",
    "DELIVERY_PASS",
    "SYNC_PASS",
    "REPORT_PASS",
    "COMMIT_READY",
    "COMPLETE",
]


@dataclass(slots=True)
class TaskIntent:
    task_id: str
    user_request_summary: str
    task_mode: str
    required_outputs: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)
    fresh_required: bool = False
    reuse_allowed: bool = True
    upload_allowed: bool = False
    sync_allowed: bool = False
    restart_allowed: bool = False
    commit_allowed: bool = False
    push_allowed: bool = False
    language_required: str = "ko"
    governance_phase: str = "LIVE_PING_VALIDATION"


@dataclass(slots=True)
class RoutingEvidence:
    requested_role: str | None = None
    delegated_role: str | None = None
    actual_executor: str | None = None
    execution_context: str = ""
    routing_reason: str = ""
    mismatch_exists: bool = False
    mismatch_allowed: bool = False
    mismatch_reason: str = ""


@dataclass(slots=True)
class ExecutionEvidence:
    boundary_reached: bool = False
    tool_called: bool = False
    provider_called: bool = False
    external_service_called: bool = False
    execution_id: str | None = None
    execution_started_at: str | None = None
    execution_finished_at: str | None = None
    fresh_output_created: bool = False
    reused_artifact: bool = False
    output_paths: list[str] = field(default_factory=list)
    actual_parameters: dict[str, Any] = field(default_factory=dict)
    requested_parameters: dict[str, Any] = field(default_factory=dict)
    parameter_mismatch: bool = False
    commit_performed: bool = False
    push_performed: bool = False
    fresh_e2e_done: bool = False


@dataclass(slots=True)
class ArtifactEvidence:
    artifacts_exist: bool = False
    canonical_primary: str | None = None
    artifact_paths: list[str] = field(default_factory=list)
    additional_artifacts: list[str] = field(default_factory=list)
    sidecar_paths: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    integrity_path: str | None = None
    hash_values: dict[str, str] = field(default_factory=dict)
    schema_valid: bool = False
    root_allowed: bool = False


@dataclass(slots=True)
class DeliveryEvidence:
    text_message_sent: bool = False
    file_upload_attempted: bool = False
    file_upload_succeeded: bool = False
    file_count: int = 0
    message_id: str | None = None
    thread_id: str | None = None
    uploaded_filenames: list[str] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)
    delivery_evidence_path: str | None = None


@dataclass(slots=True)
class SyncEvidence:
    sync_requested: bool = False
    sync_started: bool = False
    sync_finished: bool = False
    mirror_exists: bool = False
    mirror_paths: list[str] = field(default_factory=list)
    mirror_hashes: dict[str, str] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    hash_match: bool = False
    sync_evidence_path: str | None = None


@dataclass(slots=True)
class RuntimeEvidence:
    process_running: bool = False
    pid: int | None = None
    start_time: str | None = None
    source_hash: str | None = None
    config_loaded: bool = False
    policy_loaded: bool = False
    runtime_matches_repo: bool = False
    restart_required: bool = False
    restart_done: bool = False


@dataclass(slots=True)
class TestEvidence:
    __test__: ClassVar[bool] = False

    tests_run: bool = False
    tests_passed: bool = False
    tests_failed: int = 0
    focused_tests: list[str] = field(default_factory=list)
    regression_tests: list[str] = field(default_factory=list)
    lint_passed: bool = False
    diff_check_passed: bool = False


@dataclass(slots=True)
class ReportEvidence:
    report_language: str = "ko"
    omitted_present: bool = False
    protected_report: bool = True
    required_fields_present: bool = False
    unsupported_pass_claims: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    report_matches_evidence: bool = False


@dataclass(slots=True)
class ReleaseDecision:
    final_state: str
    blocking_reasons: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    stop_the_line_conditions: list[str] = field(default_factory=list)
    next_required_action: str = ""
    governance_phase: str = "LIVE_PING_VALIDATION"
    auto_blocking: bool = False
    current_state: str = ""
    blocked_state: str = ""
    violation_type: str = ""


def _has_file_upload_evidence(delivery: DeliveryEvidence) -> bool:
    return delivery.file_upload_attempted and delivery.file_upload_succeeded and delivery.file_count > 0


def _has_sync_evidence(sync: SyncEvidence) -> bool:
    return sync.sync_finished and sync.mirror_exists and sync.hash_match and bool(sync.mirror_hashes) and bool(sync.source_hashes)


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _performed_forbidden_actions(
    intent: TaskIntent,
    execution: ExecutionEvidence,
    delivery: DeliveryEvidence,
    sync: SyncEvidence,
    runtime: RuntimeEvidence,
) -> list[str]:
    performed: dict[str, bool] = {
        "upload": delivery.file_upload_attempted,
        "slack_upload": delivery.file_upload_attempted,
        "file_upload": delivery.file_upload_attempted,
        "sync": sync.sync_requested,
        "nas_sync": sync.sync_requested,
        "restart": runtime.restart_done,
        "gateway_restart": runtime.restart_done,
        "commit": execution.commit_performed,
        "push": execution.push_performed,
        "generation": execution.provider_called,
        "external_generation": execution.provider_called,
    }
    detected: list[str] = []
    for action in intent.forbidden_actions:
        action_performed = action in execution.actual_parameters or bool(performed.get(action, False))
        if action_performed:
            detected.append(action)
    return detected


def _auto_blocking_for_phase(governance_phase: str) -> bool:
    return governance_phase == "ENFORCED_BLOCKING"


def _state_for_violation(governance_phase: str, blocking: list[str], missing: list[str], stop: list[str]) -> tuple[str, str]:
    if "report_language_mismatch" in blocking and not stop and not missing:
        blocked_state = "REPORT_FAIL"
    else:
        blocked_state = "BLOCKED"
    if _auto_blocking_for_phase(governance_phase):
        return blocked_state, blocked_state
    return "GOVERNANCE_PING", blocked_state


def _violation_type(blocking: list[str], missing: list[str], stop: list[str]) -> str:
    if stop:
        return stop[0]
    if blocking:
        return blocking[0]
    if missing:
        return f"missing:{missing[0]}"
    return "none"


def evaluate_task_run(
    intent: TaskIntent,
    routing: RoutingEvidence | None,
    execution: ExecutionEvidence | None,
    artifacts: ArtifactEvidence | None,
    delivery: DeliveryEvidence | None,
    sync: SyncEvidence | None,
    runtime: RuntimeEvidence | None,
    tests: TestEvidence | None,
    report: ReportEvidence | None,
) -> ReleaseDecision:
    """기계적으로 확인된 evidence만 사용해 작업 상태를 계산한다."""

    routing = routing or RoutingEvidence()
    execution = execution or ExecutionEvidence()
    artifacts = artifacts or ArtifactEvidence()
    delivery = delivery or DeliveryEvidence()
    sync = sync or SyncEvidence()
    runtime = runtime or RuntimeEvidence()
    tests = tests or TestEvidence()
    report = report or ReportEvidence()
    governance_phase = intent.governance_phase if intent.governance_phase in {"LIVE_PING_VALIDATION", "ENFORCED_BLOCKING"} else "LIVE_PING_VALIDATION"

    blocking: list[str] = []
    missing: list[str] = []
    stop: list[str] = []

    if not routing.requested_role or not routing.actual_executor:
        _append_unique(stop, "requested_or_actual_executor_missing")
        _append_unique(missing, "routing_evidence")
    derived_mismatch = bool(
        routing.delegated_role
        and routing.actual_executor
        and routing.delegated_role != routing.actual_executor
        and routing.delegated_role not in {"any", "worker", "unknown"}
    )
    if (routing.mismatch_exists or derived_mismatch) and not routing.mismatch_allowed:
        _append_unique(blocking, "routing_mismatch_not_allowed")
        _append_unique(stop, "routing_mismatch_not_allowed")
    if not intent.task_mode:
        _append_unique(stop, "task_mode_missing")
        _append_unique(missing, "task_mode")
    elif intent.task_mode not in VALID_TASK_MODES:
        _append_unique(blocking, "unknown_task_mode")
        _append_unique(stop, "unknown_task_mode")
    if not execution.boundary_reached:
        _append_unique(stop, "execution_boundary_not_reached")
        _append_unique(missing, "execution_boundary_evidence")

    for action in _performed_forbidden_actions(intent, execution, delivery, sync, runtime):
        _append_unique(blocking, f"forbidden_action_performed:{action}")
        _append_unique(stop, "forbidden_action_performed")

    if intent.fresh_required and execution.reused_artifact:
        _append_unique(blocking, "fresh_required_but_reused_artifact")
        _append_unique(stop, "fresh_reused_artifact")
    if intent.fresh_required and not execution.fresh_output_created:
        _append_unique(blocking, "fresh_required_but_no_fresh_output")
        _append_unique(missing, "fresh_output_evidence")
    if not intent.upload_allowed and delivery.file_upload_attempted:
        _append_unique(blocking, "upload_not_allowed_but_attempted")
        _append_unique(stop, "forbidden_upload_attempted")
    if not intent.sync_allowed and sync.sync_requested:
        _append_unique(blocking, "sync_not_allowed_but_requested")
        _append_unique(stop, "forbidden_sync_requested")
    if not intent.restart_allowed and runtime.restart_done:
        _append_unique(blocking, "restart_not_allowed_but_done")
        _append_unique(stop, "forbidden_restart_done")
    if not intent.commit_allowed and execution.commit_performed:
        _append_unique(blocking, "commit_not_allowed_but_performed")
        _append_unique(stop, "forbidden_commit_performed")
    if not intent.push_allowed and execution.push_performed:
        _append_unique(blocking, "push_not_allowed_but_performed")
        _append_unique(stop, "forbidden_push_performed")

    delivery_required = "delivery" in intent.required_evidence or intent.task_mode == "delivery_verify"
    if delivery_required and not _has_file_upload_evidence(delivery):
        _append_unique(missing, "file_upload_evidence")
        _append_unique(stop, "file_delivery_pass_without_file_evidence")

    sync_required = "sync" in intent.required_evidence or intent.task_mode == "sync_verify"
    if sync_required and not _has_sync_evidence(sync):
        _append_unique(missing, "mirror_hash_evidence")
        _append_unique(stop, "sync_pass_without_mirror_evidence")

    runtime_required = "runtime" in intent.required_evidence or runtime.restart_required or intent.task_mode == "runtime_apply"
    if runtime_required and (runtime.restart_required and not runtime.restart_done or not runtime.runtime_matches_repo):
        _append_unique(missing, "runtime_apply_required")
        _append_unique(stop, "runtime_unapplied_completion_claim")

    fresh_e2e_required = "fresh_e2e" in intent.required_evidence or intent.task_mode == "fresh_e2e"
    if fresh_e2e_required and not execution.fresh_e2e_done:
        _append_unique(missing, "fresh_e2e_evidence")
        _append_unique(stop, "e2e_missing_for_commit_ready")

    tests_required = "tests" in intent.required_evidence
    if tests_required and (not tests.tests_run or not tests.tests_passed or tests.tests_failed > 0):
        _append_unique(missing, "test_evidence")
        _append_unique(stop, "test_pass_missing")

    if report.protected_report and report.omitted_present:
        _append_unique(blocking, "protected_report_omitted")
        _append_unique(stop, "protected_report_omitted")
    if report.unsupported_pass_claims and not report.report_matches_evidence:
        _append_unique(blocking, "worker_pass_without_evidence")
        _append_unique(stop, "worker_pass_without_evidence")
    if report.parse_errors:
        _append_unique(blocking, "report_parse_errors")
    if not report.required_fields_present:
        _append_unique(missing, "report_required_fields")
    if report.report_language != intent.language_required:
        _append_unique(blocking, "report_language_mismatch")

    if blocking or missing or stop:
        state, blocked_state = _state_for_violation(governance_phase, blocking, missing, stop)
        return ReleaseDecision(
            final_state=state,
            blocking_reasons=blocking,
            missing_evidence=missing,
            stop_the_line_conditions=stop,
            next_required_action="부족한 evidence를 수집하고 위반 원인을 정정해야 합니다.",
            governance_phase=governance_phase,
            auto_blocking=_auto_blocking_for_phase(governance_phase),
            current_state=state,
            blocked_state=blocked_state,
            violation_type=_violation_type(blocking, missing, stop),
        )

    if intent.task_mode == "delivery_verify":
        state = "DELIVERY_PASS"
    elif intent.task_mode == "sync_verify":
        state = "SYNC_PASS"
    elif intent.task_mode == "runtime_apply":
        state = "RUNTIME_READY"
    elif intent.task_mode == "fresh_e2e":
        state = "FRESH_E2E_PASS"
    elif intent.task_mode == "commit":
        state = "COMMIT_READY"
    elif intent.task_mode == "report_only":
        state = "REPORT_PASS"
    elif intent.task_mode == "rca":
        state = "RCA_COMPLETE"
    elif intent.task_mode == "patch":
        state = "TEST_PASS" if tests.tests_run and tests.tests_passed else "PATCH_APPLIED"
    else:
        state = "COMPLETE"

    return ReleaseDecision(
        final_state=state,
        next_required_action="none",
        governance_phase=governance_phase,
        auto_blocking=_auto_blocking_for_phase(governance_phase),
        current_state=state,
        blocked_state="",
        violation_type="none",
    )


def format_governance_ping(intent: TaskIntent, decision: ReleaseDecision) -> str:
    """Slack shortcode에 의존하지 않는 Unicode 거버넌스 PING을 만든다."""
    missing = decision.missing_evidence or []
    lines = [
        "🚨🚨🚨 GOVERNANCE PING 🚨🚨🚨",
        "⚠️ 거버넌스 법 위반 또는 증거 부족 상태가 감지되었습니다.",
        "```",
        f"task_id: {intent.task_id}",
        f"task_mode: {intent.task_mode}",
        f"violation_type: {decision.violation_type}",
        f"current_state: {decision.current_state or decision.final_state}",
        f"blocked_state: {decision.blocked_state}",
        f"missing_evidence: {missing}",
        f"next_required_action: {decision.next_required_action}",
        f"auto_blocking: {str(decision.auto_blocking).lower()}",
        f"governance_phase: {decision.governance_phase}",
        "```",
    ]
    return "\n".join(lines)
