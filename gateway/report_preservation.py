"""Shared report-preservation classifier for gateway delivery paths.

This module is intentionally about delivery semantics only: preserving a report
body does not mean trusting its claims. Artifact success validation remains the
responsibility of the existing report-integrity guard.
"""

from __future__ import annotations

import re


_EXPLICIT_REPORT_MARKERS = (
    # Existing validation/migration markers.
    "[e2e validation round v1]",
    "[e2e validation result]",
    "[comfyui restart checkpoint verify]",
    "[checkpoint migration finalization]",
    "[checkpoint selection finalization v2]",
    # Existing generation/evaluation markers.
    "[full body challenger round v1 results]",
    "[key visual challenger round v1 results]",
    "full body challenger round",
    "key visual challenger round",
    "checkpoint review",
    "portrait review",
    "nsfw-lite review",
    "nsfw lite review",
    "full body review",
    "key visual review",
    "evaluation report",
    "scoring report",
    "main / reserve review",
    "main/reserve review",
    "main reserve review",
    # Existing user/report integrity markers.
    "[nsfw stability round results]",
    "[nsfw stability result integrity rca]",
    "[angelica final response]",
    "integrity rca",
    "integrity report",
    "validation report",
    "generation result",
    "generated result",
    "생성 결과",
    "검증 보고",
    "무결성 보고",
)

_OPERATIONAL_DELIVERY_MARKERS = (
    "slack upload",
    "slack 전달 완료",
    "nas hook",
    "artifact summary",
    "provenance",
    "status summary",
    "delivery complete",
    "delivery partial",
    "delivery failed",
)

_STRUCTURED_TITLE_RE = re.compile(
    r"(?im)^\s*(?:\[[A-Z0-9][A-Z0-9 _:/.-]{3,}\]|"
    r"(?:[A-Z][A-Za-z0-9 /_-]{2,}\s+)?(?:Report|Study|Design|Database|DB|Review|Validation|Migration|Finalization|Results?)\s*(?:V\d+)?)\s*$"
)

_WORKER_RESULT_RE = re.compile(r"(?im)^\s*\[WORKER RESULT:\s*[^\]]+\]\s*$")

_GENERAL_REPORT_CLASS_TITLE_RE = re.compile(
    r"(?im)^\s*\[[^\]\n]*(?:"
    r"RCA|FIX|VERIFY|APPLY|E2E|DELIVERY|VALIDATION|INTEGRITY|HANDOFF|CHECKPOINT"
    r")[^\]\n]*\]\s*$"
)

_SECTION_HEADING_RE = re.compile(
    r"(?im)^\s*(?:"
    r"Goal|Scope|API|Cost|Commercial Use|Automation|Risk|Risks|Evidence|Result|Results|"
    r"Recommendation|Recommended Pipeline|Final Status|Status|Architecture|Migration|"
    r"Rules|Balance|Systems|Chapter|Scene|Choice|Condition|Character ID|Name|Role|Traits|"
    r"Output|Output Path|Prompt ID|file_sha256|sidecar|Slack Upload|NAS Hook|Root Cause|"
    r"Current Design|Current Weakness|Candidate Designs|Expected Benefit|Verification|검증 결과"
    r")\s*:"
)

_KEY_MARKER_RE = re.compile(
    r"(?i)\b(?:status|final status|evidence|result|results|recommendation|recommended pipeline|"
    r"root cause|risk|risks|migration|architecture|pass|fail|yes|no|ready|complete|verified)\b"
)

_RECORD_FIELD_RE = re.compile(
    r"(?im)^\s*(?:Character ID|Name|Role|Traits|Status|Chapter|Scene|Choice|Condition|Result)\s*:"
)

_ARTIFACT_REPORT_RE = re.compile(
    r"(?im)^\s*(?:output_path|prompt_id|file_sha256|prompt_hash|workflow_hash|sidecar|Slack Upload|NAS Hook)\s*:"
)


def _non_empty_lines(text: str) -> list[str]:
    return [line.rstrip() for line in str(text or "").splitlines() if line.strip()]


def _has_explicit_report_marker(text: str) -> bool:
    body = str(text or "").casefold()
    if any(marker in body for marker in _EXPLICIT_REPORT_MARKERS):
        return True
    # General report classes should not require per-marker churn.  Preserve
    # bracketed RCA/FIX/VERIFY/APPLY/E2E/DELIVERY/etc. report frames even when
    # their bodies are plain evidence lines rather than colon-labelled fields.
    raw_body = str(text or "")
    if _GENERAL_REPORT_CLASS_TITLE_RE.search(raw_body):
        return True
    # A bare worker-result frame can be an authored report, but gateway/post-upload
    # delivery summaries also use this wrapper.  Keep those operational summaries
    # compactable; preserve other long worker-result bodies.
    if _WORKER_RESULT_RE.search(raw_body):
        return not any(marker in body for marker in _OPERATIONAL_DELIVERY_MARKERS)
    return False


def looks_like_worker_result_report(text: str) -> bool:
    """Return True for worker-result frames that carry structured report bodies."""
    body = str(text or "")
    if not _WORKER_RESULT_RE.search(body):
        return False
    without_worker_header = _WORKER_RESULT_RE.sub("", body, count=1)
    return looks_like_structured_report(without_worker_header)


def looks_like_structured_report(text: str) -> bool:
    """Detect report-like deliverables without requiring a fixed marker allowlist."""
    body = str(text or "")
    lines = _non_empty_lines(body)
    if len(lines) < 12:
        return False

    # A worker-result header is a routing/attribution wrapper, not by itself a
    # report title. Strip it before scoring structure so repeated operational
    # delivery summaries like ``[WORKER RESULT: Eclipse]`` + Slack/NAS lines do
    # not become false-positive structured reports.
    analysis_body = _WORKER_RESULT_RE.sub("", body)
    title_like = bool(_STRUCTURED_TITLE_RE.search(analysis_body))
    section_labels = {
        match.group(0).split(":", 1)[0].strip().casefold()
        for match in _SECTION_HEADING_RE.finditer(analysis_body)
    }
    section_count = len(section_labels)
    key_count = len(set(match.group(0).casefold() for match in _KEY_MARKER_RE.finditer(analysis_body)))
    record_count = len(_RECORD_FIELD_RE.findall(analysis_body))
    artifact_count = len(_ARTIFACT_REPORT_RE.findall(analysis_body))

    if title_like and section_count >= 3 and key_count >= 2:
        return True
    if title_like and record_count >= 12 and key_count >= 1:
        return True
    if title_like and artifact_count >= 3 and section_count >= 2 and key_count >= 2:
        return True
    if _WORKER_RESULT_RE.search(body) and section_count >= 3 and key_count >= 2:
        return True
    return False


def should_preserve_report_body(text: str) -> bool:
    """Return True when delivery layers must not insert omission markers."""
    body = str(text or "")
    if not body:
        return False
    return (
        _has_explicit_report_marker(body)
        or looks_like_worker_result_report(body)
        or looks_like_structured_report(body)
    )


def looks_like_operational_delivery_summary(text: str) -> bool:
    """Return True for compactable delivery/provenance summaries.

    Structured reports can contain delivery words as evidence fields. Those must
    not be compacted just because they mention Slack/NAS/provenance.
    """
    body = str(text or "")
    if not body or should_preserve_report_body(body):
        return False
    folded = body.casefold()
    marker_hits = sum(1 for marker in _OPERATIONAL_DELIVERY_MARKERS if marker in folded)
    if marker_hits == 0:
        return False
    lines = _non_empty_lines(body)
    if len(lines) > 12:
        return True
    return marker_hits >= 2
