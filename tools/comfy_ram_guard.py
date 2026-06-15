#!/usr/bin/env python3
"""ComfyUI state-based Resource Guard with runtime monitoring.

The guard is intentionally conservative.  It classifies processes first,
records resource-state evidence, and produces cleanup suggestions.  In this
stage every phase is observation/dry-run only: no process termination is
performed by dry-run, safe-clean, or strict-clean.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal

try:
    import psutil
except Exception:  # pragma: no cover - psutil is a project dependency
    psutil = None  # type: ignore[assignment]

Mode = Literal["dry-run", "safe-clean", "strict-clean"]
Phase = Literal["preflight", "runtime", "post-run"]
ResourceStateName = Literal["COMFY_READY", "COMFY_TIGHT", "COMFY_STARVED"]

COMFY_READY = "COMFY_READY"
COMFY_TIGHT = "COMFY_TIGHT"
COMFY_STARVED = "COMFY_STARVED"

DEFAULT_LOG_DIR = Path("/Users/hermes/HermesWork/Logs/comfy_ram_guard")

PROTECTED_PROCESS_NAMES = {
    "kernel_task",
    "launchd",
    "WindowServer",
    "loginwindow",
    "Finder",
    "Dock",
    "SystemUIServer",
    "coreaudiod",
    "distnoted",
    "mds",
    "mds_stores",
    "bluetoothd",
}

PROTECTED_COMMAND_SUBSTRINGS = {
    "airportd",
    "networkserviceproxy",
    "networkextension",
    "wifivelocityd",
    "rapportd",
    "mDNSResponder",
    "configd",
    "sharingd",
}

ESSENTIAL_HERMES_PROFILES = {"speedy", "cron-fast", "coder"}
LLAMA_KEEP_MARKERS = {"--keep", "--no-kill", "HERMES_KEEP_LLAMA=1", "keep-alive"}

BROWSER_HELPER_NAMES = {
    "Google Chrome Helper",
    "Chrome Helper",
    "Chromium Helper",
    "Brave Browser Helper",
    "Microsoft Edge Helper",
    "FirefoxCP Web Content",
    "plugin-container",
}

IMAGE_BACKEND_MARKERS = {
    "stable-diffusion-webui",
    "webui.sh",
    "launch.py --listen",
    "invokeai",
    "kohya_gui",
}

HIGH_PROCESS_RSS_MB = 1024.0


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int | None
    name: str
    command: str
    rss_mb: float
    elapsed_seconds: int | None
    parent_name: str | None
    status: str | None = None


@dataclass(frozen=True)
class ProcessDecision:
    kind: Literal["protected", "candidate", "ignored"]
    reason: str
    killable_in_safe_clean: bool = False
    killable_in_strict_clean: bool = False


@dataclass(frozen=True)
class MemoryStatus:
    pressure: str = "unknown"
    swap_used_mb: float | None = None
    swap_delta_mb: float | None = None
    percent: float | None = None
    total_mb: float | None = None
    available_mb: float | None = None
    used_mb: float | None = None


@dataclass(frozen=True)
class ResourceState:
    state: ResourceStateName
    recommended_action: str
    reasons: list[str]
    evidence: dict[str, object]


def _lower_blob(proc: ProcessInfo) -> str:
    return f"{proc.name} {proc.command} {proc.parent_name or ''}".lower()


def _is_hermes_gateway(proc: ProcessInfo) -> bool:
    blob = _lower_blob(proc)
    return "hermes" in blob and "gateway" in blob


def _hermes_profile(proc: ProcessInfo) -> str | None:
    parts = proc.command.split()
    for i, token in enumerate(parts):
        if token in {"--profile", "-p"} and i + 1 < len(parts):
            return parts[i + 1]
        if token.startswith("--profile="):
            return token.split("=", 1)[1]
    return None


def _is_comfyui(proc: ProcessInfo) -> bool:
    blob = _lower_blob(proc)
    return "comfyui" in blob or "comfy/comfyui" in blob or (" main.py" in blob and "comfy" in blob)


def _is_llama(proc: ProcessInfo) -> bool:
    blob = _lower_blob(proc)
    return "llama-server" in blob or "llama-cli" in blob


def _has_llama_keep_marker(proc: ProcessInfo) -> bool:
    return any(marker.lower() in proc.command.lower() for marker in LLAMA_KEEP_MARKERS)


def _is_orphan(proc: ProcessInfo) -> bool:
    return proc.ppid in {0, 1, None} or (proc.parent_name or "").lower() in {
        "launchd",
        "init",
        "systemd",
    }


def classify_process(proc: ProcessInfo) -> ProcessDecision:
    """Classify a process for Resource Guard cleanup suggestions."""
    name = proc.name.strip()
    blob = _lower_blob(proc)

    if name in PROTECTED_PROCESS_NAMES:
        return ProcessDecision("protected", f"protected macOS/system process: {name}")

    for marker in PROTECTED_COMMAND_SUBSTRINGS:
        if marker.lower() in blob:
            return ProcessDecision("protected", f"protected service marker: {marker}")

    if _is_comfyui(proc):
        return ProcessDecision("protected", "ComfyUI process is protected while running")

    if _is_hermes_gateway(proc):
        profile = _hermes_profile(proc)
        if profile in ESSENTIAL_HERMES_PROFILES:
            return ProcessDecision("protected", f"essential Hermes gateway profile: {profile}")
        return ProcessDecision(
            "candidate",
            f"duplicate/non-essential Hermes gateway profile: {profile or 'unknown'}",
            killable_in_safe_clean=True,
            killable_in_strict_clean=True,
        )

    if _is_llama(proc):
        if _has_llama_keep_marker(proc):
            return ProcessDecision("protected", "llama process has explicit keep marker")
        if _is_orphan(proc):
            return ProcessDecision(
                "candidate",
                f"orphan {name} parented by launchd/init",
                killable_in_safe_clean=True,
                killable_in_strict_clean=True,
            )
        return ProcessDecision(
            "candidate",
            "llama process without keep marker; strict-clean only unless orphaned",
            killable_in_safe_clean=False,
            killable_in_strict_clean=True,
        )

    if (proc.status or "").lower() in {"zombie", "defunct"} or "<defunct>" in blob:
        return ProcessDecision(
            "candidate",
            "zombie/defunct process; recorded for operator review",
            killable_in_safe_clean=False,
            killable_in_strict_clean=False,
        )

    age = proc.elapsed_seconds or 0
    if name in BROWSER_HELPER_NAMES and age >= 3600:
        return ProcessDecision(
            "candidate",
            "old browser helper process",
            killable_in_safe_clean=True,
            killable_in_strict_clean=True,
        )

    if name.startswith("python") and age >= 6 * 3600 and any(
        marker in blob for marker in ("worker", "multiprocessing", "terminal", "agent child")
    ):
        return ProcessDecision(
            "candidate",
            "old python worker/terminal child process",
            killable_in_safe_clean=True,
            killable_in_strict_clean=True,
        )

    if any(marker in blob for marker in IMAGE_BACKEND_MARKERS) and not _is_comfyui(proc):
        return ProcessDecision(
            "candidate",
            "non-Comfy image generation backend",
            killable_in_safe_clean=True,
            killable_in_strict_clean=True,
        )

    return ProcessDecision("ignored", "not on Resource Guard cleanup allowlist")


def _normalise_pressure(pressure: str | None) -> str:
    value = (pressure or "unknown").strip().lower()
    if value in {"critical", "urgent", "starved", "red"}:
        return "critical"
    if value in {"warning", "warn", "yellow", "tight"}:
        return "warning"
    if value in {"normal", "green", "ok"}:
        return "normal"
    return "unknown"


def evaluate_resource_state(processes: Iterable[ProcessInfo], memory_status: MemoryStatus) -> ResourceState:
    """Evaluate ComfyUI readiness from resource signals, not RAM percent alone."""
    proc_list = list(processes)
    decisions = [(proc, classify_process(proc)) for proc in proc_list]
    candidates = [(proc, decision) for proc, decision in decisions if decision.kind == "candidate"]
    protected = [(proc, decision) for proc, decision in decisions if decision.kind == "protected"]

    pressure = _normalise_pressure(memory_status.pressure)
    swap_used = float(memory_status.swap_used_mb or 0.0)
    swap_delta = memory_status.swap_delta_mb
    swap_growing = swap_delta is not None and swap_delta > 0
    comfy_running = any(_is_comfyui(proc) for proc in proc_list)
    llama_running = any(_is_llama(proc) for proc in proc_list)
    keep_marker_count = sum(1 for proc in proc_list if _is_llama(proc) and _has_llama_keep_marker(proc))
    hermes_gateway_count = sum(1 for proc in proc_list if _is_hermes_gateway(proc))
    orphan_or_zombie = any(
        "orphan" in decision.reason or "zombie" in decision.reason or "defunct" in decision.reason
        for _, decision in candidates
    )
    high_memory_candidates = [
        proc for proc, decision in candidates if proc.rss_mb >= HIGH_PROCESS_RSS_MB and decision.killable_in_safe_clean
    ]
    safe_clean_candidates = [proc for proc, decision in candidates if decision.killable_in_safe_clean]

    reasons: list[str] = []
    if pressure == "critical":
        reasons.append("memory pressure is critical")
    elif pressure == "warning":
        reasons.append("memory pressure is elevated")
    if swap_used > 0:
        reasons.append("swap is in use")
    if swap_growing:
        reasons.append("swap is actively growing")
    if comfy_running:
        reasons.append("ComfyUI is running")
    if llama_running:
        reasons.append("llama-server is running")
    if hermes_gateway_count:
        reasons.append(f"Hermes gateway count: {hermes_gateway_count}")
    if orphan_or_zombie:
        reasons.append("orphan/zombie cleanup candidate exists")
    if high_memory_candidates:
        reasons.append("high-memory non-protected cleanup candidate exists")
    if keep_marker_count:
        reasons.append("keep-marker llama process is protected")

    starvation_signals = [
        pressure == "critical",
        swap_growing,
        swap_used > 0 and comfy_running and llama_running,
        bool(high_memory_candidates) and (comfy_running or llama_running),
        orphan_or_zombie and pressure in {"warning", "critical"},
    ]
    tight_signals = [
        pressure == "warning",
        swap_used > 0,
        swap_growing,
        bool(candidates),
        llama_running,
        hermes_gateway_count > len(ESSENTIAL_HERMES_PROFILES),
    ]

    if sum(1 for signal_present in starvation_signals if signal_present) >= 3:
        state: ResourceStateName = COMFY_STARVED
        recommended = "safe-clean-recommended"
    elif any(tight_signals):
        state = COMFY_TIGHT
        recommended = "review-candidates" if safe_clean_candidates else "monitor"
    else:
        state = COMFY_READY
        recommended = "ready"

    evidence: dict[str, object] = {
        "memory_pressure": pressure,
        "memory_percent": memory_status.percent,
        "swap_used_mb": memory_status.swap_used_mb,
        "swap_delta_mb": memory_status.swap_delta_mb,
        "swap_growing": swap_growing,
        "comfyui_running": comfy_running,
        "llama_running": llama_running,
        "hermes_gateway_count": hermes_gateway_count,
        "orphan_or_zombie_candidate": orphan_or_zombie,
        "candidate_count": len(candidates),
        "safe_clean_candidate_count": len(safe_clean_candidates),
        "high_memory_candidate_count": len(high_memory_candidates),
        "keep_marker_process_count": keep_marker_count,
        "protected_process_count": len(protected),
    }
    return ResourceState(state=state, recommended_action=recommended, reasons=reasons, evidence=evidence)


def _macos_memory_pressure() -> str:
    if sys.platform != "darwin":
        return "unknown"
    try:
        result = subprocess.run(["memory_pressure"], check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        return "unknown"
    output = (result.stdout + "\n" + result.stderr).lower()
    if any(token in output for token in ("critical", "urgent", "red")):
        return "critical"
    if any(token in output for token in ("warning", "warn", "yellow")):
        return "warning"
    if "normal" in output or "green" in output or result.returncode == 0:
        return "normal"
    return "unknown"


def _memory_snapshot(memory_status: MemoryStatus | None = None) -> MemoryStatus:
    if memory_status is not None:
        return memory_status
    if psutil is None:
        return MemoryStatus(pressure=_macos_memory_pressure())
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return MemoryStatus(
        pressure=_macos_memory_pressure(),
        swap_used_mb=round(swap.used / 1024 / 1024, 1),
        swap_delta_mb=None,
        percent=float(vm.percent),
        total_mb=round(vm.total / 1024 / 1024, 1),
        available_mb=round(vm.available / 1024 / 1024, 1),
        used_mb=round(vm.used / 1024 / 1024, 1),
    )


def collect_processes() -> list[ProcessInfo]:
    if psutil is None:
        raise RuntimeError("psutil is required to collect process information")

    current_pid = os.getpid()
    processes: list[ProcessInfo] = []
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "memory_info", "create_time", "status"]):
        try:
            info = process.info
            pid = int(info.get("pid") or 0)
            if pid == current_pid:
                continue
            cmdline = info.get("cmdline") or []
            command = " ".join(str(part) for part in cmdline) if cmdline else (info.get("name") or "")
            mem = info.get("memory_info")
            create_time = info.get("create_time")
            elapsed = max(0, int(datetime.now().timestamp() - float(create_time))) if create_time else None
            try:
                parent = process.parent()
                parent_name = parent.name() if parent else None
            except (psutil.Error, Exception):
                parent_name = None
            processes.append(
                ProcessInfo(
                    pid=pid,
                    ppid=info.get("ppid"),
                    name=info.get("name") or "",
                    command=command,
                    rss_mb=round((getattr(mem, "rss", 0) or 0) / 1024 / 1024, 1),
                    elapsed_seconds=elapsed,
                    parent_name=parent_name,
                    status=info.get("status"),
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return processes


def _record(proc: ProcessInfo, decision: ProcessDecision | None = None) -> dict[str, object]:
    payload = asdict(proc)
    if decision:
        payload.update(asdict(decision))
    return payload


def _normalise_mode(mode: Mode | str) -> str:
    normalized = mode.replace("_", "-").lower()
    if normalized not in {"dry-run", "safe-clean", "strict-clean"}:
        raise ValueError(f"unsupported mode: {mode}")
    return normalized


def _normalise_phase(phase: Phase | str) -> Phase:
    normalized = phase.replace("_", "-").lower()
    if normalized not in {"preflight", "runtime", "post-run"}:
        raise ValueError(f"unsupported phase: {phase}")
    return normalized  # type: ignore[return-value]


def _suggested_action(phase: Phase, state: ResourceState, cleanup_candidates: list[dict[str, object]]) -> str:
    if phase == "post-run":
        return "post-run cleanup candidates require approval" if cleanup_candidates else "post-run cleanup not needed"
    if phase == "runtime":
        if state.state == COMFY_STARVED:
            return "emit runtime alert payload and continue monitoring; do not kill or interrupt"
        if state.state == COMFY_TIGHT:
            return "monitor closely and surface cleanup candidates; do not kill"
        return "continue runtime monitor"
    if state.state == COMFY_STARVED:
        return "delay generation and request operator review"
    if state.state == COMFY_TIGHT:
        return "proceed with caution and review candidates"
    return "ready"


def _slack_payload(phase: Phase, state: ResourceState, snapshot_index: int | None = None) -> dict[str, object] | None:
    if phase != "runtime" or state.state == COMFY_READY:
        return None
    label = f" snapshot={snapshot_index}" if snapshot_index is not None else ""
    return {
        "send": False,
        "phase": phase,
        "state": state.state,
        "text": f"Comfy Resource Guard {state.state}{label}: " + "; ".join(state.reasons),
        "reasons": state.reasons,
        "evidence": state.evidence,
    }


def _build_report(
    *,
    mode: str,
    phase: Phase,
    proc_list: list[ProcessInfo],
    memory_status: MemoryStatus,
    log_dir: Path | str,
    snapshot_index: int | None = None,
    write_log: bool = True,
) -> dict[str, object]:
    state = evaluate_resource_state(proc_list, memory_status)
    decisions = [(proc, classify_process(proc)) for proc in proc_list]
    cleanup_candidates = [_record(proc, decision) for proc, decision in decisions if decision.kind == "candidate"]
    protected = [_record(proc, decision) for proc, decision in decisions if decision.kind == "protected"]
    ignored = [_record(proc, decision) for proc, decision in decisions if decision.kind == "ignored"]
    alert = _slack_payload(phase, state, snapshot_index)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, object] = {
        "timestamp": timestamp,
        "mode": mode,
        "phase": phase,
        "state": state.state,
        "resource_state": state.state,
        "reasons": state.reasons,
        "state_reasons": state.reasons,
        "state_evidence": state.evidence,
        "observed_processes": [asdict(p) for p in sorted(proc_list, key=lambda p: p.rss_mb, reverse=True)[:15]],
        "protected_processes": protected,
        "cleanup_candidates": cleanup_candidates,
        "suggested_action": _suggested_action(phase, state, cleanup_candidates),
        "kill_performed": False,
        "memory_before": asdict(memory_status),
        "memory_after": asdict(memory_status),
        "top_processes": [asdict(p) for p in sorted(proc_list, key=lambda p: p.rss_mb, reverse=True)[:15]],
        "candidates": cleanup_candidates,
        "killed_processes": [],
        "skipped_protected_processes": protected,
        "skipped_non_allowlisted_processes": [
            _record(proc, decision)
            for proc, decision in decisions
            if decision.kind == "ignored" or (decision.kind == "candidate" and not decision.killable_in_safe_clean)
        ],
        "ignored_processes_count": len(ignored),
        "slack_alert_payloads": [alert] if alert else [],
        "errors": [],
        "result": "dry_run_complete" if mode == "dry-run" else "clean_suggestion_complete",
    }
    if snapshot_index is not None:
        report["snapshot_index"] = snapshot_index

    if write_log:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        output_file = log_path / f"{timestamp}_{phase}_{mode}.json"
        output_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["log_file"] = str(output_file)
    return report


def run_guard(
    mode: Mode | str = "dry-run",
    *,
    phase: Phase | str = "preflight",
    processes: Iterable[ProcessInfo] | None = None,
    memory_status: MemoryStatus | None = None,
    log_dir: Path | str = DEFAULT_LOG_DIR,
    kill_fn: Callable[[int], None] | None = None,
    strict_confirmed: bool = False,
) -> dict[str, object]:
    """Run one Resource Guard phase snapshot; never terminates processes."""
    del kill_fn
    normalized_mode = _normalise_mode(mode)
    normalized_phase = _normalise_phase(phase)
    if normalized_mode == "strict-clean" and not strict_confirmed:
        raise PermissionError("strict-clean requires explicit --confirm-strict approval")

    proc_list = list(processes) if processes is not None else collect_processes()
    before_status = _memory_snapshot(memory_status)
    return _build_report(
        mode=normalized_mode,
        phase=normalized_phase,
        proc_list=proc_list,
        memory_status=before_status,
        log_dir=log_dir,
    )


def _live_snapshot(previous_swap_used_mb: float | None = None) -> tuple[list[ProcessInfo], MemoryStatus]:
    processes = collect_processes()
    status = _memory_snapshot()
    if status.swap_used_mb is not None and previous_swap_used_mb is not None:
        status = replace(status, swap_delta_mb=round(status.swap_used_mb - previous_swap_used_mb, 1))
    return processes, status


def run_watch(
    mode: Mode | str = "dry-run",
    *,
    interval: float = 10.0,
    duration: float = 300.0,
    log_dir: Path | str = DEFAULT_LOG_DIR,
    snapshot_provider: Callable[[int], tuple[Iterable[ProcessInfo], MemoryStatus]] | None = None,
    strict_confirmed: bool = False,
) -> dict[str, object]:
    """Run runtime monitor snapshots and record state transitions."""
    normalized_mode = _normalise_mode(mode)
    if normalized_mode == "strict-clean" and not strict_confirmed:
        raise PermissionError("strict-clean requires explicit --confirm-strict approval")

    if snapshot_provider is not None and interval == 0:
        max_snapshots = max(1, int(duration))
    else:
        max_snapshots = max(1, int(math.ceil(max(duration, interval) / max(interval, 0.001))))

    snapshots: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    alerts: list[dict[str, object]] = []
    previous_state: str | None = None
    previous_swap: float | None = None
    started = time.monotonic()

    for index in range(max_snapshots):
        if snapshot_provider is None:
            proc_list, status = _live_snapshot(previous_swap)
        else:
            provided_processes, status = snapshot_provider(index)
            proc_list = list(provided_processes)
        previous_swap = status.swap_used_mb

        snapshot = _build_report(
            mode=normalized_mode,
            phase="runtime",
            proc_list=proc_list,
            memory_status=status,
            log_dir=log_dir,
            snapshot_index=index,
            write_log=False,
        )
        snapshots.append(snapshot)
        current_state = str(snapshot["state"])
        if previous_state is not None and previous_state != current_state:
            transitions.append({"from": previous_state, "to": current_state, "snapshot_index": index})
        previous_state = current_state
        alerts.extend(snapshot["slack_alert_payloads"])  # type: ignore[arg-type]

        if snapshot_provider is None:
            elapsed = time.monotonic() - started
            if elapsed + interval > duration or index == max_snapshots - 1:
                break
            time.sleep(interval)

    final = snapshots[-1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result: dict[str, object] = {
        "timestamp": timestamp,
        "mode": normalized_mode,
        "phase": "runtime",
        "state": final["state"],
        "resource_state": final["state"],
        "reasons": final["reasons"],
        "observed_processes": final["observed_processes"],
        "protected_processes": final["protected_processes"],
        "cleanup_candidates": final["cleanup_candidates"],
        "suggested_action": final["suggested_action"],
        "kill_performed": False,
        "snapshots": snapshots,
        "state_transitions": transitions,
        "slack_alert_payloads": alerts,
        "killed_processes": [],
        "result": "runtime_monitor_complete",
    }
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    output_file = log_path / f"{timestamp}_runtime_watch_{normalized_mode}.json"
    output_file.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["log_file"] = str(output_file)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ComfyUI Resource Guard")
    parser.add_argument(
        "--mode",
        choices=("dry-run", "safe-clean", "strict-clean", "dry_run", "safe_clean", "strict_clean"),
        default="dry-run",
        help="observation mode; this stage never kills processes",
    )
    parser.add_argument(
        "--phase",
        choices=("preflight", "runtime", "post-run", "post_run"),
        default="preflight",
        help="guard phase to evaluate",
    )
    parser.add_argument("--watch", action="store_true", help="runtime monitor loop; valid with --phase runtime")
    parser.add_argument("--once", action="store_true", help="single snapshot evaluation; default when --watch is absent")
    parser.add_argument("--interval", type=float, default=10.0, help="watch interval seconds")
    parser.add_argument("--duration", type=float, default=300.0, help="watch duration seconds")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="directory for JSON guard logs")
    parser.add_argument(
        "--confirm-strict",
        action="store_true",
        help="explicit operator approval required to enter strict-clean planning mode",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        phase = _normalise_phase(args.phase)
        if args.watch:
            if phase != "runtime":
                raise ValueError("--watch is only supported with --phase runtime")
            report = run_watch(
                mode=args.mode,
                interval=args.interval,
                duration=args.duration,
                log_dir=args.log_dir,
                strict_confirmed=args.confirm_strict,
            )
        else:
            report = run_guard(
                mode=args.mode,
                phase=phase,
                log_dir=args.log_dir,
                strict_confirmed=args.confirm_strict,
            )
    except PermissionError as exc:
        print(json.dumps({"result": "refused", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"result": "error", "error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
