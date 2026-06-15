"""Tests for phased ComfyUI Resource Guard runtime monitor."""

from __future__ import annotations

import json
from pathlib import Path

from tools.comfy_ram_guard import (
    COMFY_READY,
    COMFY_STARVED,
    COMFY_TIGHT,
    ESSENTIAL_HERMES_PROFILES,
    MemoryStatus,
    ProcessInfo,
    classify_process,
    evaluate_resource_state,
    run_guard,
    run_watch,
)


def proc(
    pid: int,
    name: str,
    command: str = "",
    rss_mb: float = 100.0,
    elapsed_seconds: int = 7200,
    ppid: int = 1,
    parent_name: str = "launchd",
    status: str = "sleeping",
) -> ProcessInfo:
    return ProcessInfo(
        pid=pid,
        ppid=ppid,
        name=name,
        command=command or name,
        rss_mb=rss_mb,
        elapsed_seconds=elapsed_seconds,
        parent_name=parent_name,
        status=status,
    )


def memory(
    pressure: str = "normal",
    swap_used_mb: float = 0.0,
    swap_delta_mb: float | None = None,
    percent: float = 91.0,
) -> MemoryStatus:
    return MemoryStatus(
        pressure=pressure,
        swap_used_mb=swap_used_mb,
        swap_delta_mb=swap_delta_mb,
        percent=percent,
        total_mb=24576.0,
        available_mb=4096.0,
        used_mb=20480.0,
    )


def test_preflight_phase_outputs_required_json_shape(tmp_path: Path):
    result = run_guard(
        mode="dry-run",
        phase="preflight",
        processes=[proc(100, "kernel_task")],
        memory_status=memory(pressure="normal", swap_used_mb=0),
        log_dir=tmp_path,
    )

    assert result["phase"] == "preflight"
    assert result["state"] in {COMFY_READY, COMFY_TIGHT, COMFY_STARVED}
    assert "reasons" in result
    assert "observed_processes" in result
    assert "protected_processes" in result
    assert "cleanup_candidates" in result
    assert "suggested_action" in result
    assert result["kill_performed"] is False


def test_runtime_monitor_watch_records_multiple_snapshots_and_state_transitions(tmp_path: Path):
    snapshots = [
        ([proc(1, "kernel_task")], memory(pressure="normal", swap_used_mb=0)),
        (
            [proc(1, "kernel_task"), proc(2, "llama-server", "llama-server --model active.gguf", parent_name="bash", ppid=77)],
            memory(pressure="warning", swap_used_mb=512, swap_delta_mb=256),
        ),
        (
            [
                proc(1, "ComfyUI", "python /Volumes/SSD_Hermes/ComfyUI/main.py"),
                proc(2, "llama-server", "llama-server --model active.gguf", parent_name="bash", ppid=77, rss_mb=5000),
                proc(3, "Google Chrome Helper", rss_mb=4096, elapsed_seconds=7200),
            ],
            memory(pressure="critical", swap_used_mb=8192, swap_delta_mb=1536),
        ),
    ]

    result = run_watch(
        mode="dry-run",
        interval=0,
        duration=3,
        log_dir=tmp_path,
        snapshot_provider=lambda index: snapshots[index],
    )

    assert result["phase"] == "runtime"
    assert result["kill_performed"] is False
    assert len(result["snapshots"]) >= 2
    assert [s["state"] for s in result["snapshots"]] == [COMFY_READY, COMFY_TIGHT, COMFY_STARVED]
    assert result["state_transitions"] == [
        {"from": COMFY_READY, "to": COMFY_TIGHT, "snapshot_index": 1},
        {"from": COMFY_TIGHT, "to": COMFY_STARVED, "snapshot_index": 2},
    ]
    assert result["slack_alert_payloads"]
    alert = result["slack_alert_payloads"][-1]
    assert alert["send"] is False
    assert alert["state"] == COMFY_STARVED
    assert "COMFY_STARVED" in alert["text"]


def test_post_run_phase_is_cleanup_suggestion_only(tmp_path: Path):
    killed = []
    result = run_guard(
        mode="safe-clean",
        phase="post-run",
        processes=[proc(100, "llama-cli", "llama-cli --stale", ppid=1)],
        memory_status=memory(pressure="normal", swap_used_mb=0),
        log_dir=tmp_path,
        kill_fn=lambda pid: killed.append(pid),
    )

    assert result["phase"] == "post-run"
    assert result["state"] == COMFY_TIGHT
    assert result["cleanup_candidates"][0]["pid"] == 100
    assert result["suggested_action"] == "post-run cleanup candidates require approval"
    assert result["kill_performed"] is False
    assert killed == []


def test_ram_percent_alone_does_not_decide_state_or_action(tmp_path: Path):
    result = run_guard(
        mode="dry-run",
        phase="preflight",
        processes=[proc(200, "kernel_task")],
        memory_status=memory(pressure="normal", swap_used_mb=0, percent=99.0),
        log_dir=tmp_path,
    )

    assert result["state"] in {COMFY_READY, COMFY_TIGHT}
    assert result["state"] != COMFY_STARVED
    assert result["suggested_action"] != "safe-clean-recommended"
    assert result["memory_before"]["percent"] == 99.0


def test_swap_pressure_orphan_combination_drives_starved_state():
    state = evaluate_resource_state(
        [
            proc(1, "ComfyUI", "python /Volumes/SSD_Hermes/ComfyUI/main.py"),
            proc(2, "llama-server", "llama-server --model active.gguf", rss_mb=5000, ppid=77, parent_name="bash"),
            proc(3, "Google Chrome Helper", rss_mb=4096, elapsed_seconds=7200),
            proc(4, "llama-cli", "llama-cli --stale", ppid=1),
        ],
        memory(pressure="critical", swap_used_mb=8192, swap_delta_mb=1536, percent=51.0),
    )

    assert state.state == COMFY_STARVED
    assert "memory pressure is critical" in state.reasons
    assert "swap is actively growing" in state.reasons
    assert state.evidence["orphan_or_zombie_candidate"] is True


def test_dry_run_and_safe_clean_do_not_kill_in_this_stage(tmp_path: Path):
    killed = []
    for mode in ["dry-run", "safe-clean"]:
        result = run_guard(
            mode=mode,
            phase="runtime",
            processes=[proc(100, "llama-server", "llama-server --model stale.gguf", ppid=1)],
            memory_status=memory(pressure="warning", swap_used_mb=2048, swap_delta_mb=512),
            log_dir=tmp_path,
            kill_fn=lambda pid: killed.append(pid),
        )
        assert result["kill_performed"] is False
        assert result["killed_processes"] == []
    assert killed == []


def test_protected_processes_include_comfyui_and_keep_marker_llama():
    comfy = proc(4, "ComfyUI", "python main.py --listen 127.0.0.1")
    kept_llama = proc(8, "llama-server", "llama-server --keep --model local.gguf")

    assert classify_process(comfy).kind == "protected"
    assert classify_process(kept_llama).kind == "protected"


def test_slack_payload_generated_only_not_sent(tmp_path: Path):
    result = run_guard(
        mode="dry-run",
        phase="runtime",
        processes=[
            proc(1, "ComfyUI", "python /Volumes/SSD_Hermes/ComfyUI/main.py"),
            proc(2, "llama-server", "llama-server --model active.gguf", rss_mb=5000, ppid=77, parent_name="bash"),
            proc(3, "Google Chrome Helper", rss_mb=4096, elapsed_seconds=7200),
        ],
        memory_status=memory(pressure="critical", swap_used_mb=8192, swap_delta_mb=1536),
        log_dir=tmp_path,
    )

    payloads = result["slack_alert_payloads"]
    assert payloads
    assert payloads[0]["send"] is False
    assert payloads[0]["phase"] == "runtime"
    assert payloads[0]["state"] == COMFY_STARVED


def test_logs_include_phase_state_and_kill_false(tmp_path: Path):
    result = run_guard(
        mode="dry-run",
        phase="preflight",
        processes=[proc(100, "llama-server", "llama-server --model stale.gguf", ppid=1), proc(200, "kernel_task")],
        memory_status=memory(pressure="warning", swap_used_mb=2048, swap_delta_mb=512),
        log_dir=tmp_path,
    )

    log_files = list(tmp_path.glob("*.json"))
    assert len(log_files) == 1
    payload = json.loads(log_files[0].read_text())
    assert payload["phase"] == "preflight"
    assert payload["state"] == result["state"]
    assert payload["kill_performed"] is False
    assert payload["protected_processes"][0]["pid"] == 200


def test_essential_hermes_profiles_constant_covers_required_gateways():
    assert {"speedy", "cron-fast", "coder"}.issubset(ESSENTIAL_HERMES_PROFILES)
