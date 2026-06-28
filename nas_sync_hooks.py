from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from hermes_constants import get_default_hermes_root, get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_SECONDS = 60.0
_SUPPORTED_CATEGORIES = {"image", "documents", "story"}
_IN_PROCESS_LAST_LAUNCH: dict[str, float] = {}
_IN_PROCESS_LOCK = Lock()
HOOK_FREEZE_ENV = "HERMES_NAS_HOOKS_FROZEN"


def _hooks_frozen() -> bool:
    value = os.environ.get(HOOK_FREEZE_ENV, "").strip().casefold()
    return value in {"1", "true", "yes", "on", "frozen", "freeze"}
_STORY_SCOPE_EMPTY_NAMES = {
    "",
    ".",
    "ai_agent",
    "archive",
    "archives",
    "document",
    "documents",
    "game",
    "games",
    "image",
    "images",
    "misc",
    "root",
    "story",
    "stories",
}
_NAS_BACKUP_CONFIG_KEYS = (
    ("nas", "backup_script"),
    ("nas", "immediate_backup_script"),
    ("nas", "sync_hook_script"),
)


def _resolve_script_candidate(raw: str) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.exists():
        return path
    return None


def _resolve_configured_nas_hook_script() -> Path | None:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    except Exception:
        return None

    for section, key in _NAS_BACKUP_CONFIG_KEYS:
        value = config.get(section, {}) if isinstance(config, dict) else {}
        if isinstance(value, dict):
            candidate = _resolve_script_candidate(str(value.get(key) or "").strip())
            if candidate is not None:
                return candidate
    return None


def _resolve_nas_hook_script() -> Path | None:
    """Resolve the profile-local NAS backup script used for immediate hooks."""
    override = os.environ.get("HERMES_NAS_BACKUP_SCRIPT", "").strip()
    candidate = _resolve_script_candidate(override)
    if candidate is not None:
        return candidate

    candidate = _resolve_configured_nas_hook_script()
    if candidate is not None:
        return candidate

    common_candidate = (
        get_default_hermes_root() / "profiles" / "cron-fast" / "scripts" / "hermes_nas_backup.py"
    )
    if common_candidate.exists():
        return common_candidate

    candidate = get_hermes_home() / "scripts" / "hermes_nas_backup.py"
    if candidate.exists():
        return candidate

    return None


def _resolve_nas_hook_state_dir() -> Path:
    """Return the cron-fast NAS hook state directory used by the runner."""
    return get_default_hermes_root() / "profiles" / "cron-fast" / "state" / "nas-sync"


def _artifact_hook_key(category: str, scope: str, artifact_path: Path) -> str:
    resolved = artifact_path.resolve(strict=False)
    return f"{category}|{scope}|{resolved}"


def _source_hook_key(category: str, scope: str, source_root: Path) -> str:
    resolved = source_root.resolve(strict=False)
    return f"{category}|{scope}|{resolved}"


def _hook_key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _nas_hook_state_path(key: str) -> Path:
    return _resolve_nas_hook_state_dir() / f"{_hook_key_hash(key)}.json"


def _nas_hook_launcher_event_log_path() -> Path:
    return _resolve_nas_hook_state_dir() / "launcher_events.jsonl"


def _nas_hook_launcher_output_dir() -> Path:
    return _resolve_nas_hook_state_dir() / "launcher-output"


def _nas_hook_launcher_metadata_dir() -> Path:
    return _resolve_nas_hook_state_dir() / "launcher-metadata"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_launcher_event(payload: dict[str, object]) -> None:
    path = _nas_hook_launcher_event_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_launcher_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _normalized_story_scope(scope: str) -> str:
    normalized = scope.strip()
    if normalized.casefold() in _STORY_SCOPE_EMPTY_NAMES:
        return ""
    return normalized


def _normalized_story_source_root(source_root: Path, artifact_path: Path) -> Path:
    candidates = [source_root, source_root.parent, artifact_path, artifact_path.parent]
    candidates.extend(source_root.parents)
    candidates.extend(artifact_path.parents)
    for candidate in candidates:
        if candidate.name.casefold() == "story":
            return candidate
    return source_root if source_root.is_dir() else source_root.parent


def queue_nas_sync_hook(
    *,
    category: str,
    scope: str,
    artifact_path: Path,
    source_root: Path,
    debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
) -> bool:
    """Launch a non-blocking NAS sync hook for a published artifact.

    The helper is deliberately best-effort: if the hook script is missing,
    already triggered recently, or fails to launch, the calling publish path
    still succeeds unchanged.
    """
    normalized_category = category.strip().lower()
    if normalized_category not in _SUPPORTED_CATEGORIES:
        return False
    if _hooks_frozen():
        logger.info("NAS sync hook blocked by freeze state for category=%s scope=%s", normalized_category, scope)
        return False

    normalized_scope = _normalized_story_scope(scope) if normalized_category == "story" else scope.strip()
    normalized_source_root = (
        _normalized_story_source_root(source_root, artifact_path)
        if normalized_category == "story"
        else (source_root if source_root.is_dir() else source_root.parent)
    )

    key = _source_hook_key(normalized_category, normalized_scope, normalized_source_root)
    key_hash = _hook_key_hash(key)
    state_path = _nas_hook_state_path(key)
    now = time.monotonic()
    with _IN_PROCESS_LOCK:
        last = _IN_PROCESS_LAST_LAUNCH.get(key)
        if last is not None and (now - last) < debounce_seconds:
            logger.debug(
                "Skipping NAS sync hook for %s (debounced %.1fs < %.1fs)",
                key,
                now - last,
                debounce_seconds,
            )
            return False

        script = _resolve_nas_hook_script()
        if script is None:
            logger.warning("NAS sync hook script not found; skipping immediate sync for %s", key)
            return False

        cmd = [
            sys.executable,
            str(script),
            "--hook",
            str(normalized_source_root),
            "--category",
            normalized_category,
            "--scope",
            normalized_scope,
            "--artifact-path",
            str(artifact_path),
            "--debounce-seconds",
            str(int(debounce_seconds)),
        ]
        try:
            env = os.environ.copy()
            state_dir = _resolve_nas_hook_state_dir()
            env["HERMES_NAS_HOOK_STATE_DIR"] = str(state_dir)
            launched_at = _utc_now_iso()
            launch_token = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{key_hash[:12]}"
            stdout_path = _nas_hook_launcher_output_dir() / f"{launch_token}.stdout.log"
            stderr_path = _nas_hook_launcher_output_dir() / f"{launch_token}.stderr.log"
            metadata_path = _nas_hook_launcher_metadata_dir() / f"{launch_token}.json"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = None
            stderr_handle = None
            try:
                stdout_handle = stdout_path.open("ab")
                stderr_handle = stderr_path.open("ab")
                process = subprocess.Popen(
                    cmd,
                    start_new_session=True,
                    close_fds=True,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=env,
                )
            finally:
                if stdout_handle is not None:
                    stdout_handle.close()
                if stderr_handle is not None:
                    stderr_handle.close()
        except Exception as exc:  # noqa: BLE001
            try:
                _append_launcher_event(
                    {
                        "event": "launch_failed",
                        "launched_at": _utc_now_iso(),
                        "key": key,
                        "key_hash": key_hash,
                        "category": normalized_category,
                        "scope": normalized_scope,
                        "source_root": str(normalized_source_root),
                        "artifact_path": str(artifact_path),
                        "state_path": str(state_path),
                        "cmd": cmd,
                        "error": str(exc),
                    }
                )
            except Exception:
                pass
            logger.warning("Failed to launch NAS sync hook for %s: %s", key, exc)
            return False

        metadata = {
            "event": "launched",
            "launched_at": launched_at,
            "pid": getattr(process, "pid", None),
            "key": key,
            "key_hash": key_hash,
            "category": normalized_category,
            "scope": normalized_scope,
            "source_root": str(normalized_source_root),
            "artifact_path": str(artifact_path),
            "state_path": str(state_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "metadata_path": str(metadata_path),
            "cmd": cmd,
            "debounce_seconds": debounce_seconds,
        }
        try:
            _write_launcher_metadata(metadata_path, metadata)
            _append_launcher_event(metadata)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NAS sync hook launched for %s but metadata logging failed: %s", key, exc)

        _IN_PROCESS_LAST_LAUNCH[key] = now
        return True
