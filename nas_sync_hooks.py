from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock

from hermes_constants import get_default_hermes_root, get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_SECONDS = 60.0
_SUPPORTED_CATEGORIES = {"image", "documents", "story"}
_IN_PROCESS_LAST_LAUNCH: dict[str, float] = {}
_IN_PROCESS_LOCK = Lock()
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


def _artifact_hook_key(category: str, scope: str, artifact_path: Path) -> str:
    resolved = artifact_path.resolve(strict=False)
    return f"{category}|{scope}|{resolved}"


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

    normalized_scope = _normalized_story_scope(scope) if normalized_category == "story" else scope.strip()
    normalized_source_root = (
        _normalized_story_source_root(source_root, artifact_path)
        if normalized_category == "story"
        else (source_root if source_root.is_dir() else source_root.parent)
    )

    key = _artifact_hook_key(normalized_category, normalized_scope, artifact_path)
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
            subprocess.Popen(
                cmd,
                start_new_session=True,
                close_fds=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to launch NAS sync hook for %s: %s", key, exc)
            return False

        _IN_PROCESS_LAST_LAUNCH[key] = now
        return True
