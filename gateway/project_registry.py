from __future__ import annotations

"""Project registry and Games workspace helpers.

This module owns the first-pass project bookkeeping needed by Hermes game
workflows:

- Persist a profile-local registry under ``HERMES_HOME/state/project_registry.json``.
- Keep a stable ``project_id`` for a project name across retries and later dates.
- Materialize the canonical ``Games/<project_id>/UnityProject`` folder tree
  without creating Unity-generated files.
"""

import json
import re
import tempfile
import unicodedata
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_work_dir
from hermes_constants import get_default_hermes_root

_PROJECT_REGISTRY_FILENAME = "project_registry.json"
_PROJECT_REGISTRY_VERSION = 1
_PROJECT_LOOKUP_FALLBACK_NAME = "project"
_PROJECT_NAME_TOKEN_RE = re.compile(r"[^0-9A-Za-z가-힣]+")
_VERSIONED_STEM_RE = re.compile(r"^(?P<base>.+)_v(?P<version>\d+)$")
_DATED_PROJECT_ID_RE = re.compile(r"^\d{6}_")


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    """Canonical metadata for a registered project."""

    project_name: str
    normalized_name: str
    project_id: str
    created_on: str


def normalize_project_name(project_name: str) -> str:
    """Return a filesystem-safe project slug.

    The first-pass rule is intentionally conservative: keep alphanumerics and
    Hangul, collapse any other separator/punctuation runs to a single underscore,
    and trim leading/trailing underscores.
    """
    normalized = unicodedata.normalize("NFKC", str(project_name)).strip()
    normalized = _PROJECT_NAME_TOKEN_RE.sub("_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or _PROJECT_LOOKUP_FALLBACK_NAME


def _coerce_created_on(created_on: date | datetime | str) -> date:
    if isinstance(created_on, datetime):
        return created_on.date()
    if isinstance(created_on, date):
        return created_on
    text = str(created_on).strip()
    if not text:
        raise ValueError("created_on must not be empty")
    if "T" in text:
        text = text.split("T", 1)[0]
    return date.fromisoformat(text)


def format_project_id(project_name: str, created_on: date | datetime | str) -> str:
    """Build the canonical ``YYMMDD_project_name`` identifier."""
    created = _coerce_created_on(created_on)
    normalized_name = normalize_project_name(project_name)
    if _DATED_PROJECT_ID_RE.match(normalized_name):
        return normalized_name
    return f"{created:%y%m%d}_{normalized_name}"


def project_registry_path() -> Path:
    """Return the shared registry path under the Hermes root ``state`` dir."""
    return get_default_hermes_root() / "state" / _PROJECT_REGISTRY_FILENAME


def _project_lookup_key(project_name: str) -> str:
    return normalize_project_name(project_name).casefold()


def _record_from_payload(payload: Mapping[str, Any]) -> ProjectRecord | None:
    try:
        project_name = str(payload.get("project_name", "")).strip()
        normalized_name = str(payload.get("normalized_name", "")).strip()
        project_id = str(payload.get("project_id", "")).strip()
        created_on = str(payload.get("created_on", "")).strip()
    except Exception:
        return None
    if not (project_name and normalized_name and project_id and created_on):
        return None
    return ProjectRecord(
        project_name=project_name,
        normalized_name=normalized_name,
        project_id=project_id,
        created_on=created_on,
    )


def load_project_registry(*, registry_path: Path | None = None) -> dict[str, ProjectRecord]:
    """Load the on-disk registry, returning an empty mapping on first run."""
    path = registry_path or project_registry_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

    projects = raw.get("projects", {}) if isinstance(raw, dict) else {}
    if isinstance(projects, list):
        records = [
            record
            for record in (_record_from_payload(entry) for entry in projects if isinstance(projects, list))
            if record is not None
        ]
        return {record.normalized_name.casefold(): record for record in records}
    if not isinstance(projects, dict):
        return {}

    registry: dict[str, ProjectRecord] = {}
    for key, payload in projects.items():
        if not isinstance(payload, dict):
            continue
        record = _record_from_payload(payload)
        if record is None:
            continue
        registry[str(key).casefold()] = record
    return registry


def _write_project_registry(
    registry: Mapping[str, ProjectRecord],
    *,
    registry_path: Path | None = None,
) -> Path:
    path = registry_path or project_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": _PROJECT_REGISTRY_VERSION,
        "projects": {
            key: asdict(record)
            for key, record in sorted(registry.items(), key=lambda item: item[0])
        },
    }

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
    return path


def register_project(
    project_name: str,
    created_on: date | datetime | str,
    *,
    registry_path: Path | None = None,
) -> ProjectRecord:
    """Return a stable project record, creating it on first use.

    The first call for a given normalized project name wins. Subsequent calls
    re-use the original ``project_id`` even when ``created_on`` changes.
    """
    lookup_key = _project_lookup_key(project_name)
    registry = load_project_registry(registry_path=registry_path)
    existing = registry.get(lookup_key)
    if existing is not None:
        return existing

    created = _coerce_created_on(created_on)
    record = ProjectRecord(
        project_name=str(project_name).strip(),
        normalized_name=normalize_project_name(project_name),
        project_id=format_project_id(project_name, created),
        created_on=created.isoformat(),
    )
    registry[lookup_key] = record
    _write_project_registry(registry, registry_path=registry_path)
    return record


def _approval_proof_allows_registry_cleanup(approval_proof: Mapping[str, Any] | None) -> bool:
    return isinstance(approval_proof, Mapping) and approval_proof.get("user_approved") is True


def remove_project_registry_entry(
    project_name: str | None = None,
    *,
    project_id: str | None = None,
    approval_proof: Mapping[str, Any] | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Remove a project registry entry only when the approval proof is present.

    The registry JSON file itself is never deleted; the helper only removes the
    matching entry and rewrites the file in place.
    """
    path = registry_path or project_registry_path()
    registry = load_project_registry(registry_path=path)
    proof_payload = dict(approval_proof) if isinstance(approval_proof, Mapping) else None
    if not _approval_proof_allows_registry_cleanup(approval_proof):
        return {
            "registry_cleanup_attempted": False,
            "registry_cleanup_executed": False,
            "registry_cleanup_status": "approval_required",
            "registry_cleanup_project_id": project_id,
            "registry_cleanup_error": None,
            "registry_entry_removed": False,
            "registry_file_removed": False,
            "registry_file_path": str(path),
            "registry_file_exists_after": path.exists(),
            "approval_proof": proof_payload,
        }

    lookup_key = _project_lookup_key(project_name) if project_name is not None else None
    removed_key: str | None = None
    removed_record: ProjectRecord | None = None
    if lookup_key is not None and lookup_key in registry:
        removed_key = lookup_key
        removed_record = registry.pop(lookup_key)
    elif project_id:
        for key, record in list(registry.items()):
            if record.project_id == project_id:
                removed_key = key
                removed_record = registry.pop(key)
                break

    if removed_record is None:
        return {
            "registry_cleanup_attempted": True,
            "registry_cleanup_executed": True,
            "registry_cleanup_status": "not_found",
            "registry_cleanup_project_id": project_id,
            "registry_cleanup_error": None,
            "registry_entry_removed": False,
            "registry_file_removed": False,
            "registry_file_path": str(path),
            "registry_file_exists_after": path.exists(),
            "removed_key": removed_key,
            "approval_proof": proof_payload,
        }

    _write_project_registry(registry, registry_path=path)
    return {
        "registry_cleanup_attempted": True,
        "registry_cleanup_executed": True,
        "registry_cleanup_status": "removed",
        "registry_cleanup_project_id": removed_record.project_id,
        "registry_cleanup_error": None,
        "registry_entry_removed": True,
        "registry_file_removed": False,
        "registry_file_path": str(path),
        "registry_file_exists_after": path.exists(),
        "removed_key": removed_key,
        "removed_project_id": removed_record.project_id,
        "removed_project_name": removed_record.project_name,
        "approval_proof": proof_payload,
    }


def create_games_project_tree(
    project_name: str,
    created_on: date | datetime | str,
    *,
    work_root: Path | None = None,
    registry_path: Path | None = None,
    project_record: ProjectRecord | None = None,
) -> Path:
    """Create the canonical Games project folder tree and return its root."""
    record = project_record or register_project(project_name, created_on, registry_path=registry_path)
    games_root = Path(work_root or get_hermes_work_dir("Games")).expanduser()
    project_root = games_root / record.project_id

    folders = [
        project_root,
        project_root / "UnityProject",
        project_root / "Builds",
        project_root / "External",
        project_root / "UnityProject" / "Assets",
        project_root / "UnityProject" / "Packages",
        project_root / "UnityProject" / "ProjectSettings",
        project_root / "External" / "Spine",
        project_root / "External" / "Live2D",
        project_root / "External" / "Audio",
        project_root / "External" / "Import",
        project_root / "External" / "VFX",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    return project_root


def get_or_create_project_record(
    project_name: str,
    created_on: date | datetime | str,
    *,
    registry_path: Path | None = None,
) -> ProjectRecord:
    """Compatibility alias for callers that want the stable project metadata."""
    return register_project(project_name, created_on, registry_path=registry_path)


def _split_versioned_stem(stem: str) -> tuple[str, int]:
    match = _VERSIONED_STEM_RE.match(stem)
    if not match:
        return stem, 0
    return match.group("base"), int(match.group("version"))


def next_versioned_child_path(directory: Path, source_name: str) -> Path:
    """Return the next available ``_vN`` filename inside *directory*.

    The version component is based on the source stem when present, so a
    source that already carries ``_v3`` will not be downgraded to ``_v1`` if
    the destination directory is empty.
    """
    directory = Path(directory)
    source = Path(source_name)
    base_stem, source_version = _split_versioned_stem(source.stem)
    suffix = source.suffix

    highest_existing = 0
    pattern = f"{base_stem}_v*{suffix}" if suffix else f"{base_stem}_v*"
    for existing in directory.glob(pattern):
        if not existing.is_file():
            continue
        candidate_base, candidate_version = _split_versioned_stem(existing.stem)
        if candidate_base == base_stem and existing.suffix == suffix:
            highest_existing = max(highest_existing, candidate_version)

    next_version = max(source_version or 1, highest_existing + 1)
    return directory / f"{base_stem}_v{next_version}{suffix}"


def resolve_project_artifact_dir(
    category: str,
    project_name: str,
    created_on: date | datetime | str | None = None,
    *,
    work_root: Path | None = None,
    registry_path: Path | None = None,
) -> tuple[ProjectRecord, Path]:
    """Return the stable project record and canonical publish directory."""
    created = created_on or date.today()
    record = register_project(project_name, created, registry_path=registry_path)
    category_root = Path(work_root or get_hermes_work_dir(category)).expanduser()
    return record, category_root / record.project_id
