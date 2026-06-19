"""
Image Generation Provider ABC
=============================

Defines the pluggable-backend interface for image generation. Providers register
instances via ``PluginContext.register_image_gen_provider()``; the active one
(selected via ``image_gen.provider`` in ``config.yaml``) services every
``image_generate`` tool call.

Providers live in ``<repo>/plugins/image_gen/<name>/`` (built-in, auto-loaded
as ``kind: backend``) or ``~/.hermes/plugins/image_gen/<name>/`` (user, opt-in
via ``plugins.enabled``).

Response shape
--------------
All providers return a dict that :func:`success_response` / :func:`error_response`
produce. The tool wrapper JSON-serializes it. Keys:

    success        bool
    image          str | None       URL or absolute file path
    model          str              provider-specific model identifier
    prompt         str              echoed prompt
    aspect_ratio   str              "landscape" | "square" | "portrait"
    provider       str              provider name (for diagnostics)
    error          str              only when success=False
    error_type     str              only when success=False
"""

from __future__ import annotations

import abc
import base64
import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.project_registry import next_versioned_child_path, resolve_project_artifact_dir
from hermes_constants import get_default_hermes_root, get_hermes_home
from nas_sync_hooks import queue_nas_sync_hook

logger = logging.getLogger(__name__)


VALID_ASPECT_RATIOS: Tuple[str, ...] = ("landscape", "square", "portrait")
DEFAULT_ASPECT_RATIO = "landscape"


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class ImageGenProvider(abc.ABC):
    """Abstract base class for an image generation backend.

    Subclasses must implement :meth:`generate`. Everything else has sane
    defaults — override only what your provider needs.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used in ``image_gen.provider`` config.

        Lowercase, no spaces. Examples: ``fal``, ``openai``, ``replicate``.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``hermes tools``. Defaults to ``name.title()``."""
        return self.name.title()

    def is_available(self) -> bool:
        """Return True when this provider can service calls.

        Typically checks for a required API key. Default: True
        (providers with no external dependencies are always available).
        """
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        """Return catalog entries for ``hermes tools`` model picker.

        Each entry::

            {
                "id": "gpt-image-1.5",               # required
                "display": "GPT Image 1.5",          # optional; defaults to id
                "speed": "~10s",                     # optional
                "strengths": "...",                  # optional
                "price": "$...",                     # optional
            }

        Default: empty list (provider has no user-selectable models).
        """
        return []

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return provider metadata for the ``hermes tools`` picker.

        Used by ``tools_config.py`` to inject this provider as a row in
        the Image Generation provider list. Shape::

            {
                "name": "OpenAI",                     # picker label
                "badge": "paid",                      # optional short tag
                "tag": "One-line description...",     # optional subtitle
                "env_vars": [                         # keys to prompt for
                    {"key": "OPENAI_API_KEY",
                     "prompt": "OpenAI API key",
                     "url": "https://platform.openai.com/api-keys"},
                ],
            }

        Default: minimal entry derived from ``display_name``. Override to
        expose API key prompts and custom badges.
        """
        return {
            "name": self.display_name,
            "badge": "",
            "tag": "",
            "env_vars": [],
        }

    def default_model(self) -> Optional[str]:
        """Return the default model id, or None if not applicable."""
        models = self.list_models()
        if models:
            return models[0].get("id")
        return None

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image.

        Implementations should return the dict from :func:`success_response`
        or :func:`error_response`. ``kwargs`` may contain forward-compat
        parameters future versions of the schema will expose — implementations
        should ignore unknown keys.
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_aspect_ratio(value: Optional[str]) -> str:
    """Clamp an aspect_ratio value to the valid set, defaulting to landscape.

    Invalid values are coerced rather than rejected so the tool surface is
    forgiving of agent mistakes.
    """
    if not isinstance(value, str):
        return DEFAULT_ASPECT_RATIO
    v = value.strip().lower()
    if v in VALID_ASPECT_RATIOS:
        return v
    return DEFAULT_ASPECT_RATIO


def _images_cache_dir() -> Path:
    """Return ``$HERMES_HOME/cache/images/``, creating parents as needed."""
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "cache" / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


_SSD_IMAGE_ROOT = Path("/Volumes/SSD_Hermes/HermesWork/Image")


def _resolve_image_work_root() -> Path:
    """Resolve the image artifact root with environment override first."""
    env_override = os.environ.get("HERMES_WORK_ROOT", "").strip()
    if env_override:
        override_root = Path(env_override).expanduser()
        return override_root if override_root.name == "Image" else override_root / "Image"
    return Path("/Volumes/SSD_Hermes/HermesWork/Image")


def _verify_image_storage_root(published_dir: Path) -> Dict[str, Any]:
    """Return image-root storage diagnostics and enforce SSD anchoring in production."""
    image_root = Path(published_dir).parent
    resolved_image_root = image_root.resolve()
    resolved_ssd_root = _SSD_IMAGE_ROOT.resolve()
    is_ssd_root = (
        resolved_image_root == resolved_ssd_root
        or resolved_image_root.is_relative_to(resolved_ssd_root)
    )

    verification = {
        "logical_path": str(image_root),
        "realpath": str(resolved_image_root),
        "is_symlink": image_root.is_symlink() or image_root.parent.is_symlink(),
        "is_ssd_root": is_ssd_root,
        "expected_ssd_root": str(resolved_ssd_root),
    }

    if not is_ssd_root and "PYTEST_CURRENT_TEST" not in os.environ:
        raise RuntimeError(
            "Image artifact publish requires Image root on SSD-backed HermesWork path "
            f"({resolved_ssd_root})."
        )

    return verification


def _publish_image_artifact(
    source: Path,
    *,
    prefix: str,
    project_name: Optional[str] = None,
    artifact_name: Optional[str] = None,
) -> Path:
    """Copy an image cache file into the standard HermesWork Image tree.

    The cache copy remains the canonical source for internal cleanup, while the
    published copy is what users should keep as their final artifact.
    """
    project_key = (project_name or prefix or "image").strip() or "image"
    artifact_key = (artifact_name or prefix or source.stem or "image").strip() or "image"
    artifact_key = Path(artifact_key).name
    _, published_dir = resolve_project_artifact_dir("Image", project_key, work_root=_resolve_image_work_root())
    published_dir.mkdir(parents=True, exist_ok=True)
    published_path = next_versioned_child_path(published_dir, f"{artifact_key}{source.suffix}")
    shutil.copyfile(source, published_path)
    queue_nas_sync_hook(
        category="image",
        scope=published_path.parent.name,
        artifact_path=published_path,
        source_root=published_dir,
    )
    return published_path


def _normalize_output_signature_path(path_value: object) -> str | None:
    """Return a normalized absolute path used for duplicate detection."""
    if not path_value:
        return None
    try:
        return str(Path(path_value).expanduser().resolve())
    except Exception:
        try:
            return str(Path(path_value).expanduser())
        except Exception:
            return None


def _read_json_if_exists(path: Path) -> dict | None:
    """Read JSON defensively."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_nas_evidence(*, hook_requested: bool, mirror_path: Optional[Path] = None, mirror_sha256: Optional[str] = None) -> dict[str, Any]:
    """Return separated NAS evidence fields for image publish results.

    Hook launch/request, hook log/state, and verified mirror proof are distinct
    evidence layers.  Publishing may request a hook before the async NAS runner
    mirrors files, so mirror verification remains false until a caller proves the
    mirror path/hash separately.
    """
    home = get_hermes_home()
    return {
        "hook_requested": bool(hook_requested),
        "hook_log_path": str(home / "logs" / "nas_sync_hook.log"),
        "hook_state_dir": str(get_default_hermes_root() / "profiles" / "cron-fast" / "state" / "nas-sync"),
        "mirror_verified": bool(mirror_path and mirror_sha256),
        "mirror_path": str(mirror_path) if mirror_path else None,
        "mirror_sha256": str(mirror_sha256) if mirror_sha256 else None,
    }


def _safe_path_component(value: object, fallback: str = "run") -> str:
    raw = str(value or "").strip() or fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return safe[:96] or fallback


def _is_transient_delivery_artifact(*, project_key: str, artifact_key: str, category: str) -> bool:
    """Return True for E2E/smoke/delivery artifacts that must not share a flat publish dir."""
    haystack = " ".join([project_key, artifact_key, category]).casefold()
    tokens = re.split(r"[^a-z0-9]+", haystack)
    token_set = {token for token in tokens if token}
    return (
        "e2e" in token_set
        or "delivery" in token_set
        or "fresh_e2e" in haystack
    )


def _resolve_run_publish_dir(
    base_dir: Path,
    *,
    project_key: str,
    artifact_key: str,
    category: str,
    metadata: Dict[str, Any],
) -> Path:
    """Choose the final publish directory.

    Stable project/qualification bundles keep the historical shared directory.
    Transient E2E/smoke/delivery runs get an isolated child directory so NAS
    hooks sync only that run's canonical primary + sidecars instead of stale
    PNGs accumulated in the project root.
    """
    if not _is_transient_delivery_artifact(project_key=project_key, artifact_key=artifact_key, category=category):
        return base_dir
    run_id = (
        metadata.get("run_id")
        or metadata.get("prompt_id")
        or metadata.get("created_at")
        or uuid.uuid4().hex
    )
    return base_dir / _safe_path_component(run_id)


def _read_png_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Read PNG dimensions without importing image libraries."""
    try:
        with Path(path).open("rb") as fh:
            header = fh.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
            return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
    except Exception:
        return None, None
    return None, None


def _iter_image_bundle_manifests(image_root: Path):
    """Yield tuples of publish manifest + metadata payloads under HermesWork/Image."""
    if not image_root.exists():
        return

    for candidate in sorted(image_root.iterdir()):
        if not candidate.is_dir():
            continue
        manifest_path = candidate / "sidecar" / "manifest.json"
        manifest_payload = _read_json_if_exists(manifest_path)
        if not isinstance(manifest_payload, dict):
            legacy_manifest_path = candidate / "_sidecars" / "manifest.json"
            manifest_payload = _read_json_if_exists(legacy_manifest_path)
            if not isinstance(manifest_payload, dict):
                continue
            manifest_path = legacy_manifest_path

        sidecars = manifest_payload.get("sidecars")
        if not isinstance(sidecars, dict):
            continue

        metadata_name = sidecars.get("metadata")
        if not isinstance(metadata_name, str) or not metadata_name:
            continue

        metadata_payload = _read_json_if_exists(manifest_path.parent / metadata_name)
        if not isinstance(metadata_payload, dict):
            continue

        yield candidate, manifest_payload, metadata_payload


def _find_duplicate_bundle(
    *,
    image_root: Path,
    prompt_id: str,
    source: Path,
    output_filename: str,
) -> tuple[Path, dict, dict] | None:
    """Find an existing bundle that matches the duplicate-protection key."""
    target_prompt_id = str(prompt_id).strip()
    if not target_prompt_id:
        return None

    target_source = _normalize_output_signature_path(source)
    target_output = str(Path(output_filename).name)

    for published_dir, manifest_payload, metadata_payload in _iter_image_bundle_manifests(image_root):
        manifest_prompt_id = str(metadata_payload.get("prompt_id", "")).strip()
        if manifest_prompt_id != target_prompt_id:
            continue

        manifest_source = _normalize_output_signature_path(metadata_payload.get("output_source_path"))
        if target_source and manifest_source and manifest_source != target_source:
            continue

        manifest_output = str(manifest_payload.get("primary_image") or "").strip()
        metadata_output = str(metadata_payload.get("output_filename") or "").strip()

        if metadata_output and metadata_output != target_output:
            continue
        if metadata_output == "":
            source_stem = Path(target_output).stem
            if manifest_output and source_stem and source_stem not in Path(manifest_output).stem:
                # Keep output filename in scope when metadata is from an older shape.
                continue

        primary_image_path = published_dir / manifest_output
        if manifest_output and not primary_image_path.is_file():
            continue

        return published_dir, manifest_payload, metadata_payload

    return None


def _load_existing_bundle(
    published_dir: Path,
    manifest_payload: dict,
    metadata_payload: dict,
) -> dict[str, Any]:
    """Reconstruct the helper return value from an existing publish bundle."""
    sidecars = manifest_payload.get("sidecars")
    if not isinstance(sidecars, dict):
        sidecars = {}

    sidecar_dir = published_dir / "sidecar"
    if not (sidecar_dir / str(sidecars.get("manifest") or "manifest.json")).exists() and (published_dir / "_sidecars").exists():
        sidecar_dir = published_dir / "_sidecars"
    workflow_name = str(sidecars.get("workflow") or "")
    prompt_name = str(sidecars.get("prompt") or "")
    metadata_name = str(sidecars.get("metadata") or "")
    manifest_name = str(sidecars.get("manifest") or "manifest.json")
    integrity_name = str(sidecars.get("integrity") or "integrity.json")

    workflow_path = sidecar_dir / workflow_name
    prompt_path = sidecar_dir / prompt_name
    metadata_path = sidecar_dir / metadata_name
    manifest_path = sidecar_dir / manifest_name
    integrity_path = sidecar_dir / integrity_name

    primary_image = str(manifest_payload.get("primary_image") or "")
    nas_evidence = metadata_payload.get("nas_evidence")
    if not isinstance(nas_evidence, dict):
        nas_evidence = _build_nas_evidence(
            hook_requested=bool(metadata_payload.get("nas_hook_requested", False))
        )

    return {
        "project_id": str(manifest_payload.get("project_id") or published_dir.name),
        "published_dir": published_dir,
        "primary_image_path": published_dir / primary_image,
        "workflow_path": workflow_path,
        "prompt_path": prompt_path,
        "metadata_path": metadata_path,
        "manifest_path": manifest_path,
        "integrity_path": integrity_path,
        "primary_image": primary_image,
        "sidecars": {
            "workflow": workflow_name,
            "prompt": prompt_name,
            "metadata": metadata_name,
            "manifest": manifest_name,
            "integrity": integrity_name,
            "dir": str(sidecar_dir),
        },
        "file_sha256": str((manifest_payload.get("integrity") or {}).get("primary_image_sha256") or metadata_payload.get("file_sha256") or ""),
        "storage_verification": _verify_image_storage_root(published_dir),
        "sidecar_dir": sidecar_dir,
        "nas_hook_requested": bool(metadata_payload.get("nas_hook_requested", False)),
        "nas_evidence": nas_evidence,
    }


def publish_filesystem_image_bundle(
    source: Path,
    *,
    prefix: str,
    project_name: Optional[str],
    artifact_name: Optional[str],
    category: str,
    workflow_json: Dict[str, Any],
    prompt_payload: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Publish a filesystem-backed image plus sidecars into HermesWork/Image."""

    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Source image not found: {source}")

    project_key = (project_name or prefix or "image").strip() or "image"
    artifact_key = (artifact_name or prefix or source.stem or "image").strip() or "image"
    artifact_key = Path(artifact_key).name

    image_root = _resolve_image_work_root()
    duplicate_bundle = _find_duplicate_bundle(
        image_root=image_root,
        prompt_id=str(metadata.get("prompt_id", "") or ""),
        source=source,
        output_filename=source.name,
    )
    if duplicate_bundle is not None:
        duplicate_dir, duplicate_manifest, duplicate_metadata = duplicate_bundle
        return _load_existing_bundle(duplicate_dir, duplicate_manifest, duplicate_metadata)

    project_record, base_published_dir = resolve_project_artifact_dir("Image", project_key, work_root=image_root)
    published_dir = _resolve_run_publish_dir(
        base_published_dir,
        project_key=project_key,
        artifact_key=artifact_key,
        category=category,
        metadata=metadata,
    )
    published_dir.mkdir(parents=True, exist_ok=True)
    storage_verification = _verify_image_storage_root(published_dir)
    primary_image_path = next_versioned_child_path(published_dir, f"{artifact_key}{source.suffix}")
    shutil.copyfile(source, primary_image_path)

    actual_width, actual_height = _read_png_dimensions(primary_image_path)
    versioned_stem = primary_image_path.stem
    sidecar_dir = published_dir / "sidecar"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = sidecar_dir / "workflow.json"
    prompt_path = sidecar_dir / "prompt.json"
    metadata_path = sidecar_dir / "metadata.json"
    manifest_path = sidecar_dir / "manifest.json"
    integrity_path = sidecar_dir / "integrity.json"
    primary_image_sha256 = _sha256_file(primary_image_path)

    metadata_payload = dict(metadata)
    metadata_payload.update(
        {
            "category": category,
            "output_filename": source.name,
            "output_source_path": str(source),
            "published_primary_path": str(primary_image_path),
            "published_dir": str(published_dir),
            "published_sidecar_dir": str(sidecar_dir),
            "workflow_path": str(workflow_path),
            "prompt_path": str(prompt_path),
            "metadata_sidecar_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "integrity_path": str(integrity_path),
            "file_sha256": primary_image_sha256,
            "requested_width": prompt_payload.get("width"),
            "requested_height": prompt_payload.get("height"),
            "actual_width": actual_width,
            "actual_height": actual_height,
            "storage_verification": storage_verification,
        }
    )

    workflow_path.write_text(json.dumps(workflow_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt_path.write_text(json.dumps(prompt_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    nas_hook_requested = queue_nas_sync_hook(
        category="image",
        scope=published_dir.name,
        artifact_path=primary_image_path,
        source_root=published_dir,
    )
    nas_evidence = _build_nas_evidence(hook_requested=nas_hook_requested)
    metadata_payload["nas_hook_requested"] = nas_hook_requested
    metadata_payload["nas_evidence"] = nas_evidence
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_payload = {
        "project_id": project_record.project_id,
        "artifact_name": artifact_key,
        "version": versioned_stem,
        "category": category,
        "primary_image": primary_image_path.name,
        "files": [
            primary_image_path.name,
            f"{sidecar_dir.name}/{workflow_path.name}",
            f"{sidecar_dir.name}/{prompt_path.name}",
            f"{sidecar_dir.name}/{metadata_path.name}",
            f"{sidecar_dir.name}/{manifest_path.name}",
            f"{sidecar_dir.name}/{integrity_path.name}",
        ],
        "sidecars": {
            "workflow": workflow_path.name,
            "prompt": prompt_path.name,
            "metadata": metadata_path.name,
            "manifest": manifest_path.name,
            "integrity": integrity_path.name,
            "dir": str(sidecar_dir),
        },
        "integrity": {
            "primary_image_sha256": primary_image_sha256,
            "algorithm": "sha256",
            "status": "Pass",
        },
        "prompt_id": metadata_payload.get("prompt_id", ""),
        "engine": metadata_payload.get("provider", ""),
        "created_at": metadata_payload.get("created_at", ""),
        "dimensions": {
            "requested_width": prompt_payload.get("width"),
            "requested_height": prompt_payload.get("height"),
            "actual_width": actual_width,
            "actual_height": actual_height,
        },
        "status": {
            "local_status": metadata_payload.get("local_status", "생성 완료"),
            "publish_status": metadata_payload.get("publish_status", "HermesWork publish 완료"),
            "nas_status": "동기화 요청됨" if nas_hook_requested else "동기화 요청 실패",
            "nas_hook_requested": nas_hook_requested,
            "nas_hook_log_path": nas_evidence["hook_log_path"],
            "nas_mirror_verified": nas_evidence["mirror_verified"],
            "slack_status": metadata_payload.get("slack_status", "primary image 준비됨"),
        },
        "nas_evidence": nas_evidence,
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    integrity_payload = {
        "artifact_name": artifact_key,
        "primary_image": primary_image_path.name,
        "primary_image_sha256": primary_image_sha256,
        "files": {
            primary_image_path.name: {"sha256": primary_image_sha256, "bytes": primary_image_path.stat().st_size},
        },
        "manifest_path": str(manifest_path),
        "status": "Pass",
    }
    integrity_path.write_text(json.dumps(integrity_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "project_id": project_record.project_id,
        "published_dir": published_dir,
        "primary_image_path": primary_image_path,
        "workflow_path": workflow_path,
        "prompt_path": prompt_path,
        "metadata_path": metadata_path,
        "manifest_path": manifest_path,
        "integrity_path": integrity_path,
        "primary_image": primary_image_path.name,
        "sidecars": manifest_payload["sidecars"],
        "file_sha256": primary_image_sha256,
        "storage_verification": storage_verification,
        "sidecar_dir": sidecar_dir,
        "nas_hook_requested": nas_hook_requested,
        "nas_evidence": nas_evidence,
    }


def record_slack_upload_evidence(
    source_path: Path | str,
    *,
    message_id: str | None,
    thread_ts: str | None,
    files_count: int,
    raw_response: Any = None,
) -> Path | None:
    """Persist Slack upload evidence beside an image bundle when sidecars exist."""
    source_path = Path(source_path)
    published_dir = source_path.parent
    sidecar_dir = published_dir / "sidecar"
    manifest_path = sidecar_dir / "manifest.json"
    integrity_path = sidecar_dir / "integrity.json"
    metadata_path = sidecar_dir / "metadata.json"
    if not manifest_path.exists() or not integrity_path.exists():
        return None

    local_sha256 = _sha256_file(source_path)
    evidence = {
        "message_id": str(message_id or thread_ts or ""),
        "thread_ts": str(thread_ts or ""),
        "filename": source_path.name,
        "source_path": str(source_path),
        "local_sha256": local_sha256,
        "files_count": int(files_count),
    }
    if raw_response is not None:
        try:
            evidence["raw_response_keys"] = sorted(list(raw_response.keys())) if isinstance(raw_response, dict) else []
        except Exception:
            evidence["raw_response_keys"] = []

    evidence_path = sidecar_dir / "slack_evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for path in (manifest_path, integrity_path, metadata_path):
        if not path.exists():
            continue
        payload = _load_json_file(path)
        payload["slack_upload_evidence"] = evidence
        if path == manifest_path:
            sidecars = dict(payload.get("sidecars") or {})
            sidecars["slack_evidence"] = evidence_path.name
            payload["sidecars"] = sidecars
            files = list(payload.get("files") or [])
            slack_rel = f"{sidecar_dir.name}/{evidence_path.name}"
            if slack_rel not in files:
                files.append(slack_rel)
            payload["files"] = files
            status = dict(payload.get("status") or {})
            status["slack_status"] = "Pass" if evidence.get("message_id") and files_count > 0 else "Pending"
            payload["status"] = status
        elif path == integrity_path:
            files = dict(payload.get("files") or {})
            files[evidence_path.name] = {"sha256": _sha256_file(evidence_path), "bytes": evidence_path.stat().st_size}
            payload["files"] = files
        _write_json_file(path, payload)
    return evidence_path


def write_run_manifest(
    published_dir: Path,
    *,
    workflow_code: str,
    workflow_name: str,
    run_kind: str,
    project_name: str,
    project_id: str,
    artifacts: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Path:
    """Write a run-level manifest without changing the per-artifact manifest contract."""

    published_dir = Path(published_dir)
    published_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = published_dir / "run_manifest.json"
    payload = {
        "manifest_type": "run_summary",
        "workflow_code": workflow_code,
        "workflow_name": workflow_name,
        "run_kind": run_kind,
        "project_name": project_name,
        "project_id": project_id,
        "artifacts": list(artifacts),
        "summary": dict(summary),
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def update_run_manifest(
    published_dir: Path,
    *,
    workflow_code: str,
    workflow_name: str,
    run_kind: str,
    project_name: str,
    project_id: str,
    artifact: Dict[str, Any],
) -> Path:
    """Append or replace a single artifact entry inside ``run_manifest.json``."""

    published_dir = Path(published_dir)
    manifest_path = published_dir / "run_manifest.json"
    existing_artifacts: List[Dict[str, Any]] = []
    existing_summary: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload.get("artifacts"), list):
                existing_artifacts = [entry for entry in payload["artifacts"] if isinstance(entry, dict)]
            if isinstance(payload.get("summary"), dict):
                existing_summary = dict(payload["summary"])
        except Exception:
            existing_artifacts = []
            existing_summary = {}

    artifact_name = str(artifact.get("artifact_name") or "").strip()
    updated = False
    for idx, entry in enumerate(existing_artifacts):
        if str(entry.get("artifact_name") or "").strip() == artifact_name and artifact_name:
            existing_artifacts[idx] = dict(artifact)
            updated = True
            break
    if not updated:
        existing_artifacts.append(dict(artifact))

    summary = dict(existing_summary)
    summary["artifact_count"] = len(existing_artifacts)
    summary.setdefault("total_runs", len(existing_artifacts))
    return write_run_manifest(
        published_dir,
        workflow_code=workflow_code,
        workflow_name=workflow_name,
        run_kind=run_kind,
        project_name=project_name,
        project_id=project_id,
        artifacts=existing_artifacts,
        summary=summary,
    )


def write_qualification_report(published_dir: Path, report_payload: Dict[str, Any]) -> Path:
    """Persist qualification/QC results beside the grouped publish artifacts."""

    published_dir = Path(published_dir)
    published_dir.mkdir(parents=True, exist_ok=True)
    report_path = published_dir / "qualification_report.json"
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def _load_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_file(path: Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_delivery_status(value: Any) -> str:
    normalized = str(value or "Pending").strip().lower()
    mapping = {
        "pass": "Pass",
        "success": "Pass",
        "ok": "Pass",
        "fail": "Fail",
        "failed": "Fail",
        "error": "Fail",
        "pending": "Pending",
        "skip": "Skipped",
        "skipped": "Skipped",
    }
    return mapping.get(normalized, "Pending")


def finalize_run_manifest_status(published_dir: Path) -> Path:
    """Recompute run-level summary fields from the artifact list."""

    manifest_path = Path(published_dir) / "run_manifest.json"
    payload = _load_json_file(manifest_path)
    artifacts = [entry for entry in payload.get("artifacts", []) if isinstance(entry, dict)]
    existing_summary = dict(payload.get("summary") or {})

    artifact_count = len(artifacts)
    completed_run_count = sum(1 for entry in artifacts if str((entry.get("status") or {}).get("technical_result") or "").strip() == "Pass")
    failed_run_count = sum(1 for entry in artifacts if str((entry.get("status") or {}).get("technical_result") or "").strip() == "Fail")
    planned_run_count = existing_summary.get("planned_run_count")
    total_runs = int(planned_run_count) if isinstance(planned_run_count, int) and planned_run_count > 0 else artifact_count

    existing_summary.update(
        {
            "artifact_count": artifact_count,
            "completed_run_count": completed_run_count,
            "failed_run_count": failed_run_count,
            "total_runs": total_runs,
        }
    )
    if isinstance(planned_run_count, int) and planned_run_count > 0:
        existing_summary["planned_run_count"] = planned_run_count

    payload["summary"] = existing_summary
    return _write_json_file(manifest_path, payload)


def update_run_delivery_status(
    published_dir: Path,
    *,
    delivery_result: Dict[str, Any],
    delivery_note: Optional[str] = None,
) -> Path:
    """Update artifact-level delivery statuses in ``run_manifest.json``."""

    manifest_path = Path(published_dir) / "run_manifest.json"
    payload = _load_json_file(manifest_path)
    artifacts = [entry for entry in payload.get("artifacts", []) if isinstance(entry, dict)]

    for entry in artifacts:
        run_id = str(entry.get("run_id") or "").strip()
        artifact_name = str(entry.get("artifact_name") or "").strip()
        if run_id in delivery_result:
            status_value = delivery_result[run_id]
        elif artifact_name in delivery_result:
            status_value = delivery_result[artifact_name]
        else:
            continue
        status = dict(entry.get("status") or {})
        status["slack_status"] = _normalize_delivery_status(status_value)
        if delivery_note:
            status["delivery_note"] = delivery_note
        entry["status"] = status

    payload["artifacts"] = artifacts
    _write_json_file(manifest_path, payload)
    return finalize_run_manifest_status(published_dir)


def finalize_qualification_report_status(
    published_dir: Path,
    *,
    delivery_result: Optional[Dict[str, Any]] = None,
) -> Path:
    """Recompute delivery-related counters in ``qualification_report.json``."""

    published_dir = Path(published_dir)
    manifest_path = published_dir / "run_manifest.json"
    report_path = published_dir / "qualification_report.json"

    if delivery_result:
        update_run_delivery_status(published_dir, delivery_result=delivery_result)

    manifest_payload = _load_json_file(manifest_path)
    report_payload = _load_json_file(report_path)

    artifact_map: Dict[str, Dict[str, Any]] = {}
    for entry in manifest_payload.get("artifacts", []):
        if not isinstance(entry, dict):
            continue
        run_id = str(entry.get("run_id") or "").strip()
        artifact_name = str(entry.get("artifact_name") or "").strip()
        if run_id:
            artifact_map[run_id] = entry
        if artifact_name:
            artifact_map[artifact_name] = entry

    success_count = 0
    fail_count = 0
    pending_count = 0
    skipped_count = 0

    for run in report_payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or "").strip()
        linked = artifact_map.get(run_id)
        slack_status = "Pending"
        if isinstance(linked, dict):
            slack_status = _normalize_delivery_status((linked.get("status") or {}).get("slack_status"))
        run["slack_status"] = slack_status
        if slack_status == "Pass":
            success_count += 1
        elif slack_status == "Fail":
            fail_count += 1
        elif slack_status == "Skipped":
            skipped_count += 1
        else:
            pending_count += 1

    summary = dict(report_payload.get("summary") or {})
    summary["slack_success_count"] = success_count
    summary["slack_fail_count"] = fail_count
    summary["slack_pending_count"] = pending_count
    summary["slack_skipped_count"] = skipped_count
    report_payload["summary"] = summary

    return _write_json_file(report_path, report_payload)


def wait_for_file_stable(path: Path, *, checks: int = 2, delay_seconds: float = 0.5) -> bool:
    """Return True when the file exists and its size stays stable across checks."""

    candidate = Path(path)
    if not candidate.is_file():
        return False

    previous_size: Optional[int] = None
    stable_count = 0
    attempts = max(checks, 1) + 1
    for _ in range(attempts):
        current_size = candidate.stat().st_size
        if previous_size is not None and current_size == previous_size:
            stable_count += 1
            if stable_count >= max(checks - 1, 1):
                return True
        else:
            stable_count = 0
        previous_size = current_size
        time.sleep(delay_seconds)
    return False


def save_b64_image(
    b64_data: str,
    *,
    prefix: str = "image",
    extension: str = "png",
    project_name: Optional[str] = None,
    artifact_name: Optional[str] = None,
) -> Path:
    """Decode base64 image data, cache it, and publish it to HermesWork.

    The image is always written under ``$HERMES_HOME/cache/images/`` first so
    internal cleanup stays unchanged. A second copy is then published under
    ``HermesWork/Image/<project>/`` for the final user-visible artifact.

    Returns the published artifact path when the publish step succeeds, or the
    cache path as a safe fallback if publication fails.
    """
    raw = base64.b64decode(b64_data)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    path = _images_cache_dir() / f"{prefix}_{ts}_{short}.{extension}"
    path.write_bytes(raw)
    try:
        return _publish_image_artifact(
            path,
            prefix=prefix,
            project_name=project_name,
            artifact_name=artifact_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not publish image artifact to HermesWork: %s", exc)
        return path


# Extension inference for save_url_image — keep small and explicit.  We don't
# want to import mimetypes for a handful of formats every image_gen provider
# actually returns, and we never want to inherit a content-type that points
# at HTML or JSON when the API gives us a degenerate response.
_URL_IMAGE_CONTENT_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


def save_url_image(
    url: str,
    *,
    prefix: str = "image",
    timeout: float = 60.0,
    max_bytes: int = 25 * 1024 * 1024,
    project_name: Optional[str] = None,
    artifact_name: Optional[str] = None,
) -> Path:
    """Download an image URL and write it under ``$HERMES_HOME/cache/images/``.

    Used by providers (xAI, fallback OpenAI) whose API returns an *ephemeral*
    URL instead of inline base64 — those URLs frequently expire before a
    downstream consumer (Telegram ``send_photo``, browser fetch) can resolve
    them, so we materialise the bytes locally at tool-completion time.
    Mirrors :func:`save_b64_image`'s shape so providers can swap in one line.

    Returns the absolute :class:`Path` to the saved file.  Raises on any
    network / HTTP / oversize / non-image-content-type error so callers can
    fall back to returning the bare URL with a clear error message.
    """
    import requests

    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    # Infer extension from the response content-type, falling back to the
    # URL suffix when xAI / OpenAI omit a precise type (some CDNs return
    # ``application/octet-stream``).  Defaults to ``png``.
    content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
    extension = _URL_IMAGE_CONTENT_TYPES.get(content_type)
    if extension is None:
        url_path = url.split("?", 1)[0].lower()
        for ext in ("png", "jpg", "jpeg", "webp", "gif"):
            if url_path.endswith(f".{ext}"):
                extension = "jpg" if ext == "jpeg" else ext
                break
    if extension is None:
        extension = "png"

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    path = _images_cache_dir() / f"{prefix}_{ts}_{short}.{extension}"

    bytes_written = 0
    with path.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                fh.close()
                try:
                    path.unlink()
                except OSError:
                    pass
                raise ValueError(
                    f"Image at {url} exceeds {max_bytes // (1024 * 1024)}MB cap; refusing to cache."
                )
            fh.write(chunk)

    if bytes_written == 0:
        try:
            path.unlink()
        except OSError:
            pass
        raise ValueError(f"Image at {url} returned 0 bytes; refusing to cache.")

    try:
        return _publish_image_artifact(
            path,
            prefix=prefix,
            project_name=project_name,
            artifact_name=artifact_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not publish image artifact to HermesWork: %s", exc)
        return path


def _format_image_completion_message(local_path: Optional[str]) -> str:
    """Format the human-readable completion message for image generation."""
    local_display = local_path or "없음"
    return "\n".join([
        f"로컬 저장: {local_display}",
        "NAS 반영: 동기화 요청됨",
        "Slack 첨부: 완료",
    ])


def success_response(
    *,
    image: str,
    model: str,
    prompt: str,
    aspect_ratio: str,
    provider: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a uniform success response dict.

    ``image`` may be an HTTP URL or an absolute filesystem path (for b64
    providers like OpenAI). Callers that need to pass through additional
    backend-specific fields can supply ``extra``.
    """
    local_path = image if isinstance(image, str) and Path(image).is_absolute() else ""
    payload: Dict[str, Any] = {
        "success": True,
        "image": image,
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "provider": provider,
        "local_path": local_path,
        "nas_status": "동기화 요청됨",
        "slack_status": "완료",
        "message": _format_image_completion_message(local_path),
    }
    if extra:
        payload.update(extra)
    return payload


def error_response(
    *,
    error: str,
    error_type: str = "provider_error",
    provider: str = "",
    model: str = "",
    prompt: str = "",
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
) -> Dict[str, Any]:
    """Build a uniform error response dict."""
    return {
        "success": False,
        "image": None,
        "error": error,
        "error_type": error_type,
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "provider": provider,
    }
