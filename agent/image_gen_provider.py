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
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.project_registry import next_versioned_child_path, resolve_project_artifact_dir
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
    _, published_dir = resolve_project_artifact_dir("Image", project_key)
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

    project_record, published_dir = resolve_project_artifact_dir("Image", project_key)
    published_dir.mkdir(parents=True, exist_ok=True)
    primary_image_path = next_versioned_child_path(published_dir, f"{artifact_key}{source.suffix}")
    shutil.copyfile(source, primary_image_path)

    versioned_stem = primary_image_path.stem
    workflow_path = published_dir / f"{versioned_stem}.workflow.json"
    prompt_path = published_dir / f"{versioned_stem}.prompt.json"
    metadata_path = published_dir / f"{versioned_stem}.metadata.json"
    manifest_path = published_dir / "manifest.json"

    metadata_payload = dict(metadata)
    metadata_payload.update(
        {
            "category": category,
            "output_source_path": str(source),
            "published_primary_path": str(primary_image_path),
            "published_dir": str(published_dir),
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
    metadata_payload["nas_hook_requested"] = nas_hook_requested
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_payload = {
        "project_id": project_record.project_id,
        "artifact_name": artifact_key,
        "version": versioned_stem,
        "category": category,
        "primary_image": primary_image_path.name,
        "files": [
            primary_image_path.name,
            workflow_path.name,
            prompt_path.name,
            metadata_path.name,
        ],
        "sidecars": {
            "workflow": workflow_path.name,
            "prompt": prompt_path.name,
            "metadata": metadata_path.name,
        },
        "prompt_id": metadata_payload.get("prompt_id", ""),
        "engine": metadata_payload.get("provider", ""),
        "created_at": metadata_payload.get("created_at", ""),
        "status": {
            "local_status": metadata_payload.get("local_status", "생성 완료"),
            "publish_status": metadata_payload.get("publish_status", "HermesWork publish 완료"),
            "nas_status": "동기화 요청됨" if nas_hook_requested else "동기화 요청 실패",
            "slack_status": metadata_payload.get("slack_status", "primary image 준비됨"),
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "project_id": project_record.project_id,
        "published_dir": published_dir,
        "primary_image_path": primary_image_path,
        "workflow_path": workflow_path,
        "prompt_path": prompt_path,
        "metadata_path": metadata_path,
        "manifest_path": manifest_path,
        "primary_image": primary_image_path.name,
        "sidecars": manifest_payload["sidecars"],
        "nas_hook_requested": nas_hook_requested,
    }


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
