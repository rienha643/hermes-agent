"""NovelAI image publish backend.

This module intentionally separates the NovelAI generation result directory from
Hermes delivery artifacts. Raw generation output may live under the active
profile's ``generated/`` tree, but only a published copy under
``HermesWork/Image/NAI/<run_id>/`` is eligible for Slack/media delivery and NAS
sync hooks.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import struct
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)
from nas_sync_hooks import queue_nas_sync_hook

DEFAULT_MODEL = "nai-diffusion-4-5-curated"
NOVELAI_GENERATE_ENDPOINT = "https://image.novelai.net/ai/generate-image"
NOVELAI_GENERATE_STREAM_ENDPOINT = "https://image.novelai.net/ai/generate-image-stream"
NOVELAI_REQUEST_TIMEOUT_SECONDS = 180

# NovelAI Generation Policy V1 / Hermes Image Generation Standard V1.
NAI_ADD_QUALITY_TAGS = False
NAI_UNDESIRED_CONTENT_PRESET = "none"
NAI_UC_PRESET = 0
NAI_SAMPLER_LABEL = "DPM++ SDE"
NAI_SAMPLER = "k_dpmpp_sde"
NAI_SMEA = True
NAI_DYN = True
SAFE_1024_RANGE_MAX_PIXELS = 1024 * 1024
HIGH_RES_REQUIRES_APPROVAL = "HIGH_RES_REQUIRES_APPROVAL"
LIVE_GENERATION_REQUIRES_APPROVAL = "LIVE_GENERATION_REQUIRES_APPROVAL"
NAI_DEFAULT_WIDTH = 1024
NAI_DEFAULT_HEIGHT = 1024
NAI_DEFAULT_POSITIVE_PROMPT_PREFIX = """best quality,
high quality,
subculture illustration,
anime illustration,
clean lineart,
detailed face,
expressive eyes"""
NAI_DEFAULT_NEGATIVE_PROMPT = """normal quality,
bad quality,
low quality,
worst quality,
lowres,
bad anatomy,
bad hands,
malformed hands,
malformed fingers,
missing fingers,
extra digits,
fewer digits,
watermark,
signature,
username,
text,
blurry,
duplicate,
mutation,
deformed,
disfigured,
extra arms,
extra legs,
bad feet,
bad proportions,
JPEG artifacts,
chromatic aberration,
scan artifacts"""

_SIDECAR_FILES = (
    "request.json",
    "response.json",
    "metadata.json",
    "manifest.json",
    "integrity.json",
    "account_refresh.json",
)
_RAW_RESPONSE_FILES = ("response.bin", "response.zip")


def _resolve_image_work_root() -> Path:
    """Return the Hermes image publish root for NovelAI artifacts."""
    override = os.environ.get("HERMES_WORK_ROOT", "").strip()
    if override:
        root = Path(override).expanduser()
        return root if root.name == "Image" else root / "Image"
    # Use the Slack allow-listed operator publish root, not the active
    # profile's HOME (gateway sessions may set HOME to
    # ~/.hermes/profiles/<name>/home).
    return Path("/Users/hermes/HermesWork/Image")


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


class NovelAIResolutionApprovalRequired(ValueError):
    """Raised when a NovelAI request exceeds SAFE_1024_RANGE without approval."""

    def __init__(self, *, width: int, height: int, reason: str = "resolution") -> None:
        super().__init__(
            f"{HIGH_RES_REQUIRES_APPROVAL}: {reason} {width}x{height} exceeds SAFE_1024_RANGE; "
            "pass high_res_approved=True to build a high-resolution payload."
        )
        self.width = width
        self.height = height
        self.reason = reason
        self.policy = HIGH_RES_REQUIRES_APPROVAL


def _coerce_dimension(value: Any, *, name: str) -> int:
    try:
        dimension = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if dimension <= 0:
        raise ValueError(f"{name} must be positive")
    return dimension


def is_safe_1024_range(width: int, height: int) -> bool:
    """Return True when a NovelAI request is within SAFE_1024_RANGE."""
    return _coerce_dimension(width, name="width") * _coerce_dimension(height, name="height") <= SAFE_1024_RANGE_MAX_PIXELS


def require_safe_1024_range(
    width: int,
    height: int,
    *,
    high_res_approved: bool = False,
    reason: str = "resolution",
) -> None:
    """Enforce the NAI high-resolution approval gate."""
    width = _coerce_dimension(width, name="width")
    height = _coerce_dimension(height, name="height")
    if not high_res_approved and not is_safe_1024_range(width, height):
        raise NovelAIResolutionApprovalRequired(width=width, height=height, reason=reason)


def _merge_prompt_baseline(baseline: str, prompt: str | None, *, baseline_first: bool) -> str:
    """Merge comma/newline prompt terms while preserving the policy baseline."""
    raw_items: list[str] = []
    for text in (baseline, prompt or "") if baseline_first else (prompt or "", baseline):
        for item in str(text or "").replace("\n", ",").split(","):
            cleaned = item.strip()
            if cleaned:
                raw_items.append(cleaned)

    merged: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return ",\n".join(merged)


def _merge_positive_prompt(prompt: str) -> str:
    return _merge_prompt_baseline(NAI_DEFAULT_POSITIVE_PROMPT_PREFIX, prompt, baseline_first=True)


def _merge_negative_prompt(negative_prompt: str | None) -> str:
    return _merge_prompt_baseline(NAI_DEFAULT_NEGATIVE_PROMPT, negative_prompt, baseline_first=False)


def build_novelai_request_payload(
    *,
    prompt: str,
    width: int = NAI_DEFAULT_WIDTH,
    height: int = NAI_DEFAULT_HEIGHT,
    model: str = DEFAULT_MODEL,
    negative_prompt: str | None = None,
    high_res_approved: bool = False,
    upscale: bool = False,
    high_resolution: bool = False,
    action: str = "generate",
    seed: int | None = None,
    steps: int = 28,
    scale: float = 5.0,
    n_samples: int = 1,
    **parameter_overrides: Any,
) -> Dict[str, Any]:
    """Build a NovelAI request payload without performing any API call.

    This dry-run builder is the adapter source of truth for NovelAI policy
    defaults. It intentionally does not contact NovelAI; callers use it for
    validation, tests, and future live-generation wiring.
    """
    width = _coerce_dimension(width, name="width")
    height = _coerce_dimension(height, name="height")
    approval_reason = "upscale" if upscale else "high_resolution" if high_resolution else "resolution"
    if upscale or high_resolution:
        if not high_res_approved:
            raise NovelAIResolutionApprovalRequired(width=width, height=height, reason=approval_reason)
    require_safe_1024_range(width, height, high_res_approved=high_res_approved, reason=approval_reason)
    merged_prompt = _merge_positive_prompt(prompt)
    merged_negative_prompt = _merge_negative_prompt(negative_prompt)

    parameters: Dict[str, Any] = {
        "params_version": 3,
        "width": width,
        "height": height,
        "scale": scale,
        "sampler": NAI_SAMPLER,
        "sampler_label": NAI_SAMPLER_LABEL,
        "steps": steps,
        "n_samples": n_samples,
        "ucPreset": NAI_UC_PRESET,
        "undesired_content_preset": NAI_UNDESIRED_CONTENT_PRESET,
        "qualityToggle": NAI_ADD_QUALITY_TAGS,
        "add_quality_tags": NAI_ADD_QUALITY_TAGS,
        "sm": NAI_SMEA,
        "sm_dyn": NAI_DYN,
        "autoSmea": NAI_SMEA,
        "dynamic_thresholding": NAI_DYN,
        "noise_schedule": "karras",
        "cfg_rescale": 0,
        "skip_cfg_above_sigma": None,
        "use_coords": False,
        "legacy_uc": False,
        "normalize_reference_strength_multiple": True,
        "deliberate_euler_ancestral_bug": False,
        "prefer_brownian": False,
        "negative_prompt": merged_negative_prompt,
        "uc": merged_negative_prompt,
    }
    if seed is not None:
        parameters["seed"] = int(seed)
    parameters.update(parameter_overrides)

    if str(model).startswith("nai-diffusion-4"):
        parameters.setdefault(
            "v4_prompt",
            {
                "caption": {"base_caption": merged_prompt, "char_captions": []},
                "use_coords": False,
                "use_order": True,
            },
        )
        parameters.setdefault(
            "v4_negative_prompt",
            {
                "caption": {
                    "base_caption": merged_negative_prompt,
                    "char_captions": [],
                },
                "legacy_uc": False,
            },
        )

    return {
        "input": merged_prompt,
        "model": model,
        "action": action,
        "parameters": parameters,
        "policy": {
            "safe_range": "SAFE_1024_RANGE",
            "max_pixels": SAFE_1024_RANGE_MAX_PIXELS,
            "high_res_policy": HIGH_RES_REQUIRES_APPROVAL,
            "high_res_approved": high_res_approved,
        },
    }


@dataclass(frozen=True)
class NovelAIHTTPResponse:
    """Raw NovelAI HTTP response envelope used by the live-generation path."""

    status: int
    headers: Dict[str, str]
    body: bytes


def _novelai_api_key() -> str:
    for name in ("NOVELAI_API_KEY", "NAI_API_KEY", "NOVEL_AI_API_KEY", "NOVELAI_TOKEN", "NAI_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError("NOVELAI_API_KEY is not configured for NovelAI image generation")


def _post_novelai_generation(
    payload: Dict[str, Any],
    *,
    api_key: str,
    endpoint: str = NOVELAI_GENERATE_ENDPOINT,
    timeout: int = NOVELAI_REQUEST_TIMEOUT_SECONDS,
) -> NovelAIHTTPResponse:
    """POST a NovelAI generation request and return the raw response.

    Tests monkeypatch this function; production calls should only happen via
    explicit `image_generate` invocation from a NovelAI-enabled profile.
    """
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/zip,image/png,application/octet-stream",
            "User-Agent": "Mozilla/5.0 (Hermes Agent; NovelAI image generation)",
            "Origin": "https://novelai.net",
            "Referer": "https://novelai.net/",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-configured endpoint.
            return NovelAIHTTPResponse(
                status=int(getattr(response, "status", response.getcode())),
                headers={str(k).lower(): str(v) for k, v in response.headers.items()},
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(f"NovelAI generation HTTP {exc.code}: {body[:512]!r}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"NovelAI generation request failed: {exc.reason}") from exc


def _default_dimensions_for_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
    resolved = resolve_aspect_ratio(aspect_ratio)
    if resolved == "portrait":
        return 832, 1216
    if resolved == "landscape":
        return 1216, 832
    return NAI_DEFAULT_WIDTH, NAI_DEFAULT_HEIGHT


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _response_headers_dict(headers: Dict[str, str]) -> Dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (headers or {}).items()}


def _normalize_novelai_response_png(body: bytes, headers: Dict[str, str]) -> tuple[bytes, str, str]:
    """Return `(png_bytes, response_shape, image_name)` from NovelAI response bytes."""
    if not body:
        raise ValueError("NovelAI response body is empty")
    header_map = _response_headers_dict(headers)
    content_type = header_map.get("content-type", "").split(";", 1)[0].strip().lower()
    if body.startswith(b"\x89PNG\r\n\x1a\n") or content_type == "image/png":
        if not body.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("NovelAI response declared image/png but did not contain a PNG signature")
        return body, "png", "image_0.png"
    stream = io.BytesIO(body)
    if zipfile.is_zipfile(stream):
        stream.seek(0)
        with zipfile.ZipFile(stream) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".png"):
                    png = zf.read(name)
                    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise ValueError(f"NovelAI ZIP member is not a PNG: {name}")
                    return png, "zip", name
        raise ValueError("NovelAI ZIP response did not contain a PNG member")
    msgpack_png = _extract_final_png_from_msgpack_stream(body)
    if msgpack_png is not None:
        return msgpack_png, "msgpack", "final.png"
    sse_png = _extract_final_png_from_sse(body)
    if sse_png is not None:
        return sse_png, "sse", "final.png"
    raise ValueError(f"Unsupported NovelAI response content type: {content_type or 'unknown'}")


def _extract_final_png_from_sse(body: bytes) -> bytes | None:
    """Extract final PNG or surface stream errors from NovelAI text/event-stream bytes."""
    import base64

    text = body.decode("utf-8", errors="replace")
    final_png: bytes | None = None
    errors: list[str] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line.split(":", 1)[1].strip()
        if not data:
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            try:
                raw = base64.b64decode(data, validate=True)
            except Exception:
                continue
            if raw.startswith(b"\x89PNG\r\n\x1a\n"):
                final_png = raw
            else:
                msgpack_png = _extract_final_png_from_msgpack_stream(raw)
                if msgpack_png is not None:
                    final_png = msgpack_png
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("event_type") == "error":
            errors.append(str(obj.get("message") or obj.get("code") or "unknown stream error"))
            continue
        image_data = obj.get("image")
        if obj.get("event_type") == "final" and isinstance(image_data, str):
            raw = base64.b64decode(image_data)
            if raw.startswith(b"\x89PNG\r\n\x1a\n"):
                final_png = raw
    if final_png is None and errors:
        raise ValueError("NovelAI stream error: " + "; ".join(errors))
    return final_png


def _extract_final_png_from_msgpack_stream(body: bytes) -> bytes | None:
    """Extract the final PNG from NovelAI V4 length-prefixed msgpack stream bytes."""
    try:
        import msgpack  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency is installed in Hermes runtime
        raise ValueError("msgpack is required to parse NovelAI V4 responses") from exc

    final_png: bytes | None = None
    offset = 0
    while offset + 4 <= len(body):
        try:
            message_length = struct.unpack(">I", body[offset : offset + 4])[0]
        except struct.error:
            break
        msg_start = offset + 4
        msg_end = msg_start + message_length
        if message_length <= 0 or msg_start >= len(body) or msg_end > len(body):
            offset += 1
            continue
        try:
            obj = msgpack.unpackb(body[msg_start:msg_end], raw=False)
        except Exception:
            offset += 1
            continue
        if isinstance(obj, dict):
            event_type = obj.get("event_type")
            if event_type == "error":
                raise ValueError(f"NovelAI stream error: {obj.get('message') or obj.get('code')}")
            image_data = obj.get("image")
            if event_type == "final" and isinstance(image_data, (bytes, bytearray)):
                png = bytes(image_data)
                if png.startswith(b"\x89PNG\r\n\x1a\n"):
                    final_png = png
        offset = msg_end
    return final_png


def _write_live_generation_source_bundle(
    *,
    run_id: str,
    prompt: str,
    payload: Dict[str, Any],
    http_response: NovelAIHTTPResponse,
    png_bytes: bytes,
    response_shape: str,
    response_image_name: str,
    save_raw_response: bool,
) -> tuple[Path, Path, Dict[str, Any]]:
    from hermes_constants import get_hermes_home

    source_dir = get_hermes_home() / "generated" / "novelai" / run_id
    source_dir.mkdir(parents=True, exist_ok=True)
    source_png = source_dir / "image_000.png"
    source_png.write_bytes(png_bytes)
    _assert_png(source_png)

    sidecar = source_dir / "sidecar"
    sidecar.mkdir(parents=True, exist_ok=True)
    raw_sha = hashlib.sha256(http_response.body).hexdigest()
    png_sha = _sha256_path(source_png)
    raw_metadata = {
        "status": http_response.status,
        "response_content_type": _response_headers_dict(http_response.headers).get("content-type", "unknown"),
        "response_shape": response_shape,
        "response_image_name": response_image_name,
        "response_bytes": len(http_response.body),
        "response_sha256": raw_sha,
        "raw_response_saved": save_raw_response,
        "normalized_artifacts": {
            "count": 1,
            "files": [str(source_png)],
            "sha256": png_sha,
        },
    }

    (sidecar / "request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (sidecar / "response.json").write_text(json.dumps(raw_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (sidecar / "metadata.json").write_text(
        json.dumps(
            {
                "provider": "novelai",
                "model": payload.get("model", DEFAULT_MODEL),
                "prompt": prompt,
                "run_id": run_id,
                "width": payload.get("parameters", {}).get("width"),
                "height": payload.get("parameters", {}).get("height"),
                "raw_response_saved": save_raw_response,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (sidecar / "integrity.json").write_text(
        json.dumps({"sha256": png_sha, "files": {"png": str(source_png)}}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (sidecar / "manifest.json").write_text(
        json.dumps(
            {
                "files": {"png": str(source_png)},
                "checks": {"generation": "PASS", "png": "PASS", "sidecar": "PASS", "hash": "PASS"},
                "raw_response_saved": save_raw_response,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if save_raw_response:
        raw_name = "response.zip" if response_shape == "zip" else "response.bin"
        (sidecar / raw_name).write_bytes(http_response.body)
    return source_png, sidecar, raw_metadata


def _write_live_response_metadata(published_sidecar_dir: Path, raw_metadata: Dict[str, Any], published_png: Path) -> None:
    response_path = published_sidecar_dir / "response.json"
    response = _load_json(response_path)
    response.update(raw_metadata)
    response["normalized_artifacts"] = {
        "count": 1,
        "files": [str(published_png)],
        "sha256": _sha256_path(published_png),
    }
    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path = published_sidecar_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest["raw_response"] = response
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        artifacts[0]["sha256"] = _sha256_path(published_png)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    integrity_path = published_sidecar_dir / "integrity.json"
    integrity = _load_json(integrity_path)
    integrity["sha256"] = _sha256_path(published_png)
    integrity["files"] = {"png": str(published_png)}
    integrity_path.write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _generate_live_novelai_image(
    *,
    prompt: str,
    aspect_ratio: str,
    model: str,
    run_id: str,
    width: int,
    height: int,
    negative_prompt: str | None,
    high_res_approved: bool,
    upscale: bool,
    high_resolution: bool,
    save_raw_response: bool,
    endpoint: str,
    timeout: int,
    parameter_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    payload = build_novelai_request_payload(
        prompt=prompt,
        width=width,
        height=height,
        model=model,
        negative_prompt=negative_prompt,
        high_res_approved=high_res_approved,
        upscale=upscale,
        high_resolution=high_resolution,
        **parameter_overrides,
    )
    if str(model).startswith("nai-diffusion-4"):
        payload = json.loads(json.dumps(payload))
        payload.get("parameters", {}).pop("sm", None)
        payload.get("parameters", {}).pop("sm_dyn", None)
    http_response = _post_novelai_generation(
        payload,
        api_key=_novelai_api_key(),
        endpoint=endpoint,
        timeout=timeout,
    )
    png_bytes, response_shape, response_image_name = _normalize_novelai_response_png(
        http_response.body,
        http_response.headers,
    )
    source_png, source_sidecar, raw_metadata = _write_live_generation_source_bundle(
        run_id=run_id,
        prompt=prompt,
        payload=payload,
        http_response=http_response,
        png_bytes=png_bytes,
        response_shape=response_shape,
        response_image_name=response_image_name,
        save_raw_response=save_raw_response,
    )
    result = publish_existing_generation(
        source_png=source_png,
        source_sidecar_dir=source_sidecar,
        run_id=run_id,
        prompt=prompt,
        model=model,
        aspect_ratio=aspect_ratio,
        artifact_name="image_000.png",
        save_raw_response=save_raw_response,
    )
    _write_live_response_metadata(Path(result["sidecar_dir"]), raw_metadata, Path(result["image"]))
    return result


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_png(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"NovelAI source PNG not found: {path}")
    with path.open("rb") as fh:
        signature = fh.read(8)
    if signature != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"NovelAI source is not a PNG: {path}")


def _raw_response_path(source_sidecar_dir: Path) -> Path | None:
    for name in _RAW_RESPONSE_FILES:
        candidate = source_sidecar_dir / name
        if candidate.is_file():
            return candidate
    return None


def _response_shape(path: Path | None) -> str:
    if path is None or not path.is_file():
        return "unknown"
    if path.suffix.lower() == ".zip":
        return "zip"
    try:
        if zipfile.is_zipfile(path):
            return "zip"
    except OSError:
        pass
    return "binary"


def _response_content_type(path: Path | None, shape: str) -> str:
    if shape == "zip":
        return "application/zip"
    if path is None:
        return "unknown"
    return "application/octet-stream"


def _raw_response_metadata(source_sidecar_dir: Path, *, raw_response_saved: bool) -> Dict[str, Any]:
    raw_path = _raw_response_path(source_sidecar_dir)
    shape = _response_shape(raw_path)
    return {
        "response_content_type": _response_content_type(raw_path, shape),
        "response_shape": shape,
        "response_bytes": raw_path.stat().st_size if raw_path is not None and raw_path.is_file() else 0,
        "response_sha256": _sha256_path(raw_path) if raw_path is not None and raw_path.is_file() else "",
        "raw_response_saved": raw_response_saved,
    }


def _copy_existing_sidecars(
    source_sidecar_dir: Path,
    published_sidecar_dir: Path,
    *,
    save_raw_response: bool = False,
) -> Dict[str, str]:
    published_sidecar_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    if not save_raw_response:
        for raw_name in _RAW_RESPONSE_FILES:
            try:
                (published_sidecar_dir / raw_name).unlink()
            except FileNotFoundError:
                pass

    if not source_sidecar_dir.is_dir():
        return copied

    for child in source_sidecar_dir.iterdir():
        if not child.is_file():
            continue
        if child.name in _RAW_RESPONSE_FILES and not save_raw_response:
            continue
        if child.name in _SIDECAR_FILES or (save_raw_response and child.name in _RAW_RESPONSE_FILES):
            destination = published_sidecar_dir / child.name
            shutil.copy2(child, destination)
            copied[child.name] = str(destination.resolve())
    return copied


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _write_response_metadata(
    *,
    source_sidecar_dir: Path,
    published_sidecar_dir: Path,
    published_png: Path,
    raw_response_saved: bool,
) -> Dict[str, Any]:
    response_path = published_sidecar_dir / "response.json"
    response = _load_json(response_path)
    metadata = _raw_response_metadata(source_sidecar_dir, raw_response_saved=raw_response_saved)
    metadata["normalized_artifacts"] = {
        "count": 1,
        "files": [str(published_png)],
    }
    response.update(metadata)
    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def _write_metadata_raw_flag(published_sidecar_dir: Path, *, raw_response_saved: bool) -> None:
    metadata_path = published_sidecar_dir / "metadata.json"
    metadata = _load_json(metadata_path)
    metadata["raw_response_saved"] = raw_response_saved
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_extended_manifest(
    *,
    source_png: Path,
    published_png: Path,
    published_sidecar_dir: Path,
    sidecar_files: Dict[str, str],
    nas_hook_requested: bool,
    response_metadata: Dict[str, Any],
) -> None:
    manifest_path = published_sidecar_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    raw_checks = manifest.get("checks")
    checks: Dict[str, Any] = raw_checks if isinstance(raw_checks, dict) else {}

    sidecar_manifest_files = {
        name: path
        for name, path in sidecar_files.items()
        if name != "manifest.json" and name not in _RAW_RESPONSE_FILES
    }
    files: Dict[str, Any] = {"png": str(published_png)}
    if sidecar_manifest_files:
        files["sidecar"] = sidecar_manifest_files
    if response_metadata.get("raw_response_saved"):
        raw_sidecars = {name: path for name, path in sidecar_files.items() if name in _RAW_RESPONSE_FILES}
        if raw_sidecars:
            files.setdefault("sidecar", {}).update(raw_sidecars)

    checks.update(
        {
            "png": "PASS",
            "publish": "PASS",
            "slack_upload": "READY",
            "nas_hook": "REQUESTED" if nas_hook_requested else "NOT_REQUESTED",
        }
    )
    manifest.update(
        {
            "source_path": str(source_png),
            "published_path": str(published_png),
            "published_resolved_path": str(published_png.resolve()),
            "files": files,
            "artifacts": [
                {
                    "path": str(published_png),
                    "kind": "normalized_png",
                    "sha256": _sha256_path(published_png),
                }
            ],
            "raw_response_saved": bool(response_metadata.get("raw_response_saved")),
            "raw_response": response_metadata,
            "checks": checks,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def publish_existing_generation(
    *,
    source_png: str | Path,
    source_sidecar_dir: str | Path | None = None,
    run_id: str,
    prompt: str = "",
    model: str = DEFAULT_MODEL,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    artifact_name: str = "image_000.png",
    work_root: str | Path | None = None,
    save_raw_response: bool | None = None,
) -> Dict[str, Any]:
    """Publish an existing NovelAI PNG into HermesWork and return provider output.

    The source PNG is never modified or regenerated. The published path is the
    only path returned in ``image``/``media_files`` so Slack delivery uses the
    allow-listed HermesWork artifact instead of the raw profile ``generated/``
    path. Raw NovelAI response bodies are not copied by default; set
    ``save_raw_response`` or ``HERMES_NAI_SAVE_RAW_RESPONSE=1`` for explicit
    debug retention.
    """
    if not run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be a non-empty single path segment")

    source = Path(source_png).expanduser().resolve(strict=False)
    _assert_png(source)
    sidecar_source = (
        Path(source_sidecar_dir).expanduser().resolve(strict=False)
        if source_sidecar_dir is not None
        else source.parent / "sidecar"
    )

    image_root = Path(work_root).expanduser() if work_root is not None else _resolve_image_work_root()
    published_dir = image_root / "NAI" / run_id
    published_dir.mkdir(parents=True, exist_ok=True)
    published_png = published_dir / artifact_name
    shutil.copy2(source, published_png)

    published_sidecar_dir = published_dir / "sidecar"
    raw_response_saved = _env_truthy("HERMES_NAI_SAVE_RAW_RESPONSE") if save_raw_response is None else save_raw_response
    sidecar_files = _copy_existing_sidecars(
        sidecar_source,
        published_sidecar_dir,
        save_raw_response=raw_response_saved,
    )
    sidecar_files.setdefault("manifest.json", str((published_sidecar_dir / "manifest.json").resolve()))
    response_metadata = _write_response_metadata(
        source_sidecar_dir=sidecar_source,
        published_sidecar_dir=published_sidecar_dir,
        published_png=published_png,
        raw_response_saved=raw_response_saved,
    )
    _write_metadata_raw_flag(published_sidecar_dir, raw_response_saved=raw_response_saved)

    nas_hook_requested = queue_nas_sync_hook(
        category="image",
        scope=f"NAI/{run_id}",
        artifact_path=published_png,
        source_root=published_dir,
    )
    _write_extended_manifest(
        source_png=source,
        published_png=published_png,
        published_sidecar_dir=published_sidecar_dir,
        sidecar_files=sidecar_files,
        nas_hook_requested=nas_hook_requested,
        response_metadata=response_metadata,
    )

    return success_response(
        image=str(published_png),
        model=model,
        prompt=prompt,
        aspect_ratio=resolve_aspect_ratio(aspect_ratio),
        provider="novelai",
        extra={
            "media_files": [str(published_png)],
            "published_path": str(published_png),
            "source_path": str(source),
            "sidecar_dir": str(published_sidecar_dir),
            "nas_hook_requested": nas_hook_requested,
            "raw_response_saved": raw_response_saved,
        },
    )


class NovelAIImageGenProvider(ImageGenProvider):
    """NovelAI adapter shell with HermesWork publish contract support."""

    @property
    def name(self) -> str:
        return "novelai"

    @property
    def display_name(self) -> str:
        return "NovelAI"

    def is_available(self) -> bool:
        return bool(
            os.environ.get("NOVELAI_API_KEY")
            or os.environ.get("NAI_API_KEY")
            or os.environ.get("NOVEL_AI_API_KEY")
            or os.environ.get("NOVELAI_TOKEN")
            or os.environ.get("NAI_TOKEN")
        )

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "NovelAI Diffusion 4.5 Curated",
                "speed": "varies",
                "strengths": "NovelAI generation; published via HermesWork/Image/NAI",
                "price": "subscription/anlas dependent",
            }
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "NovelAI",
            "badge": "paid",
            "tag": "NovelAI image generation with HermesWork publishing",
            "env_vars": [
                {
                    "key": "NOVELAI_API_KEY",
                    "prompt": "NovelAI API token",
                    "url": "https://novelai.net/",
                }
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        source_png = kwargs.get("source_png")
        run_id = str(kwargs.get("run_id") or "").strip() or _default_run_id()
        model = str(kwargs.get("model") or DEFAULT_MODEL)
        if kwargs.get("dry_run_request"):
            width = kwargs.get("width", NAI_DEFAULT_WIDTH)
            height = kwargs.get("height", NAI_DEFAULT_HEIGHT)
            payload = build_novelai_request_payload(
                prompt=prompt,
                width=width,
                height=height,
                model=model,
                negative_prompt=kwargs.get("negative_prompt"),
                high_res_approved=bool(kwargs.get("high_res_approved", False)),
                upscale=bool(kwargs.get("upscale", False)),
                high_resolution=bool(kwargs.get("high_resolution", False)),
            )
            return success_response(
                image="",
                model=payload["model"],
                prompt=prompt,
                aspect_ratio=resolve_aspect_ratio(aspect_ratio),
                provider="novelai",
                extra={"request_payload": payload, "dry_run_request": True},
            )
        if source_png:
            return publish_existing_generation(
                source_png=source_png,
                source_sidecar_dir=kwargs.get("source_sidecar_dir"),
                run_id=run_id,
                prompt=prompt,
                model=model,
                aspect_ratio=aspect_ratio,
                artifact_name=str(kwargs.get("artifact_name") or "image_000.png"),
                save_raw_response=kwargs.get("save_raw_response"),
            )

        if not bool(kwargs.get("live_generation_approved", False)):
            return error_response(
                error=(
                    f"{LIVE_GENERATION_REQUIRES_APPROVAL}: NovelAI live generation is disabled unless "
                    "live_generation_approved=True is provided by an explicitly approved operator action."
                ),
                error_type="approval_required",
                provider="novelai",
                model=model,
                prompt=prompt,
                aspect_ratio=resolve_aspect_ratio(aspect_ratio),
            )

        default_width, default_height = _default_dimensions_for_aspect_ratio(aspect_ratio)
        width = _coerce_dimension(kwargs.get("width", default_width), name="width")
        height = _coerce_dimension(kwargs.get("height", default_height), name="height")
        reserved = {
            "artifact_name",
            "dry_run_request",
            "endpoint",
            "height",
            "high_res_approved",
            "high_resolution",
            "live_generation_approved",
            "model",
            "negative_prompt",
            "run_id",
            "save_raw_response",
            "source_png",
            "source_sidecar_dir",
            "timeout",
            "upscale",
            "width",
        }
        parameter_overrides = {key: value for key, value in kwargs.items() if key not in reserved}
        try:
            return _generate_live_novelai_image(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                model=model,
                run_id=run_id,
                width=width,
                height=height,
                negative_prompt=kwargs.get("negative_prompt"),
                high_res_approved=bool(kwargs.get("high_res_approved", False)),
                upscale=bool(kwargs.get("upscale", False)),
                high_resolution=bool(kwargs.get("high_resolution", False)),
                save_raw_response=(
                    _env_truthy("HERMES_NAI_SAVE_RAW_RESPONSE")
                    if kwargs.get("save_raw_response") is None
                    else bool(kwargs.get("save_raw_response"))
                ),
                endpoint=str(
                    kwargs.get("endpoint")
                    or (NOVELAI_GENERATE_STREAM_ENDPOINT if model.startswith("nai-diffusion-4") else NOVELAI_GENERATE_ENDPOINT)
                ),
                timeout=int(kwargs.get("timeout") or NOVELAI_REQUEST_TIMEOUT_SECONDS),
                parameter_overrides=parameter_overrides,
            )
        except NovelAIResolutionApprovalRequired:
            raise
        except Exception as exc:
            return error_response(
                error=str(exc),
                error_type="generation_failed",
                provider="novelai",
                model=model,
                prompt=prompt,
                aspect_ratio=resolve_aspect_ratio(aspect_ratio),
            )


def register(ctx) -> None:
    """Plugin entry point — wire ``NovelAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(NovelAIImageGenProvider())
