"""Local ComfyUI image generation backend.

This provider talks to a macOS local ComfyUI API, with legacy Windows/WSL
fallback support, waits for `/history/<prompt_id>` success, verifies the exact
output file for the run, then publishes the result into
`HermesWork/Image/<project_id>/` with workflow / prompt / metadata sidecars
before queueing the existing NAS sync hook.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    publish_filesystem_image_bundle,
    resolve_aspect_ratio,
    success_response,
    update_run_manifest,
    wait_for_file_stable,
    write_qualification_report,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_OUTPUT_DIR = "/Volumes/SSD_Hermes/ComfyUI/output"
DEFAULT_OUTPUT_DIR_FALLBACKS: Tuple[str, ...] = (
    "/Volumes/SSD_Hermes/ComfyUI/output",
    "/Volumes/SSD_Hermes/ComfyUI/ComfyUI/output",
    "/mnt/c/AI/ComfyUI/output",
    "/mnt/d/AI/ComfyUI/output",
    "/mnt/e/AI/ComfyUI/output",
)
DEFAULT_CHECKPOINT = "AOM3A1_orangemixs.safetensors"
DEFAULT_SMOKE_E2E_CHECKPOINT = "animagine-xl-4.0-opt.safetensors"
DEFAULT_STEPS = 12
DEFAULT_CFG = 7.0
DEFAULT_SAMPLER = "euler"
DEFAULT_DENOISE = 1.0
DEFAULT_SEED = 123456789
DEFAULT_CATEGORY = "txt2img"

CHARACTER_PRODUCTION_PRESET = "character_production"
CHARACTER_PRODUCTION_STEPS = 28
CHARACTER_PRODUCTION_CFG = 5.0
CHARACTER_PRODUCTION_SAMPLER = "dpmpp_2m"
CHARACTER_PRODUCTION_SCHEDULER = "karras"
CHARACTER_PRODUCTION_POSITIVE_SKELETON = (
    "1girl, solo, full body, standing, looking at viewer, character focus, centered character, "
    "large character, detailed face, detailed eyes, detailed outfit, beautiful young adult woman, "
    "gacha game heroine, RPG protagonist, protagonist-grade heroine, attractive face, clear facial features, "
    "readable expression, refined anime illustration, premium game character illustration, ornate fantasy outfit, "
    "detailed costume design, elegant silhouette, clean silhouette, full-body character art, vertical portrait, "
    "simple background, background secondary, safe, masterpiece, high score, great score, absurdres"
)
CHARACTER_PRODUCTION_NEGATIVE_BASELINE = (
    "low quality, worst quality, bad quality, normal quality, lowres, blurry, watermark, text, signature, "
    "username, bad anatomy, bad hands, malformed hands, malformed fingers, missing fingers, extra digits, "
    "fewer digits, bad feet, bad proportions, duplicate, mutation, disfigured, deformed, extra arms, extra legs, "
    "abstract, symbolic, tiny person, tiny character, distant character, silhouette-only character, background focus, "
    "scenery focus, environment focus, unreadable face, face out of frame, head out of frame, cropped body, "
    "cropped legs, covered face, wings covering body, overwhelming background, symbolic art first, "
    "scenery-first composition"
)
CHARACTER_PRODUCTION_KEYWORDS: Tuple[str, ...] = (
    "캐릭터",
    "미소녀",
    "전신",
    "RPG",
    "수집형",
    "주인공급",
    "heroine",
    "gacha",
    "full body",
)

_ASPECT_DIMENSIONS: dict[str, tuple[int, int]] = {
    "landscape": (1024, 768),
    "square": (1024, 1024),
    "portrait": (768, 1024),
}


def _load_image_gen_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _provider_cfg() -> Dict[str, Any]:
    cfg = _load_image_gen_config()
    for key in ("comfy-local", "comfy_local"):
        block = cfg.get(key)
        if isinstance(block, dict):
            return block
    return {}


def _helper_base_url() -> Optional[str]:
    try:
        from hermes_constants import get_hermes_home

        helper = get_hermes_home() / "scripts" / "comfy_api_url.sh"
        if helper.is_file():
            return subprocess.check_output([str(helper)], text=True, timeout=5).strip() or None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not resolve Comfy helper base URL: %s", exc)
    return None


def _resolve_base_url() -> str:
    env_override = os.environ.get("COMFY_LOCAL_IMAGE_BASE_URL")
    if env_override and env_override.strip():
        return env_override.strip().rstrip("/")
    cfg = _provider_cfg()
    base_url = cfg.get("base_url") if isinstance(cfg.get("base_url"), str) else None
    if base_url and base_url.strip():
        return base_url.strip().rstrip("/")
    helper_url = _helper_base_url()
    if helper_url:
        return helper_url.rstrip("/")
    return DEFAULT_BASE_URL


def _resolve_output_dir() -> Path:
    # Keep backward-compatible behavior: explicit override first, then first
    # configured/default candidate.
    env_override = os.environ.get("COMFY_LOCAL_OUTPUT_DIR")
    if env_override and env_override.strip():
        return Path(env_override.strip())
    cfg = _provider_cfg()
    output_dir = cfg.get("output_dir") if isinstance(cfg.get("output_dir"), str) else None
    if output_dir and output_dir.strip():
        return Path(output_dir.strip())
    return Path(DEFAULT_OUTPUT_DIR)


def _candidate_output_dirs() -> List[Path]:
    """Return ordered candidate directories for ComfyUI output lookup.

    Real deployments can emit outputs under WSL mount points, SMB mount points,
    or legacy absolute paths. Try configured/override values first, then a list of
    known fallback paths.
    """
    candidates: List[Path] = []
    seen: set[str] = set()

    env_override = os.environ.get("COMFY_LOCAL_OUTPUT_DIR")
    if env_override and env_override.strip():
        candidates.append(Path(env_override.strip()))

    cfg = _provider_cfg()
    output_dir = cfg.get("output_dir") if isinstance(cfg.get("output_dir"), str) else None
    if output_dir and output_dir.strip():
        candidates.append(Path(output_dir.strip()))

    candidates.append(Path(DEFAULT_OUTPUT_DIR))
    candidates.extend(Path(item) for item in DEFAULT_OUTPUT_DIR_FALLBACKS)

    normalized: List[Path] = []
    for candidate in candidates:
        try:
            p = candidate.expanduser()
        except Exception:
            p = candidate
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            normalized.append(p)
            seen.add(key)
    return normalized


def _find_comfy_output_file(output_image: Dict[str, Any]) -> Optional[Path]:
    filename = str(output_image.get("filename") or "").strip()
    if not filename:
        return None

    subfolder = str(output_image.get("subfolder") or "").strip()
    filename = Path(filename).name
    searched: List[Path] = []
    for output_dir in _candidate_output_dirs():
        candidate = output_dir
        if subfolder:
            candidate = candidate / subfolder
        candidate = candidate / filename
        searched.append(candidate)
        if candidate.is_file():
            return candidate

    logger.warning(
        "ComfyUI output file not found. filename=%s subfolder=%s candidates=%s",
        filename,
        subfolder or "(none)",
        [str(p) for p in searched],
    )
    return None


def _remote_output_cache_dir() -> Path:
    """Return a controlled cache dir for remote ComfyUI /view downloads."""
    try:
        from hermes_constants import get_hermes_home

        root = get_hermes_home() / "cache" / "images" / "comfy-remote"
    except Exception:  # noqa: BLE001
        root = Path("/tmp") / "hermes-comfy-remote"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _download_comfy_output_file(base_url: str, output_image: Dict[str, Any], prompt_id: str) -> Optional[Path]:
    """Download a remote ComfyUI output via /view when no local output path exists.

    Remote Windows workers expose generated files through the ComfyUI REST API,
    but their output directory is not necessarily mounted on the Mac running
    Hermes. Save the bytes under the profile cache, then publish that cached file
    through the normal HermesWork/Image bundle path.
    """
    filename = Path(str(output_image.get("filename") or "").strip()).name
    if not filename:
        return None
    output_type = str(output_image.get("type") or "output").strip() or "output"
    subfolder = str(output_image.get("subfolder") or "").strip()
    query = f"filename={quote(filename)}&type={quote(output_type)}&subfolder={quote(subfolder)}"
    url = f"{base_url.rstrip('/')}/view?{query}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("remote_comfy_view_download_failed prompt_id=%s filename=%s err=%s", prompt_id, filename, exc)
        return None
    if not isinstance(content, (bytes, bytearray)) or not content.startswith(b"\x89PNG\r\n\x1a\n"):
        logger.warning("remote_comfy_view_download_invalid_png prompt_id=%s filename=%s", prompt_id, filename)
        return None

    safe_prompt = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prompt_id)[:80] or "remote"
    cache_path = _remote_output_cache_dir() / f"{safe_prompt}_{filename}"
    cache_path.write_bytes(bytes(content))
    return cache_path


def _resolve_model() -> str:
    env_override = os.environ.get("COMFY_LOCAL_IMAGE_MODEL")
    if env_override and env_override.strip():
        return env_override.strip()
    cfg = _provider_cfg()
    model = cfg.get("model") if isinstance(cfg.get("model"), str) else None
    if model and model.strip():
        return model.strip()
    return DEFAULT_CHECKPOINT


def _is_character_production_request(prompt_text: str) -> bool:
    lowered = str(prompt_text or "").casefold()
    return any(keyword.casefold() in lowered for keyword in CHARACTER_PRODUCTION_KEYWORDS)


def _translate_character_production_prompt(prompt_text: str) -> str:
    translated = str(prompt_text or "").strip()
    if not translated:
        return translated
    for pattern, replacement in (
        (r"주인공급", "heroine-grade"),
        (r"미소녀", "beautiful girl"),
        (r"캐릭터", "character"),
        (r"전신", "full body"),
        (r"수집형", "gacha"),
    ):
        translated = re.sub(pattern, replacement, translated, flags=re.IGNORECASE)
    translated = re.sub(r"\s+", " ", translated).strip()
    return _merge_comma_terms(CHARACTER_PRODUCTION_POSITIVE_SKELETON, translated)


def _merge_comma_terms(*segments: str) -> str:
    merged: List[str] = []
    seen: set[str] = set()
    for segment in segments:
        for item in re.split(r"[\n,]", str(segment or "")):
            term = item.strip()
            if not term:
                continue
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(term)
    return ", ".join(merged)


def _normalize_subject_dominance(subject_dominance: Any, *, default: float = 80.0) -> tuple[float, float, str]:
    if subject_dominance is None or (isinstance(subject_dominance, str) and not subject_dominance.strip()):
        value = default
    else:
        try:
            value = float(subject_dominance)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid subject_dominance value: {subject_dominance!r}") from exc
    if value <= 0:
        raise ValueError("subject_dominance must be greater than 0")
    if value <= 1.0:
        normalized = value
        dominance_pct = value * 100.0
    elif value <= 100.0:
        normalized = value / 100.0
        dominance_pct = value
    else:
        raise ValueError("subject_dominance must be between 0-1 or 0-100")
    if dominance_pct < 70.0:
        raise ValueError("character_production preset requires subject_dominance of at least 70")
    if dominance_pct >= 90.0:
        rule = "single character, centered full-body composition, dominant subject, minimal background clutter"
    elif dominance_pct >= 80.0:
        rule = "single character focus, centered composition, clear silhouette, minimal background clutter"
    else:
        rule = "subject-focused composition, clear separation from background"
    return dominance_pct, normalized, rule


def _build_character_production_runtime(
    prompt: str,
    *,
    negative_prompt: str,
    subject_dominance: Any,
) -> Optional[Dict[str, Any]]:
    if not _is_character_production_request(prompt):
        return None
    negative_prompt_text = str(negative_prompt or "").strip()
    dominance_pct, normalized_subject_dominance, subject_rule = _normalize_subject_dominance(subject_dominance)
    translated_prompt = _translate_character_production_prompt(prompt)
    final_prompt = f"{translated_prompt}, {subject_rule}" if subject_rule else translated_prompt
    final_negative_prompt = CHARACTER_PRODUCTION_NEGATIVE_BASELINE
    if negative_prompt_text:
        final_negative_prompt = _merge_comma_terms(CHARACTER_PRODUCTION_NEGATIVE_BASELINE, negative_prompt_text)
    return {
        "preset": CHARACTER_PRODUCTION_PRESET,
        "source_prompt": str(prompt or "").strip(),
        "translated_prompt": translated_prompt,
        "prompt": final_prompt,
        "negative_prompt": final_negative_prompt,
        "negative_baseline": CHARACTER_PRODUCTION_NEGATIVE_BASELINE,
        "subject_dominance": dominance_pct,
        "subject_dominance_ratio": normalized_subject_dominance,
        "subject_dominance_rule": subject_rule,
        "steps": CHARACTER_PRODUCTION_STEPS,
        "cfg": CHARACTER_PRODUCTION_CFG,
        "sampler_name": CHARACTER_PRODUCTION_SAMPLER,
        "scheduler": CHARACTER_PRODUCTION_SCHEDULER,
        "prompt_translation_policy": "character-skeleton + keyword-translate + subject-dominance guidance",
    }


def _is_smoke_or_e2e_context(task_context: str) -> bool:
    lowered = str(task_context or "").casefold()
    return any(
        token in lowered
        for token in (
            "smoke",
            "e2e",
            "delivery verification",
            "delivery_verification",
            "fresh e2e",
            "스모크",
            "전달 검증",
            "딜리버리 검증",
        )
    )


def _normalize_checkpoint_list(checkpoints: Any) -> List[str]:
    if not isinstance(checkpoints, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()
    for item in checkpoints:
        if not isinstance(item, str) or not item.strip():
            continue
        name = item.strip()
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return normalized


def _checkpoint_stem(name: str) -> str:
    lowered = str(name or "").strip()
    for suffix in (".safetensors", ".ckpt"):
        if lowered.casefold().endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _checkpoint_key(name: str) -> str:
    """Normalize checkpoint aliases without banning any model-name string."""
    return "".join(ch.casefold() for ch in _checkpoint_stem(name) if ch.isalnum())


def _partial_alias_keys(requested: str) -> List[str]:
    key = _checkpoint_key(requested)
    keys = [key] if key else []
    # Some installed checkpoint names abbreviate Illustrious as "Illus"; allow
    # a deterministic stem alias while still requiring unique remote-list match.
    if len(key) >= 6:
        keys.append(key[:5])
    deduped: List[str] = []
    for item in keys:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _resolution_payload(
    *,
    base: Dict[str, Any],
    ok: bool,
    mode: str,
    candidates: List[str],
    resolved_checkpoint: Optional[str],
    error_type: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        **base,
        "ok": ok,
        "resolved_checkpoint": resolved_checkpoint,
        "resolution_mode": mode,
        # Backward-compatible alias for existing report consumers.
        "resolution_reason": mode,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    if error_type:
        payload["error_type"] = error_type
    if error:
        payload["error"] = error
    return payload


def resolve_checkpoint(
    requested_checkpoint: str,
    checkpoints: Any,
    *,
    task_context: str = "",
    allow_smoke_default: bool = True,
) -> Dict[str, Any]:
    """Resolve a requested checkpoint using only the remote ComfyUI model list.

    Resolution is list-based, not blacklist-based: exact remote filenames win,
    user-shortened aliases are accepted only when they resolve uniquely, and
    ambiguous aliases never auto-select a checkpoint.
    """
    requested = str(requested_checkpoint or "").strip() or DEFAULT_CHECKPOINT
    available = _normalize_checkpoint_list(checkpoints)
    base: Dict[str, Any] = {
        "requested_checkpoint": requested,
        "available_checkpoints_count": len(available),
        "available_checkpoints": available,
    }

    if requested in available:
        return _resolution_payload(
            base=base,
            ok=True,
            mode="exact",
            candidates=[requested],
            resolved_checkpoint=requested,
        )

    extension_matches = [name for name in available if requested == _checkpoint_stem(name)]
    if len(extension_matches) == 1:
        return _resolution_payload(
            base=base,
            ok=True,
            mode="extension_insensitive",
            candidates=extension_matches,
            resolved_checkpoint=extension_matches[0],
        )
    if len(extension_matches) > 1:
        return _resolution_payload(
            base=base,
            ok=False,
            mode="ambiguous",
            candidates=extension_matches,
            resolved_checkpoint=None,
            error_type="checkpoint_ambiguous",
            error=f"Requested checkpoint alias is ambiguous: {requested}",
        )

    requested_key = _checkpoint_key(requested)
    normalized_matches = [name for name in available if requested_key and requested_key == _checkpoint_key(name)]
    if len(normalized_matches) == 1:
        return _resolution_payload(
            base=base,
            ok=True,
            mode="normalized",
            candidates=normalized_matches,
            resolved_checkpoint=normalized_matches[0],
        )
    if len(normalized_matches) > 1:
        return _resolution_payload(
            base=base,
            ok=False,
            mode="ambiguous",
            candidates=normalized_matches,
            resolved_checkpoint=None,
            error_type="checkpoint_ambiguous",
            error=f"Requested checkpoint alias is ambiguous: {requested}",
        )

    alias_keys = _partial_alias_keys(requested)
    partial_matches: List[str] = []
    for name in available:
        checkpoint_key = _checkpoint_key(name)
        if any(alias_key in checkpoint_key for alias_key in alias_keys):
            partial_matches.append(name)
    if len(partial_matches) == 1:
        return _resolution_payload(
            base=base,
            ok=True,
            mode="unique_partial",
            candidates=partial_matches,
            resolved_checkpoint=partial_matches[0],
        )
    if len(partial_matches) > 1:
        return _resolution_payload(
            base=base,
            ok=False,
            mode="ambiguous",
            candidates=partial_matches,
            resolved_checkpoint=None,
            error_type="checkpoint_ambiguous",
            error=f"Requested checkpoint alias is ambiguous: {requested}",
        )

    if allow_smoke_default and _is_smoke_or_e2e_context(task_context) and DEFAULT_SMOKE_E2E_CHECKPOINT in available:
        return _resolution_payload(
            base=base,
            ok=True,
            mode="default_for_smoke",
            candidates=[],
            resolved_checkpoint=DEFAULT_SMOKE_E2E_CHECKPOINT,
        )
    return _resolution_payload(
        base=base,
        ok=False,
        mode="not_found",
        candidates=available,
        resolved_checkpoint=None,
        error_type="checkpoint_not_found",
        error=f"Requested checkpoint not found in remote ComfyUI model list: {requested}",
    )


def _resolve_vae() -> Optional[str]:
    env_override = os.environ.get("COMFY_LOCAL_IMAGE_VAE")
    if env_override and env_override.strip():
        return env_override.strip()
    cfg = _provider_cfg()
    vae = cfg.get("vae") if isinstance(cfg.get("vae"), str) else None
    if vae and vae.strip():
        return vae.strip()
    return None


def _resolve_dimensions(aspect_ratio: str, width: Any = None, height: Any = None) -> tuple[int, int]:
    if isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0:
        return width, height
    aspect = resolve_aspect_ratio(aspect_ratio)
    return _ASPECT_DIMENSIONS.get(aspect, _ASPECT_DIMENSIONS[DEFAULT_ASPECT_RATIO])


def _history_url(base_url: str, prompt_id: str) -> str:
    return f"{base_url}/history/{prompt_id}"


def _first_output_image(history_payload: Dict[str, Any], prompt_id: str) -> Optional[Dict[str, Any]]:
    entry = history_payload.get(prompt_id)
    if not isinstance(entry, dict):
        return None
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for node_data in outputs.values():
        if not isinstance(node_data, dict):
            continue
        images = node_data.get("images")
        if not isinstance(images, list):
            continue
        for image in images:
            if isinstance(image, dict) and isinstance(image.get("filename"), str):
                return image
    return None


def _history_completed_successfully(history_payload: Dict[str, Any], prompt_id: str) -> bool:
    entry = history_payload.get(prompt_id)
    if not isinstance(entry, dict):
        return False
    status = entry.get("status")
    if not isinstance(status, dict):
        return False
    return bool(status.get("completed")) and str(status.get("status_str") or "").lower() == "success"


class ComfyLocalImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "comfy-local"

    @property
    def display_name(self) -> str:
        return "Comfy Local"

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Comfy Local",
            "badge": "local",
            "tag": "Windows-hosted ComfyUI API with HermesWork publish + NAS sidecars.",
            "env_vars": [],
        }

    def is_available(self) -> bool:
        base_url = _resolve_base_url()
        try:
            response = requests.get(f"{base_url}/system_stats", timeout=5)
            response.raise_for_status()
            data = response.json()
            return isinstance(data, dict) and isinstance(data.get("system"), dict)
        except Exception:  # noqa: BLE001
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        base_url = _resolve_base_url()
        try:
            response = requests.get(f"{base_url}/models/checkpoints", timeout=10)
            response.raise_for_status()
            models = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Comfy checkpoint lookup failed: %s", exc)
            models = [DEFAULT_CHECKPOINT]

        entries: List[Dict[str, Any]] = []
        if isinstance(models, list):
            for item in models:
                if not isinstance(item, str) or not item.strip():
                    continue
                name = item.strip()
                entries.append(
                    {
                        "id": name,
                        "display": name,
                        "speed": "local",
                        "strengths": "comfyui checkpoint",
                    }
                )
        if not entries:
            entries.append(
                {
                    "id": DEFAULT_CHECKPOINT,
                    "display": DEFAULT_CHECKPOINT,
                    "speed": "local",
                    "strengths": "comfyui checkpoint",
                }
            )
        return entries

    def default_model(self) -> Optional[str]:
        return _resolve_model()

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=self.name,
                aspect_ratio=aspect,
            )

        base_url = _resolve_base_url()
        checkpoint = str(kwargs.get("model") or _resolve_model()).strip() or DEFAULT_CHECKPOINT
        requested_checkpoint = checkpoint
        vae = str(kwargs.get("vae") or _resolve_vae() or "").strip() or None
        project_name = kwargs.get("project_name")
        artifact_name = kwargs.get("artifact_name")
        category = str(kwargs.get("category") or DEFAULT_CATEGORY).strip() or DEFAULT_CATEGORY
        width, height = _resolve_dimensions(aspect, kwargs.get("width"), kwargs.get("height"))
        steps = int(kwargs.get("steps")) if isinstance(kwargs.get("steps"), int) and int(kwargs.get("steps")) > 0 else DEFAULT_STEPS
        cfg = float(kwargs.get("cfg_scale")) if isinstance(kwargs.get("cfg_scale"), (int, float)) and float(kwargs.get("cfg_scale")) > 0 else DEFAULT_CFG
        denoise = float(kwargs.get("denoise")) if isinstance(kwargs.get("denoise"), (int, float)) and float(kwargs.get("denoise")) > 0 else DEFAULT_DENOISE
        sampler_name = str(kwargs.get("sampler_name") or DEFAULT_SAMPLER).strip() or DEFAULT_SAMPLER
        scheduler = str(kwargs.get("scheduler") or "normal").strip() or "normal"
        seed = int(kwargs.get("seed")) if isinstance(kwargs.get("seed"), int) else DEFAULT_SEED
        negative_prompt = str(kwargs.get("negative_prompt") or "").strip()
        subject_dominance = kwargs.get("subject_dominance")
        runtime_preset: Optional[Dict[str, Any]] = None
        try:
            runtime_preset = _build_character_production_runtime(
                prompt,
                negative_prompt=negative_prompt,
                subject_dominance=subject_dominance,
            )
        except ValueError as exc:
            return error_response(
                error=str(exc),
                error_type="invalid_argument",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        prompt_for_generation = prompt
        source_prompt = prompt
        preset_name = "default"
        prompt_translation_policy = "raw"
        subject_dominance_value: Optional[float] = None
        subject_dominance_rule: Optional[str] = None
        if runtime_preset is not None:
            preset_name = str(runtime_preset.get("preset") or CHARACTER_PRODUCTION_PRESET)
            prompt_for_generation = str(runtime_preset.get("prompt") or prompt)
            negative_prompt = str(runtime_preset.get("negative_prompt") or negative_prompt)
            steps = int(runtime_preset.get("steps") or steps)
            cfg = float(runtime_preset.get("cfg") or cfg)
            sampler_name = str(runtime_preset.get("sampler_name") or sampler_name)
            scheduler = str(runtime_preset.get("scheduler") or scheduler)
            prompt_translation_policy = str(runtime_preset.get("prompt_translation_policy") or prompt_translation_policy)
            subject_dominance_value = runtime_preset.get("subject_dominance")
            subject_dominance_rule = runtime_preset.get("subject_dominance_rule")

        filename_prefix = Path(str(artifact_name or kwargs.get("filename_prefix") or "angelica_txt2img").strip() or "angelica_txt2img").name

        try:
            system_response = requests.get(f"{base_url}/system_stats", timeout=10)
            system_response.raise_for_status()
            system_payload = system_response.json()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"ComfyUI system_stats check failed: {exc}",
                error_type="connection_error",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        checkpoint_resolution: Dict[str, Any] = {}
        try:
            checkpoint_response = requests.get(f"{base_url}/models/checkpoints", timeout=10)
            checkpoint_response.raise_for_status()
            checkpoints = checkpoint_response.json()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"ComfyUI checkpoint lookup failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        task_context_parts = [
            prompt,
            str(project_name or ""),
            str(artifact_name or ""),
            category,
            str(kwargs.get("filename_prefix") or ""),
            str(kwargs.get("task_context") or ""),
        ]
        qualification_context = kwargs.get("qualification_context") if isinstance(kwargs.get("qualification_context"), dict) else None
        if qualification_context:
            task_context_parts.extend(
                str(qualification_context.get(key) or "")
                for key in ("run_kind", "run_id", "workflow_code", "workflow_name")
            )
        checkpoint_resolution = resolve_checkpoint(
            checkpoint,
            checkpoints,
            task_context="\n".join(part for part in task_context_parts if part),
            allow_smoke_default=runtime_preset is None,
        )
        if not checkpoint_resolution.get("ok"):
            result = error_response(
                error=str(checkpoint_resolution.get("error") or f"Requested checkpoint not found in remote ComfyUI model list: {checkpoint}"),
                error_type=str(checkpoint_resolution.get("error_type") or "checkpoint_not_found"),
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )
            result.update(
                {
                    "requested_checkpoint": checkpoint_resolution.get("requested_checkpoint"),
                    "resolved_checkpoint": checkpoint_resolution.get("resolved_checkpoint"),
                    "resolution_mode": checkpoint_resolution.get("resolution_mode"),
                    "resolution_reason": checkpoint_resolution.get("resolution_reason"),
                    "source_model_rejected": checkpoint_resolution.get("source_model_rejected"),
                    "available_checkpoints_count": checkpoint_resolution.get("available_checkpoints_count"),
                    "candidate_count": checkpoint_resolution.get("candidate_count"),
                    "candidates": checkpoint_resolution.get("candidates"),
                }
            )
            return result
        checkpoint = str(checkpoint_resolution.get("resolved_checkpoint") or checkpoint)

        if vae is not None:
            try:
                vae_response = requests.get(f"{base_url}/models/vae", timeout=10)
                vae_response.raise_for_status()
                vaes = vae_response.json()
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"ComfyUI VAE lookup failed: {exc}",
                    error_type="api_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            if vae not in vaes:
                return error_response(
                    error=f"Requested VAE not found in ComfyUI: {vae}",
                    error_type="invalid_argument",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        workflow: Dict[str, Any] = {
            "1": {"inputs": {"ckpt_name": checkpoint}, "class_type": "CheckpointLoaderSimple"},
            "2": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
            "3": {"inputs": {"text": prompt_for_generation, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
            "4": {"inputs": {"text": negative_prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
            "5": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": denoise,
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["2", 0],
                },
                "class_type": "KSampler",
            },
            "7": {
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2] if vae is None else ["6", 0],
                },
                "class_type": "VAEDecode",
            },
            "8": {"inputs": {"filename_prefix": filename_prefix, "images": ["7", 0]}, "class_type": "SaveImage"},
        }

        if vae is not None:
            workflow["6"] = {"inputs": {"vae_name": vae}, "class_type": "VAELoader"}
        payload = {"prompt": workflow}

        try:
            response = requests.post(f"{base_url}/prompt", json=payload, timeout=30)
            response.raise_for_status()
            submit = response.json()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"ComfyUI prompt submission failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        prompt_id = str(submit.get("prompt_id") or "").strip()
        if not prompt_id:
            return error_response(
                error="ComfyUI prompt response did not include prompt_id",
                error_type="invalid_response",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        history_payload: Optional[Dict[str, Any]] = None
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                history_response = requests.get(_history_url(base_url, prompt_id), timeout=15)
                history_response.raise_for_status()
                history_payload = history_response.json()
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"ComfyUI history lookup failed: {exc}",
                    error_type="api_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            if isinstance(history_payload, dict) and _history_completed_successfully(history_payload, prompt_id):
                break
            time.sleep(1)
        else:
            return error_response(
                error=f"ComfyUI history timed out before success for prompt_id={prompt_id}",
                error_type="timeout",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        output_image = _first_output_image(history_payload or {}, prompt_id)
        if not output_image:
            return error_response(
                error="ComfyUI history success did not contain an output image",
                error_type="invalid_response",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        source_path = _find_comfy_output_file(output_image)
        source_origin = "local_output_dir"
        if source_path is None:
            source_path = _download_comfy_output_file(base_url, output_image, prompt_id)
            source_origin = "remote_view_download" if source_path is not None else "missing"
        if source_path is None:
            candidate_dirs = ", ".join(str(candidate) for candidate in _candidate_output_dirs())
            logger.warning(
                "publish_source_missing=%s prompt_id=%s prompt=%s model=%s candidates=[%s]",
                output_image,
                prompt_id,
                prompt,
                checkpoint,
                candidate_dirs,
            )
            return error_response(
                error="ComfyUI output file not found after history success",
                error_type="io_error",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        if not wait_for_file_stable(source_path, checks=2, delay_seconds=0.1):
            return error_response(
                error=f"ComfyUI output file did not stabilize: {source_path}",
                error_type="io_error",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        created_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        raw_prompt_payload = {
            "submit_payload": payload,
            "submit_response": submit,
            "history_status": (history_payload or {}).get(prompt_id, {}).get("status", {}),
            "output_image": output_image,
            "system_stats": system_payload,
            "runtime_preset": preset_name,
            "prompt_translation_policy": prompt_translation_policy,
        }
        prompt_payload = {
            "prompt": prompt_for_generation,
            "source_prompt": source_prompt,
            "translated_prompt": prompt_for_generation,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": 1,
            "seed": seed,
            "sampler": sampler_name,
            "steps": steps,
            "cfg": cfg,
            "denoise": denoise,
            "scheduler": scheduler,
            "runtime_preset": preset_name,
            "subject_dominance": subject_dominance_value,
            "subject_dominance_rule": subject_dominance_rule,
            "prompt_translation_policy": prompt_translation_policy,
            "raw_prompt_payload": raw_prompt_payload,
            "requested_checkpoint": checkpoint_resolution.get("requested_checkpoint", requested_checkpoint),
            "resolved_checkpoint": checkpoint,
            "resolution_mode": checkpoint_resolution.get("resolution_mode"),
            "resolution_reason": checkpoint_resolution.get("resolution_reason"),
            "source_model_rejected": checkpoint_resolution.get("source_model_rejected"),
            "candidate_count": checkpoint_resolution.get("candidate_count"),
            "candidates": checkpoint_resolution.get("candidates"),
        }
        metadata = {
            "provider": self.name,
            "prompt_id": prompt_id,
            "api_base_url": base_url,
            "checkpoint": checkpoint,
            "requested_checkpoint": checkpoint_resolution.get("requested_checkpoint", requested_checkpoint),
            "resolved_checkpoint": checkpoint,
            "resolution_mode": checkpoint_resolution.get("resolution_mode"),
            "resolution_reason": checkpoint_resolution.get("resolution_reason"),
            "source_model_rejected": checkpoint_resolution.get("source_model_rejected"),
            "available_checkpoints_count": checkpoint_resolution.get("available_checkpoints_count"),
            "candidate_count": checkpoint_resolution.get("candidate_count"),
            "candidates": checkpoint_resolution.get("candidates"),
            "preset": preset_name,
            "source_prompt": source_prompt,
            "translated_prompt": prompt_for_generation,
            "prompt_translation_policy": prompt_translation_policy,
            "subject_dominance": subject_dominance_value,
            "subject_dominance_rule": subject_dominance_rule,
            "negative_baseline": CHARACTER_PRODUCTION_NEGATIVE_BASELINE if preset_name == CHARACTER_PRODUCTION_PRESET else None,
            "negative_prompt": negative_prompt,
            "vae": vae,
            "loras": [],
            "controlnet_used": False,
            "seed": seed,
            "sampler": sampler_name,
            "steps": steps,
            "cfg": cfg,
            "denoise": denoise,
            "created_at": created_at,
            "category": category,
            "output_source_path": str(source_path),
            "output_source_origin": source_origin,
            "local_status": "생성 완료",
            "publish_status": "HermesWork publish 완료",
            "slack_status": "primary image 준비됨",
            "requested_width": width,
            "requested_height": height,
        }

        try:
            bundle = publish_filesystem_image_bundle(
                source_path,
                prefix=filename_prefix,
                project_name=project_name,
                artifact_name=artifact_name or filename_prefix,
                category=category,
                workflow_json=workflow,
                prompt_payload=prompt_payload,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Could not publish ComfyUI image bundle to HermesWork: prompt_id=%s path=%s err=%s",
                prompt_id,
                source_path,
                exc,
            )
            return error_response(
                error=f"Could not publish ComfyUI image bundle to HermesWork: {exc}",
                error_type="io_error",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        qualification_context = kwargs.get("qualification_context") if isinstance(kwargs.get("qualification_context"), dict) else None
        run_manifest_path: Optional[Path] = None
        qualification_report_path: Optional[Path] = None
        if qualification_context:
            artifact_entry = {
                "run_id": str(qualification_context.get("run_id") or artifact_name or filename_prefix),
                "artifact_name": str(artifact_name or filename_prefix),
                "primary_image": bundle["primary_image"],
                "workflow_json": Path(bundle["workflow_path"]).name,
                "prompt_json": Path(bundle["prompt_path"]).name,
                "metadata_json": Path(bundle["metadata_path"]).name,
                "category": category,
                "seed": seed,
                "status": {
                    "technical_result": "Pass",
                    "publish_status": "Pass",
                    "nas_hook_status": "Pass" if bundle["nas_hook_requested"] else "Fail",
                    "slack_status": "Pending",
                },
            }
            run_manifest_path = update_run_manifest(
                bundle["published_dir"],
                workflow_code=str(qualification_context.get("workflow_code") or ""),
                workflow_name=str(qualification_context.get("workflow_name") or ""),
                run_kind=str(qualification_context.get("run_kind") or "qualification"),
                project_name=str(project_name or filename_prefix),
                project_id=str(bundle["project_id"]),
                artifact=artifact_entry,
            )
            report_payload = qualification_context.get("report")
            if isinstance(report_payload, dict):
                qualification_report_path = write_qualification_report(bundle["published_dir"], report_payload)

        nas_status = "동기화 요청됨" if bundle["nas_hook_requested"] else "동기화 요청 실패"
        return success_response(
            image=str(bundle["primary_image_path"]),
            model=checkpoint,
            prompt=prompt_for_generation,
            aspect_ratio=aspect,
            provider=self.name,
            extra={
                "base_url": base_url,
                "requested_checkpoint": checkpoint_resolution.get("requested_checkpoint", requested_checkpoint),
                "resolved_checkpoint": checkpoint,
                "resolution_mode": checkpoint_resolution.get("resolution_mode"),
                "resolution_reason": checkpoint_resolution.get("resolution_reason"),
                "source_model_rejected": checkpoint_resolution.get("source_model_rejected"),
                "available_checkpoints_count": checkpoint_resolution.get("available_checkpoints_count"),
                "candidate_count": checkpoint_resolution.get("candidate_count"),
                "candidates": checkpoint_resolution.get("candidates"),
                "preset": preset_name,
                "source_prompt": source_prompt,
                "translated_prompt": prompt_for_generation,
                "prompt_translation_policy": prompt_translation_policy,
                "subject_dominance": subject_dominance_value,
                "subject_dominance_rule": subject_dominance_rule,
                "negative_baseline": CHARACTER_PRODUCTION_NEGATIVE_BASELINE if preset_name == CHARACTER_PRODUCTION_PRESET else None,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "local_status": "생성 완료",
                "publish_status": "HermesWork publish 완료",
                "nas_status": nas_status,
                "slack_status": "primary image 준비됨",
                "workflow_path": str(bundle["workflow_path"]),
                "prompt_path": str(bundle["prompt_path"]),
                "metadata_path": str(bundle["metadata_path"]),
                "manifest_path": str(bundle["manifest_path"]),
                "integrity_path": str(bundle["integrity_path"]),
                "run_manifest_path": str(run_manifest_path) if run_manifest_path else None,
                "qualification_report_path": str(qualification_report_path) if qualification_report_path else None,
                "primary_image": bundle["primary_image"],
                "media_files": [str(bundle["primary_image_path"])],
                "sidecars": bundle["sidecars"],
                "artifact_path": str(bundle["primary_image_path"]),
                "artifact_files": [
                    str(bundle["primary_image_path"]),
                    str(bundle["workflow_path"]),
                    str(bundle["prompt_path"]),
                    str(bundle["metadata_path"]),
                    str(bundle["manifest_path"]),
                    str(bundle["integrity_path"]),
                ],
                "file_sha256": bundle.get("file_sha256"),
                "nas_hook_requested": bundle["nas_hook_requested"],
                "nas_evidence": bundle.get("nas_evidence"),
                "slack_upload_evidence": False,
                "output_source_origin": source_origin,
                "prompt_id": prompt_id,
                "category": category,
                "api_base_url": base_url,
            },
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(ComfyLocalImageGenProvider())
