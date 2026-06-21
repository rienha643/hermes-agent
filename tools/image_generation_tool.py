#!/usr/bin/env python3
"""
Image Generation Tools Module

Provides image generation via FAL.ai. Multiple FAL models are supported and
selectable via ``hermes tools`` → Image Generation; the active model is
persisted to ``image_gen.model`` in ``config.yaml``.

Architecture:
- ``FAL_MODELS`` is a catalog of supported models with per-model metadata
  (size-style family, defaults, ``supports`` whitelist, upscaler flag).
- ``_build_fal_payload()`` translates the agent's unified inputs (prompt +
  aspect_ratio) into the model-specific payload and filters to the
  ``supports`` whitelist so models never receive rejected keys.
- Upscaling via FAL's Clarity Upscaler is gated per-model via the ``upscale``
  flag — on for FLUX 2 Pro (backward-compat), off for all faster/newer models
  where upscaling would either hurt latency or add marginal quality.

Pricing shown in UI strings is as-of the initial commit; we accept drift and
update when it's noticed.
"""

import json
import logging
import os
import re
import datetime
import threading
import uuid
from typing import Any, Dict, Optional

# fal_client is imported lazily — see _load_fal_client(). Pulling it
# eagerly added ~64 ms to every CLI cold start because
# discover_builtin_tools() imports this module unconditionally during
# the registry walk, even when image generation is never used.
#
# Tests that monkeypatch this attribute (e.g.
# ``monkeypatch.setattr(image_tool, "fal_client", fake_fal_client)``)
# still work: _load_fal_client() short-circuits when the attribute is
# anything truthy, so a test-installed mock is not overwritten by a
# subsequent real import.
fal_client: Any = None

_IMAGE_TASK_METADATA_LOCK = threading.Lock()
_IMAGE_TASK_METADATA_BY_TASK_ID: Dict[str, Dict[str, Optional[str]]] = {}


def register_image_task_metadata(
    task_id: Optional[str],
    *,
    project_name: Optional[str] = None,
    artifact_name: Optional[str] = None,
) -> None:
    """Register delegated image routing hints for a task id."""
    task_key = str(task_id or "").strip()
    if not task_key:
        return
    payload = {
        "project_name": str(project_name).strip() if isinstance(project_name, str) and project_name.strip() else None,
        "artifact_name": str(artifact_name).strip() if isinstance(artifact_name, str) and artifact_name.strip() else None,
    }
    if not payload["project_name"] and not payload["artifact_name"]:
        return
    with _IMAGE_TASK_METADATA_LOCK:
        _IMAGE_TASK_METADATA_BY_TASK_ID[task_key] = payload


def _consume_image_task_metadata(task_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    task_key = str(task_id or "").strip()
    if not task_key:
        return None, None
    with _IMAGE_TASK_METADATA_LOCK:
        payload = _IMAGE_TASK_METADATA_BY_TASK_ID.pop(task_key, None)
    if not payload:
        return None, None
    return payload.get("project_name"), payload.get("artifact_name")


def _load_fal_client() -> Any:
    """Lazily import fal_client and rebind the module global on first use.

    Idempotent. Returns the (now-loaded) ``fal_client`` module reference.
    Skips the import if the global is already truthy — this preserves the
    test pattern of monkeypatching the module global to install a mock.
    """
    global fal_client
    if fal_client is not None:
        return fal_client
    from tools.fal_common import import_fal_client
    fal_client = import_fal_client()
    return fal_client


from tools.debug_helpers import DebugSession
from tools.fal_common import (
    _ManagedFalSyncClient,
    _extract_http_status,
    _normalize_fal_queue_url_format,  # noqa: F401 — re-exported for tests
)
from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import (
    fal_key_is_configured,
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    prefers_gateway,
)

logger = logging.getLogger(__name__)

# Single-pass guard for local Forge sessions: when a singular request has
# already produced one image in the current task, later image_generate calls
# reuse the same result instead of regenerating new candidates.
_FORGE_LOCAL_SINGLE_PASS_RESULTS: Dict[str, str] = {}
_FORGE_LOCAL_SINGLE_PASS_LOCK = threading.Lock()
_SINGLE_OUTPUT_TASKS: set[str] = set()
_SINGLE_OUTPUT_TASK_RESULTS: Dict[str, str] = {}
_SINGLE_OUTPUT_TASK_LOCK = threading.Lock()


def enable_single_output_task_mode(task_id: str) -> None:
    task = str(task_id or "").strip()
    if not task:
        return
    with _SINGLE_OUTPUT_TASK_LOCK:
        _SINGLE_OUTPUT_TASKS.add(task)


def disable_single_output_task_mode(task_id: str) -> None:
    task = str(task_id or "").strip()
    if not task:
        return
    with _SINGLE_OUTPUT_TASK_LOCK:
        _SINGLE_OUTPUT_TASKS.discard(task)


def _single_output_task_cached_result(task_id: str | None) -> str | None:
    task = str(task_id or "").strip()
    if not task:
        return None
    with _SINGLE_OUTPUT_TASK_LOCK:
        if task not in _SINGLE_OUTPUT_TASKS:
            return None
        return _SINGLE_OUTPUT_TASK_RESULTS.get(task)


def _store_single_output_task_result(task_id: str | None, result_json: str) -> None:
    task = str(task_id or "").strip()
    if not task:
        return
    with _SINGLE_OUTPUT_TASK_LOCK:
        if task in _SINGLE_OUTPUT_TASKS:
            _SINGLE_OUTPUT_TASK_RESULTS[task] = result_json


# ---------------------------------------------------------------------------
# FAL model catalog
# ---------------------------------------------------------------------------
#
# Each entry declares how to translate our unified inputs into the model's
# native payload shape. Size specification falls into three families:
#
#   "image_size_preset" — preset enum ("square_hd", "landscape_16_9", ...)
#                          used by the flux family, z-image, qwen, recraft,
#                          ideogram.
#   "aspect_ratio"      — aspect ratio enum ("16:9", "1:1", ...) used by
#                          nano-banana (Gemini).
#   "gpt_literal"       — literal dimension strings ("1024x1024", etc.)
#                          used by gpt-image-1.5.
#
# ``supports`` is a whitelist of keys allowed in the outgoing payload — any
# key outside this set is stripped before submission so models never receive
# rejected parameters (each FAL model rejects unknown keys differently).
#
# ``upscale`` controls whether to chain Clarity Upscaler after generation.

FAL_MODELS: Dict[str, Dict[str, Any]] = {
    "fal-ai/flux-2/klein/9b": {
        "display": "FLUX 2 Klein 9B",
        "speed": "<1s",
        "strengths": "Fast, crisp text",
        "price": "$0.006/MP",
        "size_style": "image_size_preset",
        "sizes": {
            "landscape": "landscape_16_9",
            "square": "square_hd",
            "portrait": "portrait_16_9",
        },
        "defaults": {
            "num_inference_steps": 4,
            "output_format": "png",
            "enable_safety_checker": False,
        },
        "supports": {
            "prompt", "image_size", "num_inference_steps", "seed",
            "output_format", "enable_safety_checker",
        },
        "upscale": False,
    },
    "fal-ai/flux-2-pro": {
        "display": "FLUX 2 Pro",
        "speed": "~6s",
        "strengths": "Studio photorealism",
        "price": "$0.03/MP",
        "size_style": "image_size_preset",
        "sizes": {
            "landscape": "landscape_16_9",
            "square": "square_hd",
            "portrait": "portrait_16_9",
        },
        "defaults": {
            "num_inference_steps": 50,
            "guidance_scale": 4.5,
            "num_images": 1,
            "output_format": "png",
            "enable_safety_checker": False,
            "safety_tolerance": "5",
            "sync_mode": True,
        },
        "supports": {
            "prompt", "image_size", "num_inference_steps", "guidance_scale",
            "num_images", "output_format", "enable_safety_checker",
            "safety_tolerance", "sync_mode", "seed",
        },
        "upscale": True,   # Backward-compat: current default behavior.
    },
    "fal-ai/z-image/turbo": {
        "display": "Z-Image Turbo",
        "speed": "~2s",
        "strengths": "Bilingual EN/CN, 6B",
        "price": "$0.005/MP",
        "size_style": "image_size_preset",
        "sizes": {
            "landscape": "landscape_16_9",
            "square": "square_hd",
            "portrait": "portrait_16_9",
        },
        "defaults": {
            "num_inference_steps": 8,
            "num_images": 1,
            "output_format": "png",
            "enable_safety_checker": False,
            "enable_prompt_expansion": False,  # avoid the extra per-request charge
        },
        "supports": {
            "prompt", "image_size", "num_inference_steps", "num_images",
            "seed", "output_format", "enable_safety_checker",
            "enable_prompt_expansion",
        },
        "upscale": False,
    },
    "fal-ai/nano-banana-pro": {
        "display": "Nano Banana Pro (Gemini 3 Pro Image)",
        "speed": "~8s",
        "strengths": "Gemini 3 Pro, reasoning depth, text rendering",
        "price": "$0.15/image (1K)",
        "size_style": "aspect_ratio",
        "sizes": {
            "landscape": "16:9",
            "square": "1:1",
            "portrait": "9:16",
        },
        "defaults": {
            "num_images": 1,
            "output_format": "png",
            "safety_tolerance": "5",
            # "1K" is the cheapest tier; 4K doubles the per-image cost.
            # Users on Nous Subscription should stay at 1K for predictable billing.
            "resolution": "1K",
        },
        "supports": {
            "prompt", "aspect_ratio", "num_images", "output_format",
            "safety_tolerance", "seed", "sync_mode", "resolution",
            "enable_web_search", "limit_generations",
        },
        "upscale": False,
    },
    "fal-ai/gpt-image-1.5": {
        "display": "GPT Image 1.5",
        "speed": "~15s",
        "strengths": "Prompt adherence",
        "price": "$0.034/image",
        "size_style": "gpt_literal",
        "sizes": {
            "landscape": "1536x1024",
            "square": "1024x1024",
            "portrait": "1024x1536",
        },
        "defaults": {
            # Quality is pinned to medium to keep portal billing predictable
            # across all users (low is too rough, high is 4-6x more expensive).
            "quality": "medium",
            "num_images": 1,
            "output_format": "png",
        },
        "supports": {
            "prompt", "image_size", "quality", "num_images", "output_format",
            "background", "sync_mode",
        },
        "upscale": False,
    },
    "fal-ai/gpt-image-2": {
        "display": "GPT Image 2",
        "speed": "~20s",
        "strengths": "SOTA text rendering + CJK, world-aware photorealism",
        "price": "$0.04–0.06/image",
        # GPT Image 2 uses FAL's standard preset enum (unlike 1.5's literal
        # dimensions). We map to the 4:3 variants — the 16:9 presets
        # (1024x576) fall below GPT-Image-2's 655,360 min-pixel requirement
        # and would be rejected. 4:3 keeps us above the minimum on all
        # three aspect ratios.
        "size_style": "image_size_preset",
        "sizes": {
            "landscape": "landscape_4_3",   # 1024x768
            "square": "square_hd",            # 1024x1024
            "portrait": "portrait_4_3",       # 768x1024
        },
        "defaults": {
            # Same quality pinning as gpt-image-1.5: medium keeps Nous
            # Portal billing predictable. "high" is 3-4x the per-image
            # cost at the same size; "low" is too rough for production use.
            "quality": "medium",
            "num_images": 1,
            "output_format": "png",
        },
        "supports": {
            "prompt", "image_size", "quality", "num_images", "output_format",
            "sync_mode",
            # openai_api_key (BYOK) intentionally omitted — all users go
            # through the shared FAL billing path.
        },
        "upscale": False,
    },
    "fal-ai/ideogram/v3": {
        "display": "Ideogram V3",
        "speed": "~5s",
        "strengths": "Best typography",
        "price": "$0.03-0.09/image",
        "size_style": "image_size_preset",
        "sizes": {
            "landscape": "landscape_16_9",
            "square": "square_hd",
            "portrait": "portrait_16_9",
        },
        "defaults": {
            "rendering_speed": "BALANCED",
            "expand_prompt": True,
            "style": "AUTO",
        },
        "supports": {
            "prompt", "image_size", "rendering_speed", "expand_prompt",
            "style", "seed",
        },
        "upscale": False,
    },
    "fal-ai/recraft/v4/pro/text-to-image": {
        "display": "Recraft V4 Pro",
        "speed": "~8s",
        "strengths": "Design, brand systems, production-ready",
        "price": "$0.25/image",
        "size_style": "image_size_preset",
        "sizes": {
            "landscape": "landscape_16_9",
            "square": "square_hd",
            "portrait": "portrait_16_9",
        },
        "defaults": {
            # V4 Pro dropped V3's required `style` enum — defaults handle taste now.
            "enable_safety_checker": False,
        },
        "supports": {
            "prompt", "image_size", "enable_safety_checker",
            "colors", "background_color",
        },
        "upscale": False,
    },
    "fal-ai/qwen-image": {
        "display": "Qwen Image",
        "speed": "~12s",
        "strengths": "LLM-based, complex text",
        "price": "$0.02/MP",
        "size_style": "image_size_preset",
        "sizes": {
            "landscape": "landscape_16_9",
            "square": "square_hd",
            "portrait": "portrait_16_9",
        },
        "defaults": {
            "num_inference_steps": 30,
            "guidance_scale": 2.5,
            "num_images": 1,
            "output_format": "png",
            "acceleration": "regular",
        },
        "supports": {
            "prompt", "image_size", "num_inference_steps", "guidance_scale",
            "num_images", "output_format", "acceleration", "seed", "sync_mode",
        },
        "upscale": False,
    },
    # Krea 2 — Krea's first foundation image model, day-0 partner launch on
    # fal (2026-05-27). Same model family as our direct ``plugins/image_gen/krea``
    # backend, exposed here for users who prefer to bill through their
    # existing FAL key / Nous Portal subscription rather than register
    # directly with Krea.  Both variants share the same parameter schema —
    # only model id, price, and recommended use case differ.
    "fal-ai/krea/v2/medium/text-to-image": {
        "display": "Krea 2 Medium",
        "speed": "~15-25s",
        "strengths": "Illustration, anime, painting, expressive/artistic styles",
        "price": "$0.030 (text) / $0.035 (style refs)",
        "size_style": "aspect_ratio",
        # Krea natively accepts 1:1, 4:3, 3:2, 16:9, 2.35:1, 4:5, 2:3, 9:16 —
        # we map our 3 abstract ratios to the closest match.
        "sizes": {
            "landscape": "16:9",
            "square": "1:1",
            "portrait": "9:16",
        },
        "defaults": {
            "creativity": "medium",
        },
        "supports": {
            "prompt", "aspect_ratio", "creativity", "seed",
            "image_style_references",
        },
        "upscale": False,
    },
    "fal-ai/krea/v2/large/text-to-image": {
        "display": "Krea 2 Large",
        "speed": "~25-60s",
        "strengths": "Photorealism, raw textured looks (motion blur, grain, film)",
        "price": "$0.060 (text) / $0.065 (style refs)",
        "size_style": "aspect_ratio",
        "sizes": {
            "landscape": "16:9",
            "square": "1:1",
            "portrait": "9:16",
        },
        "defaults": {
            "creativity": "medium",
        },
        "supports": {
            "prompt", "aspect_ratio", "creativity", "seed",
            "image_style_references",
        },
        "upscale": False,
    },
}

# Default model is the fastest reasonable option. Kept cheap and sub-1s.
DEFAULT_MODEL = "fal-ai/flux-2/klein/9b"

DEFAULT_ASPECT_RATIO = "landscape"
VALID_ASPECT_RATIOS = ("landscape", "square", "portrait")


# ---------------------------------------------------------------------------
# Upscaler (Clarity Upscaler — unchanged from previous implementation)
# ---------------------------------------------------------------------------
UPSCALER_MODEL = "fal-ai/clarity-upscaler"
UPSCALER_FACTOR = 2
UPSCALER_SAFETY_CHECKER = False
UPSCALER_DEFAULT_PROMPT = "masterpiece, best quality, highres"
UPSCALER_NEGATIVE_PROMPT = "(worst quality, low quality, normal quality:2)"
UPSCALER_CREATIVITY = 0.35
UPSCALER_RESEMBLANCE = 0.6
UPSCALER_GUIDANCE_SCALE = 4
UPSCALER_NUM_INFERENCE_STEPS = 18


_debug = DebugSession("image_tools", env_var="IMAGE_TOOLS_DEBUG")
_managed_fal_client = None
_managed_fal_client_config = None
_managed_fal_client_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Managed FAL gateway (Nous Subscription)
# ---------------------------------------------------------------------------
def _resolve_managed_fal_gateway():
    """Return managed fal-queue gateway config when the user prefers the gateway
    or direct FAL credentials are absent."""
    if fal_key_is_configured() and not prefers_gateway("image_gen"):
        return None
    return resolve_managed_tool_gateway("fal-queue")


def _get_managed_fal_client(managed_gateway):
    """Reuse the managed FAL client so its internal httpx.Client is not leaked per call."""
    global _managed_fal_client, _managed_fal_client_config

    client_config = (
        managed_gateway.gateway_origin.rstrip("/"),
        managed_gateway.nous_user_token,
    )
    with _managed_fal_client_lock:
        if _managed_fal_client is not None and _managed_fal_client_config == client_config:
            return _managed_fal_client

        # Resolve fal_client on the legacy module — preserves the test
        # pattern of monkey-patching ``image_generation_tool.fal_client``.
        _load_fal_client()
        _managed_fal_client = _ManagedFalSyncClient(
            fal_client,
            key=managed_gateway.nous_user_token,
            queue_run_origin=managed_gateway.gateway_origin,
        )
        _managed_fal_client_config = client_config
        return _managed_fal_client


def _submit_fal_request(model: str, arguments: Dict[str, Any]):
    """Submit a FAL request using direct credentials or the managed queue gateway."""
    # Trigger the lazy import on first call. Idempotent.
    _load_fal_client()
    request_headers = {"x-idempotency-key": str(uuid.uuid4())}
    managed_gateway = _resolve_managed_fal_gateway()
    if managed_gateway is None:
        return fal_client.submit(model, arguments=arguments, headers=request_headers)

    managed_client = _get_managed_fal_client(managed_gateway)
    try:
        return managed_client.submit(
            model,
            arguments=arguments,
            headers=request_headers,
        )
    except Exception as exc:
        # 4xx from the managed gateway typically means the portal doesn't
        # currently proxy this model (allowlist miss, billing gate, etc.)
        # — surface a clearer message with actionable remediation instead
        # of a raw HTTP error from httpx.
        status = _extract_http_status(exc)
        if status is not None and 400 <= status < 500:
            gateway_message = ""
            if status in {401, 402, 403}:
                gateway_message = (
                    "\n\n"
                    + nous_tool_gateway_unavailable_message(
                        "managed FAL image generation",
                        force_fresh=True,
                    )
                )
            raise ValueError(
                f"Nous Subscription gateway rejected model '{model}' "
                f"(HTTP {status}). This model may not yet be enabled on "
                f"the Nous Portal's FAL proxy. Either:\n"
                f"  • Set FAL_KEY in your environment to use FAL.ai directly, or\n"
                f"  • Pick a different model via `hermes tools` → Image Generation."
                f"{gateway_message}"
            ) from exc
        raise


# ---------------------------------------------------------------------------
# Model resolution + payload construction
# ---------------------------------------------------------------------------
def _resolve_fal_model() -> tuple:
    """Resolve the active FAL model from config.yaml (primary) or default.

    Returns (model_id, metadata_dict). Falls back to DEFAULT_MODEL if the
    configured model is unknown (logged as a warning).
    """
    model_id = ""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        img_cfg = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(img_cfg, dict):
            raw = img_cfg.get("model")
            if isinstance(raw, str):
                model_id = raw.strip()
    except Exception as exc:
        logger.debug("Could not load image_gen.model from config: %s", exc)

    # Env var escape hatch (undocumented; backward-compat for tests/scripts).
    if not model_id:
        model_id = os.getenv("FAL_IMAGE_MODEL", "").strip()

    if not model_id:
        return DEFAULT_MODEL, FAL_MODELS[DEFAULT_MODEL]

    if model_id not in FAL_MODELS:
        logger.warning(
            "Unknown FAL model '%s' in config; falling back to %s",
            model_id, DEFAULT_MODEL,
        )
        return DEFAULT_MODEL, FAL_MODELS[DEFAULT_MODEL]

    return model_id, FAL_MODELS[model_id]


def _build_fal_payload(
    model_id: str,
    prompt: str,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    seed: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a FAL request payload for `model_id` from unified inputs.

    Translates aspect_ratio into the model's native size spec (preset enum,
    aspect-ratio enum, or GPT literal string), merges model defaults, applies
    caller overrides, then filters to the model's ``supports`` whitelist.
    """
    meta = FAL_MODELS[model_id]
    size_style = meta["size_style"]
    sizes = meta["sizes"]

    aspect = (aspect_ratio or DEFAULT_ASPECT_RATIO).lower().strip()
    if aspect not in sizes:
        aspect = DEFAULT_ASPECT_RATIO

    payload: Dict[str, Any] = dict(meta.get("defaults", {}))
    payload["prompt"] = (prompt or "").strip()

    if size_style in {"image_size_preset", "gpt_literal"}:
        payload["image_size"] = sizes[aspect]
    elif size_style == "aspect_ratio":
        payload["aspect_ratio"] = sizes[aspect]
    else:
        raise ValueError(f"Unknown size_style: {size_style!r}")

    if seed is not None and isinstance(seed, int):
        payload["seed"] = seed

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                payload[k] = v

    supports = meta["supports"]
    return {k: v for k, v in payload.items() if k in supports}


# ---------------------------------------------------------------------------
# Upscaler
# ---------------------------------------------------------------------------
def _upscale_image(image_url: str, original_prompt: str) -> Optional[Dict[str, Any]]:
    """Upscale an image using FAL.ai's Clarity Upscaler.

    Returns upscaled image dict, or None on failure (caller falls back to
    the original image).
    """
    try:
        logger.info("Upscaling image with Clarity Upscaler...")

        upscaler_arguments = {
            "image_url": image_url,
            "prompt": f"{UPSCALER_DEFAULT_PROMPT}, {original_prompt}",
            "upscale_factor": UPSCALER_FACTOR,
            "negative_prompt": UPSCALER_NEGATIVE_PROMPT,
            "creativity": UPSCALER_CREATIVITY,
            "resemblance": UPSCALER_RESEMBLANCE,
            "guidance_scale": UPSCALER_GUIDANCE_SCALE,
            "num_inference_steps": UPSCALER_NUM_INFERENCE_STEPS,
            "enable_safety_checker": UPSCALER_SAFETY_CHECKER,
        }

        handler = _submit_fal_request(UPSCALER_MODEL, arguments=upscaler_arguments)
        result = handler.get()

        if result and "image" in result:
            upscaled_image = result["image"]
            logger.info(
                "Image upscaled successfully to %sx%s",
                upscaled_image.get("width", "unknown"),
                upscaled_image.get("height", "unknown"),
            )
            return {
                "url": upscaled_image["url"],
                "width": upscaled_image.get("width", 0),
                "height": upscaled_image.get("height", 0),
                "upscaled": True,
                "upscale_factor": UPSCALER_FACTOR,
            }
        logger.error("Upscaler returned invalid response")
        return None

    except Exception as e:
        logger.error("Error upscaling image: %s", e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------
def image_generate_tool(
    prompt: str,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_images: Optional[int] = None,
    output_format: Optional[str] = None,
    seed: Optional[int] = None,
    project_name: Optional[str] = None,
    artifact_name: Optional[str] = None,
) -> str:
    """Generate an image from a text prompt using the configured FAL model.

    The agent-facing schema exposes only ``prompt`` and ``aspect_ratio``; the
    remaining kwargs are overrides for direct Python callers and are filtered
    per-model via the ``supports`` whitelist (unsupported overrides are
    silently dropped so legacy callers don't break when switching models).

    Returns a JSON string with ``{"success": bool, "image": url | None,
    "error": str, "error_type": str}``.
    """
    model_id, meta = _resolve_fal_model()

    debug_call_data = {
        "model": model_id,
        "parameters": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "output_format": output_format,
            "seed": seed,
            "project_name": project_name,
            "artifact_name": artifact_name,
        },
        "error": None,
        "success": False,
        "images_generated": 0,
        "generation_time": 0,
    }

    start_time = datetime.datetime.now()

    try:
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt is required and must be a non-empty string")

        if not (fal_key_is_configured() or _resolve_managed_fal_gateway()):
            raise ValueError(_build_no_backend_setup_message())

        aspect_lc = (aspect_ratio or DEFAULT_ASPECT_RATIO).lower().strip()
        if aspect_lc not in VALID_ASPECT_RATIOS:
            logger.warning(
                "Invalid aspect_ratio '%s', defaulting to '%s'",
                aspect_ratio, DEFAULT_ASPECT_RATIO,
            )
            aspect_lc = DEFAULT_ASPECT_RATIO

        project_name, artifact_name = _infer_image_project_metadata(
            prompt,
            project_name=project_name,
            artifact_name=artifact_name,
        )

        overrides: Dict[str, Any] = {}
        if num_inference_steps is not None:
            overrides["num_inference_steps"] = num_inference_steps
        if guidance_scale is not None:
            overrides["guidance_scale"] = guidance_scale
        if num_images is not None:
            overrides["num_images"] = num_images
        if output_format is not None:
            overrides["output_format"] = output_format

        arguments = _build_fal_payload(
            model_id, prompt, aspect_lc, seed=seed, overrides=overrides,
        )

        logger.info(
            "Generating image with %s (%s) — prompt: %s",
            meta.get("display", model_id), model_id, prompt[:80],
        )

        handler = _submit_fal_request(model_id, arguments=arguments)
        result = handler.get()

        generation_time = (datetime.datetime.now() - start_time).total_seconds()

        if not result or "images" not in result:
            raise ValueError("Invalid response from FAL.ai API — no images returned")

        images = result.get("images", [])
        if not images:
            raise ValueError("No images were generated")

        should_upscale = bool(meta.get("upscale", False))

        formatted_images = []
        for img in images:
            if not (isinstance(img, dict) and "url" in img):
                continue
            original_image = {
                "url": img["url"],
                "width": img.get("width", 0),
                "height": img.get("height", 0),
            }

            if should_upscale:
                upscaled_image = _upscale_image(img["url"], prompt.strip())
                if upscaled_image:
                    formatted_images.append(upscaled_image)
                    continue
                logger.warning("Using original image as fallback (upscale failed)")

            original_image["upscaled"] = False
            formatted_images.append(original_image)

        if not formatted_images:
            raise ValueError("No valid image URLs returned from API")

        upscaled_count = sum(1 for img in formatted_images if img.get("upscaled"))
        logger.info(
            "Generated %s image(s) in %.1fs (%s upscaled) via %s",
            len(formatted_images), generation_time, upscaled_count, model_id,
        )

        response_data = {
            "success": True,
            "image": formatted_images[0]["url"] if formatted_images else None,
        }

        debug_call_data["success"] = True
        debug_call_data["images_generated"] = len(formatted_images)
        debug_call_data["generation_time"] = generation_time
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()

        return json.dumps(response_data, indent=2, ensure_ascii=False)

    except Exception as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        error_msg = f"Error generating image: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)

        response_data = {
            "success": False,
            "image": None,
            "error": str(e),
            "error_type": type(e).__name__,
        }

        debug_call_data["error"] = error_msg
        debug_call_data["generation_time"] = generation_time
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()

        return json.dumps(response_data, indent=2, ensure_ascii=False)


def check_fal_api_key() -> bool:
    """True if the FAL.ai API key (direct or managed gateway) is available."""
    return bool(fal_key_is_configured() or _resolve_managed_fal_gateway())


def _build_no_backend_setup_message() -> str:
    """Build an actionable error string when no FAL backend is reachable.

    Used by the in-tree FAL path. Mentions:
      - FAL_KEY signup link
      - managed-gateway status (if Nous tools are enabled)
      - plugin alternative pointer (so users on a stale ``image_gen.provider``
        know the registry exists and how to inspect it)
    """
    lines = ["Image generation is unavailable in this environment.", ""]
    lines.append("Missing requirements:")
    if managed_nous_tools_enabled():
        lines.append(
            "  - FAL_KEY is not set and the managed FAL gateway is unreachable"
        )
    else:
        lines.append("  - FAL_KEY environment variable is not set")
        gateway_message = nous_tool_gateway_unavailable_message(
            "managed FAL image generation",
        )
        if gateway_message:
            lines.append(f"  - {gateway_message}")
    lines.append("")
    lines.append("To enable image generation, do one of:")
    lines.append(
        "  1. Get a free API key at https://fal.ai and set "
        "FAL_KEY=<your-key> (then restart the session)"
    )
    if managed_nous_tools_enabled():
        lines.append(
            "  2. Sign in to a Nous account that has the managed FAL "
            "gateway enabled (`hermes setup`)"
        )
    lines.append(
        "  3. Configure a different image_gen provider via `hermes tools` "
        "→ Image Generation (run `hermes plugins list` to see installed "
        "backends)"
    )
    return "\n".join(lines)


def _is_openai_image_provider(configured_provider: Optional[str]) -> bool:
    """Return True when the configured provider is an explicit OpenAI route.

    The OpenAI and OpenAI Codex image backends are remote HTTP paths, so we
    can skip the local HTTP backend availability probe when either one is
    explicitly selected in config.
    """
    if not isinstance(configured_provider, str):
        return False
    normalized = configured_provider.strip().lower()
    return normalized == "openai" or normalized.startswith("openai-codex")


def check_image_generation_requirements() -> bool:
    """True if any image gen backend is available.

    Providers are considered in this order:

    1. The in-tree FAL backend (FAL_KEY or managed gateway).
    2. Any plugin-registered provider whose ``is_available()`` returns True.

    Plugins win only when the in-tree FAL path is NOT ready, which matches
    the historical behavior: shipping hermes with a FAL key configured
    should still expose the tool. The active selection among ready
    providers is resolved per-call by ``image_gen.provider``.
    """
    configured_provider = _read_configured_image_provider()
    if _is_openai_image_provider(configured_provider):
        return True

    try:
        if check_fal_api_key():
            # Trigger the lazy fal_client import here as the SDK presence
            # check. Raises ImportError if the optional ``fal-client``
            # package isn't installed; the caller's except ImportError
            # below catches that and continues to plugin probing.
            _load_fal_client()
            return True
    except ImportError:
        pass

    # Probe plugin providers. Discovery is idempotent and cheap.
    try:
        from agent.image_gen_registry import list_providers
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        for provider in list_providers():
            try:
                if provider.is_available():
                    return True
            except Exception:
                continue
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Demo / CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🎨 Image Generation Tools — FAL.ai multi-model support")
    print("=" * 60)

    if not check_fal_api_key():
        print("❌ FAL_KEY environment variable not set")
        print("   Set it via: export FAL_KEY='your-key-here'")
        print("   Get a key: https://fal.ai/")
        raise SystemExit(1)
    print("✅ FAL.ai API key found")

    try:
        import fal_client  # noqa: F401
        print("✅ fal_client library available")
    except ImportError:
        print("❌ fal_client library not found — pip install fal-client")
        raise SystemExit(1)

    model_id, meta = _resolve_fal_model()
    print(f"🤖 Active model: {meta.get('display', model_id)} ({model_id})")
    print(f"   Speed: {meta.get('speed', '?')}  ·  Price: {meta.get('price', '?')}")
    print(f"   Upscaler: {'on' if meta.get('upscale') else 'off'}")

    print("\nAvailable models:")
    for mid, m in FAL_MODELS.items():
        marker = " ← active" if mid == model_id else ""
        print(f"  {mid:<32}  {m.get('speed', '?'):<6}  {m.get('price', '?')}{marker}")

    if _debug.active:
        print(f"\n🐛 Debug mode enabled — session {_debug.session_id}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

IMAGE_GENERATE_SCHEMA = {
    "name": "image_generate",
    "description": (
        "Generate high-quality images from text prompts. The underlying "
        "backend (FAL, OpenAI, etc.) and model are user-configured and not "
        "selectable by the agent. Returns either a URL or an absolute file "
        "path in the `image` field; display it with markdown "
        "![description](url-or-path) and the gateway will deliver it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text prompt describing the desired image. Be detailed and descriptive.",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(VALID_ASPECT_RATIOS),
                "description": "The aspect ratio of the generated image. 'landscape' is 16:9 wide, 'portrait' is 16:9 tall, 'square' is 1:1.",
                "default": DEFAULT_ASPECT_RATIO,
            },
            "project_name": {
                "type": "string",
                "description": "Optional project name used to publish the image into the matching HermesWork/Image project folder.",
            },
            "artifact_name": {
                "type": "string",
                "description": "Optional artifact base name used for the published filename.",
            },
            "negative_prompt": {
                "type": "string",
                "description": "Optional negative prompt. Required by the ComfyUI character_production preset and merged with its negative baseline when used.",
            },
            "subject_dominance": {
                "type": "number",
                "description": "Optional subject-dominance score (0-1 or 0-100). Required by the ComfyUI character_production preset.",
            },
            "width": {
                "type": "integer",
                "description": "Optional explicit output width in pixels. When set with height, provider defaults such as square 1024x1024 must not override it.",
            },
            "height": {
                "type": "integer",
                "description": "Optional explicit output height in pixels. When set with width, provider defaults such as square 1024x1024 must not override it.",
            },
            "live_generation_approved": {
                "type": "boolean",
                "description": (
                    "Operator approval flag for providers that require explicit live-generation approval, "
                    "such as NovelAI. Default is false. If the current user/operator message explicitly "
                    "approves live generation or directly asks the image worker to generate/run/proceed, "
                    "pass true for that approved request; do not require the user to name this internal flag."
                ),
                "default": False,
            },
            "high_res_approved": {
                "type": "boolean",
                "description": "Operator approval flag for high-resolution provider requests. Default is false.",
                "default": False,
            },
        },
        "required": ["prompt"],
    },
}


def _infer_image_project_metadata(
    prompt: str,
    project_name: Optional[str] = None,
    artifact_name: Optional[str] = None,
) -> tuple[str | None, str | None]:
    """Best-effort project routing for image worker calls.

    The worker path may omit project metadata entirely. When that happens, we
    infer the project and artifact names from the prompt using the minimal
    rules requested by the image routing workflow.
    """
    prompt_text = str(prompt or "").strip()
    resolved_project = str(project_name).strip() if isinstance(project_name, str) else ""
    resolved_artifact = str(artifact_name).strip() if isinstance(artifact_name, str) else ""

    if not resolved_project and re.search(r"(?<!\w)망각구역(?!\w)", prompt_text):
        resolved_project = "망각구역"

    if not resolved_artifact:
        if re.search(r"(?<!\w)주인공(?!\w)", prompt_text):
            resolved_artifact = "주인공"
        elif re.search(r"검증용\s*이미지", prompt_text):
            resolved_artifact = "검증용이미지"
        else:
            phrase = None
            image_phrase = re.search(r"(.+?)\s*(?:이미지|image)\b", prompt_text, re.IGNORECASE | re.DOTALL)
            if image_phrase:
                phrase = image_phrase.group(1).strip()
                phrase = re.sub(
                    r"(?:컨셉|concept|그림|illustration|render|이미지|image)\s*$",
                    "",
                    phrase,
                    flags=re.IGNORECASE,
                )
            if phrase:
                tokens = [token for token in re.split(r"[\s/_-]+", phrase) if token]
                for token in reversed(tokens):
                    lowered = token.lower()
                    if lowered in {"컨셉", "concept", "그림", "illustration", "render", "이미지", "image"}:
                        continue
                    resolved_artifact = token
                    break
            if not resolved_artifact:
                resolved_artifact = "이미지"

    return (resolved_project or None, resolved_artifact or None)


def _read_configured_image_model():
    """Return the value of ``image_gen.model`` from config.yaml, or None."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            value = section.get("model")
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception as exc:
        logger.debug("Could not read image_gen.model: %s", exc)
    return None


def _read_configured_image_provider():
    """Return the value of ``image_gen.provider`` from config.yaml, or None.

    We only consult the plugin registry when this is explicitly set — an
    unset value keeps users on the in-tree FAL fallback even when other
    providers happen to be registered (e.g. a user has OPENAI_API_KEY set
    for other features but never asked for OpenAI image gen). ``"fal"``
    explicitly routes through ``plugins/image_gen/fal/`` (which delegates
    back into this module's pipeline via call-time indirection — see
    issue #26241).
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            value = section.get("provider")
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception as exc:
        logger.debug("Could not read image_gen.provider: %s", exc)
    return None


def _dispatch_to_plugin_provider(
    prompt: str,
    aspect_ratio: str,
    task_id: str | None = None,
    project_name: str | None = None,
    artifact_name: str | None = None,
    negative_prompt: str | None = None,
    subject_dominance: float | int | None = None,
    width: int | None = None,
    height: int | None = None,
    live_generation_approved: bool | None = None,
    high_res_approved: bool | None = None,
):
    """Route the call to a plugin-registered provider when one is selected.

    Returns a JSON string on dispatch, or ``None`` to fall through to the
    in-tree FAL fallback in ``image_generate_tool``.

    Dispatch fires when ``image_gen.provider`` is explicitly set — including
    ``"fal"`` itself, which now resolves to the
    ``plugins/image_gen/fal/`` plugin (the plugin re-enters this module's
    pipeline via ``_it`` indirection so behavior is identical to the
    direct call, just routed through the registry).
    """
    configured = _read_configured_image_provider()
    if not configured:
        return None

    cached_single_output = _single_output_task_cached_result(task_id)
    if cached_single_output:
        logger.info(
            "single-output image guard: reusing cached image for task_id=%s",
            task_id,
        )
        return cached_single_output

    # Local Forge singular requests are single-pass: once a task has already
    # produced one image, return the same result on later calls rather than
    # generating new candidates.
    cache_key = None
    if configured == "forge-local" and isinstance(task_id, str) and task_id.strip():
        cache_key = task_id.strip()
        with _FORGE_LOCAL_SINGLE_PASS_LOCK:
            cached = _FORGE_LOCAL_SINGLE_PASS_RESULTS.get(cache_key)
        if cached:
            logger.info(
                "forge-local single-pass guard: reusing cached image for task_id=%s",
                cache_key,
            )
            return cached

    # Also read configured model so we can pass it to the plugin
    configured_model = _read_configured_image_model()

    try:
        # Import locally so plugin discovery isn't triggered just by
        # importing this module (tests rely on that).
        from agent.image_gen_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        provider = get_provider(configured)
    except Exception as exc:
        logger.debug("image_gen plugin dispatch skipped: %s", exc)
        return None

    if provider is None:
        try:
            # Long-lived sessions may have discovered plugins before a bundled
            # backend was patched in or before config changed. Retry once with
            # a forced refresh before surfacing a missing-provider error.
            _ensure_plugins_discovered(force=True)
            provider = get_provider(configured)
        except Exception as exc:
            logger.debug("image_gen plugin force-refresh skipped: %s", exc)

    if provider is None:
        return json.dumps({
            "success": False,
            "image": None,
            "error": (
                f"image_gen.provider='{configured}' is set but no plugin "
                f"registered that name. Run `hermes plugins list` to see "
                f"available image gen backends."
            ),
            "error_type": "provider_not_registered",
        })

    try:
        kwargs: Dict[str, Any] = {"prompt": prompt, "aspect_ratio": aspect_ratio}
        if project_name is not None:
            kwargs["project_name"] = project_name
        if artifact_name is not None:
            kwargs["artifact_name"] = artifact_name
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        if subject_dominance is not None:
            kwargs["subject_dominance"] = subject_dominance
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if live_generation_approved is not None:
            kwargs["live_generation_approved"] = live_generation_approved
        if high_res_approved is not None:
            kwargs["high_res_approved"] = high_res_approved
        if configured_model:
            kwargs["model"] = configured_model
        result = provider.generate(**kwargs)
    except Exception as exc:
        logger.warning(
            "Image gen provider '%s' raised: %s",
            getattr(provider, "name", "?"), exc,
        )
        return json.dumps({
            "success": False,
            "image": None,
            "error": f"Provider '{getattr(provider, 'name', '?')}' error: {exc}",
            "error_type": "provider_exception",
        })
    if not isinstance(result, dict):
        return json.dumps({
            "success": False,
            "image": None,
            "error": "Provider returned a non-dict result",
            "error_type": "provider_contract",
        })

    result_json = json.dumps(result)
    if result.get("success") and result.get("image"):
        _store_single_output_task_result(task_id, result_json)
    if configured == "forge-local" and cache_key and result.get("success") and result.get("image"):
        with _FORGE_LOCAL_SINGLE_PASS_LOCK:
            _FORGE_LOCAL_SINGLE_PASS_RESULTS[cache_key] = result_json
    return result_json


def _handle_image_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return tool_error("prompt is required for image generation")
    aspect_ratio = args.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
    project_name = args.get("project_name")
    artifact_name = args.get("artifact_name")
    negative_prompt = args.get("negative_prompt")
    subject_dominance = args.get("subject_dominance")
    width = args.get("width")
    height = args.get("height")
    live_generation_approved = args.get("live_generation_approved")
    high_res_approved = args.get("high_res_approved")
    task_id = kw.get("task_id")

    task_project_name, task_artifact_name = _consume_image_task_metadata(task_id)
    if task_project_name and not project_name:
        project_name = task_project_name
    if task_artifact_name and not artifact_name:
        artifact_name = task_artifact_name

    project_name, artifact_name = _infer_image_project_metadata(
        prompt,
        project_name=project_name,
        artifact_name=artifact_name,
    )

    # Route to a plugin-registered provider if one is active (and it's
    # not the in-tree FAL path).
    dispatched = _dispatch_to_plugin_provider(
        prompt,
        aspect_ratio,
        task_id=task_id,
        project_name=project_name,
        artifact_name=artifact_name,
        negative_prompt=str(negative_prompt) if isinstance(negative_prompt, str) and negative_prompt.strip() else None,
        subject_dominance=subject_dominance if isinstance(subject_dominance, (int, float)) else None,
        width=width if isinstance(width, int) and width > 0 else None,
        height=height if isinstance(height, int) and height > 0 else None,
        live_generation_approved=live_generation_approved if isinstance(live_generation_approved, bool) else None,
        high_res_approved=high_res_approved if isinstance(high_res_approved, bool) else None,
    )
    if dispatched is not None:
        return dispatched

    return image_generate_tool(
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        project_name=project_name,
        artifact_name=artifact_name,
    )


registry.register(
    name="image_generate",
    toolset="image_gen",
    schema=IMAGE_GENERATE_SCHEMA,
    handler=_handle_image_generate,
    check_fn=check_image_generation_requirements,
    requires_env=[],
    is_async=False,   # sync fal_client API to avoid "Event loop is closed" in gateway
    emoji="🎨",
)
