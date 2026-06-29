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
_IMAGE_TASK_METADATA_BY_TASK_ID: Dict[str, Dict[str, Any]] = {}

_COMMANDER_IMAGE_ARG_KEYS = {
    "operation",
    "workflow_key",
    "reference_image_path",
    "reference_image",
    "source_image_path",
    "source_image",
    "project_name",
    "artifact_name",
    "output_type",
    "prompt",
    "negative_prompt",
    "live_generation_approved",
    "experimental_reference_identity",
    "allow_reference_workflow_family_change",
    "style_preset",
    "lora_preset",
    "loras",
    "vae",
    "model",
    "checkpoint",
    "aspect_ratio",
    "width",
    "height",
    "steps",
    "cfg_scale",
    "sampler_name",
    "scheduler",
    "seed",
    "denoise",
    "postprocess_preset",
    "upscale_model",
    "mask_target",
    "mask_source",
    "mask_box",
    "mask_feather_px",
    "grow_mask_by",
    "sam3_positive_coords",
    "sam3_negative_coords",
    "sam3_threshold",
    "sam3_refine_iterations",
    "sam3_detail_denoise",
}


def register_image_task_metadata(
    task_id: Optional[str],
    *,
    project_name: Optional[str] = None,
    artifact_name: Optional[str] = None,
    image_args: Optional[Dict[str, Any]] = None,
) -> None:
    """Register delegated image routing hints for a task id."""
    task_key = str(task_id or "").strip()
    if not task_key:
        return
    payload = {
        "project_name": str(project_name).strip() if isinstance(project_name, str) and project_name.strip() else None,
        "artifact_name": str(artifact_name).strip() if isinstance(artifact_name, str) and artifact_name.strip() else None,
    }
    if isinstance(image_args, dict):
        filtered_args = {
            key: value
            for key, value in image_args.items()
            if key in _COMMANDER_IMAGE_ARG_KEYS and value not in (None, "", [], {})
        }
        if filtered_args:
            payload["image_args"] = filtered_args
    if not payload["project_name"] and not payload["artifact_name"] and not payload.get("image_args"):
        return
    with _IMAGE_TASK_METADATA_LOCK:
        _IMAGE_TASK_METADATA_BY_TASK_ID[task_key] = payload


def _consume_image_task_metadata(task_id: Optional[str]) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
    task_key = str(task_id or "").strip()
    if not task_key:
        return None, None, {}
    with _IMAGE_TASK_METADATA_LOCK:
        payload = _IMAGE_TASK_METADATA_BY_TASK_ID.pop(task_key, None)
    if not payload:
        return None, None, {}
    image_args = payload.get("image_args")
    return (
        payload.get("project_name"),
        payload.get("artifact_name"),
        image_args if isinstance(image_args, dict) else {},
    )


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
    output_type: Optional[str] = None,
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
            "output_type": output_type,
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
            "model": {
                "type": "string",
                "description": "Optional provider-specific model/checkpoint name. For ComfyUI this maps to the checkpoint filename.",
            },
            "vae": {
                "type": "string",
                "description": "Optional provider-specific VAE filename. For ComfyUI this maps to models/vae.",
            },
            "loras": {
                "type": "array",
                "description": "Optional provider-specific LoRA stack. For ComfyUI items may include name, weight, and clip_weight.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "lora_name": {"type": "string"},
                        "weight": {"type": "number"},
                        "clip_weight": {"type": "number"},
                        "strength_model": {"type": "number"},
                        "strength_clip": {"type": "number"},
                        "preset": {"type": "string"},
                        "use_case": {"type": "string"}
                    }
                },
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
            "seed": {
                "type": "integer",
                "description": "Optional deterministic generation seed. NovelAI preserves this in request sidecars for reproducible comparisons.",
            },
            "steps": {
                "type": "integer",
                "description": "Optional provider sampling step count. For NovelAI this maps to the `steps` request parameter.",
            },
            "scale": {
                "type": "number",
                "description": "Optional provider guidance scale. For NovelAI this maps to the `scale` request parameter.",
            },
            "sampler": {
                "type": "string",
                "description": "Optional provider sampler id. For NovelAI use ids such as `k_dpmpp_sde` only when a specific sampler is intended.",
            },
            "cfg_rescale": {
                "type": "number",
                "description": "Optional NovelAI CFG rescale value. Omit to use the provider default.",
            },
            "uc_preset": {
                "type": "integer",
                "description": "Optional NovelAI undesired-content preset value. Omit to use the Hermes baseline.",
            },
            "noise_schedule": {
                "type": "string",
                "description": "Optional NovelAI noise schedule, for example `karras`. Omit to use the provider default.",
            },
            "add_quality_tags": {
                "type": "boolean",
                "description": "Optional NovelAI quality tag toggle. Omit to use the Hermes baseline.",
            },
            "auto_smea": {
                "type": "boolean",
                "description": "Optional NovelAI SMEA toggle. Omit to use the provider default.",
            },
            "dynamic_thresholding": {
                "type": "boolean",
                "description": "Optional NovelAI dynamic-thresholding toggle. Omit to use the provider default.",
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
            "operation": {
                "type": "string",
                "enum": ["generate", "txt2img", "postprocess", "source_preserving_postprocess", "upscale", "masked_inpaint", "reference_identity_txt2img"],
                "description": "Optional provider operation. Use `source_preserving_postprocess` only when the user explicitly asks to preserve an existing source image and apply broad postprocess/detail correction. Use `masked_inpaint` only for local source-image edits with mask_source/mask_target or mask_box. Use `upscale` only for source-image upscaling. Use `reference_identity_txt2img` only with an explicit temporary reference identity experiment workflow and reference_image_path.",
                "default": "generate",
            },
            "output_type": {
                "type": "string",
                "description": "Optional image-router output type, for example `profile_icon`, `portrait`, `dialogue_bust`, `upper_body`, `fullbody`, `standing_sprite`, `ingame_cg`, or `key_visual`. For ComfyUI this is forwarded to provider routing and evidence metadata.",
            },
            "source_image_path": {
                "type": "string",
                "description": "Optional absolute local source image path for source-preserving ComfyUI postprocess or upscale workflows.",
            },
            "reference_image_path": {
                "type": "string",
                "description": "Optional reference image path or registered reference alias. For NovelAI this may be a policy-managed alias such as `style_ref_00001` when `experimental_reference_images=true` is explicitly requested. For ComfyUI this is allowed only with explicit operation=`reference_identity_txt2img` and workflow_key=`character_reference_key_visual_experimental_v1` or `fullbody_v8_reference_identity_experimental_v1`; it must not be used for default generation. Match the reference image workflow family unless `allow_reference_workflow_family_change=true` is intentionally set.",
            },
            "experimental_reference_images": {
                "type": "boolean",
                "description": "NovelAI-only explicit validation flag for Precise Reference / vibe reference image generation. Use only when the user has explicitly requested a NovelAI reference-image test or workflow.",
                "default": False,
            },
            "reference_strength": {
                "type": "number",
                "description": "NovelAI-only reference image strength for Precise Reference / vibe reference tests. Omit to use the provider default.",
            },
            "reference_information_extracted": {
                "type": "number",
                "description": "NovelAI-only information/fidelity extraction value for Precise Reference / vibe reference tests. Omit to use the provider default.",
            },
            "experimental_reference_identity": {
                "type": "boolean",
                "description": "Optional audit flag for the temporary ComfyUI reference identity experiment. This does not make the route valid without operation, workflow_key, and reference_image_path.",
                "default": False,
            },
            "allow_reference_workflow_family_change": {
                "type": "boolean",
                "description": "Optional safety override for temporary ComfyUI reference identity experiments. Default false. Set true only when intentionally converting a reference image from one workflow family, such as fullbody, into another target family, such as key_visual.",
                "default": False,
            },
            "postprocess_preset": {
                "type": "string",
                "enum": [
                    "face8m_d035_hand9c_d025",
                    "face8m_d035_pithand_d025",
                    "hand9c_d025_only",
                    "pithand_d025_only",
                    "depth50_canny100_face8m_hand9c_v1",
                    "sam3_local_hand_tight_v1",
                ],
                "description": "Optional ComfyUI source-preserving postprocess preset. `face8m_d035_hand9c_d025` applies FaceDetailer face_yolov8m denoise 0.35 plus hand_yolov9c denoise 0.25. `face8m_d035_pithand_d025` keeps the same face route and uses PitHandDetailer-v1 segmentation for hand detail correction as a candidate route. `hand9c_d025_only` and `pithand_d025_only` skip face detail correction and apply only hand-region detail correction. `depth50_canny100_face8m_hand9c_v1` adds the promoted Depth+Canny structure-preserving stack before face/hand detail correction. `sam3_local_hand_tight_v1` is an experimental coordinate-guided SAM3 local hand mask route and requires sam3_positive_coords.",
            },
            "upscale_model": {
                "type": "string",
                "description": "Optional ComfyUI upscale model for operation=`upscale`, for example `4x-UltraSharp.pth`.",
            },
            "mask_source": {
                "type": "string",
                "enum": ["rectangle", "detailer_bbox"],
                "description": "Optional ComfyUI mask source for operation=`masked_inpaint`. Use `detailer_bbox` for detector-derived face/hand masks; use `rectangle` only when mask_box is explicitly supplied.",
            },
            "mask_target": {
                "type": "string",
                "description": "Optional target label for operation=`masked_inpaint`, for example `hand`, `left_hand`, `right_hand`, or `face`. With mask_source=`detailer_bbox`, this selects the detector family.",
            },
            "mask_box": {
                "description": "Optional normalized rectangle for operation=`masked_inpaint` with mask_source=`rectangle`. Accepts an object with x/y/w/h, a 4-item array, or an 'x,y,w,h' string. Not required for mask_source=`detailer_bbox`.",
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "w": {"type": "number"},
                            "h": {"type": "number"},
                            "width": {"type": "number"},
                            "height": {"type": "number"}
                        }
                    },
                    {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4
                    },
                    {"type": "string"}
                ],
            },
            "mask_feather_px": {
                "type": "integer",
                "description": "Optional feather radius in pixels for operation=`masked_inpaint`.",
            },
            "grow_mask_by": {
                "type": "integer",
                "description": "Optional mask growth in pixels for operation=`masked_inpaint`.",
            },
            "sam3_positive_coords": {
                "description": "Optional coordinate points for postprocess_preset=`sam3_local_hand_tight_v1`. Use source-image pixel coordinates, for example [{\"x\": 388, \"y\": 464}]. Required for the SAM3 local hand preset.",
                "oneOf": [
                    {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"}
                            },
                            "required": ["x", "y"]
                        }
                    },
                    {"type": "string"}
                ],
            },
            "sam3_negative_coords": {
                "description": "Optional negative coordinate points for postprocess_preset=`sam3_local_hand_tight_v1`. Use source-image pixel coordinates to exclude forearm, props, background, or adjacent body regions.",
                "oneOf": [
                    {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"}
                            },
                            "required": ["x", "y"]
                        }
                    },
                    {"type": "string"}
                ],
            },
            "sam3_threshold": {
                "type": "number",
                "description": "Optional SAM3 detection threshold for postprocess_preset=`sam3_local_hand_tight_v1`. Higher values tend to tighten masks.",
            },
            "sam3_refine_iterations": {
                "type": "integer",
                "description": "Optional SAM3 refine iteration count for postprocess_preset=`sam3_local_hand_tight_v1`.",
            },
            "sam3_detail_denoise": {
                "type": "number",
                "description": "Optional detailer denoise for postprocess_preset=`sam3_local_hand_tight_v1`. Use lower values for conservative local repair.",
            },
            "workflow_key": {
                "type": "string",
                "description": "Optional ComfyUI workflow key override for evidence and provider routing.",
            },
            "style_preset": {
                "type": "string",
                "description": "Optional ComfyUI style/LoRA preset name, for example `stable`, `glossy_skin`, `video_source`, or `eye_gloss`.",
            },
            "lora_preset": {
                "type": "string",
                "description": "Optional ComfyUI LoRA preset alias. Prefer `style_preset` for user-facing style choices.",
            },
            "cfg_scale": {
                "type": "number",
                "description": "Optional ComfyUI CFG scale. This is forwarded separately from NovelAI `scale`.",
            },
            "sampler_name": {
                "type": "string",
                "description": "Optional ComfyUI sampler name, for example `dpmpp_2m`.",
            },
            "scheduler": {
                "type": "string",
                "description": "Optional ComfyUI scheduler name, for example `karras`.",
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
    seed: int | None = None,
    steps: int | None = None,
    scale: float | int | None = None,
    sampler: str | None = None,
    cfg_rescale: float | int | None = None,
    uc_preset: int | None = None,
    noise_schedule: str | None = None,
    add_quality_tags: bool | None = None,
    auto_smea: bool | None = None,
    dynamic_thresholding: bool | None = None,
    live_generation_approved: bool | None = None,
    high_res_approved: bool | None = None,
    operation: str | None = None,
    output_type: str | None = None,
    source_image_path: str | None = None,
    reference_image_path: str | None = None,
    reference_image: str | None = None,
    experimental_reference_images: bool | None = None,
    reference_strength: float | int | None = None,
    reference_information_extracted: float | int | None = None,
    experimental_reference_identity: bool | None = None,
    allow_reference_workflow_family_change: bool | None = None,
    postprocess_preset: str | None = None,
    upscale_model: str | None = None,
    model: str | None = None,
    vae: str | None = None,
    loras: list[dict[str, Any]] | None = None,
    workflow_key: str | None = None,
    style_preset: str | None = None,
    lora_preset: str | None = None,
    cfg_scale: float | int | None = None,
    denoise: float | int | None = None,
    sampler_name: str | None = None,
    scheduler: str | None = None,
    mask_target: str | None = None,
    mask_source: str | None = None,
    mask_box: Any | None = None,
    mask_feather_px: int | None = None,
    grow_mask_by: int | None = None,
    sam3_positive_coords: Any | None = None,
    sam3_negative_coords: Any | None = None,
    sam3_threshold: float | int | None = None,
    sam3_refine_iterations: int | None = None,
    sam3_detail_denoise: float | int | None = None,
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
        if seed is not None:
            kwargs["seed"] = seed
        if steps is not None:
            kwargs["steps"] = steps
        if scale is not None:
            kwargs["scale"] = scale
        if sampler is not None:
            kwargs["sampler"] = sampler
        if cfg_rescale is not None:
            kwargs["cfg_rescale"] = cfg_rescale
        if uc_preset is not None:
            kwargs["ucPreset"] = uc_preset
        if noise_schedule is not None:
            kwargs["noise_schedule"] = noise_schedule
        if add_quality_tags is not None:
            kwargs["qualityToggle"] = add_quality_tags
            kwargs["add_quality_tags"] = add_quality_tags
        if auto_smea is not None:
            kwargs["autoSmea"] = auto_smea
        if dynamic_thresholding is not None:
            kwargs["dynamic_thresholding"] = dynamic_thresholding
        if live_generation_approved is not None:
            kwargs["live_generation_approved"] = live_generation_approved
        if high_res_approved is not None:
            kwargs["high_res_approved"] = high_res_approved
        if operation is not None:
            kwargs["operation"] = operation
        if output_type is not None:
            kwargs["output_type"] = output_type
        if source_image_path is not None:
            kwargs["source_image_path"] = source_image_path
        if reference_image_path is not None:
            kwargs["reference_image_path"] = reference_image_path
        if reference_image is not None:
            kwargs["reference_image"] = reference_image
        if experimental_reference_images is not None:
            kwargs["experimental_reference_images"] = experimental_reference_images
        if reference_strength is not None:
            kwargs["reference_strength"] = reference_strength
        if reference_information_extracted is not None:
            kwargs["reference_information_extracted"] = reference_information_extracted
        if experimental_reference_identity is not None:
            kwargs["experimental_reference_identity"] = experimental_reference_identity
        if allow_reference_workflow_family_change is not None:
            kwargs["allow_reference_workflow_family_change"] = allow_reference_workflow_family_change
        if postprocess_preset is not None:
            kwargs["postprocess_preset"] = postprocess_preset
        if upscale_model is not None:
            kwargs["upscale_model"] = upscale_model
        if model is not None:
            kwargs["model"] = model
        elif configured_model:
            kwargs["model"] = configured_model
        if vae is not None:
            kwargs["vae"] = vae
        if loras is not None:
            kwargs["loras"] = loras
        if workflow_key is not None:
            kwargs["workflow_key"] = workflow_key
        if style_preset is not None:
            kwargs["style_preset"] = style_preset
        if lora_preset is not None:
            kwargs["lora_preset"] = lora_preset
        if cfg_scale is not None:
            kwargs["cfg_scale"] = cfg_scale
        if denoise is not None:
            kwargs["denoise"] = denoise
        if sampler_name is not None:
            kwargs["sampler_name"] = sampler_name
        if scheduler is not None:
            kwargs["scheduler"] = scheduler
        if mask_target is not None:
            kwargs["mask_target"] = mask_target
        if mask_source is not None:
            kwargs["mask_source"] = mask_source
        if mask_box is not None:
            kwargs["mask_box"] = mask_box
        if mask_feather_px is not None:
            kwargs["mask_feather_px"] = mask_feather_px
        if grow_mask_by is not None:
            kwargs["grow_mask_by"] = grow_mask_by
        if sam3_positive_coords is not None:
            kwargs["sam3_positive_coords"] = sam3_positive_coords
        if sam3_negative_coords is not None:
            kwargs["sam3_negative_coords"] = sam3_negative_coords
        if sam3_threshold is not None:
            kwargs["sam3_threshold"] = sam3_threshold
        if sam3_refine_iterations is not None:
            kwargs["sam3_refine_iterations"] = sam3_refine_iterations
        if sam3_detail_denoise is not None:
            kwargs["sam3_detail_denoise"] = sam3_detail_denoise
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
    task_id = kw.get("task_id")
    task_project_name, task_artifact_name, task_image_args = _consume_image_task_metadata(task_id)
    if task_image_args:
        merged_args = dict(args or {})
        merged_args.update(task_image_args)
        args = merged_args
        logger.info(
            "Commander image task args applied for task_id=%s keys=%s",
            task_id,
            sorted(task_image_args),
        )
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
    seed = args.get("seed")
    steps = args.get("steps")
    scale = args.get("scale")
    sampler = args.get("sampler")
    cfg_rescale = args.get("cfg_rescale")
    uc_preset = args.get("uc_preset")
    noise_schedule = args.get("noise_schedule")
    add_quality_tags = args.get("add_quality_tags")
    auto_smea = args.get("auto_smea")
    dynamic_thresholding = args.get("dynamic_thresholding")
    live_generation_approved = args.get("live_generation_approved")
    high_res_approved = args.get("high_res_approved")
    operation = args.get("operation")
    output_type = args.get("output_type")
    source_image_path = args.get("source_image_path")
    reference_image_path = args.get("reference_image_path")
    reference_image = args.get("reference_image")
    experimental_reference_images = args.get("experimental_reference_images")
    reference_strength = args.get("reference_strength")
    reference_information_extracted = args.get("reference_information_extracted")
    experimental_reference_identity = args.get("experimental_reference_identity")
    allow_reference_workflow_family_change = args.get("allow_reference_workflow_family_change")
    postprocess_preset = args.get("postprocess_preset")
    upscale_model = args.get("upscale_model")
    model = args.get("model")
    vae = args.get("vae")
    loras = args.get("loras")
    workflow_key = args.get("workflow_key")
    style_preset = args.get("style_preset")
    lora_preset = args.get("lora_preset")
    cfg_scale = args.get("cfg_scale")
    denoise = args.get("denoise")
    sampler_name = args.get("sampler_name")
    scheduler = args.get("scheduler")
    mask_target = args.get("mask_target")
    mask_source = args.get("mask_source")
    mask_box = args.get("mask_box")
    mask_feather_px = args.get("mask_feather_px")
    grow_mask_by = args.get("grow_mask_by")
    sam3_positive_coords = args.get("sam3_positive_coords")
    sam3_negative_coords = args.get("sam3_negative_coords")
    sam3_threshold = args.get("sam3_threshold")
    sam3_refine_iterations = args.get("sam3_refine_iterations")
    sam3_detail_denoise = args.get("sam3_detail_denoise")
    if task_project_name:
        project_name = task_project_name
    if task_artifact_name:
        artifact_name = task_artifact_name

    project_name, artifact_name = _infer_image_project_metadata(
        prompt,
        project_name=project_name,
        artifact_name=artifact_name,
    )

    # Route to a plugin-registered provider if one is active (and it's
    # not the in-tree FAL path).
    def _coerce_nonnegative_int(value):
        if isinstance(value, bool):
            return None
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return None
        return coerced if coerced >= 0 else None

    def _coerce_positive_float(value):
        if isinstance(value, bool):
            return None
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        return coerced if coerced > 0 else None

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
        seed=seed if isinstance(seed, int) and seed >= 0 else None,
        steps=steps if isinstance(steps, int) and steps > 0 else None,
        scale=scale if isinstance(scale, (int, float)) and scale > 0 else None,
        sampler=str(sampler).strip() if isinstance(sampler, str) and sampler.strip() else None,
        cfg_rescale=cfg_rescale if isinstance(cfg_rescale, (int, float)) and cfg_rescale >= 0 else None,
        uc_preset=uc_preset if isinstance(uc_preset, int) and uc_preset >= 0 else None,
        noise_schedule=str(noise_schedule).strip() if isinstance(noise_schedule, str) and noise_schedule.strip() else None,
        add_quality_tags=add_quality_tags if isinstance(add_quality_tags, bool) else None,
        auto_smea=auto_smea if isinstance(auto_smea, bool) else None,
        dynamic_thresholding=dynamic_thresholding if isinstance(dynamic_thresholding, bool) else None,
        live_generation_approved=live_generation_approved if isinstance(live_generation_approved, bool) else None,
        high_res_approved=high_res_approved if isinstance(high_res_approved, bool) else None,
        operation=str(operation).strip() if isinstance(operation, str) and operation.strip() else None,
        output_type=str(output_type).strip() if isinstance(output_type, str) and output_type.strip() else None,
        source_image_path=str(source_image_path).strip() if isinstance(source_image_path, str) and source_image_path.strip() else None,
        reference_image_path=str(reference_image_path).strip() if isinstance(reference_image_path, str) and reference_image_path.strip() else None,
        reference_image=str(reference_image).strip() if isinstance(reference_image, str) and reference_image.strip() else None,
        experimental_reference_images=experimental_reference_images if isinstance(experimental_reference_images, bool) else None,
        reference_strength=_coerce_positive_float(reference_strength),
        reference_information_extracted=_coerce_positive_float(reference_information_extracted),
        experimental_reference_identity=experimental_reference_identity if isinstance(experimental_reference_identity, bool) else None,
        allow_reference_workflow_family_change=allow_reference_workflow_family_change if isinstance(allow_reference_workflow_family_change, bool) else None,
        postprocess_preset=str(postprocess_preset).strip() if isinstance(postprocess_preset, str) and postprocess_preset.strip() else None,
        upscale_model=str(upscale_model).strip() if isinstance(upscale_model, str) and upscale_model.strip() else None,
        model=str(model).strip() if isinstance(model, str) and model.strip() else None,
        vae=str(vae).strip() if isinstance(vae, str) and vae.strip() else None,
        loras=loras if isinstance(loras, list) else None,
        workflow_key=str(workflow_key).strip() if isinstance(workflow_key, str) and workflow_key.strip() else None,
        style_preset=str(style_preset).strip() if isinstance(style_preset, str) and style_preset.strip() else None,
        lora_preset=str(lora_preset).strip() if isinstance(lora_preset, str) and lora_preset.strip() else None,
        cfg_scale=_coerce_positive_float(cfg_scale),
        denoise=_coerce_positive_float(denoise),
        sampler_name=str(sampler_name).strip() if isinstance(sampler_name, str) and sampler_name.strip() else None,
        scheduler=str(scheduler).strip() if isinstance(scheduler, str) and scheduler.strip() else None,
        mask_target=str(mask_target).strip() if isinstance(mask_target, str) and mask_target.strip() else None,
        mask_source=str(mask_source).strip() if isinstance(mask_source, str) and mask_source.strip() else None,
        mask_box=mask_box if mask_box not in (None, "", [], {}) else None,
        mask_feather_px=_coerce_nonnegative_int(mask_feather_px),
        grow_mask_by=_coerce_nonnegative_int(grow_mask_by),
        sam3_positive_coords=sam3_positive_coords if sam3_positive_coords not in (None, "", [], {}) else None,
        sam3_negative_coords=sam3_negative_coords if sam3_negative_coords not in (None, "", []) else None,
        sam3_threshold=_coerce_positive_float(sam3_threshold),
        sam3_refine_iterations=_coerce_nonnegative_int(sam3_refine_iterations),
        sam3_detail_denoise=_coerce_positive_float(sam3_detail_denoise),
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
