"""Local Forge / Automatic1111 image generation backend.

This provider talks directly to a local Forge WebUI-compatible sdapi endpoint
(default: ``http://172.22.224.1:7860``), submits ``txt2img`` requests, and
materialises the returned base64 image under HermesWork/Image via the shared
image artifact helpers.

It is intentionally small and deterministic:

- no API key requirement
- minimal txt2img support with sane defaults
- absolute file path returned for Slack / gateway attachment use
- image publication follows the HermesWork artifact policy
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://172.22.224.1:7860"
DEFAULT_MODEL = "Nullstyle_v20"
DEFAULT_STEPS = 20
DEFAULT_CFG_SCALE = 7.0
DEFAULT_SAMPLER = "DPM++ 2M"

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
    for key in ("forge-local", "forge_local"):
        block = cfg.get(key)
        if isinstance(block, dict):
            return block
    return {}


def _resolve_base_url() -> str:
    env_override = os.environ.get("FORGE_LOCAL_IMAGE_BASE_URL")
    if env_override and env_override.strip():
        return env_override.strip().rstrip("/")
    cfg = _provider_cfg()
    base_url = cfg.get("base_url") if isinstance(cfg.get("base_url"), str) else None
    if base_url and base_url.strip():
        return base_url.strip().rstrip("/")
    return DEFAULT_BASE_URL


def _resolve_model() -> str:
    env_override = os.environ.get("FORGE_LOCAL_IMAGE_MODEL")
    if env_override and env_override.strip():
        return env_override.strip()
    cfg = _provider_cfg()
    model = cfg.get("model") if isinstance(cfg.get("model"), str) else None
    if model and model.strip():
        return model.strip()
    return DEFAULT_MODEL


def _resolve_dimensions(aspect_ratio: str) -> tuple[int, int]:
    aspect = resolve_aspect_ratio(aspect_ratio)
    return _ASPECT_DIMENSIONS.get(aspect, _ASPECT_DIMENSIONS[DEFAULT_ASPECT_RATIO])


def _extract_first_image(result: Dict[str, Any]) -> Optional[str]:
    images = result.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    # A1111/Forge responses usually return ``images`` but keep a fallback for
    # variants that nest payloads a bit differently.
    data = result.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            image = first.get("image") or first.get("b64_json")
            if isinstance(image, str) and image.strip():
                return image.strip()
    return None


class ForgeLocalImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "forge-local"

    @property
    def display_name(self) -> str:
        return "Forge Local"

    def is_available(self) -> bool:
        base_url = _resolve_base_url()
        try:
            response = requests.get(f"{base_url}/sdapi/v1/sd-models", timeout=5)
            response.raise_for_status()
            return isinstance(response.json(), list)
        except Exception:  # noqa: BLE001
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        base_url = _resolve_base_url()
        try:
            response = requests.get(f"{base_url}/sdapi/v1/sd-models", timeout=10)
            response.raise_for_status()
            models = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Forge model lookup failed: %s", exc)
            return [
                {
                    "id": DEFAULT_MODEL,
                    "display": DEFAULT_MODEL,
                    "speed": "local",
                    "strengths": "forge webui checkpoint",
                }
            ]

        entries: List[Dict[str, Any]] = []
        if isinstance(models, list):
            for item in models:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or item.get("model_name") or "").strip()
                if not title:
                    continue
                entries.append(
                    {
                        "id": str(item.get("model_name") or title).strip(),
                        "display": title,
                        "speed": "local",
                        "strengths": "forge webui checkpoint",
                    }
                )
        if not entries:
            entries.append(
                {
                    "id": DEFAULT_MODEL,
                    "display": DEFAULT_MODEL,
                    "speed": "local",
                    "strengths": "forge webui checkpoint",
                }
            )
        return entries

    def default_model(self) -> Optional[str]:
        models = self.list_models()
        if models:
            return models[0].get("id")
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Forge Local",
            "badge": "local",
            "tag": "Local Forge / A1111 sdapi txt2img endpoint (no API key).",
            "env_vars": [],
        }

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
        model = _resolve_model()
        width, height = _resolve_dimensions(aspect)
        steps = kwargs.get("steps")
        if not isinstance(steps, int) or steps <= 0:
            steps = DEFAULT_STEPS
        project_name = kwargs.get("project_name")
        artifact_name = kwargs.get("artifact_name")
        cfg_scale = kwargs.get("cfg_scale")
        if not isinstance(cfg_scale, (int, float)) or cfg_scale <= 0:
            cfg_scale = DEFAULT_CFG_SCALE
        sampler_name = kwargs.get("sampler_name")
        if not isinstance(sampler_name, str) or not sampler_name.strip():
            sampler_name = DEFAULT_SAMPLER

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": str(kwargs.get("negative_prompt") or "").strip(),
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler_name,
            "width": width,
            "height": height,
            "batch_size": 1,
            "n_iter": 1,
            "seed": int(kwargs["seed"]) if isinstance(kwargs.get("seed"), int) else -1,
            "subseed": int(kwargs["subseed"]) if isinstance(kwargs.get("subseed"), int) else -1,
            "do_not_save_samples": True,
            "do_not_save_grid": True,
            "send_images": True,
            "override_settings": {
                "sd_model_checkpoint": model,
            },
            "override_settings_restore_afterwards": True,
        }

        if not payload["negative_prompt"]:
            payload.pop("negative_prompt")

        try:
            response = requests.post(
                f"{base_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            resp = exc.response
            status = resp.status_code if resp is not None else 0
            body = ""
            if resp is not None:
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        body = str(data.get("detail") or data.get("error") or data)
                    else:
                        body = str(data)
                except Exception:  # noqa: BLE001
                    body = resp.text[:500]
            return error_response(
                error=f"Forge txt2img failed ({status}): {body or exc}",
                error_type="api_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error="Forge txt2img timed out",
                error_type="timeout",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"Forge connection error: {exc}",
                error_type="connection_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            result = response.json()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"Forge returned invalid JSON: {exc}",
                error_type="invalid_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        b64_image = _extract_first_image(result)
        if not b64_image:
            return error_response(
                error="Forge response contained no image data",
                error_type="empty_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(
                b64_image,
                prefix="forge_test",
                project_name=project_name,
                artifact_name=artifact_name,
            )
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"Could not save Forge image to HermesWork: {exc}",
                error_type="io_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            extra={
                "base_url": base_url,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler_name": sampler_name,
            },
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(ForgeLocalImageGenProvider())
