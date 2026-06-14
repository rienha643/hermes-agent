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
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
DEFAULT_STEPS = 12
DEFAULT_CFG = 7.0
DEFAULT_SAMPLER = "euler"
DEFAULT_DENOISE = 1.0
DEFAULT_SEED = 123456789
DEFAULT_CATEGORY = "txt2img"

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


def _resolve_model() -> str:
    env_override = os.environ.get("COMFY_LOCAL_IMAGE_MODEL")
    if env_override and env_override.strip():
        return env_override.strip()
    cfg = _provider_cfg()
    model = cfg.get("model") if isinstance(cfg.get("model"), str) else None
    if model and model.strip():
        return model.strip()
    return DEFAULT_CHECKPOINT


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
        if checkpoint not in checkpoints:
            return error_response(
                error=f"Requested checkpoint not found in ComfyUI: {checkpoint}",
                error_type="invalid_argument",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
            )

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
            "3": {"inputs": {"text": prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
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
        }
        prompt_payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": 1,
            "seed": seed,
            "sampler": sampler_name,
            "steps": steps,
            "cfg": cfg,
            "denoise": denoise,
            "raw_prompt_payload": raw_prompt_payload,
        }
        metadata = {
            "provider": self.name,
            "prompt_id": prompt_id,
            "api_base_url": base_url,
            "checkpoint": checkpoint,
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
            "local_status": "생성 완료",
            "publish_status": "HermesWork publish 완료",
            "slack_status": "primary image 준비됨",
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
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            extra={
                "base_url": base_url,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg,
                "sampler_name": sampler_name,
                "denoise": denoise,
                "local_status": "생성 완료",
                "publish_status": "HermesWork publish 완료",
                "nas_status": nas_status,
                "slack_status": "primary image 준비됨",
                "workflow_path": str(bundle["workflow_path"]),
                "prompt_path": str(bundle["prompt_path"]),
                "metadata_path": str(bundle["metadata_path"]),
                "manifest_path": str(bundle["manifest_path"]),
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
                ],
                "prompt_id": prompt_id,
                "category": category,
                "api_base_url": base_url,
            },
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(ComfyLocalImageGenProvider())
