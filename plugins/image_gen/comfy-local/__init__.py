"""Local ComfyUI image generation backend.

This provider talks to a macOS local ComfyUI API, with legacy Windows/WSL
fallback support, waits for `/history/<prompt_id>` success, verifies the exact
output file for the run, then publishes the result into
`HermesWork/Image/<project_id>/` with workflow / prompt / metadata sidecars
before queueing the existing NAS sync hook.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import unicodedata
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
POSTPROCESS_CATEGORY = "postprocess"

DEFAULT_STABLE_STYLE_LORA = r"00_illustrious_style_candidates\pornmaster-Aesthetics-v2-lora.safetensors"
DEFAULT_STABLE_STYLE_LORA_WEIGHT = 0.15
DEFAULT_KEY_ART_LORA = r"00_illustrious_style_candidates\pornmaster-Aesthetics-v2-lora.safetensors"
DEFAULT_KEY_ART_LORA_WEIGHT = 0.20
DEFAULT_MATTE_PRODUCTION_LORA = r"04_utility_skin_gloss\Matte_Skin_Illustrious_v4.safetensors"
DEFAULT_MATTE_PRODUCTION_LORA_WEIGHT = 0.35
DEFAULT_EYE_DETAIL_LORA = r"02_utility_eyes\EyeDetail_Z_Turbo.safetensors"
DEFAULT_EYE_DETAIL_LORA_WEIGHT = 0.25
DEFAULT_DETAIL_SMOOTH_LORA = r"03_utility_detail_enhancer\Smooth_Booster_v5.safetensors"
DEFAULT_DETAIL_SMOOTH_LORA_WEIGHT = 0.20
DEFAULT_DETAIL_ENHANCER_LORA = r"03_utility_detail_enhancer\Detail_enhancer_IL_v2.safetensors"
DEFAULT_DETAIL_ENHANCER_LORA_WEIGHT = 0.20
DEFAULT_GLOSSY_SKIN_LORA = r"04_utility_skin_gloss\shiny_nai_ilxl_goofy_remade.safetensors"
DEFAULT_GLOSSY_SKIN_LORA_WEIGHT = 0.20
DEFAULT_VIDEO_SOURCE_STYLE_LORA = r"00_illustrious_style_candidates\K NAI Style.safetensors"
DEFAULT_VIDEO_SOURCE_STYLE_LORA_WEIGHT = 0.65
DEFAULT_ADD_MICRO_DETAILS_LORA = r"03_utility_detail_enhancer\AddMicroDetails_Illustrious_v6.safetensors"
DEFAULT_ADD_MICRO_DETAILS_LORA_WEIGHT = 0.20
DEFAULT_DNF_ANIMA_EXPERIMENTAL_LORA = r"90_compatibility_review_non_illustrious_or_anima\DNF Anima-v2.safetensors"
DEFAULT_DNF_ANIMA_EXPERIMENTAL_LORA_WEIGHT = 0.45


def _lora_preset_entry(preset: str, name: str, weight: float, use_case: str) -> Dict[str, Any]:
    return {
        "preset": preset,
        "name": name,
        "weight": weight,
        "use_case": use_case,
    }


def _stable_lora_preset_entry() -> Dict[str, Any]:
    return _lora_preset_entry(
        "stable",
        DEFAULT_STABLE_STYLE_LORA,
        DEFAULT_STABLE_STYLE_LORA_WEIGHT,
        "default user-approved subculture character illustration",
    )


def _portrait_primary_lora_preset_entries() -> List[Dict[str, Any]]:
    return [
        _lora_preset_entry(
            "portrait_primary",
            DEFAULT_VIDEO_SOURCE_STYLE_LORA,
            DEFAULT_VIDEO_SOURCE_STYLE_LORA_WEIGHT,
            "user-selected top portrait/close-up style baseline from 2026-06-22 F candidate",
        ),
        _lora_preset_entry(
            "portrait_primary_detail",
            DEFAULT_ADD_MICRO_DETAILS_LORA,
            DEFAULT_ADD_MICRO_DETAILS_LORA_WEIGHT,
            "portrait/close-up micro-detail companion from WAI cross-format confirmation",
        ),
    ]


STYLE_PRESET_LORAS: Dict[str, Any] = {
    "stable": _stable_lora_preset_entry(),
    "default": _stable_lora_preset_entry(),
    "portrait_primary": _portrait_primary_lora_preset_entries(),
    "portrait_closeup": _portrait_primary_lora_preset_entries(),
    "key_art": {
        "preset": "key_art",
        "name": DEFAULT_KEY_ART_LORA,
        "weight": DEFAULT_KEY_ART_LORA_WEIGHT,
        "use_case": "key visual or intentional image distortion effect",
    },
    "keyart": {
        "preset": "key_art",
        "name": DEFAULT_KEY_ART_LORA,
        "weight": DEFAULT_KEY_ART_LORA_WEIGHT,
        "use_case": "key visual or intentional image distortion effect",
    },
    "dramatic": {
        "preset": "key_art",
        "name": DEFAULT_KEY_ART_LORA,
        "weight": DEFAULT_KEY_ART_LORA_WEIGHT,
        "use_case": "key visual or intentional image distortion effect",
    },
    "matte_skin": [
        _stable_lora_preset_entry(),
        {
            "preset": "matte_skin",
            "name": DEFAULT_MATTE_PRODUCTION_LORA,
            "weight": DEFAULT_MATTE_PRODUCTION_LORA_WEIGHT,
            "use_case": "deprecated/review: matte lighting may remove too much illumination",
        },
    ],
    "matte": [
        _stable_lora_preset_entry(),
        {
            "preset": "matte_skin",
            "name": DEFAULT_MATTE_PRODUCTION_LORA,
            "weight": DEFAULT_MATTE_PRODUCTION_LORA_WEIGHT,
            "use_case": "deprecated/review: matte lighting may remove too much illumination",
        },
    ],
    "standing_matte": [
        _stable_lora_preset_entry(),
        {
            "preset": "matte_skin",
            "name": DEFAULT_MATTE_PRODUCTION_LORA,
            "weight": DEFAULT_MATTE_PRODUCTION_LORA_WEIGHT,
            "use_case": "deprecated/review: matte lighting may remove too much illumination",
        },
    ],
    "eye_detail": [
        _stable_lora_preset_entry(),
        {
            "preset": "eye_detail",
            "name": DEFAULT_EYE_DETAIL_LORA,
            "weight": DEFAULT_EYE_DETAIL_LORA_WEIGHT,
            "use_case": "review/no-op alone in template check; prefer eye_gloss when eye improvement is desired",
        },
    ],
    "detail_smooth": [
        _stable_lora_preset_entry(),
        {
            "preset": "detail_smooth",
            "name": DEFAULT_DETAIL_SMOOTH_LORA,
            "weight": DEFAULT_DETAIL_SMOOTH_LORA_WEIGHT,
            "use_case": "optional outfit/material detail; do not default because face/character identity may drift",
        },
    ],
    "detail_enhancer": [
        _stable_lora_preset_entry(),
        {
            "preset": "detail_enhancer",
            "name": DEFAULT_DETAIL_ENHANCER_LORA,
            "weight": DEFAULT_DETAIL_ENHANCER_LORA_WEIGHT,
            "use_case": "review/rejected for default: worst in second character recheck",
        },
    ],
    "glossy_skin": [
        _stable_lora_preset_entry(),
        {
            "preset": "glossy_skin",
            "name": DEFAULT_GLOSSY_SKIN_LORA,
            "weight": DEFAULT_GLOSSY_SKIN_LORA_WEIGHT,
            "use_case": "best skin/outfit gloss candidate",
        },
    ],
    "eye_gloss": [
        _stable_lora_preset_entry(),
        {
            "preset": "eye_detail",
            "name": DEFAULT_EYE_DETAIL_LORA,
            "weight": DEFAULT_EYE_DETAIL_LORA_WEIGHT,
            "use_case": "eye detail component for eye_gloss composite",
        },
        {
            "preset": "glossy_skin",
            "name": DEFAULT_GLOSSY_SKIN_LORA,
            "weight": DEFAULT_GLOSSY_SKIN_LORA_WEIGHT,
            "use_case": "gloss/skin component for eye_gloss composite",
        },
    ],
    "eye_detail_glossy": [
        _stable_lora_preset_entry(),
        {
            "preset": "eye_detail",
            "name": DEFAULT_EYE_DETAIL_LORA,
            "weight": DEFAULT_EYE_DETAIL_LORA_WEIGHT,
            "use_case": "eye detail component for eye_gloss composite",
        },
        {
            "preset": "glossy_skin",
            "name": DEFAULT_GLOSSY_SKIN_LORA,
            "weight": DEFAULT_GLOSSY_SKIN_LORA_WEIGHT,
            "use_case": "gloss/skin component for eye_gloss composite",
        },
    ],
    "video_source": {
        "preset": "video_source",
        "name": DEFAULT_VIDEO_SOURCE_STYLE_LORA,
        "weight": DEFAULT_VIDEO_SOURCE_STYLE_LORA_WEIGHT,
        "use_case": "best hand/lighting candidate; stable style candidate for image-to-video source material",
    },
    "k_nai": {
        "preset": "video_source",
        "name": DEFAULT_VIDEO_SOURCE_STYLE_LORA,
        "weight": DEFAULT_VIDEO_SOURCE_STYLE_LORA_WEIGHT,
        "use_case": "best hand/lighting candidate; stable style candidate for image-to-video source material",
    },
    "dnf_anima_experimental": {
        "preset": "dnf_anima_experimental",
        "name": DEFAULT_DNF_ANIMA_EXPERIMENTAL_LORA,
        "weight": DEFAULT_DNF_ANIMA_EXPERIMENTAL_LORA_WEIGHT,
        "use_case": "experimental alternate taste; not a default preset",
    },
}

CHARACTER_PRODUCTION_PRESET = "character_production"
PROFILE_ICON_PRODUCTION_PRESET = "profile_icon_production"
PORTRAIT_PRODUCTION_PRESET = "portrait_production"
DIALOGUE_BUST_PRODUCTION_PRESET = "dialogue_bust_production"
UPPER_BODY_PRODUCTION_PRESET = "upper_body_production"
FULLBODY_PRODUCTION_PRESET = "fullbody_production"
STANDING_SPRITE_PRODUCTION_PRESET = "standing_sprite_production"
INGAME_CG_PRODUCTION_PRESET = "ingame_cg_production"
V8_STYLE_WORKFLOW_PRESET = "v8_style_workflow"
KEY_VISUAL_SUBCULTURE_PRESET = "key_visual_subculture_v1"
REFERENCE_IDENTITY_EXPERIMENTAL_PRESET = "reference_identity_experimental_v1"
REFERENCE_IDENTITY_FULLBODY_EXPERIMENTAL_PRESET = "reference_identity_fullbody_experimental_v1"
SOURCE_PRESERVING_POSTPROCESS_OPERATION = "source_preserving_postprocess"
LOCAL_RETOUCH_OPERATION = "local_retouch"
MASKED_INPAINT_OPERATION = "masked_inpaint"
UPSCALE_OPERATION = "upscale"
REFERENCE_IDENTITY_TXT2IMG_OPERATION = "reference_identity_txt2img"
FACE8M_HAND9C_POSTPROCESS_PRESET = "face8m_d035_hand9c_d025"
DEPTH50_CANNY100_FACE8M_HAND9C_POSTPROCESS_PRESET = "depth50_canny100_face8m_hand9c_v1"
DEFAULT_UPSCALE_MODEL = "4x-UltraSharp.pth"
DEFAULT_WORKFLOW_KEY = "txt2img_minimal_v1"
CHARACTER_KEY_VISUAL_WORKFLOW_KEY = "character_key_visual_txt2img_v1"
CHARACTER_REFERENCE_KEY_VISUAL_EXPERIMENTAL_WORKFLOW_KEY = "character_reference_key_visual_experimental_v1"
CHARACTER_REFERENCE_FULLBODY_EXPERIMENTAL_WORKFLOW_KEY = "fullbody_v8_reference_identity_experimental_v1"
PORTRAIT_WORKFLOW_KEY = "portrait_round_v1_txt2img_v1"
FULLBODY_V8_WORKFLOW_KEY = "fullbody_v8_scene_txt2img_v2"
SOURCE_PRESERVING_FACE_HAND_WORKFLOW_KEY = "source_preserving_face8m_hand9c_v1"
SOURCE_PRESERVING_DEPTH50_CANNY100_FACE_HAND_WORKFLOW_KEY = "source_preserving_depth50_canny100_face8m_hand9c_v1"
SOURCE_MASKED_INPAINT_WORKFLOW_KEY = "source_masked_inpaint_v1"
SOURCE_DETAILER_BBOX_MASKED_INPAINT_WORKFLOW_KEY = "source_detailer_bbox_masked_inpaint_v1"
SOURCE_IMAGE_UPSCALE_WORKFLOW_KEY = "source_image_4x_ultrasharp_v1"
SOURCE_PRESERVING_DEPTH_CANNY_CHECKPOINT = "waiIllustriousSDXL_v170.safetensors"
SOURCE_PRESERVING_DEPTH_CANNY_VAE = "Anime SDXL VAE DPipe Prototype.safetensors"
SOURCE_PRESERVING_DEPTH_CANNY_STYLE_LORA = r"00_illustrious_style_candidates\K NAI Style.safetensors"
SOURCE_PRESERVING_DEPTH_CANNY_STYLE_LORA_WEIGHT = 0.65
SOURCE_PRESERVING_DEPTH_CANNY_UTILITY_LORA = r"03_utility_detail_enhancer\AddMicroDetails_Illustrious_v6.safetensors"
SOURCE_PRESERVING_DEPTH_CANNY_UTILITY_LORA_WEIGHT = 0.20
PORTRAIT_PRIMARY_CHECKPOINT = SOURCE_PRESERVING_DEPTH_CANNY_CHECKPOINT
PORTRAIT_PRIMARY_VAE = SOURCE_PRESERVING_DEPTH_CANNY_VAE
PORTRAIT_PRIMARY_WIDTH = 1024
PORTRAIT_PRIMARY_HEIGHT = 1216
WIDER_CHARACTER_PRIMARY_CHECKPOINT = "pornmasterAnime_ilV5.safetensors"
WIDER_CHARACTER_PRIMARY_VAE = SOURCE_PRESERVING_DEPTH_CANNY_VAE
FULLBODY_PRODUCTION_WIDTH = 1024
FULLBODY_PRODUCTION_HEIGHT = 1536
SOURCE_PRESERVING_DEPTH_CONTROLNET = "controlnet_zoe_depth_sdxl_1_0.safetensors"
SOURCE_PRESERVING_CANNY_CONTROLNET = "illustriousXLCanny_v10.safetensors"
REFERENCE_IDENTITY_IPADAPTER_MODEL = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
REFERENCE_IDENTITY_CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
REFERENCE_IDENTITY_IPADAPTER_WEIGHT = 0.42
REFERENCE_IDENTITY_IPADAPTER_WEIGHT_TYPE = "linear"
REFERENCE_IDENTITY_IPADAPTER_START_AT = 0.0
REFERENCE_IDENTITY_IPADAPTER_END_AT = 0.55
REFERENCE_IDENTITY_IPADAPTER_EMBEDS_SCALING = "V only"
REFERENCE_IDENTITY_EXPERIMENTAL_WORKFLOW_KEYS = {
    CHARACTER_REFERENCE_KEY_VISUAL_EXPERIMENTAL_WORKFLOW_KEY,
    CHARACTER_REFERENCE_FULLBODY_EXPERIMENTAL_WORKFLOW_KEY,
}
REFERENCE_IDENTITY_NEGATIVE_GUARD = (
    "copied reference background, overpowering reference background, style transfer artifact, "
    "stained glass pattern, mosaic background, flat stained glass colors, posterized colors, "
    "overexposed background, lineart only, under-rendered skin, flat childish drawing, "
    "background person, distant person, tiny background character, silhouette person, extra girl in background"
)
LOCALIZED_RETOUCH_TARGET_KEYWORDS: Tuple[str, ...] = (
    "hand",
    "finger",
    "wrist",
    "face",
    "eye",
    "mouth",
    "left_hand",
    "right_hand",
    "손",
    "손가락",
    "손목",
    "얼굴",
    "눈",
    "입",
)
CHARACTER_PRODUCTION_STEPS = 28
CHARACTER_PRODUCTION_CFG = 5.0
CHARACTER_PRODUCTION_SAMPLER = "dpmpp_2m"
CHARACTER_PRODUCTION_SCHEDULER = "karras"
KEY_VISUAL_SUBCULTURE_STEPS = 32
KEY_VISUAL_SUBCULTURE_CFG = 6.5
KEY_VISUAL_SUBCULTURE_SAMPLER = "dpmpp_2m"
KEY_VISUAL_SUBCULTURE_SCHEDULER = "karras"
PORTRAIT_PRODUCTION_WIDTH = 1024
PORTRAIT_PRODUCTION_HEIGHT = 1536
PROFILE_ICON_PRODUCTION_WIDTH = 512
PROFILE_ICON_PRODUCTION_HEIGHT = 512
PORTRAIT_PRODUCTION_CFG = 6.0
PORTRAIT_PRODUCTION_SAMPLER = "euler"
PORTRAIT_PRODUCTION_SCHEDULER = "normal"
PORTRAIT_PRODUCTION_SEED = 12345
V8_STYLE_WORKFLOW_KEYWORDS: Tuple[str, ...] = (
    "v8_style_workflow",
    "v8 workflow",
    "use v8",
    "이미지_v8",
    "v8 화풍",
    "v8 워크플로우",
)
V8_STYLE_SUBJECT_BASELINE = (
    "1girl, solo, anime girl, athletic anime heroine, visible human face, "
    "beautiful face, polished eyes, detailed hair, sporty outfit, "
    "subculture anime game illustration, clean skin lighting, crisp linework, "
    "bright indoor gym or sports lounge background"
)
V8_STYLE_COMPOSITION_GUARD = (
    "medium full shot, cowboy shot, standing pose, camera at chest height, "
    "character centered, full torso and upper thighs visible, balanced perspective, "
    "no first-person perspective"
)
KEY_VISUAL_SUBCULTURE_STYLE_BASELINE = (
    "subculture anime game illustration, light novel cover art, anime key visual, "
    "premium mobile game promotional art, clean lineart, polished eyes, expressive faces, "
    "refined hair flow, rich costume detail, cinematic bloom lighting, commercial splash art finish"
)
CHARACTER_PRODUCTION_POSITIVE_SKELETON = (
    "1girl, solo, full body, standing, looking at viewer, character focus, centered character, "
    "large character, detailed face, detailed eyes, detailed outfit, beautiful young adult woman, "
    "gacha game heroine, RPG protagonist, protagonist-grade heroine, attractive face, clear facial features, "
    "readable expression, refined anime illustration, premium game character illustration, ornate fantasy outfit, "
    "detailed costume design, elegant silhouette, clean silhouette, full-body character art, vertical portrait, "
    "simple background, background secondary, safe, masterpiece, high score, great score, absurdres"
)
PROFILE_ICON_PRODUCTION_POSITIVE_SKELETON = (
    "subculture anime game profile icon, square avatar composition, single character, centered face, "
    "head and shoulders, clear readable face, polished eyes, clean hair silhouette, simple clean background, "
    "icon-ready crop, commercial quality, clean lineart"
)
PORTRAIT_PRODUCTION_POSITIVE_SKELETON = (
    "single adult character, beautiful face, expressive face, detailed expressive eyes, detailed outfit, "
    "anime illustration, upper body portrait, commercial quality, clean lineart, "
    "subculture illustration, light novel cover art, anime key visual"
)
DIALOGUE_BUST_PRODUCTION_POSITIVE_SKELETON = (
    "subculture anime game dialogue bust, waist-up character art, upper body visible, expressive face, "
    "readable emotion, clean outfit upper details, visual novel dialogue portrait source, "
    "simple clean background, polished cel shading, clean lineart, commercial quality"
)
UPPER_BODY_PRODUCTION_POSITIVE_SKELETON = (
    "subculture anime game illustration, upper body to knee-up composition, single character focus, "
    "visible face, visible hands, hand and prop readable, detailed fingers, detailed outfit, "
    "readable silhouette, polished cel shading, clean lineart, commercial quality"
)
FULLBODY_PRODUCTION_POSITIVE_SKELETON = (
    "1girl, solo, only one person, anime girl, subculture mobile RPG heroine, "
    "finished colored anime illustration, premium game character illustration, polished cel shading, "
    "clean confident lineart, crisp linework, clean skin lighting, polished eyes, detailed hair, "
    "detailed costume rendering, finished rendering, high detail background, atmospheric scene background, "
    "dimensional lighting, rich but controlled lighting, commercial quality, "
    "full body character art, head-to-toe visible, full feet visible, centered character, "
    "shoes fully visible, bottom margin around shoes, readable full silhouette, balanced body proportions, "
    "visible face, visible hands"
)
STANDING_SPRITE_PRODUCTION_POSITIVE_SKELETON = (
    "subculture anime game illustration, game standing sprite, full body standing character art, "
    "single character focus, centered character, neutral standing pose, readable full silhouette, "
    "visible face, visible hands, clean outfit shapes, production-ready sprite source, "
    "simple clean background, polished cel shading, clean lineart, commercial quality"
)
INGAME_CG_PRODUCTION_POSITIVE_SKELETON = (
    "subculture anime game event CG, cinematic story scene, character and background relationship readable, "
    "expressive face, clear focal character, rich but controlled background, dramatic lighting, "
    "visual novel CG quality, polished cel shading, clean lineart, commercial quality"
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
PORTRAIT_PRODUCTION_NEGATIVE_BASELINE = (
    "low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, missing fingers, "
    "distorted face, text, watermark"
)
PROFILE_ICON_PRODUCTION_NEGATIVE_BASELINE = (
    "low quality, worst quality, blurry, bad anatomy, distorted face, asymmetrical eyes, unreadable face, "
    "full body, tiny face, multiple characters, text, logo, watermark, UI, border, card frame, cropped head"
)
DIALOGUE_BUST_PRODUCTION_NEGATIVE_BASELINE = (
    "low quality, worst quality, bad quality, lowres, blurry, watermark, text, logo, UI, textbox, "
    "full body, tiny face, face out of frame, cropped head, bad anatomy, bad hands, extra fingers, "
    "missing fingers, photorealistic, 3d render"
)
UPPER_BODY_PRODUCTION_NEGATIVE_BASELINE = (
    "low quality, worst quality, bad quality, normal quality, lowres, blurry, watermark, text, "
    "logo, signature, UI, title, textbox, stats panel, trading card, card frame, character sheet, "
    "turnaround sheet, photorealistic, 3d render, bad anatomy, bad hands, malformed hands, "
    "extra fingers, missing fingers, fused fingers, extra arms, cropped face, face out of frame, "
    "unreadable face, black face, asymmetrical eyes"
)
FULLBODY_PRODUCTION_NEGATIVE_BASELINE = (
    "low quality, worst quality, bad quality, lowres, blurry, watermark, text, letters, typography, "
    "caption, title, logo, UI, card frame, "
    "character sheet, turnaround sheet, cropped feet, cropped legs, cropped body, head out of frame, "
    "tiny face, unreadable face, black face, bad anatomy, bad hands, malformed hands, extra fingers, "
    "missing fingers, extra arms, extra legs, bad feet, multiple characters, 2girls, two girls, duo, "
    "twins, companion, partner, group, another person, holding hands, overlapping bodies, "
    "photorealistic, 3d render, sketch, rough sketch, unfinished, draft, doodle, monochrome, grayscale, "
    "flat colors, uncolored, rough lineart, messy lineart, pencil sketch, storyboard, lineart only, "
    "rough coloring, under-rendered, flat patterned background, wallpaper background, blank background, "
    "simple background, close-up, portrait crop, cowboy shot, knee crop, missing feet, hidden hands, fused fingers"
)
INGAME_CG_PRODUCTION_NEGATIVE_BASELINE = (
    "low quality, worst quality, bad quality, lowres, blurry, watermark, text, logo, UI, textbox, "
    "card frame, character sheet, turnaround sheet, multiple views, poster collage, tiny face, "
    "unreadable face, black face, bad anatomy, bad hands, malformed hands, extra fingers, missing fingers, "
    "photorealistic, 3d render"
)
KEY_VISUAL_SUBCULTURE_NEGATIVE_BASELINE = (
    "low quality, worst quality, bad quality, normal quality, lowres, blurry, watermark, text, logo, "
    "signature, letters, typography, UI, title, textbox, stats panel, trading card, card frame, "
    "character sheet, turnaround sheet, copied layout, copied character design, photorealistic, 3d render, "
    "bad anatomy, bad hands, malformed hands, extra fingers, missing fingers, "
    "fused fingers, extra arms, cropped face, face out of frame, unreadable face, black face, asymmetrical eyes"
)
V8_STYLE_NEGATIVE_COMPOSITION_GUARD = (
    "extreme close-up, close-up, headshot, bust shot, cropped body, pov, first-person view, "
    "knees foreground, fisheye, face filling frame, body out of frame, male, muscular man, "
    "camera head, faceless, silhouette-only character, abstract body, monster body"
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
PORTRAIT_PRODUCTION_KEYWORDS: Tuple[str, ...] = (
    "초상",
    "얼굴",
    "상반신",
    "흉상",
    "프로필",
    "프로필사진",
    "portrait",
    "upper body",
    "bust",
    "headshot",
    "face focus",
    "icon",
    "avatar",
)
NSFW_WEIGHTED_TAG_RE = re.compile(r"(?:^|,\s*)\(?\s*NSFW\s*:\s*[-+]?\d+(?:\.\d+)?\s*\)?", re.IGNORECASE)
NSFW_TEXT_TAG_RE = re.compile(r"(?:^|,\s*)\(?\s*(?:no\s+)?NSFW\s*\)?", re.IGNORECASE)

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


def _make_slack_preview_image(image_path: Path, *, max_edge: int = 2048) -> Optional[Path]:
    """Create a smaller PNG preview for Slack when a generated PNG is too large."""
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        logger.warning("slack_preview_pillow_unavailable path=%s err=%s", image_path, exc)
        return None
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            largest = max(width, height)
            if largest <= max_edge:
                return None
            scale = max_edge / float(largest)
            preview_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            preview = image.convert("RGBA")
            preview.thumbnail(preview_size, Image.Resampling.LANCZOS)
            preview_path = image_path.with_name(f"{image_path.stem}_slack_preview.png")
            preview.save(preview_path, "PNG", optimize=True)
            return preview_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("slack_preview_create_failed path=%s err=%s", image_path, exc)
        return None


def _upload_comfy_input_file(base_url: str, source_path: Path) -> Dict[str, Any]:
    """Upload a local source image to ComfyUI input storage for LoadImage."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Source image not found: {source_path}")
    with source_path.open("rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}/upload/image",
            files={"image": (source_path.name, handle, "image/png")},
            data={"overwrite": "true", "type": "input"},
            timeout=60,
        )
    response.raise_for_status()
    payload = response.json()
    name = str(payload.get("name") or payload.get("filename") or source_path.name).strip()
    if not name:
        raise ValueError("ComfyUI upload response did not include an input image name")
    return {
        "name": name,
        "subfolder": str(payload.get("subfolder") or ""),
        "type": str(payload.get("type") or "input"),
        "source_path": str(source_path),
    }


def _image_work_root_from_source(source_path: Path) -> Optional[Path]:
    parts = source_path.parts
    for idx, part in enumerate(parts):
        if part == "Image" and idx > 0:
            return Path(*parts[: idx + 1])
    env_root = os.getenv("HERMES_WORK_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser() / "Image"
        if candidate.is_dir():
            return candidate
    fallback = Path("/Volumes/SSD_Hermes/HermesWork/Image")
    return fallback if fallback.is_dir() else None


def _candidate_normalized_paths(path: Path) -> List[Path]:
    raw = str(path)
    candidates = [path]
    for form in ("NFC", "NFD"):
        normalized = Path(unicodedata.normalize(form, raw)).expanduser()
        if normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _loose_source_filename_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", str(value or "")).casefold()
    return re.sub(r"[\s_-]+", "", normalized)


def _find_unique_source_by_project_version(source_path: Path) -> Optional[Path]:
    image_root = _image_work_root_from_source(source_path)
    if image_root is None or not image_root.is_dir():
        return None
    parent_name = unicodedata.normalize("NFC", source_path.parent.name)
    date_match = re.match(r"^(\d{6})[_-]", parent_name)
    version_match = re.search(
        r"(_v\d+)(\.(?:png|jpg|jpeg|webp))$",
        unicodedata.normalize("NFC", source_path.name),
        re.IGNORECASE,
    )
    if not date_match or not version_match:
        return None
    date_prefix = date_match.group(1)
    version_suffix = version_match.group(1).casefold()
    extension = version_match.group(2).casefold()
    matches: List[Path] = []
    for project_dir in image_root.glob(f"{date_prefix}*"):
        if not project_dir.is_dir():
            continue
        for candidate in project_dir.rglob("*"):
            if not candidate.is_file():
                continue
            name = unicodedata.normalize("NFC", candidate.name).casefold()
            if name.endswith(f"{version_suffix}{extension}"):
                matches.append(candidate)
    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in matches:
        key = str(candidate.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else None


def _resolve_existing_source_image_path(source_image_path: str | Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    requested = Path(source_image_path).expanduser()
    attempts: List[str] = []
    for candidate in _candidate_normalized_paths(requested):
        attempts.append(str(candidate))
        if candidate.is_file():
            return candidate, {
                "requested_source_image_path": str(requested),
                "resolved_source_image_path": str(candidate),
                "source_path_resolution": "exact" if candidate == requested else "unicode_normalized",
                "source_path_resolution_attempts": attempts,
            }
        parent = candidate.parent
        if parent.is_dir():
            wanted = unicodedata.normalize("NFC", candidate.name)
            sibling_matches = [
                item for item in parent.iterdir()
                if item.is_file() and unicodedata.normalize("NFC", item.name) == wanted
            ]
            if len(sibling_matches) == 1:
                resolved = sibling_matches[0]
                return resolved, {
                    "requested_source_image_path": str(requested),
                    "resolved_source_image_path": str(resolved),
                    "source_path_resolution": "sibling_unicode_normalized",
                    "source_path_resolution_attempts": attempts,
                }
            wanted_loose = _loose_source_filename_key(candidate.name)
            loose_matches = [
                item for item in parent.iterdir()
                if item.is_file() and _loose_source_filename_key(item.name) == wanted_loose
            ]
            if len(loose_matches) == 1:
                resolved = loose_matches[0]
                return resolved, {
                    "requested_source_image_path": str(requested),
                    "resolved_source_image_path": str(resolved),
                    "source_path_resolution": "sibling_loose_filename_match",
                    "source_path_resolution_attempts": attempts,
                }
    project_version_match = _find_unique_source_by_project_version(requested)
    if project_version_match is not None:
        return project_version_match, {
            "requested_source_image_path": str(requested),
            "resolved_source_image_path": str(project_version_match),
            "source_path_resolution": "unique_date_version_match",
            "source_path_resolution_attempts": attempts,
        }
    return None, {
        "requested_source_image_path": str(requested),
        "resolved_source_image_path": None,
        "source_path_resolution": "not_found",
        "source_path_resolution_attempts": attempts,
    }


_SOURCE_IMAGE_PROMPT_ONLY_RE = re.compile(
    r"(source_image_path\s*:|source_image\s*:|operation\s*:\s*[\"']?(?:source_preserving_postprocess|postprocess|local_retouch|masked_inpaint|upscale)|"
    r"source[-_ ]preserving[-_ ]postprocess)",
    re.IGNORECASE,
)


def _looks_like_source_image_task_prompt_without_args(prompt: str, *, source_image_path: str, operation: str) -> bool:
    """Detect source-image jobs that were accidentally passed as txt2img prose."""
    if source_image_path or operation:
        return False
    return bool(_SOURCE_IMAGE_PROMPT_ONLY_RE.search(str(prompt or "")))


def _build_face8m_hand9c_postprocess_workflow(
    *,
    checkpoint: str,
    source_image_name: str,
    positive_prompt: str,
    negative_prompt: str,
    filename_prefix: str,
) -> Dict[str, Any]:
    return {
        "1": {"inputs": {"ckpt_name": checkpoint}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"image": source_image_name}, "class_type": "LoadImage"},
        "3": {"inputs": {"text": positive_prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "4": {"inputs": {"text": negative_prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "5": {"inputs": {"model_name": "bbox/face_yolov8m.pt"}, "class_type": "UltralyticsDetectorProvider"},
        "6": {
            "inputs": {
                "image": ["2", 0],
                "model": ["1", 0],
                "clip": ["1", 1],
                "vae": ["1", 2],
                "guide_size": 512,
                "guide_size_for": True,
                "max_size": 1024,
                "seed": 20260643,
                "steps": 16,
                "cfg": 5.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["3", 0],
                "negative": ["4", 0],
                "denoise": 0.35,
                "feather": 6,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": 0.45,
                "bbox_dilation": 8,
                "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1",
                "sam_dilation": 0,
                "sam_threshold": 0.93,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "bbox_detector": ["5", 0],
                "wildcard": "",
                "cycle": 1,
            },
            "class_type": "FaceDetailer",
        },
        "7": {
            "inputs": {
                "text": (
                    "masterpiece, best quality, high quality, anime illustration, clean hand anatomy, "
                    "elegant fingers, natural hand pose, five fingers, clean lineart, polished cel shading, "
                    "consistent character art style, preserve original character identity and lighting"
                ),
                "clip": ["1", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "8": {
            "inputs": {
                "text": (
                    "low quality, blurry, bad hands, malformed hands, extra fingers, missing fingers, "
                    "fused fingers, broken fingers, mutated fingers, deformed palm, distorted hand, "
                    "photorealistic, 3d render, changing pose, changing character identity"
                ),
                "clip": ["1", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "9": {"inputs": {"model_name": "bbox/hand_yolov9c.pt"}, "class_type": "UltralyticsDetectorProvider"},
        "10": {
            "inputs": {
                "bbox_detector": ["9", 0],
                "image": ["6", 0],
                "threshold": 0.35,
                "dilation": 10,
                "crop_factor": 3.0,
                "drop_size": 12,
                "labels": "all",
            },
            "class_type": "BboxDetectorSEGS",
        },
        "11": {
            "inputs": {
                "image": ["6", 0],
                "segs": ["10", 0],
                "model": ["1", 0],
                "clip": ["1", 1],
                "vae": ["1", 2],
                "guide_size": 384,
                "guide_size_for": True,
                "max_size": 768,
                "seed": 20260697,
                "steps": 14,
                "cfg": 5.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["7", 0],
                "negative": ["8", 0],
                "denoise": 0.25,
                "feather": 5,
                "noise_mask": True,
                "force_inpaint": True,
                "wildcard": "",
                "cycle": 1,
            },
            "class_type": "DetailerForEach",
        },
        "99": {"inputs": {"images": ["11", 0], "filename_prefix": filename_prefix}, "class_type": "SaveImage"},
    }


def _build_depth50_canny100_face8m_hand9c_postprocess_workflow(
    *,
    checkpoint: str,
    source_image_name: str,
    positive_prompt: str,
    negative_prompt: str,
    filename_prefix: str,
    seed: int,
) -> Dict[str, Any]:
    return {
        "1": {"inputs": {"ckpt_name": checkpoint}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"vae_name": SOURCE_PRESERVING_DEPTH_CANNY_VAE}, "class_type": "VAELoader"},
        "3": {
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": SOURCE_PRESERVING_DEPTH_CANNY_STYLE_LORA,
                "strength_model": SOURCE_PRESERVING_DEPTH_CANNY_STYLE_LORA_WEIGHT,
                "strength_clip": SOURCE_PRESERVING_DEPTH_CANNY_STYLE_LORA_WEIGHT,
            },
            "class_type": "LoraLoader",
        },
        "4": {
            "inputs": {
                "model": ["3", 0],
                "clip": ["3", 1],
                "lora_name": SOURCE_PRESERVING_DEPTH_CANNY_UTILITY_LORA,
                "strength_model": SOURCE_PRESERVING_DEPTH_CANNY_UTILITY_LORA_WEIGHT,
                "strength_clip": SOURCE_PRESERVING_DEPTH_CANNY_UTILITY_LORA_WEIGHT,
            },
            "class_type": "LoraLoader",
        },
        "5": {"inputs": {"text": positive_prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "6": {"inputs": {"text": negative_prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "7": {"inputs": {"width": 1024, "height": 1536, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "20": {"inputs": {"image": source_image_name}, "class_type": "LoadImage"},
        "21": {"inputs": {"image": ["20", 0], "resolution": 1024}, "class_type": "Zoe_DepthAnythingPreprocessor"},
        "22": {
            "inputs": {
                "image": ["20", 0],
                "low_threshold": 100,
                "high_threshold": 200,
                "resolution": 1024,
            },
            "class_type": "CannyEdgePreprocessor",
        },
        "25": {"inputs": {"control_net_name": SOURCE_PRESERVING_DEPTH_CONTROLNET}, "class_type": "ControlNetLoader"},
        "26": {
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "control_net": ["25", 0],
                "image": ["21", 0],
                "strength": 0.5,
                "start_percent": 0.0,
                "end_percent": 0.5,
            },
            "class_type": "ControlNetApplyAdvanced",
        },
        "27": {"inputs": {"control_net_name": SOURCE_PRESERVING_CANNY_CONTROLNET}, "class_type": "ControlNetLoader"},
        "28": {
            "inputs": {
                "positive": ["26", 0],
                "negative": ["26", 1],
                "control_net": ["27", 0],
                "image": ["22", 0],
                "strength": 1.0,
                "start_percent": 0.0,
                "end_percent": 0.4,
            },
            "class_type": "ControlNetApplyAdvanced",
        },
        "8": {
            "inputs": {
                "seed": seed,
                "steps": 28,
                "cfg": 5.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["28", 0],
                "negative": ["28", 1],
                "latent_image": ["7", 0],
            },
            "class_type": "KSampler",
        },
        "9": {"inputs": {"samples": ["8", 0], "vae": ["2", 0]}, "class_type": "VAEDecode"},
        "10": {
            "inputs": {
                "text": (
                    "masterpiece, best quality, high quality, anime illustration, clean hand anatomy, "
                    "elegant fingers, natural hand pose, five fingers, clean lineart, polished cel shading, "
                    "consistent character art style, preserve original character identity and lighting"
                ),
                "clip": ["4", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "11": {
            "inputs": {
                "text": (
                    "low quality, blurry, bad hands, malformed hands, extra fingers, missing fingers, "
                    "fused fingers, broken fingers, mutated fingers, deformed palm, distorted hand, "
                    "photorealistic, 3d render, changing pose, changing character identity"
                ),
                "clip": ["4", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "30": {"inputs": {"model_name": "bbox/face_yolov8m.pt"}, "class_type": "UltralyticsDetectorProvider"},
        "31": {
            "inputs": {
                "image": ["9", 0],
                "model": ["4", 0],
                "clip": ["4", 1],
                "vae": ["2", 0],
                "guide_size": 512,
                "guide_size_for": True,
                "max_size": 1024,
                "seed": seed + 17,
                "steps": 16,
                "cfg": 5.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["5", 0],
                "negative": ["6", 0],
                "denoise": 0.35,
                "feather": 6,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": 0.45,
                "bbox_dilation": 8,
                "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1",
                "sam_dilation": 0,
                "sam_threshold": 0.93,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "bbox_detector": ["30", 0],
                "wildcard": "",
                "cycle": 1,
            },
            "class_type": "FaceDetailer",
        },
        "32": {"inputs": {"model_name": "bbox/hand_yolov9c.pt"}, "class_type": "UltralyticsDetectorProvider"},
        "33": {
            "inputs": {
                "bbox_detector": ["32", 0],
                "image": ["31", 0],
                "threshold": 0.35,
                "dilation": 10,
                "crop_factor": 3.0,
                "drop_size": 12,
                "labels": "all",
            },
            "class_type": "BboxDetectorSEGS",
        },
        "34": {
            "inputs": {
                "image": ["31", 0],
                "segs": ["33", 0],
                "model": ["4", 0],
                "clip": ["4", 1],
                "vae": ["2", 0],
                "guide_size": 384,
                "guide_size_for": True,
                "max_size": 768,
                "seed": seed + 71,
                "steps": 14,
                "cfg": 5.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["10", 0],
                "negative": ["11", 0],
                "denoise": 0.25,
                "feather": 5,
                "noise_mask": True,
                "force_inpaint": True,
                "wildcard": "",
                "cycle": 1,
            },
            "class_type": "DetailerForEach",
        },
        "99": {"inputs": {"images": ["34", 0], "filename_prefix": filename_prefix}, "class_type": "SaveImage"},
    }


def _build_source_image_upscale_workflow(
    *,
    source_image_name: str,
    upscale_model: str,
    filename_prefix: str,
) -> Dict[str, Any]:
    return {
        "1": {"inputs": {"image": source_image_name}, "class_type": "LoadImage"},
        "2": {"inputs": {"model_name": upscale_model}, "class_type": "UpscaleModelLoader"},
        "3": {"inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}, "class_type": "ImageUpscaleWithModel"},
        "99": {"inputs": {"images": ["3", 0], "filename_prefix": filename_prefix}, "class_type": "SaveImage"},
    }


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "on", "allow", "enabled"}
    return False


def _reference_identity_workflow_family(workflow_key: Any) -> Optional[str]:
    key = str(workflow_key or "").strip()
    if key in {FULLBODY_V8_WORKFLOW_KEY, CHARACTER_REFERENCE_FULLBODY_EXPERIMENTAL_WORKFLOW_KEY}:
        return "fullbody"
    if key in {CHARACTER_KEY_VISUAL_WORKFLOW_KEY, CHARACTER_REFERENCE_KEY_VISUAL_EXPERIMENTAL_WORKFLOW_KEY}:
        return "key_visual"
    if key == PORTRAIT_WORKFLOW_KEY:
        return "portrait"
    return None


def _load_reference_identity_source_metadata(reference_input_path: Path) -> Dict[str, Any]:
    """Best-effort read of a published HermesWork image sidecar metadata file."""
    candidates = [
        reference_input_path.parent / "sidecar" / "metadata.json",
        reference_input_path.parent / "metadata.json",
    ]
    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            logger.debug("Could not read reference identity metadata: %s", candidate, exc_info=True)
    return {}


def _build_reference_identity_experimental_workflow(
    *,
    checkpoint: str,
    vae: Optional[str],
    lora_stack: List[Dict[str, Any]],
    reference_image_name: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    filename_prefix: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    workflow: Dict[str, Any] = {
        "1": {"inputs": {"ckpt_name": checkpoint}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "3": {"inputs": {"text": prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "4": {"inputs": {"text": negative_prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "50": {"inputs": {"ipadapter_file": REFERENCE_IDENTITY_IPADAPTER_MODEL}, "class_type": "IPAdapterModelLoader"},
        "51": {"inputs": {"clip_name": REFERENCE_IDENTITY_CLIP_VISION}, "class_type": "CLIPVisionLoader"},
        "52": {"inputs": {"image": reference_image_name}, "class_type": "LoadImage"},
        "53": {
            "inputs": {
                "model": ["1", 0],
                "ipadapter": ["50", 0],
                "image": ["52", 0],
                "weight": REFERENCE_IDENTITY_IPADAPTER_WEIGHT,
                "weight_type": REFERENCE_IDENTITY_IPADAPTER_WEIGHT_TYPE,
                "combine_embeds": "average",
                "start_at": REFERENCE_IDENTITY_IPADAPTER_START_AT,
                "end_at": REFERENCE_IDENTITY_IPADAPTER_END_AT,
                "embeds_scaling": REFERENCE_IDENTITY_IPADAPTER_EMBEDS_SCALING,
                "clip_vision": ["51", 0],
            },
            "class_type": "IPAdapterAdvanced",
        },
        "5": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": ["53", 0],
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

    model_source = ["1", 0]
    clip_source = ["1", 1]
    for index, lora in enumerate(lora_stack, start=1):
        node_id = str(20 + index)
        workflow[node_id] = {
            "inputs": {
                "model": model_source,
                "clip": clip_source,
                "lora_name": lora["name"],
                "strength_model": lora["weight"],
                "strength_clip": lora.get("clip_weight", lora["weight"]),
            },
            "class_type": "LoraLoader",
        }
        model_source = [node_id, 0]
        clip_source = [node_id, 1]
    if lora_stack:
        workflow["3"]["inputs"]["clip"] = clip_source
        workflow["4"]["inputs"]["clip"] = clip_source
        workflow["53"]["inputs"]["model"] = model_source

    workflow_node_audit = {
        "audit_version": "comfy_reference_identity_experimental_v1",
        "checkpoint_node": "1",
        "checkpoint": workflow.get("1", {}).get("inputs", {}).get("ckpt_name"),
        "vae_node": "6" if vae is not None else "1",
        "vae": workflow.get("6", {}).get("inputs", {}).get("vae_name") if vae is not None else "checkpoint_builtin_vae",
        "lora_nodes": [
            {
                "node": node_id,
                "name": node.get("inputs", {}).get("lora_name"),
                "weight": node.get("inputs", {}).get("strength_model"),
                "clip_weight": node.get("inputs", {}).get("strength_clip"),
            }
            for node_id, node in sorted(workflow.items(), key=lambda item: item[0])
            if isinstance(node, dict) and node.get("class_type") == "LoraLoader"
        ],
        "ipadapter": {
            "model_loader_node": "50",
            "clip_vision_node": "51",
            "reference_image_node": "52",
            "apply_node": "53",
            "model": REFERENCE_IDENTITY_IPADAPTER_MODEL,
            "clip_vision": REFERENCE_IDENTITY_CLIP_VISION,
            "weight": REFERENCE_IDENTITY_IPADAPTER_WEIGHT,
            "weight_type": REFERENCE_IDENTITY_IPADAPTER_WEIGHT_TYPE,
            "start_at": REFERENCE_IDENTITY_IPADAPTER_START_AT,
            "end_at": REFERENCE_IDENTITY_IPADAPTER_END_AT,
            "embeds_scaling": REFERENCE_IDENTITY_IPADAPTER_EMBEDS_SCALING,
        },
        "ksampler_node": "5",
        "steps": workflow.get("5", {}).get("inputs", {}).get("steps"),
        "cfg": workflow.get("5", {}).get("inputs", {}).get("cfg"),
        "sampler_name": workflow.get("5", {}).get("inputs", {}).get("sampler_name"),
        "scheduler": workflow.get("5", {}).get("inputs", {}).get("scheduler"),
        "width": workflow.get("2", {}).get("inputs", {}).get("width"),
        "height": workflow.get("2", {}).get("inputs", {}).get("height"),
    }
    model_stack_verified = (
        workflow_node_audit["checkpoint"] == checkpoint
        and workflow_node_audit["vae"] == (vae if vae is not None else "checkpoint_builtin_vae")
        and [item["name"] for item in workflow_node_audit["lora_nodes"]] == [item["name"] for item in lora_stack]
        and workflow_node_audit["ipadapter"]["model"] == REFERENCE_IDENTITY_IPADAPTER_MODEL
        and workflow_node_audit["ipadapter"]["clip_vision"] == REFERENCE_IDENTITY_CLIP_VISION
    )
    return workflow, workflow_node_audit, model_stack_verified


def _parse_mask_box(mask_box: Any) -> Tuple[float, float, float, float]:
    if isinstance(mask_box, dict):
        raw = (
            mask_box.get("x"),
            mask_box.get("y"),
            mask_box.get("w", mask_box.get("width")),
            mask_box.get("h", mask_box.get("height")),
        )
    elif isinstance(mask_box, (list, tuple)) and len(mask_box) == 4:
        raw = tuple(mask_box)
    elif isinstance(mask_box, str) and mask_box.strip():
        parts = [part.strip() for part in re.split(r"[\s,]+", mask_box.strip()) if part.strip()]
        if len(parts) != 4:
            raise ValueError("mask_box string must contain exactly 4 numbers: x,y,w,h")
        raw = tuple(parts)
    else:
        raise ValueError("mask_box is required and must be dict, list, tuple, or 'x,y,w,h' string")

    try:
        x, y, w, h = (float(value) for value in raw)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid mask_box values: {mask_box!r}") from exc

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("mask_box values must satisfy x>=0, y>=0, w>0, h>0")
    if x >= 1 or y >= 1 or w > 1 or h > 1:
        raise ValueError("mask_box values must be normalized ratios between 0 and 1")
    if x + w > 1 or y + h > 1:
        raise ValueError("mask_box must stay within the image bounds when using normalized ratios")
    return x, y, w, h


def _normalized_mask_box_to_pixels(
    mask_box: Tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    x, y, w, h = mask_box
    left = max(0, min(image_width - 1, int(round(x * image_width))))
    top = max(0, min(image_height - 1, int(round(y * image_height))))
    right = max(left + 1, min(image_width, int(round((x + w) * image_width))))
    bottom = max(top + 1, min(image_height, int(round((y + h) * image_height))))
    return left, top, right, bottom


def _create_mask_image_for_box(
    *,
    source_image_path: Path,
    mask_box: Tuple[float, float, float, float],
    feather_px: int,
    temp_prefix: str,
) -> Tuple[Path, Dict[str, Any]]:
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow is required to build a local mask image") from exc

    with Image.open(source_image_path) as source_image:
        width, height = source_image.size

    left, top, right, bottom = _normalized_mask_box_to_pixels(mask_box, image_width=width, image_height=height)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle((left, top, right - 1, bottom - 1), fill=255)
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))

    fd, temp_path = tempfile.mkstemp(prefix=f"{temp_prefix}_", suffix="_mask.png")
    os.close(fd)
    mask_path = Path(temp_path)
    mask.save(mask_path, "PNG")
    return mask_path, {
        "normalized": {"x": mask_box[0], "y": mask_box[1], "w": mask_box[2], "h": mask_box[3]},
        "pixel_box": {"left": left, "top": top, "right": right, "bottom": bottom},
        "image_size": {"width": width, "height": height},
        "feather_px": feather_px,
        "mask_shape": "rectangle",
        "mask_coverage_ratio": round(((right - left) * (bottom - top)) / max(1, width * height), 6),
    }


def _is_localized_retouch_target(mask_target: Optional[str]) -> bool:
    lowered = str(mask_target or "").casefold()
    return bool(lowered) and any(keyword.casefold() in lowered for keyword in LOCALIZED_RETOUCH_TARGET_KEYWORDS)


def _apply_masked_inpaint_safety_defaults(
    *,
    mask_target: Optional[str],
    kwargs: Dict[str, Any],
    denoise: float,
    feather_px: int,
    grow_mask_by: int,
) -> Tuple[float, int, int, Dict[str, Any]]:
    localized_target = _is_localized_retouch_target(mask_target)
    adjustments: List[str] = []
    denoise_was_explicit = isinstance(kwargs.get("denoise"), (int, float))
    feather_was_explicit = "mask_feather_px" in kwargs
    grow_was_explicit = "grow_mask_by" in kwargs

    if localized_target:
        if not grow_was_explicit and grow_mask_by != 0:
            grow_mask_by = 0
            adjustments.append("localized_target_default_grow_mask_by_0")
        if not feather_was_explicit:
            feather_px = 6
            adjustments.append("localized_target_default_mask_feather_px_6")
        if not denoise_was_explicit:
            denoise = min(denoise, 0.55)
            adjustments.append("localized_target_default_denoise_max_0_55")
        elif denoise > 0.65 and not bool(kwargs.get("allow_high_denoise")):
            denoise = 0.65
            adjustments.append("localized_target_capped_high_denoise_0_65")

    safety = {
        "localized_target": localized_target,
        "adjustments": adjustments,
        "requires_visual_review": localized_target,
        "visual_review_checks": [
            "masked region only changed",
            "no rectangular seam or patch boundary",
            "no background/outfit/forearm drift outside intended local area",
        ]
        if localized_target
        else [],
    }
    return denoise, max(0, feather_px), max(0, grow_mask_by), safety


def _build_masked_inpaint_workflow(
    *,
    checkpoint: str,
    source_image_name: str,
    mask_image_name: str,
    positive_prompt: str,
    negative_prompt: str,
    filename_prefix: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    grow_mask_by: int,
) -> Dict[str, Any]:
    return {
        "1": {"inputs": {"image": source_image_name}, "class_type": "LoadImage"},
        "2": {"inputs": {"image": mask_image_name, "channel": "red"}, "class_type": "LoadImageMask"},
        "3": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["12", 0],
            },
            "class_type": "KSampler",
        },
        "4": {"inputs": {"ckpt_name": checkpoint}, "class_type": "CheckpointLoaderSimple"},
        "6": {"inputs": {"text": positive_prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "7": {"inputs": {"text": negative_prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
        "12": {
            "inputs": {"pixels": ["1", 0], "mask": ["2", 0], "vae": ["4", 2], "grow_mask_by": grow_mask_by},
            "class_type": "VAEEncodeForInpaint",
        },
        "99": {"inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}, "class_type": "SaveImage"},
    }


def _detailer_bbox_detector_for_target(mask_target: Optional[str]) -> Dict[str, Any]:
    lowered = str(mask_target or "").casefold()
    if any(keyword in lowered for keyword in ("face", "eye", "mouth", "얼굴", "눈", "입")):
        return {
            "model_name": "bbox/face_yolov8m.pt",
            "threshold": 0.45,
            "dilation": 8,
            "crop_factor": 3.0,
            "drop_size": 10,
        }
    return {
        "model_name": "bbox/hand_yolov9c.pt",
        "threshold": 0.35,
        "dilation": 10,
        "crop_factor": 3.0,
        "drop_size": 12,
    }


def _build_detailer_bbox_masked_inpaint_workflow(
    *,
    checkpoint: str,
    source_image_name: str,
    positive_prompt: str,
    negative_prompt: str,
    filename_prefix: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    grow_mask_by: int,
    feather_px: int,
    mask_target: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    detector = _detailer_bbox_detector_for_target(mask_target)
    workflow = {
        "1": {"inputs": {"image": source_image_name}, "class_type": "LoadImage"},
        "3": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["12", 0],
            },
            "class_type": "KSampler",
        },
        "4": {"inputs": {"ckpt_name": checkpoint}, "class_type": "CheckpointLoaderSimple"},
        "5": {"inputs": {"model_name": detector["model_name"]}, "class_type": "UltralyticsDetectorProvider"},
        "6": {"inputs": {"text": positive_prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "7": {"inputs": {"text": negative_prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
        "9": {
            "inputs": {
                "bbox_detector": ["5", 0],
                "image": ["1", 0],
                "threshold": detector["threshold"],
                "dilation": detector["dilation"],
                "crop_factor": detector["crop_factor"],
                "drop_size": detector["drop_size"],
                "labels": "all",
            },
            "class_type": "BboxDetectorSEGS",
        },
        "10": {"inputs": {"segs": ["9", 0]}, "class_type": "SegsToCombinedMask"},
        "11": {
            "inputs": {"mask": ["10", 0], "expand": grow_mask_by, "tapered_corners": True},
            "class_type": "GrowMask",
        },
        "12": {
            "inputs": {"pixels": ["1", 0], "mask": ["13", 0], "vae": ["4", 2], "grow_mask_by": 0},
            "class_type": "VAEEncodeForInpaint",
        },
        "13": {
            "inputs": {
                "mask": ["11", 0],
                "left": feather_px,
                "top": feather_px,
                "right": feather_px,
                "bottom": feather_px,
            },
            "class_type": "FeatherMask",
        },
        "99": {"inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}, "class_type": "SaveImage"},
    }
    return workflow, detector


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


def _normalize_output_type_token(output_type: Any) -> str:
    return str(output_type or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _is_profile_icon_production_request(*, output_type: Any = None) -> bool:
    requested_output_type = _normalize_output_type_token(output_type)
    return requested_output_type in {"profile_icon", "profileicon", "avatar", "icon"}


def _is_portrait_production_request(prompt_text: str, *, output_type: Any = None) -> bool:
    requested_output_type = _normalize_output_type_token(output_type)
    if requested_output_type in {"portrait", "face_portrait", "bust_portrait"}:
        return True
    lowered = str(prompt_text or "").casefold()
    return any(keyword.casefold() in lowered for keyword in PORTRAIT_PRODUCTION_KEYWORDS)


def _is_v8_style_workflow_request(prompt_text: str) -> bool:
    lowered = str(prompt_text or "").casefold()
    return any(keyword.casefold() in lowered for keyword in V8_STYLE_WORKFLOW_KEYWORDS)


def _is_key_visual_subculture_request(prompt_text: str, *, workflow_key: Any = None, output_type: Any = None) -> bool:
    requested_workflow = str(workflow_key or "").strip()
    requested_output_type = _normalize_output_type_token(output_type)
    if requested_workflow in {
        CHARACTER_KEY_VISUAL_WORKFLOW_KEY,
        CHARACTER_REFERENCE_KEY_VISUAL_EXPERIMENTAL_WORKFLOW_KEY,
    }:
        return True
    if requested_output_type in {"key_visual", "keyvisual", "promotional_key_visual"}:
        return True
    lowered = str(prompt_text or "").casefold()
    return any(
        keyword in lowered
        for keyword in (
            "key visual",
            "key_visual",
            "키비주얼",
            "promotional poster",
            "poster artwork",
            "multi-zone",
            "three stacked",
            "3구역",
            "3개 영역",
        )
    )


def _is_upper_body_production_request(*, output_type: Any = None) -> bool:
    requested_output_type = _normalize_output_type_token(output_type)
    return requested_output_type in {"upper_body", "upperbody", "half_body", "knee_up", "knees_up"}


def _is_dialogue_bust_production_request(*, output_type: Any = None) -> bool:
    requested_output_type = _normalize_output_type_token(output_type)
    return requested_output_type in {"dialogue_bust", "dialoguebust", "bust", "visual_novel_bust", "talking_bust"}


def _is_fullbody_production_request(*, output_type: Any = None) -> bool:
    requested_output_type = _normalize_output_type_token(output_type)
    return requested_output_type in {"fullbody", "full_body", "full_body_shot", "fullbody_shot", "whole_body"}


def _is_standing_sprite_production_request(*, output_type: Any = None) -> bool:
    requested_output_type = _normalize_output_type_token(output_type)
    return requested_output_type in {
        "standing_sprite",
        "standingsprite",
        "standing",
        "standing_character",
        "game_sprite",
        "sprite",
    }


def _is_ingame_cg_production_request(*, output_type: Any = None) -> bool:
    requested_output_type = _normalize_output_type_token(output_type)
    return requested_output_type in {"ingame_cg", "in_game_cg", "event_cg", "story_cg", "cg"}


POSITIVE_PROMPT_NEGATION_PATTERNS: Tuple[str, ...] = (
    r"\bno\s+text\b",
    r"\bno\s+logo\b",
    r"\bno\s+watermark\b",
    r"\bno\s+signature\b",
    r"\bno\s+letters?\b",
    r"\bno\s+typography\b",
    r"\bno\s+caption\b",
    r"\bno\s+title\b",
    r"\bno\s+ui\b",
    r"\bno\s+speech\s+bubble\b",
    r"\bno\s+text\s*box\b",
    r"\bwithout\s+text\b",
    r"\bwithout\s+logo\b",
    r"\bwithout\s+watermark\b",
)


def _strip_positive_prompt_negation_terms(prompt_text: str) -> str:
    sanitized = str(prompt_text or "")
    for pattern in POSITIVE_PROMPT_NEGATION_PATTERNS:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s*,\s*,+", ", ", sanitized)
    sanitized = re.sub(r"^\s*,\s*|\s*,\s*$", "", sanitized)
    return re.sub(r"\s+", " ", sanitized).strip()


def _sanitize_sfw_prompt_terms(prompt_text: str) -> str:
    sanitized = NSFW_WEIGHTED_TAG_RE.sub("", str(prompt_text or ""))
    sanitized = NSFW_TEXT_TAG_RE.sub("", sanitized)
    sanitized = _strip_positive_prompt_negation_terms(sanitized)
    sanitized = re.sub(r"\s*,\s*,+", ", ", sanitized)
    sanitized = re.sub(r"^\s*,\s*|\s*,\s*$", "", sanitized)
    return re.sub(r"\s+", " ", sanitized).strip()


REFERENCE_IDENTITY_META_PROMPT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\buse\s+the\s+reference\s+image\s+only\s+as\s+temporary\s+identity\s+guidance\s+for\s+this\s+experiment\b", ", "),
    (r"\buse\s+the\s+reference\s+image\s+only\s+as\s+identity\s+guidance\b", ", "),
    (r"\buse\s+the\s+reference\s+image\b", ", "),
    (r"\breference\s+image\b", ", "),
    (r"\btemporary\s+identity\s+guidance\b", ", "),
    (r"\bthis\s+experiment\b", ", "),
    (r"\bsame\s+original\s+heroine\s+identity\b", ", "),
    (r"\bsame\s+character\s+identity\s+as\s+reference\b", ", "),
    (r"\bsame\s+character\s+as\s+reference\s+image\b", ", "),
    (r"\bkeep\s+(her|him|them|the\s+character)\s+recognizable\s+as\s+the\s+same\s+character\b", ", recognizable character identity, "),
    (r"\bpreserve\s+the\s+fullbody_v8\s+image\s+type\b", ", fullbody_v8 image type, "),
    (r"\bpreserve\s+the\s+[^,.;:]+?\s+image\s+type\b", ", "),
    (r"\bnew\s+scene\b", ", scene, "),
    (r"\bwhile\s+changing\s+only\s+scene\s+and\s+poster\s+composition\b", ", "),
    (r"\bwhile\s+changing\s+only\s+[^,.;:]+", ", "),
    (r"\bsame\s+(short|long|pink|blue|teal|navy|white|black|red|blonde|green|golden|youthful|elegant|academy|uniform|hair|eyes|outfit|costume|sailor)\b", r"\1"),
)


def _sanitize_reference_identity_prompt_terms(prompt_text: str) -> str:
    """Strip routing prose from reference prompts before they reach CLIPTextEncode."""
    sanitized = _sanitize_sfw_prompt_terms(prompt_text)
    sanitized = re.sub(r"[.;:]+", ", ", sanitized)
    for pattern, replacement in REFERENCE_IDENTITY_META_PROMPT_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\b(no|not|without)\s+(text|logo|watermark|caption|title|ui|speech bubble)\b", "", sanitized, flags=re.IGNORECASE)
    while re.search(r",\s*,", sanitized):
        sanitized = re.sub(r"\s*,\s*,+", ", ", sanitized)
    sanitized = re.sub(r"^\s*,\s*|\s*,\s*$", "", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def _sanitize_portrait_prompt_terms(prompt_text: str) -> str:
    sanitized = _sanitize_sfw_prompt_terms(prompt_text)
    sanitized = re.sub(r"\(?\s*no\s+full\s+body\s*\)?", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\(?\s*no\s+wide\s+shot\s*\)?", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s*,\s*,+", ", ", sanitized)
    sanitized = re.sub(r"^\s*,\s*|\s*,\s*$", "", sanitized)
    return re.sub(r"\s+", " ", sanitized).strip()


def _translate_character_production_prompt(prompt_text: str) -> str:
    translated = _sanitize_sfw_prompt_terms(prompt_text)
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


def _translate_portrait_production_prompt(prompt_text: str) -> str:
    translated = _sanitize_portrait_prompt_terms(prompt_text)
    if not translated:
        return translated
    for pattern, replacement in (
        (r"주인공급", "heroine-grade"),
        (r"미소녀", "beautiful girl"),
        (r"캐릭터", "character"),
        (r"초상", "portrait"),
        (r"얼굴", "face focus"),
        (r"상반신", "upper body"),
        (r"흉상", "bust portrait"),
        (r"프로필사진", "profile portrait"),
        (r"프로필", "profile portrait"),
    ):
        translated = re.sub(pattern, replacement, translated, flags=re.IGNORECASE)
    translated = re.sub(r"\bfull body\b", "upper body portrait", translated, flags=re.IGNORECASE)
    translated = re.sub(r"\s+", " ", translated).strip()
    return _merge_comma_terms(PORTRAIT_PRODUCTION_POSITIVE_SKELETON, translated)


def _translate_profile_icon_production_prompt(prompt_text: str) -> str:
    translated = _sanitize_portrait_prompt_terms(prompt_text)
    if not translated:
        return translated
    return _merge_comma_terms(PROFILE_ICON_PRODUCTION_POSITIVE_SKELETON, translated)


def _translate_dialogue_bust_production_prompt(prompt_text: str) -> str:
    translated = _sanitize_portrait_prompt_terms(prompt_text)
    if not translated:
        return translated
    return _merge_comma_terms(DIALOGUE_BUST_PRODUCTION_POSITIVE_SKELETON, translated)


def _translate_upper_body_production_prompt(prompt_text: str) -> str:
    translated = _sanitize_sfw_prompt_terms(prompt_text)
    if not translated:
        return translated
    return _merge_comma_terms(UPPER_BODY_PRODUCTION_POSITIVE_SKELETON, translated)


def _translate_fullbody_production_prompt(prompt_text: str) -> str:
    translated = _sanitize_sfw_prompt_terms(prompt_text)
    if not translated:
        return translated
    return _merge_comma_terms(FULLBODY_PRODUCTION_POSITIVE_SKELETON, translated)


def _translate_standing_sprite_production_prompt(prompt_text: str) -> str:
    translated = _sanitize_sfw_prompt_terms(prompt_text)
    if not translated:
        return translated
    return _merge_comma_terms(STANDING_SPRITE_PRODUCTION_POSITIVE_SKELETON, translated)


def _translate_ingame_cg_production_prompt(prompt_text: str) -> str:
    translated = _sanitize_sfw_prompt_terms(prompt_text)
    if not translated:
        return translated
    return _merge_comma_terms(INGAME_CG_PRODUCTION_POSITIVE_SKELETON, translated)


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
    workflow_key: Any = None,
    output_type: Any = None,
) -> Optional[Dict[str, Any]]:
    if _is_v8_style_workflow_request(prompt):
        negative_prompt_text = str(negative_prompt or "").strip()
        source_prompt = _merge_comma_terms(
            _sanitize_sfw_prompt_terms(prompt),
            V8_STYLE_SUBJECT_BASELINE,
            V8_STYLE_COMPOSITION_GUARD,
        )
        final_negative_prompt = _merge_comma_terms(
            PORTRAIT_PRODUCTION_NEGATIVE_BASELINE,
            V8_STYLE_NEGATIVE_COMPOSITION_GUARD,
        )
        if negative_prompt_text:
            final_negative_prompt = _merge_comma_terms(final_negative_prompt, negative_prompt_text)
        return {
            "preset": V8_STYLE_WORKFLOW_PRESET,
            "workflow_key": PORTRAIT_WORKFLOW_KEY,
            "source_prompt": source_prompt,
            "translated_prompt": source_prompt,
            "prompt": source_prompt,
            "negative_prompt": final_negative_prompt,
            "negative_baseline": PORTRAIT_PRODUCTION_NEGATIVE_BASELINE,
            "subject_dominance": None,
            "subject_dominance_ratio": None,
            "subject_dominance_rule": None,
            "steps": CHARACTER_PRODUCTION_STEPS,
            "cfg": PORTRAIT_PRODUCTION_CFG,
            "sampler_name": PORTRAIT_PRODUCTION_SAMPLER,
            "scheduler": PORTRAIT_PRODUCTION_SCHEDULER,
            "use_checkpoint_vae": True,
            "prompt_translation_policy": "v8-style-workflow + sfw-sanitize + composition-guard + no portrait rewrite",
        }
    is_reference_key_visual = str(workflow_key or "").strip() == CHARACTER_REFERENCE_KEY_VISUAL_EXPERIMENTAL_WORKFLOW_KEY
    if _is_key_visual_subculture_request(prompt, workflow_key=workflow_key, output_type=output_type):
        negative_prompt_text = str(negative_prompt or "").strip()
        sanitized_prompt = (
            _sanitize_reference_identity_prompt_terms(prompt)
            if is_reference_key_visual
            else _sanitize_sfw_prompt_terms(prompt)
        )
        source_prompt = _merge_comma_terms(
            sanitized_prompt,
            KEY_VISUAL_SUBCULTURE_STYLE_BASELINE,
        )
        final_negative_prompt = KEY_VISUAL_SUBCULTURE_NEGATIVE_BASELINE
        if is_reference_key_visual:
            final_negative_prompt = _merge_comma_terms(final_negative_prompt, REFERENCE_IDENTITY_NEGATIVE_GUARD)
        if negative_prompt_text:
            final_negative_prompt = _merge_comma_terms(final_negative_prompt, negative_prompt_text)
        return {
            "preset": KEY_VISUAL_SUBCULTURE_PRESET,
            "workflow_key": CHARACTER_KEY_VISUAL_WORKFLOW_KEY,
            "source_prompt": source_prompt,
            "translated_prompt": source_prompt,
            "prompt": source_prompt,
            "negative_prompt": final_negative_prompt,
            "negative_baseline": KEY_VISUAL_SUBCULTURE_NEGATIVE_BASELINE,
            "subject_dominance": None,
            "subject_dominance_ratio": None,
            "subject_dominance_rule": None,
            "checkpoint": WIDER_CHARACTER_PRIMARY_CHECKPOINT,
            "vae": WIDER_CHARACTER_PRIMARY_VAE,
            "steps": KEY_VISUAL_SUBCULTURE_STEPS,
            "cfg": KEY_VISUAL_SUBCULTURE_CFG,
            "sampler_name": KEY_VISUAL_SUBCULTURE_SAMPLER,
            "scheduler": KEY_VISUAL_SUBCULTURE_SCHEDULER,
            "prompt_translation_policy": (
                "key-visual-subculture-v1 + "
                + ("reference-identity-prompt-scrub + low-style-ipadapter-guard + " if is_reference_key_visual else "")
                + "sfw-sanitize + v8-style-anchors-only + no subject/composition rewrite"
            ),
        }
    if _is_profile_icon_production_request(output_type=output_type):
        negative_prompt_text = str(negative_prompt or "").strip()
        source_prompt = _sanitize_portrait_prompt_terms(prompt)
        translated_prompt = _translate_profile_icon_production_prompt(source_prompt)
        final_negative_prompt = PROFILE_ICON_PRODUCTION_NEGATIVE_BASELINE
        if negative_prompt_text:
            final_negative_prompt = _merge_comma_terms(PROFILE_ICON_PRODUCTION_NEGATIVE_BASELINE, negative_prompt_text)
        return {
            "preset": PROFILE_ICON_PRODUCTION_PRESET,
            "workflow_key": DEFAULT_WORKFLOW_KEY,
            "source_prompt": source_prompt,
            "translated_prompt": translated_prompt,
            "prompt": translated_prompt,
            "negative_prompt": final_negative_prompt,
            "negative_baseline": PROFILE_ICON_PRODUCTION_NEGATIVE_BASELINE,
            "subject_dominance": None,
            "subject_dominance_ratio": None,
            "subject_dominance_rule": None,
            "width": PROFILE_ICON_PRODUCTION_WIDTH,
            "height": PROFILE_ICON_PRODUCTION_HEIGHT,
            "steps": CHARACTER_PRODUCTION_STEPS,
            "cfg": PORTRAIT_PRODUCTION_CFG,
            "sampler_name": PORTRAIT_PRODUCTION_SAMPLER,
            "scheduler": PORTRAIT_PRODUCTION_SCHEDULER,
            "prompt_translation_policy": "profile-icon-v1 + sfw-sanitize + square-avatar skeleton",
        }
    if _is_dialogue_bust_production_request(output_type=output_type):
        negative_prompt_text = str(negative_prompt or "").strip()
        source_prompt = _sanitize_portrait_prompt_terms(prompt)
        translated_prompt = _translate_dialogue_bust_production_prompt(source_prompt)
        final_negative_prompt = DIALOGUE_BUST_PRODUCTION_NEGATIVE_BASELINE
        if negative_prompt_text:
            final_negative_prompt = _merge_comma_terms(DIALOGUE_BUST_PRODUCTION_NEGATIVE_BASELINE, negative_prompt_text)
        return {
            "preset": DIALOGUE_BUST_PRODUCTION_PRESET,
            "workflow_key": DEFAULT_WORKFLOW_KEY,
            "source_prompt": source_prompt,
            "translated_prompt": translated_prompt,
            "prompt": translated_prompt,
            "negative_prompt": final_negative_prompt,
            "negative_baseline": DIALOGUE_BUST_PRODUCTION_NEGATIVE_BASELINE,
            "subject_dominance": None,
            "subject_dominance_ratio": None,
            "subject_dominance_rule": None,
            "steps": CHARACTER_PRODUCTION_STEPS,
            "cfg": PORTRAIT_PRODUCTION_CFG,
            "sampler_name": PORTRAIT_PRODUCTION_SAMPLER,
            "scheduler": PORTRAIT_PRODUCTION_SCHEDULER,
            "prompt_translation_policy": "dialogue-bust-v1 + sfw-sanitize + visual-novel bust skeleton",
        }
    if _is_upper_body_production_request(output_type=output_type):
        negative_prompt_text = str(negative_prompt or "").strip()
        source_prompt = _sanitize_sfw_prompt_terms(prompt)
        translated_prompt = _translate_upper_body_production_prompt(source_prompt)
        final_negative_prompt = UPPER_BODY_PRODUCTION_NEGATIVE_BASELINE
        if negative_prompt_text:
            final_negative_prompt = _merge_comma_terms(UPPER_BODY_PRODUCTION_NEGATIVE_BASELINE, negative_prompt_text)
        return {
            "preset": UPPER_BODY_PRODUCTION_PRESET,
            "workflow_key": DEFAULT_WORKFLOW_KEY,
            "source_prompt": source_prompt,
            "translated_prompt": translated_prompt,
            "prompt": translated_prompt,
            "negative_prompt": final_negative_prompt,
            "negative_baseline": UPPER_BODY_PRODUCTION_NEGATIVE_BASELINE,
            "subject_dominance": None,
            "subject_dominance_ratio": None,
            "subject_dominance_rule": None,
            "checkpoint": WIDER_CHARACTER_PRIMARY_CHECKPOINT,
            "vae": WIDER_CHARACTER_PRIMARY_VAE,
            "steps": KEY_VISUAL_SUBCULTURE_STEPS,
            "cfg": KEY_VISUAL_SUBCULTURE_CFG,
            "sampler_name": KEY_VISUAL_SUBCULTURE_SAMPLER,
            "scheduler": KEY_VISUAL_SUBCULTURE_SCHEDULER,
            "prompt_translation_policy": "upper-body-v1 + sfw-sanitize + no portrait skeleton rewrite",
        }
    is_reference_fullbody = str(workflow_key or "").strip() == CHARACTER_REFERENCE_FULLBODY_EXPERIMENTAL_WORKFLOW_KEY
    if (
        _is_fullbody_production_request(output_type=output_type)
        or is_reference_fullbody
    ):
        negative_prompt_text = str(negative_prompt or "").strip()
        source_prompt = (
            _sanitize_reference_identity_prompt_terms(prompt)
            if is_reference_fullbody
            else _sanitize_sfw_prompt_terms(prompt)
        )
        translated_prompt = _translate_fullbody_production_prompt(source_prompt)
        final_negative_prompt = FULLBODY_PRODUCTION_NEGATIVE_BASELINE
        if is_reference_fullbody:
            final_negative_prompt = _merge_comma_terms(final_negative_prompt, REFERENCE_IDENTITY_NEGATIVE_GUARD)
        if negative_prompt_text:
            final_negative_prompt = _merge_comma_terms(FULLBODY_PRODUCTION_NEGATIVE_BASELINE, negative_prompt_text)
            if is_reference_fullbody:
                final_negative_prompt = _merge_comma_terms(
                    FULLBODY_PRODUCTION_NEGATIVE_BASELINE,
                    REFERENCE_IDENTITY_NEGATIVE_GUARD,
                    negative_prompt_text,
                )
        return {
            "preset": FULLBODY_PRODUCTION_PRESET,
            "workflow_key": FULLBODY_V8_WORKFLOW_KEY,
            "source_prompt": source_prompt,
            "translated_prompt": translated_prompt,
            "prompt": translated_prompt,
            "negative_prompt": final_negative_prompt,
            "negative_baseline": FULLBODY_PRODUCTION_NEGATIVE_BASELINE,
            "subject_dominance": None,
            "subject_dominance_ratio": None,
            "subject_dominance_rule": None,
            "checkpoint": WIDER_CHARACTER_PRIMARY_CHECKPOINT,
            "vae": WIDER_CHARACTER_PRIMARY_VAE,
            "width": FULLBODY_PRODUCTION_WIDTH,
            "height": FULLBODY_PRODUCTION_HEIGHT,
            "steps": CHARACTER_PRODUCTION_STEPS,
            "cfg": PORTRAIT_PRODUCTION_CFG,
            "sampler_name": PORTRAIT_PRODUCTION_SAMPLER,
            "scheduler": PORTRAIT_PRODUCTION_SCHEDULER,
            "prompt_translation_policy": (
                "fullbody-v8-scene-v2 + sfw-sanitize + solo/head-to-toe guard + "
                "finished-color anti-sketch guard + scene/full-feet anti-wallpaper guard"
                + (" + reference-identity-prompt-scrub + low-style-ipadapter-guard" if is_reference_fullbody else "")
            ),
        }
    if _is_standing_sprite_production_request(output_type=output_type):
        negative_prompt_text = str(negative_prompt or "").strip()
        source_prompt = _sanitize_sfw_prompt_terms(prompt)
        translated_prompt = _translate_standing_sprite_production_prompt(source_prompt)
        final_negative_prompt = CHARACTER_PRODUCTION_NEGATIVE_BASELINE
        if negative_prompt_text:
            final_negative_prompt = _merge_comma_terms(CHARACTER_PRODUCTION_NEGATIVE_BASELINE, negative_prompt_text)
        return {
            "preset": STANDING_SPRITE_PRODUCTION_PRESET,
            "workflow_key": FULLBODY_V8_WORKFLOW_KEY,
            "source_prompt": source_prompt,
            "translated_prompt": translated_prompt,
            "prompt": translated_prompt,
            "negative_prompt": final_negative_prompt,
            "negative_baseline": CHARACTER_PRODUCTION_NEGATIVE_BASELINE,
            "subject_dominance": None,
            "subject_dominance_ratio": None,
            "subject_dominance_rule": None,
            "checkpoint": WIDER_CHARACTER_PRIMARY_CHECKPOINT,
            "vae": WIDER_CHARACTER_PRIMARY_VAE,
            "width": FULLBODY_PRODUCTION_WIDTH,
            "height": FULLBODY_PRODUCTION_HEIGHT,
            "steps": CHARACTER_PRODUCTION_STEPS,
            "cfg": CHARACTER_PRODUCTION_CFG,
            "sampler_name": CHARACTER_PRODUCTION_SAMPLER,
            "scheduler": CHARACTER_PRODUCTION_SCHEDULER,
            "prompt_translation_policy": (
                "standing-sprite-v1 + fullbody-v8-workflow + sfw-sanitize + "
                "production sprite skeleton + no portrait rewrite"
            ),
        }
    if _is_ingame_cg_production_request(output_type=output_type):
        negative_prompt_text = str(negative_prompt or "").strip()
        source_prompt = _sanitize_sfw_prompt_terms(prompt)
        translated_prompt = _translate_ingame_cg_production_prompt(source_prompt)
        final_negative_prompt = INGAME_CG_PRODUCTION_NEGATIVE_BASELINE
        if negative_prompt_text:
            final_negative_prompt = _merge_comma_terms(INGAME_CG_PRODUCTION_NEGATIVE_BASELINE, negative_prompt_text)
        return {
            "preset": INGAME_CG_PRODUCTION_PRESET,
            "workflow_key": DEFAULT_WORKFLOW_KEY,
            "source_prompt": source_prompt,
            "translated_prompt": translated_prompt,
            "prompt": translated_prompt,
            "negative_prompt": final_negative_prompt,
            "negative_baseline": INGAME_CG_PRODUCTION_NEGATIVE_BASELINE,
            "subject_dominance": None,
            "subject_dominance_ratio": None,
            "subject_dominance_rule": None,
            "checkpoint": WIDER_CHARACTER_PRIMARY_CHECKPOINT,
            "vae": WIDER_CHARACTER_PRIMARY_VAE,
            "steps": KEY_VISUAL_SUBCULTURE_STEPS,
            "cfg": KEY_VISUAL_SUBCULTURE_CFG,
            "sampler_name": KEY_VISUAL_SUBCULTURE_SAMPLER,
            "scheduler": KEY_VISUAL_SUBCULTURE_SCHEDULER,
            "prompt_translation_policy": "ingame-cg-v1 + sfw-sanitize + story-scene skeleton",
        }
    if _is_portrait_production_request(prompt, output_type=output_type):
        negative_prompt_text = str(negative_prompt or "").strip()
        source_prompt = _sanitize_portrait_prompt_terms(prompt)
        translated_prompt = _translate_portrait_production_prompt(source_prompt)
        final_negative_prompt = PORTRAIT_PRODUCTION_NEGATIVE_BASELINE
        if negative_prompt_text:
            final_negative_prompt = _merge_comma_terms(PORTRAIT_PRODUCTION_NEGATIVE_BASELINE, negative_prompt_text)
        return {
            "preset": PORTRAIT_PRODUCTION_PRESET,
            "workflow_key": PORTRAIT_WORKFLOW_KEY,
            "source_prompt": source_prompt,
            "translated_prompt": translated_prompt,
            "prompt": translated_prompt,
            "negative_prompt": final_negative_prompt,
            "negative_baseline": PORTRAIT_PRODUCTION_NEGATIVE_BASELINE,
            "subject_dominance": None,
            "subject_dominance_ratio": None,
            "subject_dominance_rule": None,
            "checkpoint": PORTRAIT_PRIMARY_CHECKPOINT,
            "vae": PORTRAIT_PRIMARY_VAE,
            "width": PORTRAIT_PRIMARY_WIDTH,
            "height": PORTRAIT_PRIMARY_HEIGHT,
            "steps": CHARACTER_PRODUCTION_STEPS,
            "cfg": PORTRAIT_PRODUCTION_CFG,
            "sampler_name": PORTRAIT_PRODUCTION_SAMPLER,
            "scheduler": PORTRAIT_PRODUCTION_SCHEDULER,
            "seed": PORTRAIT_PRODUCTION_SEED,
            "prompt_translation_policy": "portrait-round-v1-skeleton + keyword-translate + sfw-sanitize + portrait-primary-wai-knai-addmicro",
        }
    if not _is_character_production_request(prompt):
        return None
    negative_prompt_text = str(negative_prompt or "").strip()
    dominance_pct, normalized_subject_dominance, subject_rule = _normalize_subject_dominance(subject_dominance)
    source_prompt = _sanitize_sfw_prompt_terms(prompt)
    translated_prompt = _translate_character_production_prompt(source_prompt)
    final_prompt = f"{translated_prompt}, {subject_rule}" if subject_rule else translated_prompt
    final_negative_prompt = CHARACTER_PRODUCTION_NEGATIVE_BASELINE
    if negative_prompt_text:
        final_negative_prompt = _merge_comma_terms(CHARACTER_PRODUCTION_NEGATIVE_BASELINE, negative_prompt_text)
    return {
        "preset": CHARACTER_PRODUCTION_PRESET,
        "workflow_key": CHARACTER_KEY_VISUAL_WORKFLOW_KEY,
        "source_prompt": source_prompt,
        "translated_prompt": translated_prompt,
        "prompt": final_prompt,
        "negative_prompt": final_negative_prompt,
        "negative_baseline": CHARACTER_PRODUCTION_NEGATIVE_BASELINE,
        "subject_dominance": dominance_pct,
        "subject_dominance_ratio": normalized_subject_dominance,
        "subject_dominance_rule": subject_rule,
        "checkpoint": WIDER_CHARACTER_PRIMARY_CHECKPOINT,
        "vae": WIDER_CHARACTER_PRIMARY_VAE,
        "steps": CHARACTER_PRODUCTION_STEPS,
        "cfg": CHARACTER_PRODUCTION_CFG,
        "sampler_name": CHARACTER_PRODUCTION_SAMPLER,
        "scheduler": CHARACTER_PRODUCTION_SCHEDULER,
        "prompt_translation_policy": "character-skeleton + keyword-translate + subject-dominance guidance",
    }


def _resolve_lora_stack(kwargs: Dict[str, Any], *, runtime_preset: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Resolve ComfyUI LoRA stack from explicit kwargs or Angelica defaults."""
    def normalize_lora_name(value: str) -> str:
        # Commander JSON/text handoffs may double-escape Windows-style separators.
        # ComfyUI model lists use a single backslash as the relative path separator.
        return re.sub(r"\\+", r"\\", value.strip().replace("/", "\\"))

    explicit_stack = kwargs.get("loras")
    if isinstance(explicit_stack, list):
        resolved: List[Dict[str, Any]] = []
        for item in explicit_stack:
            if not isinstance(item, dict):
                continue
            name = normalize_lora_name(str(item.get("name") or item.get("lora_name") or ""))
            if not name:
                continue
            try:
                weight = float(item.get("weight", item.get("strength_model", 1.0)))
            except Exception:
                weight = 1.0
            try:
                clip_weight = float(item.get("clip_weight", item.get("strength_clip", weight)))
            except Exception:
                clip_weight = weight
            resolved.append({
                "preset": str(item.get("preset") or "custom").strip() or "custom",
                "name": name,
                "weight": weight,
                "clip_weight": clip_weight,
                "use_case": str(item.get("use_case") or "explicit stack").strip() or "explicit stack",
            })
        return resolved

    explicit_name = normalize_lora_name(str(kwargs.get("lora_name") or ""))
    if explicit_name:
        try:
            weight = float(kwargs.get("lora_weight", kwargs.get("strength_model", 1.0)))
        except Exception:
            weight = 1.0
        try:
            clip_weight = float(kwargs.get("lora_clip_weight", kwargs.get("strength_clip", weight)))
        except Exception:
            clip_weight = weight
        return [{
            "preset": str(kwargs.get("lora_preset") or kwargs.get("style_preset") or "custom").strip() or "custom",
            "name": explicit_name,
            "weight": weight,
            "clip_weight": clip_weight,
            "use_case": "explicit lora",
        }]

    preset_key = str(kwargs.get("lora_preset") or kwargs.get("style_preset") or "").strip().casefold().replace("-", "_").replace(" ", "_")
    runtime_preset_name = str((runtime_preset or {}).get("preset") or "")
    if not preset_key:
        if runtime_preset_name == PORTRAIT_PRODUCTION_PRESET:
            preset_key = "portrait_primary"
        elif runtime_preset_name in {
            KEY_VISUAL_SUBCULTURE_PRESET,
            REFERENCE_IDENTITY_EXPERIMENTAL_PRESET,
            REFERENCE_IDENTITY_FULLBODY_EXPERIMENTAL_PRESET,
        }:
            preset_key = "stable"
        elif runtime_preset_name in {
            CHARACTER_PRODUCTION_PRESET,
            PROFILE_ICON_PRODUCTION_PRESET,
            DIALOGUE_BUST_PRODUCTION_PRESET,
            UPPER_BODY_PRODUCTION_PRESET,
            FULLBODY_PRODUCTION_PRESET,
            STANDING_SPRITE_PRODUCTION_PRESET,
            INGAME_CG_PRODUCTION_PRESET,
        }:
            preset_key = "stable"
    if preset_key in STYLE_PRESET_LORAS:
        preset_config = STYLE_PRESET_LORAS[preset_key]
        preset_items = preset_config if isinstance(preset_config, list) else [preset_config]
        stack: List[Dict[str, Any]] = []
        for item in preset_items:
            preset = dict(item)
            preset["clip_weight"] = preset["weight"]
            stack.append(preset)
        return stack
    return []


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


def _read_png_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        with path.open("rb") as fh:
            header = fh.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
            return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
    except Exception:  # noqa: BLE001
        return None, None
    return None, None


def _update_metadata_report_evidence(metadata_path: Path, report_evidence: Dict[str, Any]) -> None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["report_evidence"] = report_evidence
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not update Comfy metadata report_evidence: %s", exc)


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
        configured_checkpoint = _resolve_model()
        explicit_checkpoint = isinstance(kwargs.get("model"), str) and bool(str(kwargs.get("model")).strip())
        checkpoint = str(kwargs.get("model") or configured_checkpoint).strip() or DEFAULT_CHECKPOINT
        requested_checkpoint = checkpoint
        explicit_vae = isinstance(kwargs.get("vae"), str) and bool(str(kwargs.get("vae")).strip())
        vae = str(kwargs.get("vae") or _resolve_vae() or "").strip() or None
        project_name = kwargs.get("project_name")
        artifact_name = kwargs.get("artifact_name")
        category = str(kwargs.get("category") or DEFAULT_CATEGORY).strip() or DEFAULT_CATEGORY
        explicit_dimensions = isinstance(kwargs.get("width"), int) and int(kwargs.get("width")) > 0 and isinstance(kwargs.get("height"), int) and int(kwargs.get("height")) > 0
        width, height = _resolve_dimensions(aspect, kwargs.get("width"), kwargs.get("height"))
        explicit_steps = isinstance(kwargs.get("steps"), int) and int(kwargs.get("steps")) > 0
        explicit_cfg = isinstance(kwargs.get("cfg_scale"), (int, float)) and float(kwargs.get("cfg_scale")) > 0
        explicit_sampler = isinstance(kwargs.get("sampler_name"), str) and bool(str(kwargs.get("sampler_name")).strip())
        explicit_scheduler = isinstance(kwargs.get("scheduler"), str) and bool(str(kwargs.get("scheduler")).strip())
        steps = int(kwargs.get("steps")) if explicit_steps else DEFAULT_STEPS
        cfg = float(kwargs.get("cfg_scale")) if explicit_cfg else DEFAULT_CFG
        denoise = float(kwargs.get("denoise")) if isinstance(kwargs.get("denoise"), (int, float)) and float(kwargs.get("denoise")) > 0 else DEFAULT_DENOISE
        sampler_name = str(kwargs.get("sampler_name") or DEFAULT_SAMPLER).strip() or DEFAULT_SAMPLER
        scheduler = str(kwargs.get("scheduler") or "normal").strip() or "normal"
        explicit_seed = isinstance(kwargs.get("seed"), int)
        seed = int(kwargs.get("seed")) if explicit_seed else DEFAULT_SEED
        negative_prompt = str(kwargs.get("negative_prompt") or "").strip()
        subject_dominance = kwargs.get("subject_dominance")
        requested_workflow_key = str(kwargs.get("workflow_key") or "").strip()
        output_type = kwargs.get("output_type")
        requested_output_type = str(output_type or "").strip() or None
        normalized_output_type = (
            requested_output_type.casefold().replace("-", "_").replace(" ", "_")
            if requested_output_type
            else None
        )
        operation = str(kwargs.get("operation") or "").strip()
        source_image_path = str(kwargs.get("source_image_path") or kwargs.get("source_image") or "").strip()
        reference_image_path = str(kwargs.get("reference_image_path") or kwargs.get("reference_image") or "").strip()
        reference_identity_experiment_enabled = _truthy_flag(kwargs.get("experimental_reference_identity"))
        allow_reference_workflow_family_change = _truthy_flag(kwargs.get("allow_reference_workflow_family_change"))
        reference_identity_requested = (
            operation == REFERENCE_IDENTITY_TXT2IMG_OPERATION
            or requested_workflow_key in REFERENCE_IDENTITY_EXPERIMENTAL_WORKFLOW_KEYS
            or reference_identity_experiment_enabled
        )
        reference_identity_explicit_route = (
            bool(reference_image_path)
            and operation == REFERENCE_IDENTITY_TXT2IMG_OPERATION
            and requested_workflow_key in REFERENCE_IDENTITY_EXPERIMENTAL_WORKFLOW_KEYS
        )
        if reference_image_path and not reference_identity_requested:
            return error_response(
                error=(
                    "reference_image_path is accepted only by the explicit experimental reference identity route. "
                    "Set operation=reference_identity_txt2img, "
                    "and workflow_key=character_reference_key_visual_experimental_v1. "
                    "experimental_reference_identity=true is recorded when provided, but is not the routing key."
                ),
                error_type="reference_identity_requires_explicit_experiment",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
                extra={
                    "reference_identity_status": "blocked_to_prevent_default_route_contamination",
                    "required_arguments": [
                        "operation=reference_identity_txt2img",
                        "workflow_key=character_reference_key_visual_experimental_v1 or fullbody_v8_reference_identity_experimental_v1",
                        "reference_image_path",
                    ],
                    "optional_audit_flag": "experimental_reference_identity=true",
                },
            )
        if reference_identity_requested and not reference_identity_explicit_route:
            return error_response(
                error=(
                    "Experimental reference identity generation requires explicit operation, workflow_key, and reference_image_path. "
                    "This route is intentionally not available by partial inference."
                ),
                error_type="reference_identity_incomplete_guard",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
                extra={
                    "reference_identity_status": "blocked_to_prevent_partial_activation",
                    "operation": operation or None,
                    "workflow_key": requested_workflow_key or None,
                    "experimental_reference_identity": reference_identity_experiment_enabled,
                    "has_reference_image_path": bool(reference_image_path),
                    "required_arguments": [
                        "operation=reference_identity_txt2img",
                        "workflow_key=character_reference_key_visual_experimental_v1 or fullbody_v8_reference_identity_experimental_v1",
                        "reference_image_path",
                    ],
                    "optional_audit_flag": "experimental_reference_identity=true",
                },
            )
        if _looks_like_source_image_task_prompt_without_args(
            prompt,
            source_image_path=source_image_path,
            operation=operation,
        ):
            return error_response(
                error=(
                    "source-image task text was passed as a plain prompt. "
                    "Call image_generate with explicit operation and source_image_path instead of txt2img."
                ),
                error_type="source_image_task_prompt_only",
                provider=self.name,
                model=checkpoint,
                prompt=prompt,
                aspect_ratio=aspect,
                extra={
                    "detected_source_image_task_prompt_only": True,
                    "required_arguments": ["operation", "source_image_path"],
                    "final_report_allowed": False,
                    "completion_report_forbidden": True,
                },
            )
        postprocess_preset = str(kwargs.get("postprocess_preset") or "").strip()
        masked_inpaint = operation == MASKED_INPAINT_OPERATION
        source_image_upscale = operation == UPSCALE_OPERATION
        source_preserving_postprocess = (
            operation in {"postprocess", SOURCE_PRESERVING_POSTPROCESS_OPERATION, LOCAL_RETOUCH_OPERATION}
            or postprocess_preset in {
                FACE8M_HAND9C_POSTPROCESS_PRESET,
                DEPTH50_CANNY100_FACE8M_HAND9C_POSTPROCESS_PRESET,
            }
            or (bool(source_image_path) and not source_image_upscale and not masked_inpaint)
        )
        source_preserving_operation = operation
        if source_preserving_postprocess and operation not in {
            "postprocess",
            SOURCE_PRESERVING_POSTPROCESS_OPERATION,
            LOCAL_RETOUCH_OPERATION,
        }:
            source_preserving_operation = LOCAL_RETOUCH_OPERATION
        runtime_preset: Optional[Dict[str, Any]] = None
        try:
            runtime_preset = _build_character_production_runtime(
                prompt,
                negative_prompt=negative_prompt,
                subject_dominance=subject_dominance,
                workflow_key=requested_workflow_key,
                output_type=output_type,
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
        workflow_key = requested_workflow_key or DEFAULT_WORKFLOW_KEY
        if runtime_preset is not None:
            preset_name = str(runtime_preset.get("preset") or CHARACTER_PRODUCTION_PRESET)
            source_prompt = str(runtime_preset.get("source_prompt") or source_prompt)
            prompt_for_generation = str(runtime_preset.get("prompt") or prompt)
            negative_prompt = str(runtime_preset.get("negative_prompt") or negative_prompt)
            honor_explicit_generation_settings = preset_name in {PORTRAIT_PRODUCTION_PRESET}
            if not explicit_steps or not honor_explicit_generation_settings:
                steps = int(runtime_preset.get("steps") or steps)
            if not explicit_cfg or not honor_explicit_generation_settings:
                cfg = float(runtime_preset.get("cfg") or cfg)
            if not explicit_sampler or not honor_explicit_generation_settings:
                sampler_name = str(runtime_preset.get("sampler_name") or sampler_name)
            if not explicit_scheduler or not honor_explicit_generation_settings:
                scheduler = str(runtime_preset.get("scheduler") or scheduler)
            prompt_translation_policy = str(runtime_preset.get("prompt_translation_policy") or prompt_translation_policy)
            subject_dominance_value = runtime_preset.get("subject_dominance")
            subject_dominance_rule = runtime_preset.get("subject_dominance_rule")
            workflow_key = str(runtime_preset.get("workflow_key") or workflow_key)
            preset_checkpoint = str(runtime_preset.get("checkpoint") or "").strip()
            if preset_checkpoint and (
                not explicit_checkpoint
                or checkpoint in {DEFAULT_CHECKPOINT, configured_checkpoint}
            ):
                checkpoint = preset_checkpoint
                requested_checkpoint = checkpoint
            preset_vae = str(runtime_preset.get("vae") or "").strip()
            if preset_vae and not explicit_vae:
                vae = preset_vae
            if (
                not explicit_dimensions
                and isinstance(runtime_preset.get("width"), int)
                and isinstance(runtime_preset.get("height"), int)
            ):
                width, height = int(runtime_preset["width"]), int(runtime_preset["height"])
            elif preset_name in {PORTRAIT_PRODUCTION_PRESET, V8_STYLE_WORKFLOW_PRESET} and not explicit_dimensions:
                width, height = PORTRAIT_PRODUCTION_WIDTH, PORTRAIT_PRODUCTION_HEIGHT
            if not explicit_seed and isinstance(runtime_preset.get("seed"), int):
                seed = int(runtime_preset["seed"])
            if runtime_preset.get("use_checkpoint_vae") is True and not explicit_vae:
                vae = None

        lora_stack = _resolve_lora_stack(kwargs, runtime_preset=runtime_preset)

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
        if (
            not checkpoint_resolution.get("ok")
            and postprocess_preset == DEPTH50_CANNY100_FACE8M_HAND9C_POSTPROCESS_PRESET
            and SOURCE_PRESERVING_DEPTH_CANNY_CHECKPOINT in checkpoints
        ):
            checkpoint_resolution = _resolution_payload(
                base=checkpoint_resolution,
                ok=True,
                mode="preset_override",
                candidates=[],
                resolved_checkpoint=SOURCE_PRESERVING_DEPTH_CANNY_CHECKPOINT,
            )
            checkpoint_resolution["source_model_rejected"] = checkpoint
            checkpoint_resolution["resolution_reason"] = "source_preserving_depth_canny_promoted_preset"
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

        if source_image_upscale:
            if not source_image_path:
                return error_response(
                    error="source_image_path is required for ComfyUI source-image upscale",
                    error_type="invalid_argument",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            source_input_path, source_resolution = _resolve_existing_source_image_path(source_image_path)
            if source_input_path is None:
                return error_response(
                    error=f"source_image_path not found for ComfyUI source-image upscale: {source_image_path}",
                    error_type="source_image_not_found",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    extra=source_resolution,
                )
            upscale_model = str(kwargs.get("upscale_model") or DEFAULT_UPSCALE_MODEL).strip() or DEFAULT_UPSCALE_MODEL
            try:
                uploaded_source = _upload_comfy_input_file(base_url, source_input_path)
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"ComfyUI source image upload failed: {exc}",
                    error_type="api_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    extra=source_resolution,
                )

            workflow_key = SOURCE_IMAGE_UPSCALE_WORKFLOW_KEY
            workflow = _build_source_image_upscale_workflow(
                source_image_name=str(uploaded_source["name"]),
                upscale_model=upscale_model,
                filename_prefix=filename_prefix,
            )
            payload = {"prompt": workflow}

            try:
                response = requests.post(f"{base_url}/prompt", json=payload, timeout=30)
                response.raise_for_status()
                submit = response.json()
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"ComfyUI upscale prompt submission failed: {exc}",
                    error_type="api_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            prompt_id = str(submit.get("prompt_id") or "").strip()
            if not prompt_id:
                return error_response(
                    error="ComfyUI upscale prompt response did not include prompt_id",
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
                        error=f"ComfyUI upscale history lookup failed: {exc}",
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
                    error=f"ComfyUI upscale history timed out before success for prompt_id={prompt_id}",
                    error_type="timeout",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            output_image = _first_output_image(history_payload or {}, prompt_id)
            if not output_image:
                return error_response(
                    error="ComfyUI upscale history success did not contain an output image",
                    error_type="invalid_response",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            output_source_path = _find_comfy_output_file(output_image)
            source_origin = "local_output_dir"
            if output_source_path is None:
                output_source_path = _download_comfy_output_file(base_url, output_image, prompt_id)
                source_origin = "remote_view_download" if output_source_path is not None else "missing"
            if output_source_path is None:
                return error_response(
                    error="ComfyUI upscale output file not found after history success",
                    error_type="io_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            if not wait_for_file_stable(output_source_path, checks=2, delay_seconds=0.1):
                return error_response(
                    error=f"ComfyUI upscale output file did not stabilize: {output_source_path}",
                    error_type="io_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            actual_width, actual_height = _read_png_dimensions(output_source_path)
            output_resolution = (
                f"{actual_width}x{actual_height}"
                if actual_width is not None and actual_height is not None
                else None
            )

            created_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
            prompt_payload = {
                "prompt": prompt_for_generation,
                "source_prompt": source_prompt,
                "source_image_path": str(source_input_path),
                "requested_operation": UPSCALE_OPERATION,
                "raw_requested_operation": operation or None,
                "canonical_operation": UPSCALE_OPERATION,
                "uploaded_source": uploaded_source,
                "runtime_preset": "source_image_upscale",
                "workflow_key": workflow_key,
                "upscale_model": upscale_model,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "output_resolution": output_resolution,
                "raw_prompt_payload": {
                    "submit_payload": payload,
                    "submit_response": submit,
                    "history_status": (history_payload or {}).get(prompt_id, {}).get("status", {}),
                    "output_image": output_image,
                    "system_stats": system_payload,
                    "workflow_key": workflow_key,
                },
            }
            metadata = {
                "provider": self.name,
                "prompt_id": prompt_id,
                "api_base_url": base_url,
                "workflow_key": workflow_key,
                "requested_operation": UPSCALE_OPERATION,
                "raw_requested_operation": operation or None,
                "canonical_operation": UPSCALE_OPERATION,
                "checkpoint": None,
                "source_image_path": str(source_input_path),
                "uploaded_source": uploaded_source,
                "upscale_model": upscale_model,
                "created_at": created_at,
                "category": "upscale",
                "output_source_path": str(output_source_path),
                "output_source_origin": source_origin,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "output_resolution": output_resolution,
                "local_status": "업스케일 완료",
                "publish_status": "HermesWork publish 완료",
                "slack_status": "primary image 준비됨",
            }
            try:
                bundle = publish_filesystem_image_bundle(
                    output_source_path,
                    prefix=filename_prefix,
                    project_name=project_name,
                    artifact_name=artifact_name or filename_prefix,
                    category="upscale",
                    workflow_json=workflow,
                    prompt_payload=prompt_payload,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"Could not publish ComfyUI upscale bundle to HermesWork: {exc}",
                    error_type="io_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            evidence = {
                "workflow_key": workflow_key,
                "workflow_path": str(bundle["workflow_path"]),
                "prompt_id": prompt_id,
                "requested_operation": UPSCALE_OPERATION,
                "raw_requested_operation": operation or None,
                "canonical_operation": UPSCALE_OPERATION,
                "source_image": str(source_input_path),
                "uploaded_source": uploaded_source,
                "upscale_model": upscale_model,
                "output_image": output_image,
                "artifact_path": str(bundle["primary_image_path"]),
                "output_source_origin": source_origin,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "output_resolution": output_resolution,
            }
            slack_preview_path = _make_slack_preview_image(bundle["primary_image_path"])
            media_files = [str(slack_preview_path or bundle["primary_image_path"])]
            artifact_files = [
                str(bundle["primary_image_path"]),
                str(bundle["workflow_path"]),
                str(bundle["prompt_path"]),
                str(bundle["metadata_path"]),
                str(bundle["manifest_path"]),
                str(bundle["integrity_path"]),
            ]
            if slack_preview_path is not None:
                evidence["slack_preview_image"] = str(slack_preview_path)
                artifact_files.append(str(slack_preview_path))
            delivery_image_path = slack_preview_path or bundle["primary_image_path"]
            report_evidence = {
                "operation": UPSCALE_OPERATION,
                "canonical_operation": UPSCALE_OPERATION,
                "upscale_model": upscale_model,
                "workflow_key": workflow_key,
                "workflow_path": str(bundle["workflow_path"]),
                "prompt_id": prompt_id,
                "source_image_path": str(source_input_path),
                "output_image": bundle["primary_image"],
                "artifact_path": str(bundle["primary_image_path"]),
                "output_resolution": output_resolution,
                "actual_width": actual_width,
                "actual_height": actual_height,
            }
            _update_metadata_report_evidence(bundle["metadata_path"], report_evidence)
            nas_status = "동기화 요청됨" if bundle["nas_hook_requested"] else "동기화 요청 실패"
            return success_response(
                image=str(delivery_image_path),
                model=upscale_model,
                prompt=prompt_for_generation,
                aspect_ratio=aspect,
                provider=self.name,
                extra={
                    "base_url": base_url,
                    "preset": "source_image_upscale",
                    "workflow_key": workflow_key,
                    "evidence": evidence,
                    "report_evidence": report_evidence,
                    "requested_operation": UPSCALE_OPERATION,
                    "raw_requested_operation": operation or None,
                    "canonical_operation": UPSCALE_OPERATION,
                    "source_image": str(source_input_path),
                    "upscale_model": upscale_model,
                    "local_status": "업스케일 완료",
                    "publish_status": "HermesWork publish 완료",
                    "nas_status": nas_status,
                    "slack_status": "primary image 준비됨",
                    "workflow_path": str(bundle["workflow_path"]),
                    "prompt_path": str(bundle["prompt_path"]),
                    "metadata_path": str(bundle["metadata_path"]),
                    "manifest_path": str(bundle["manifest_path"]),
                    "artifact_path": str(bundle["primary_image_path"]),
                    "original_image_path": str(bundle["primary_image_path"]),
                    "primary_image": bundle["primary_image"],
                    "media_files": media_files,
                    "slack_preview_image": str(slack_preview_path) if slack_preview_path is not None else None,
                    "sidecars": bundle["sidecars"],
                    "artifact_files": artifact_files,
                    "file_sha256": bundle.get("file_sha256"),
                    "integrity_path": str(bundle["integrity_path"]),
                    "nas_hook_requested": bundle["nas_hook_requested"],
                    "nas_evidence": bundle.get("nas_evidence"),
                    "slack_upload_evidence": False,
                    "output_source_origin": source_origin,
                    "output_image": output_image,
                    "prompt_id": prompt_id,
                    "category": "upscale",
                    "actual_width": actual_width,
                    "actual_height": actual_height,
                    "output_resolution": output_resolution,
                    "api_base_url": base_url,
                },
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

        if masked_inpaint:
            if not source_image_path:
                return error_response(
                    error="source_image_path is required for masked inpaint",
                    error_type="invalid_argument",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            source_input_path, source_resolution = _resolve_existing_source_image_path(source_image_path)
            if source_input_path is None:
                return error_response(
                    error=f"source_image_path not found for masked inpaint: {source_image_path}",
                    error_type="source_image_not_found",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    extra=source_resolution,
                )
            feather_px = int(kwargs.get("mask_feather_px") or 10)
            grow_mask_by = int(kwargs.get("grow_mask_by") or 8)
            inpaint_target = str(kwargs.get("mask_target") or kwargs.get("mask_label") or "").strip() or None
            mask_source = str(kwargs.get("mask_source") or kwargs.get("mask_mode") or "").strip()
            if not mask_source:
                mask_source = "rectangle"
            if mask_source not in {"rectangle", "detailer_bbox"}:
                return error_response(
                    error=f"Unsupported mask_source for masked inpaint: {mask_source}",
                    error_type="invalid_argument",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            mask_box: Optional[Tuple[float, float, float, float]] = None
            if mask_source == "rectangle":
                try:
                    mask_box = _parse_mask_box(kwargs.get("mask_box"))
                except ValueError as exc:
                    return error_response(
                        error=str(exc),
                        error_type="invalid_argument",
                        provider=self.name,
                        model=checkpoint,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
            denoise, feather_px, grow_mask_by, mask_safety = _apply_masked_inpaint_safety_defaults(
                mask_target=inpaint_target,
                kwargs=kwargs,
                denoise=denoise,
                feather_px=feather_px,
                grow_mask_by=grow_mask_by,
            )
            mask_safety["mask_source"] = mask_source
            if mask_source == "detailer_bbox":
                mask_safety.setdefault("warnings", []).append(
                    "experimental_detailer_bbox_mask_source_requires_visual_review"
                )
            mask_path: Optional[Path] = None
            mask_info: Dict[str, Any]
            uploaded_mask: Optional[Dict[str, Any]] = None
            detailer_detector: Optional[Dict[str, Any]] = None
            try:
                uploaded_source = _upload_comfy_input_file(base_url, source_input_path)
                if mask_source == "rectangle":
                    if mask_box is None:
                        raise ValueError("mask_box is required for rectangle masked inpaint")
                    mask_path, mask_info = _create_mask_image_for_box(
                        source_image_path=source_input_path,
                        mask_box=mask_box,
                        feather_px=feather_px,
                        temp_prefix=filename_prefix,
                    )
                    coverage_ratio = mask_info.get("mask_coverage_ratio")
                    mask_safety["mask_shape"] = mask_info.get("mask_shape")
                    mask_safety["mask_coverage_ratio"] = coverage_ratio
                    if mask_safety["localized_target"] and isinstance(coverage_ratio, (int, float)) and coverage_ratio > 0.04:
                        mask_safety.setdefault("warnings", []).append("localized_target_large_rectangular_mask")
                    uploaded_mask = _upload_comfy_input_file(base_url, mask_path)
                else:
                    detailer_detector = _detailer_bbox_detector_for_target(inpaint_target)
                    mask_info = {
                        "normalized": None,
                        "pixel_box": None,
                        "image_size": None,
                        "feather_px": feather_px,
                        "mask_shape": "detailer_bbox_segs",
                        "mask_coverage_ratio": None,
                    }
                    mask_safety["mask_shape"] = "detailer_bbox_segs"
                    mask_safety["mask_coverage_ratio"] = None
                    mask_safety["detailer_detector"] = detailer_detector
            except Exception as exc:  # noqa: BLE001
                if mask_path is not None:
                    try:
                        mask_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                return error_response(
                    error=f"ComfyUI masked inpaint setup failed: {exc}",
                    error_type="api_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    extra=source_resolution,
                )

            workflow_key = (
                SOURCE_DETAILER_BBOX_MASKED_INPAINT_WORKFLOW_KEY
                if mask_source == "detailer_bbox"
                else SOURCE_MASKED_INPAINT_WORKFLOW_KEY
            )
            if not negative_prompt:
                negative_prompt = (
                    "low quality, blurry, bad anatomy, extra fingers, missing fingers, malformed hands, "
                    "distorted face, identity drift, changing outfit outside mask, changing background outside mask"
                )
            if mask_source == "detailer_bbox":
                workflow, detailer_detector = _build_detailer_bbox_masked_inpaint_workflow(
                    checkpoint=checkpoint,
                    source_image_name=str(uploaded_source["name"]),
                    positive_prompt=prompt_for_generation,
                    negative_prompt=negative_prompt,
                    filename_prefix=filename_prefix,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    denoise=denoise,
                    grow_mask_by=max(0, grow_mask_by),
                    feather_px=feather_px,
                    mask_target=inpaint_target,
                )
            else:
                if uploaded_mask is None:
                    return error_response(
                        error="ComfyUI masked inpaint setup failed: rectangle mask upload missing",
                        error_type="api_error",
                        provider=self.name,
                        model=checkpoint,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                workflow = _build_masked_inpaint_workflow(
                    checkpoint=checkpoint,
                    source_image_name=str(uploaded_source["name"]),
                    mask_image_name=str(uploaded_mask["name"]),
                    positive_prompt=prompt_for_generation,
                    negative_prompt=negative_prompt,
                    filename_prefix=filename_prefix,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    denoise=denoise,
                    grow_mask_by=max(0, grow_mask_by),
                )
            payload = {"prompt": workflow}

            try:
                response = requests.post(f"{base_url}/prompt", json=payload, timeout=30)
                response.raise_for_status()
                submit = response.json()
            except Exception as exc:  # noqa: BLE001
                if mask_path is not None:
                    try:
                        mask_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                return error_response(
                    error=f"ComfyUI masked inpaint prompt submission failed: {exc}",
                    error_type="api_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            prompt_id = str(submit.get("prompt_id") or "").strip()
            if not prompt_id:
                if mask_path is not None:
                    try:
                        mask_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                return error_response(
                    error="ComfyUI masked inpaint prompt response did not include prompt_id",
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
                    if mask_path is not None:
                        try:
                            mask_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                    return error_response(
                        error=f"ComfyUI masked inpaint history lookup failed: {exc}",
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
                if mask_path is not None:
                    try:
                        mask_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                return error_response(
                    error=f"ComfyUI masked inpaint history timed out before success for prompt_id={prompt_id}",
                    error_type="timeout",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            output_image = _first_output_image(history_payload or {}, prompt_id)
            if not output_image:
                if mask_path is not None:
                    try:
                        mask_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                return error_response(
                    error="ComfyUI masked inpaint history success did not contain an output image",
                    error_type="invalid_response",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            output_source_path = _find_comfy_output_file(output_image)
            source_origin = "local_output_dir"
            if output_source_path is None:
                output_source_path = _download_comfy_output_file(base_url, output_image, prompt_id)
                source_origin = "remote_view_download" if output_source_path is not None else "missing"
            if output_source_path is None:
                if mask_path is not None:
                    try:
                        mask_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                return error_response(
                    error="ComfyUI masked inpaint output file not found after history success",
                    error_type="io_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            if not wait_for_file_stable(output_source_path, checks=2, delay_seconds=0.1):
                if mask_path is not None:
                    try:
                        mask_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                return error_response(
                    error=f"ComfyUI masked inpaint output file did not stabilize: {output_source_path}",
                    error_type="io_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            actual_width, actual_height = _read_png_dimensions(output_source_path)
            output_resolution = (
                f"{actual_width}x{actual_height}"
                if actual_width is not None and actual_height is not None
                else None
            )
            created_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
            prompt_payload = {
                "prompt": prompt_for_generation,
                "source_prompt": source_prompt,
                "negative_prompt": negative_prompt,
                "source_image_path": str(source_input_path),
                "mask_path": str(mask_path) if mask_path is not None else None,
                "uploaded_source": uploaded_source,
                "uploaded_mask": uploaded_mask,
                "runtime_preset": "masked_inpaint",
                "workflow_key": workflow_key,
                "mask_target": inpaint_target,
                "mask_box": mask_info["normalized"],
                "mask_box_px": mask_info["pixel_box"],
                "mask_feather_px": mask_info["feather_px"],
                "mask_shape": mask_info["mask_shape"],
                "mask_coverage_ratio": mask_info["mask_coverage_ratio"],
                "grow_mask_by": max(0, grow_mask_by),
                "masked_inpaint_safety": mask_safety,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "output_resolution": output_resolution,
                "raw_prompt_payload": {
                    "submit_payload": payload,
                    "submit_response": submit,
                    "history_status": (history_payload or {}).get(prompt_id, {}).get("status", {}),
                    "output_image": output_image,
                    "system_stats": system_payload,
                    "workflow_key": workflow_key,
                },
            }
            metadata = {
                "provider": self.name,
                "prompt_id": prompt_id,
                "api_base_url": base_url,
                "workflow_key": workflow_key,
                "checkpoint": checkpoint,
                "requested_checkpoint": checkpoint_resolution.get("requested_checkpoint", requested_checkpoint),
                "resolved_checkpoint": checkpoint,
                "preset": "masked_inpaint",
                "source_image_path": str(source_input_path),
                "mask_target": inpaint_target,
                "mask_box": mask_info["normalized"],
                "mask_box_px": mask_info["pixel_box"],
                "mask_feather_px": mask_info["feather_px"],
                "mask_shape": mask_info["mask_shape"],
                "mask_coverage_ratio": mask_info["mask_coverage_ratio"],
                "grow_mask_by": max(0, grow_mask_by),
                "masked_inpaint_safety": mask_safety,
                "uploaded_source": uploaded_source,
                "uploaded_mask": uploaded_mask,
                "negative_prompt": negative_prompt,
                "vae": None,
                "loras": [],
                "controlnet_used": False,
                "created_at": created_at,
                "category": POSTPROCESS_CATEGORY,
                "output_source_path": str(output_source_path),
                "output_source_origin": source_origin,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "output_resolution": output_resolution,
                "local_status": "국소 인페인트 완료",
                "publish_status": "HermesWork publish 완료",
                "slack_status": "primary image 준비됨",
            }
            try:
                bundle = publish_filesystem_image_bundle(
                    output_source_path,
                    prefix=filename_prefix,
                    project_name=project_name,
                    artifact_name=artifact_name or filename_prefix,
                    category=POSTPROCESS_CATEGORY,
                    workflow_json=workflow,
                    prompt_payload=prompt_payload,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                if mask_path is not None:
                    try:
                        mask_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                return error_response(
                    error=f"Could not publish ComfyUI masked inpaint bundle to HermesWork: {exc}",
                    error_type="io_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            evidence = {
                "workflow_key": workflow_key,
                "workflow_path": str(bundle["workflow_path"]),
                "prompt_id": prompt_id,
                "source_image": str(source_input_path),
                "mask_image": str(mask_path) if mask_path is not None else None,
                "uploaded_source": uploaded_source,
                "uploaded_mask": uploaded_mask,
                "mask_target": inpaint_target,
                "mask_box": mask_info["normalized"],
                "mask_box_px": mask_info["pixel_box"],
                "mask_feather_px": mask_info["feather_px"],
                "mask_shape": mask_info["mask_shape"],
                "mask_coverage_ratio": mask_info["mask_coverage_ratio"],
                "grow_mask_by": max(0, grow_mask_by),
                "masked_inpaint_safety": mask_safety,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "output_image": output_image,
                "artifact_path": str(bundle["primary_image_path"]),
                "output_source_origin": source_origin,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "output_resolution": output_resolution,
            }
            report_evidence = {
                "operation": MASKED_INPAINT_OPERATION,
                "workflow_key": workflow_key,
                "workflow_path": str(bundle["workflow_path"]),
                "prompt_id": prompt_id,
                "source_image_path": str(source_input_path),
                "mask_target": inpaint_target,
                "mask_box": mask_info["normalized"],
                "mask_box_px": mask_info["pixel_box"],
                "mask_shape": mask_info["mask_shape"],
                "mask_coverage_ratio": mask_info["mask_coverage_ratio"],
                "masked_inpaint_requires_visual_review": mask_safety["requires_visual_review"],
                "output_image": bundle["primary_image"],
                "artifact_path": str(bundle["primary_image_path"]),
                "output_resolution": output_resolution,
                "actual_width": actual_width,
                "actual_height": actual_height,
            }
            _update_metadata_report_evidence(bundle["metadata_path"], report_evidence)
            nas_status = "동기화 요청됨" if bundle["nas_hook_requested"] else "동기화 요청 실패"
            if mask_path is not None:
                try:
                    mask_path.unlink(missing_ok=True)
                except Exception:
                    pass
            return success_response(
                image=str(bundle["primary_image_path"]),
                model=checkpoint,
                prompt=prompt_for_generation,
                aspect_ratio=aspect,
                provider=self.name,
                extra={
                    "base_url": base_url,
                    "preset": "masked_inpaint",
                    "workflow_key": workflow_key,
                    "evidence": evidence,
                    "report_evidence": report_evidence,
                    "source_image": str(source_input_path),
                    "mask_target": inpaint_target,
                    "mask_box": mask_info["normalized"],
                    "mask_box_px": mask_info["pixel_box"],
                    "mask_feather_px": mask_info["feather_px"],
                    "mask_shape": mask_info["mask_shape"],
                    "mask_coverage_ratio": mask_info["mask_coverage_ratio"],
                    "grow_mask_by": max(0, grow_mask_by),
                    "masked_inpaint_safety": mask_safety,
                    "masked_inpaint_requires_visual_review": mask_safety["requires_visual_review"],
                    "local_status": "국소 인페인트 완료",
                    "publish_status": "HermesWork publish 완료",
                    "nas_status": nas_status,
                    "slack_status": "primary image 준비됨",
                    "workflow_path": str(bundle["workflow_path"]),
                    "prompt_path": str(bundle["prompt_path"]),
                    "metadata_path": str(bundle["metadata_path"]),
                    "manifest_path": str(bundle["manifest_path"]),
                    "artifact_path": str(bundle["primary_image_path"]),
                    "primary_image": bundle["primary_image"],
                    "media_files": [str(bundle["primary_image_path"])],
                    "sidecars": bundle["sidecars"],
                    "artifact_files": [
                        str(bundle["primary_image_path"]),
                        str(bundle["workflow_path"]),
                        str(bundle["prompt_path"]),
                        str(bundle["metadata_path"]),
                        str(bundle["manifest_path"]),
                        str(bundle["integrity_path"]),
                    ],
                    "file_sha256": bundle.get("file_sha256"),
                    "integrity_path": str(bundle["integrity_path"]),
                    "nas_hook_requested": bundle["nas_hook_requested"],
                    "nas_evidence": bundle.get("nas_evidence"),
                    "slack_upload_evidence": False,
                    "output_source_origin": source_origin,
                    "output_image": output_image,
                    "prompt_id": prompt_id,
                    "category": POSTPROCESS_CATEGORY,
                    "api_base_url": base_url,
                    "actual_width": actual_width,
                    "actual_height": actual_height,
                    "output_resolution": output_resolution,
                },
            )

        if source_preserving_postprocess:
            supported_postprocess_presets = {
                FACE8M_HAND9C_POSTPROCESS_PRESET,
                DEPTH50_CANNY100_FACE8M_HAND9C_POSTPROCESS_PRESET,
            }
            if postprocess_preset and postprocess_preset not in supported_postprocess_presets:
                return error_response(
                    error=f"Unsupported ComfyUI postprocess preset: {postprocess_preset}",
                    error_type="invalid_argument",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            if not source_image_path:
                return error_response(
                    error="source_image_path is required for source-preserving postprocess",
                    error_type="invalid_argument",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            source_input_path, source_resolution = _resolve_existing_source_image_path(source_image_path)
            if source_input_path is None:
                return error_response(
                    error=f"source_image_path not found for source-preserving postprocess: {source_image_path}",
                    error_type="source_image_not_found",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    extra=source_resolution,
                )
            try:
                uploaded_source = _upload_comfy_input_file(base_url, source_input_path)
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"ComfyUI source image upload failed: {exc}",
                    error_type="api_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    extra=source_resolution,
                )

            preset_name = postprocess_preset or FACE8M_HAND9C_POSTPROCESS_PRESET
            workflow_key = SOURCE_PRESERVING_FACE_HAND_WORKFLOW_KEY
            effective_checkpoint = checkpoint
            effective_vae: Optional[str] = None
            effective_loras: List[Dict[str, Any]] = []
            controlnet_used = False
            controlnet_stack: List[Dict[str, Any]] = []
            if preset_name == DEPTH50_CANNY100_FACE8M_HAND9C_POSTPROCESS_PRESET:
                workflow_key = SOURCE_PRESERVING_DEPTH50_CANNY100_FACE_HAND_WORKFLOW_KEY
                effective_checkpoint = SOURCE_PRESERVING_DEPTH_CANNY_CHECKPOINT
                effective_vae = SOURCE_PRESERVING_DEPTH_CANNY_VAE
                effective_loras = [
                    {
                        "name": SOURCE_PRESERVING_DEPTH_CANNY_STYLE_LORA,
                        "weight": SOURCE_PRESERVING_DEPTH_CANNY_STYLE_LORA_WEIGHT,
                    },
                    {
                        "name": SOURCE_PRESERVING_DEPTH_CANNY_UTILITY_LORA,
                        "weight": SOURCE_PRESERVING_DEPTH_CANNY_UTILITY_LORA_WEIGHT,
                    },
                ]
                controlnet_used = True
                controlnet_stack = [
                    {
                        "name": SOURCE_PRESERVING_DEPTH_CONTROLNET,
                        "preprocessor": "Zoe_DepthAnythingPreprocessor",
                        "strength": 0.5,
                        "start_percent": 0.0,
                        "end_percent": 0.5,
                    },
                    {
                        "name": SOURCE_PRESERVING_CANNY_CONTROLNET,
                        "preprocessor": "CannyEdgePreprocessor",
                        "strength": 1.0,
                        "start_percent": 0.0,
                        "end_percent": 0.4,
                    },
                ]
                if not negative_prompt:
                    negative_prompt = (
                        "low quality, worst quality, normal quality, lowres, blurry, jpeg artifacts, watermark, "
                        "text, logo, signature, card, trading card, card frame, ornate frame, border, UI, title area, "
                        "textbox, stats panel, multiple characters, cropped body, cropped legs, feet out of frame, "
                        "tiny face, unreadable face, black face, shadowed face, bad face, extra arms, extra hands, "
                        "bad anatomy, bad hands, malformed hands, extra fingers, missing fingers, fused fingers, "
                        "bad legs, bad feet, broken silhouette, distorted costume, photorealistic, 3d render, "
                        "realistic skin texture, plastic skin, over-sharpened, crunchy details, noisy texture, muddy colors"
                    )
                workflow = _build_depth50_canny100_face8m_hand9c_postprocess_workflow(
                    checkpoint=effective_checkpoint,
                    source_image_name=str(uploaded_source["name"]),
                    positive_prompt=prompt_for_generation,
                    negative_prompt=negative_prompt,
                    filename_prefix=filename_prefix,
                    seed=DEFAULT_SEED,
                )
            else:
                if not negative_prompt:
                    negative_prompt = (
                        "low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, "
                        "missing fingers, distorted face, text, watermark, changing pose, changing character identity"
                    )
                workflow = _build_face8m_hand9c_postprocess_workflow(
                    checkpoint=effective_checkpoint,
                    source_image_name=str(uploaded_source["name"]),
                    positive_prompt=prompt_for_generation,
                    negative_prompt=negative_prompt,
                    filename_prefix=filename_prefix,
                )
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

            output_source_path = _find_comfy_output_file(output_image)
            source_origin = "local_output_dir"
            if output_source_path is None:
                output_source_path = _download_comfy_output_file(base_url, output_image, prompt_id)
                source_origin = "remote_view_download" if output_source_path is not None else "missing"
            if output_source_path is None:
                return error_response(
                    error="ComfyUI output file not found after history success",
                    error_type="io_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            if not wait_for_file_stable(output_source_path, checks=2, delay_seconds=0.1):
                return error_response(
                    error=f"ComfyUI output file did not stabilize: {output_source_path}",
                    error_type="io_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            actual_width, actual_height = _read_png_dimensions(output_source_path)
            output_resolution = (
                f"{actual_width}x{actual_height}"
                if actual_width is not None and actual_height is not None
                else None
            )
            face_detailer_config = {"model": "bbox/face_yolov8m.pt", "denoise": 0.35, "steps": 16, "cfg": 5.5}
            hand_detailer_config = {"model": "bbox/hand_yolov9c.pt", "denoise": 0.25, "steps": 14, "cfg": 5.5}

            created_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
            prompt_payload = {
                "prompt": prompt_for_generation,
                "source_prompt": source_prompt,
                "negative_prompt": negative_prompt,
                "source_image_path": str(source_input_path),
                "uploaded_source": uploaded_source,
                "requested_operation": source_preserving_operation,
                "raw_requested_operation": operation or None,
                "runtime_preset": preset_name,
                "workflow_key": workflow_key,
                "postprocess_preset": preset_name,
                "source_path_resolution": source_resolution,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "output_resolution": output_resolution,
                "face_detailer": face_detailer_config,
                "hand_detailer": hand_detailer_config,
                "vae": effective_vae,
                "loras": effective_loras,
                "controlnet_used": controlnet_used,
                "controlnet_stack": controlnet_stack,
                "raw_prompt_payload": {
                    "submit_payload": payload,
                    "submit_response": submit,
                    "history_status": (history_payload or {}).get(prompt_id, {}).get("status", {}),
                    "output_image": output_image,
                    "system_stats": system_payload,
                    "workflow_key": workflow_key,
                },
            }
            metadata = {
                "provider": self.name,
                "prompt_id": prompt_id,
                "api_base_url": base_url,
                "workflow_key": workflow_key,
                "requested_operation": source_preserving_operation,
                "raw_requested_operation": operation or None,
                "checkpoint": effective_checkpoint,
                "requested_checkpoint": checkpoint_resolution.get("requested_checkpoint", requested_checkpoint),
                "resolved_checkpoint": effective_checkpoint,
                "preset": preset_name,
                "postprocess_preset": preset_name,
                "source_image_path": str(source_input_path),
                "source_path_resolution": source_resolution,
                "uploaded_source": uploaded_source,
                "negative_prompt": negative_prompt,
                "vae": effective_vae,
                "loras": effective_loras,
                "controlnet_used": controlnet_used,
                "controlnet_stack": controlnet_stack,
                "created_at": created_at,
                "category": POSTPROCESS_CATEGORY,
                "output_source_path": str(output_source_path),
                "output_source_origin": source_origin,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "output_resolution": output_resolution,
                "face_detailer": face_detailer_config,
                "hand_detailer": hand_detailer_config,
                "local_status": "후보정 완료",
                "publish_status": "HermesWork publish 완료",
                "slack_status": "primary image 준비됨",
            }
            try:
                bundle = publish_filesystem_image_bundle(
                    output_source_path,
                    prefix=filename_prefix,
                    project_name=project_name,
                    artifact_name=artifact_name or filename_prefix,
                    category=POSTPROCESS_CATEGORY,
                    workflow_json=workflow,
                    prompt_payload=prompt_payload,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"Could not publish ComfyUI postprocess bundle to HermesWork: {exc}",
                    error_type="io_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            evidence = {
                "workflow_key": workflow_key,
                "workflow_path": str(bundle["workflow_path"]),
                "prompt_id": prompt_id,
                "requested_operation": source_preserving_operation,
                "raw_requested_operation": operation or None,
                "source_image": str(source_input_path),
                "requested_source_image_path": source_resolution.get("requested_source_image_path"),
                "resolved_source_image_path": source_resolution.get("resolved_source_image_path"),
                "source_path_resolution": source_resolution.get("source_path_resolution"),
                "uploaded_source": uploaded_source,
                "seed": None,
                "vae": effective_vae,
                "vae_report_value": effective_vae or "checkpoint_builtin_vae",
                "loras": effective_loras,
                "controlnet_used": controlnet_used,
                "controlnet_stack": controlnet_stack,
                "face_detailer": face_detailer_config,
                "hand_detailer": hand_detailer_config,
                "output_image": output_image,
                "artifact_path": str(bundle["primary_image_path"]),
                "output_source_origin": source_origin,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "output_resolution": output_resolution,
            }
            report_evidence = {
                "operation": source_preserving_operation,
                "canonical_operation": SOURCE_PRESERVING_POSTPROCESS_OPERATION,
                "postprocess_preset": preset_name,
                "workflow_key": workflow_key,
                "workflow_path": str(bundle["workflow_path"]),
                "prompt_id": prompt_id,
                "source_image_path": str(source_input_path),
                "requested_source_image_path": source_resolution.get("requested_source_image_path"),
                "resolved_source_image_path": source_resolution.get("resolved_source_image_path"),
                "source_path_resolution": source_resolution.get("source_path_resolution"),
                "output_image": bundle["primary_image"],
                "artifact_path": str(bundle["primary_image_path"]),
                "output_resolution": output_resolution,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "face_detailer": face_detailer_config,
                "hand_detailer": hand_detailer_config,
            }
            _update_metadata_report_evidence(bundle["metadata_path"], report_evidence)
            nas_status = "동기화 요청됨" if bundle["nas_hook_requested"] else "동기화 요청 실패"
            return success_response(
                image=str(bundle["primary_image_path"]),
                model=effective_checkpoint,
                prompt=prompt_for_generation,
                aspect_ratio=aspect,
                provider=self.name,
                extra={
                    "base_url": base_url,
                    "preset": preset_name,
                    "requested_operation": source_preserving_operation,
                    "raw_requested_operation": operation or None,
                    "canonical_operation": SOURCE_PRESERVING_POSTPROCESS_OPERATION,
                    "postprocess_preset": preset_name,
                    "workflow_key": workflow_key,
                    "evidence": evidence,
                    "report_evidence": report_evidence,
                    "source_image": str(source_input_path),
                    "requested_source_image_path": source_resolution.get("requested_source_image_path"),
                    "resolved_source_image_path": source_resolution.get("resolved_source_image_path"),
                    "source_path_resolution": source_resolution.get("source_path_resolution"),
                    "vae": effective_vae,
                    "vae_report_value": effective_vae or "checkpoint_builtin_vae",
                    "loras": effective_loras,
                    "controlnet_used": controlnet_used,
                    "controlnet_stack": controlnet_stack,
                    "local_status": "후보정 완료",
                    "publish_status": "HermesWork publish 완료",
                    "nas_status": nas_status,
                    "slack_status": "primary image 준비됨",
                    "workflow_path": str(bundle["workflow_path"]),
                    "prompt_path": str(bundle["prompt_path"]),
                    "metadata_path": str(bundle["metadata_path"]),
                    "manifest_path": str(bundle["manifest_path"]),
                    "artifact_path": str(bundle["primary_image_path"]),
                    "primary_image": bundle["primary_image"],
                    "media_files": [str(bundle["primary_image_path"])],
                    "sidecars": bundle["sidecars"],
                    "artifact_files": [
                        str(bundle["primary_image_path"]),
                        str(bundle["workflow_path"]),
                        str(bundle["prompt_path"]),
                        str(bundle["metadata_path"]),
                        str(bundle["manifest_path"]),
                        str(bundle["integrity_path"]),
                    ],
                    "file_sha256": bundle.get("file_sha256"),
                    "integrity_path": str(bundle["integrity_path"]),
                    "nas_hook_requested": bundle["nas_hook_requested"],
                    "nas_evidence": bundle.get("nas_evidence"),
                    "slack_upload_evidence": False,
                    "output_source_origin": source_origin,
                    "output_image": output_image,
                    "actual_width": actual_width,
                    "actual_height": actual_height,
                    "output_resolution": output_resolution,
                    "prompt_id": prompt_id,
                    "category": POSTPROCESS_CATEGORY,
                    "api_base_url": base_url,
                },
            )

        reference_identity_evidence: Optional[Dict[str, Any]] = None
        if reference_identity_requested:
            reference_input_path, reference_resolution = _resolve_existing_source_image_path(reference_image_path)
            if reference_input_path is None:
                return error_response(
                    error=f"reference_image_path not found for experimental reference identity route: {reference_image_path}",
                    error_type="reference_image_not_found",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    extra={
                        "reference_identity_status": "blocked_missing_reference_image",
                        "requested_reference_image_path": reference_image_path,
                        "reference_path_resolution": reference_resolution,
                    },
                )
            reference_source_metadata = _load_reference_identity_source_metadata(reference_input_path)
            reference_source_workflow_key = str(
                reference_source_metadata.get("workflow_key")
                or (reference_source_metadata.get("report_evidence") or {}).get("workflow_key")
                or ""
            ).strip()
            reference_source_audit = reference_source_metadata.get("workflow_node_audit")
            if not isinstance(reference_source_audit, dict):
                reference_source_audit = (reference_source_metadata.get("report_evidence") or {}).get("workflow_node_audit")
            if not isinstance(reference_source_audit, dict):
                reference_source_audit = {}
            reference_source_model_stack_verified = reference_source_metadata.get("model_stack_verified")
            if reference_source_model_stack_verified is None:
                reference_source_model_stack_verified = (reference_source_metadata.get("report_evidence") or {}).get("model_stack_verified")
            reference_source_family = _reference_identity_workflow_family(reference_source_workflow_key)
            requested_reference_family = _reference_identity_workflow_family(requested_workflow_key)
            if reference_source_metadata and reference_source_model_stack_verified is False:
                return error_response(
                    error=(
                        "Reference identity source image metadata reports an unverified model stack. "
                        "Use a verified source image or explicitly regenerate the source before reference identity work."
                    ),
                    error_type="reference_identity_source_model_stack_unverified",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    extra={
                        "reference_identity_status": "blocked_unverified_reference_source",
                        "requested_reference_image_path": reference_image_path,
                        "resolved_reference_image_path": str(reference_input_path),
                        "reference_source_workflow_key": reference_source_workflow_key or None,
                        "reference_source_model_stack_verified": reference_source_model_stack_verified,
                    },
                )
            if (
                reference_source_family
                and requested_reference_family
                and reference_source_family != requested_reference_family
                and not allow_reference_workflow_family_change
            ):
                return error_response(
                    error=(
                        "Reference identity workflow family mismatch. "
                        "The reference image was generated on a different workflow family than the requested reference route. "
                        "Set allow_reference_workflow_family_change=true only when intentionally converting between image types."
                    ),
                    error_type="reference_identity_workflow_family_mismatch",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    extra={
                        "reference_identity_status": "blocked_workflow_family_mismatch",
                        "requested_reference_image_path": reference_image_path,
                        "resolved_reference_image_path": str(reference_input_path),
                        "reference_source_workflow_key": reference_source_workflow_key or None,
                        "reference_source_family": reference_source_family,
                        "requested_workflow_key": requested_workflow_key,
                        "requested_reference_family": requested_reference_family,
                        "allow_reference_workflow_family_change": allow_reference_workflow_family_change,
                    },
                )
            try:
                uploaded_reference = _upload_comfy_input_file(base_url, reference_input_path)
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"ComfyUI reference image upload failed: {exc}",
                    error_type="api_error",
                    provider=self.name,
                    model=checkpoint,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    extra={
                        "reference_identity_status": "blocked_reference_upload_failed",
                        "requested_reference_image_path": reference_image_path,
                        "resolved_reference_image_path": str(reference_input_path),
                    },
                )
            workflow_key = requested_workflow_key
            preset_name = (
                REFERENCE_IDENTITY_FULLBODY_EXPERIMENTAL_PRESET
                if requested_workflow_key == CHARACTER_REFERENCE_FULLBODY_EXPERIMENTAL_WORKFLOW_KEY
                else REFERENCE_IDENTITY_EXPERIMENTAL_PRESET
            )
            prompt_translation_policy = (
                f"{prompt_translation_policy} + experimental-reference-identity-ipadapter-guarded"
            )
            if category == DEFAULT_CATEGORY:
                category = "experimental_reference_identity"
            workflow, workflow_node_audit, model_stack_verified = _build_reference_identity_experimental_workflow(
                checkpoint=checkpoint,
                vae=vae,
                lora_stack=lora_stack,
                reference_image_name=str(uploaded_reference["name"]),
                prompt=prompt_for_generation,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                filename_prefix=filename_prefix,
            )
            reference_identity_evidence = {
                "reference_identity_status": "experimental_only",
                "default_route_contamination_guard": True,
                "explicit_route_guard": True,
                "required_explicit_operation": REFERENCE_IDENTITY_TXT2IMG_OPERATION,
                "required_explicit_workflow_key": requested_workflow_key,
                "experimental_reference_identity": reference_identity_experiment_enabled,
                "optional_audit_flag_present": reference_identity_experiment_enabled,
                "reference_source_workflow_key": reference_source_workflow_key or None,
                "reference_source_family": reference_source_family,
                "reference_source_model_stack_verified": reference_source_model_stack_verified,
                "reference_source_workflow_node_audit": reference_source_audit,
                "reference_target_stack_policy": "same workflow family required unless allow_reference_workflow_family_change=true; IPAdapter is identity helper only, not production style source",
                "requested_reference_family": requested_reference_family,
                "allow_reference_workflow_family_change": allow_reference_workflow_family_change,
                "reference_image": str(reference_input_path),
                "reference_image_path": str(reference_input_path),
                "requested_reference_image_path": reference_resolution.get("requested_source_image_path"),
                "resolved_reference_image_path": reference_resolution.get("resolved_source_image_path"),
                "reference_path_resolution": reference_resolution.get("source_path_resolution"),
                "uploaded_reference": uploaded_reference,
                "ipadapter": workflow_node_audit.get("ipadapter"),
            }
        else:
            workflow = {
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

            model_source = ["1", 0]
            clip_source = ["1", 1]
            for index, lora in enumerate(lora_stack, start=1):
                # ComfyUI's prompt API expects numeric node identifiers. Non-numeric
                # IDs such as L1 validate poorly on some servers and can cause 400s.
                node_id = str(20 + index)
                workflow[node_id] = {
                    "inputs": {
                        "model": model_source,
                        "clip": clip_source,
                        "lora_name": lora["name"],
                        "strength_model": lora["weight"],
                        "strength_clip": lora.get("clip_weight", lora["weight"]),
                    },
                    "class_type": "LoraLoader",
                }
                model_source = [node_id, 0]
                clip_source = [node_id, 1]
            if lora_stack:
                workflow["3"]["inputs"]["clip"] = clip_source
                workflow["4"]["inputs"]["clip"] = clip_source
                workflow["5"]["inputs"]["model"] = model_source
            workflow_node_audit = {
                "audit_version": "comfy_txt2img_model_stack_v1",
                "checkpoint_node": "1",
                "checkpoint": workflow.get("1", {}).get("inputs", {}).get("ckpt_name"),
                "vae_node": "6" if vae is not None else "1",
                "vae": workflow.get("6", {}).get("inputs", {}).get("vae_name") if vae is not None else "checkpoint_builtin_vae",
                "lora_nodes": [
                    {
                        "node": node_id,
                        "name": node.get("inputs", {}).get("lora_name"),
                        "weight": node.get("inputs", {}).get("strength_model"),
                        "clip_weight": node.get("inputs", {}).get("strength_clip"),
                    }
                    for node_id, node in sorted(workflow.items(), key=lambda item: item[0])
                    if isinstance(node, dict) and node.get("class_type") == "LoraLoader"
                ],
                "ksampler_node": "5",
                "steps": workflow.get("5", {}).get("inputs", {}).get("steps"),
                "cfg": workflow.get("5", {}).get("inputs", {}).get("cfg"),
                "sampler_name": workflow.get("5", {}).get("inputs", {}).get("sampler_name"),
                "scheduler": workflow.get("5", {}).get("inputs", {}).get("scheduler"),
                "width": workflow.get("2", {}).get("inputs", {}).get("width"),
                "height": workflow.get("2", {}).get("inputs", {}).get("height"),
            }
            model_stack_verified = (
                workflow_node_audit["checkpoint"] == checkpoint
                and workflow_node_audit["vae"] == (vae if vae is not None else "checkpoint_builtin_vae")
                and [
                    item["name"]
                    for item in workflow_node_audit["lora_nodes"]
                ] == [item["name"] for item in lora_stack]
            )
        payload = {"prompt": workflow}

        try:
            response = requests.post(f"{base_url}/prompt", json=payload, timeout=30)
            response.raise_for_status()
            submit = response.json()
        except Exception as exc:  # noqa: BLE001
            response_body = ""
            response_obj = getattr(exc, "response", None)
            if response_obj is not None:
                try:
                    response_body = str(response_obj.text or "").strip()
                except Exception:  # noqa: BLE001
                    response_body = ""
            try:
                debug_dir = Path(
                    os.environ.get(
                        "HERMES_COMFY_PROMPT_ERROR_DIR",
                        "/Volumes/SSD_Hermes/HermesCodexControl/debug/comfy_prompt_errors",
                    )
                )
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{filename_prefix}.json"
                debug_path.write_text(
                    json.dumps(
                        {
                            "error": str(exc),
                            "response_body": response_body,
                            "base_url": base_url,
                            "checkpoint": checkpoint,
                            "vae": vae,
                            "loras": lora_stack,
                            "workflow": workflow,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except Exception as debug_exc:  # noqa: BLE001
                logger.warning("Could not write ComfyUI prompt error debug file: %s", debug_exc)
            detail = f"; response_body={response_body[:2000]}" if response_body else ""
            return error_response(
                error=f"ComfyUI prompt submission failed: {exc}{detail}",
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

        actual_width, actual_height = _read_png_dimensions(source_path)
        output_resolution = (
            f"{actual_width}x{actual_height}"
            if actual_width is not None and actual_height is not None
            else None
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
            "loras": lora_stack,
            "workflow_node_audit": workflow_node_audit,
            "model_stack_verified": model_stack_verified,
            "workflow_key": workflow_key,
            "requested_output_type": requested_output_type,
            "output_type": normalized_output_type,
            "reference_identity_evidence": reference_identity_evidence,
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
            "actual_width": actual_width,
            "actual_height": actual_height,
            "output_resolution": output_resolution,
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
            "loras": lora_stack,
            "workflow_node_audit": workflow_node_audit,
            "model_stack_verified": model_stack_verified,
            "workflow_key": workflow_key,
            "requested_output_type": requested_output_type,
            "output_type": normalized_output_type,
            "reference_identity_evidence": reference_identity_evidence,
        }
        metadata = {
            "provider": self.name,
            "prompt_id": prompt_id,
            "api_base_url": base_url,
            "workflow_key": workflow_key,
            "requested_workflow_key": requested_workflow_key or None,
            "requested_output_type": requested_output_type,
            "output_type": normalized_output_type,
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
            "negative_baseline": runtime_preset.get("negative_baseline") if runtime_preset is not None else None,
            "negative_prompt": negative_prompt,
            "vae": vae,
            "loras": lora_stack,
            "workflow_node_audit": workflow_node_audit,
            "model_stack_verified": model_stack_verified,
            "reference_identity_evidence": reference_identity_evidence,
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
            "actual_width": actual_width,
            "actual_height": actual_height,
            "output_resolution": output_resolution,
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
        evidence = {
            "workflow_key": workflow_key,
            "workflow_path": str(bundle["workflow_path"]),
            "prompt_id": prompt_id,
            "seed": seed,
            "vae": vae,
            "vae_report_value": vae if vae is not None else "checkpoint_builtin_vae",
            "loras": lora_stack,
            "workflow_node_audit": workflow_node_audit,
            "model_stack_verified": model_stack_verified,
            "preset": preset_name,
            "requested_output_type": requested_output_type,
            "output_type": normalized_output_type,
            "output_image": output_image,
            "artifact_path": str(bundle["primary_image_path"]),
            "output_source_origin": source_origin,
            "actual_width": actual_width,
            "actual_height": actual_height,
            "output_resolution": output_resolution,
            "technical_execution_status": "COMPLETE",
            "visual_quality_status": "USER_REVIEW_REQUIRED",
            "visual_quality_note": "Tool evidence verifies model stack and file output only; visual style/composition must be reviewed separately.",
            "reporting_note": (
                "Use evidence.workflow_key for the workflow key. "
                "Do not substitute workflow_path when reporting workflow_key."
            ),
            "reference_identity_evidence": reference_identity_evidence,
        }
        report_evidence = {
            "operation": "txt2img",
            "workflow_key": workflow_key,
            "workflow_path": str(bundle["workflow_path"]),
            "prompt_id": prompt_id,
            "preset": preset_name,
            "requested_output_type": requested_output_type,
            "output_type": normalized_output_type,
            "checkpoint": checkpoint,
            "vae": vae,
            "loras": lora_stack,
            "workflow_node_audit": workflow_node_audit,
            "model_stack_verified": model_stack_verified,
            "technical_execution_status": "COMPLETE",
            "visual_quality_status": "USER_REVIEW_REQUIRED",
            "visual_quality_note": "Automatic report confirms execution evidence only. Do not claim visual PASS without user/Codex review.",
            "reporting_contract": {
                "version": "image_generate_report_evidence_v2",
                "required_fields": [
                    "operation",
                    "preset",
                    "output_type",
                    "workflow_key",
                    "checkpoint",
                    "vae",
                    "loras",
                    "workflow_node_audit",
                    "model_stack_verified",
                    "technical_execution_status",
                    "visual_quality_status",
                    "output_resolution",
                    "seed",
                    "artifact_path",
                ],
                "instruction": "Final Slack report must copy report_evidence fields; do not infer or omit model stack fields.",
            },
            "seed": seed,
            "artifact_path": str(bundle["primary_image_path"]),
            "output_resolution": output_resolution,
            "actual_width": actual_width,
            "actual_height": actual_height,
            "reference_identity_evidence": reference_identity_evidence,
        }
        _update_metadata_report_evidence(bundle["metadata_path"], report_evidence)
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
                "negative_baseline": runtime_preset.get("negative_baseline") if runtime_preset is not None else None,
                "negative_prompt": negative_prompt,
                "evidence": evidence,
                "report_evidence": report_evidence,
                "workflow_key": workflow_key,
                "requested_output_type": requested_output_type,
                "output_type": normalized_output_type,
                "reference_identity_evidence": reference_identity_evidence,
                "seed": seed,
                "vae": vae,
                "vae_report_value": evidence["vae_report_value"],
                "loras": lora_stack,
                "workflow_node_audit": workflow_node_audit,
                "model_stack_verified": model_stack_verified,
                "width": width,
                "height": height,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "output_resolution": output_resolution,
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
                "output_image": output_image,
                "prompt_id": prompt_id,
                "category": category,
                "api_base_url": base_url,
            },
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(ComfyLocalImageGenProvider())
