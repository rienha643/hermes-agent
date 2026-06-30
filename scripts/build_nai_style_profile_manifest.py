#!/usr/bin/env python3
"""Build Hermes NovelAI style-reference profile manifests.

The generated manifest doubles as a reference alias manifest for the NovelAI
adapter and as a compact per-reference style policy table for Seir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LANE_POLICIES: dict[str, dict[str, Any]] = {
    "default_subculture": {
        "label_ko": "기본 서브컬쳐/게임 일러스트",
        "reference_strength_default": 0.45,
        "reference_information_extracted_default": 0.75,
        "positive_style_tags": [
            "polished anime game illustration",
            "rich game illustration",
            "soft gradient anime shading",
            "clear readable face",
        ],
        "negative_terms_to_suppress": [],
        "usage_note_ko": "일반 초상/전신/인게임 CG 기본 후보. 외형은 새 지시를 따르고 그림체만 참고한다.",
    },
    "high_fx_keyvisual": {
        "label_ko": "고효과 키비주얼/홍보 일러스트",
        "reference_strength_default": 0.43,
        "reference_information_extracted_default": 0.82,
        "positive_style_tags": [
            "cinematic lighting",
            "volumetric lighting",
            "dramatic rim light",
            "layered detailed background",
            "premium mobile game key art",
        ],
        "negative_terms_to_suppress": ["flat lighting", "low-detail background", "plain background", "empty background"],
        "usage_note_ko": "광원/배경 밀도/마법 이펙트가 중요한 키비주얼용. 양산형 기본값으로 자동 사용하지 않는다.",
    },
    "hybrid_3d_painterly_keyvisual": {
        "label_ko": "3D/블렌더+회화 혼합 키비주얼",
        "reference_strength_default": 0.35,
        "reference_information_extracted_default": 0.85,
        "positive_style_tags": [
            "stylized 3D anime game illustration",
            "blender medium",
            "oil painting medium",
            "cinematic volumetric lighting",
            "delicate hues",
        ],
        "negative_terms_to_suppress": [
            "photorealistic",
            "realistic skin texture",
            "flat lighting",
            "flat cel coloring",
            "flat anime screenshot",
            "low color depth",
            "posterized shading",
        ],
        "usage_note_ko": "3D/회화 혼합 레퍼런스. strength를 과하게 올리면 뭉개질 수 있어 낮은 s와 높은 i를 기본으로 둔다.",
    },
    "three_d_cg_render": {
        "label_ko": "3D/CG 렌더/반실사 게임 CG",
        "reference_strength_default": 0.40,
        "reference_information_extracted_default": 0.85,
        "positive_style_tags": [
            "stylized semi-realistic 3D anime game render",
            "blender medium",
            "MMD-like anime CG",
            "glossy detailed skin",
            "volumetric lights",
        ],
        "negative_terms_to_suppress": [
            "photorealistic",
            "realistic skin texture",
            "flat lighting",
            "flat cel coloring",
            "flat anime screenshot",
            "thick black outline",
            "low color depth",
            "posterized shading",
        ],
        "usage_note_ko": "3D/CG 화풍을 참고할 때만 사용. NSFW 태그는 가져오지 않고 렌더링/광원/질감 태그만 참고한다.",
    },
    "painterly_ink_atmosphere": {
        "label_ko": "페인터리/수묵/몽환 분위기",
        "reference_strength_default": 0.45,
        "reference_information_extracted_default": 0.88,
        "positive_style_tags": [
            "painterly anime illustration",
            "ink wash painting",
            "oriental painting",
            "muted colors",
            "ornate atmospheric background",
        ],
        "negative_terms_to_suppress": [
            "flat cel coloring",
            "flat anime screenshot",
            "thick black outline",
            "posterized shading",
            "simple gradient background",
        ],
        "usage_note_ko": "몽환/퇴폐/회상/특수 장면용. 선명한 양산형 게임 일러스트 프리셋이 분위기를 밀어내지 않도록 한다.",
    },
    "sd_chibi_gag_only": {
        "label_ko": "SD/치비/개그씬 전용",
        "reference_strength_default": 0.50,
        "reference_information_extracted_default": 0.75,
        "positive_style_tags": ["chibi anime style", "cute simplified proportions", "expressive gag scene"],
        "negative_terms_to_suppress": ["chibi"],
        "usage_note_ko": "개그씬/SD 컷인 전용. 일반 전신/초상/키비주얼에는 자동 사용하지 않는다.",
    },
    "pixel_sprite_simplified": {
        "label_ko": "픽셀/스프라이트/간이 도트풍",
        "reference_strength_default": 0.52,
        "reference_information_extracted_default": 0.72,
        "positive_style_tags": ["simplified sprite-like anime rendering", "compact game sprite feel", "clean readable silhouette"],
        "negative_terms_to_suppress": ["low color depth", "posterized shading"],
        "usage_note_ko": "순수 픽셀보다 간이 도트화된 스프라이트풍. 필요할 때만 명시 사용한다.",
    },
    "character_sheet_reference": {
        "label_ko": "캐릭터 시트/전신 참고",
        "reference_strength_default": 0.35,
        "reference_information_extracted_default": 0.78,
        "positive_style_tags": ["clean full body character reference", "stable silhouette", "simple readable character presentation"],
        "negative_terms_to_suppress": [],
        "usage_note_ko": "정식 캐릭터 시트 생성 전 임시 참고용. 운영 캐릭터 확정 후 별도 Char.Ref 시트를 만드는 것이 우선이다.",
    },
}

STYLE_KEYWORDS = {
    "3d": ["3d", "blender", "mmd", "ray tracing", "subsurface scattering", "game cg", "gmae cg"],
    "realistic": ["photorealistic", "photo realistic", "realistic", "glossy skin", "shiny skin", "detailed skin"],
    "painterly": ["oil painting", "painterly", "oriental painting", "ink wash", "ink", "no lineart", "muted colors"],
    "high_fx": ["cinematic", "volumetric", "lens flare", "bloom", "rim light", "magic", "dramatic"],
    "flat_clean": ["flat colors", "cel shading", "crisp outlines", "clean lineart"],
}

STYLE_TAG_WHITELIST = [
    "3d",
    "blender",
    "mmd",
    "game cg",
    "oil painting",
    "painterly",
    "oriental painting",
    "ink wash painting",
    "no lineart",
    "flat colors",
    "cel shading",
    "cinematic lighting",
    "volumetric lighting",
    "volumetric lights",
    "lens flare",
    "bloom",
    "ray tracing",
    "subsurface scattering",
    "shiny skin",
    "glossy skin",
    "detailed skin",
    "soft shading",
    "smooth skin texture",
    "natural skin glow",
    "muted colors",
    "saturated colors",
    "delicate hues",
    "vibrant colors",
    "ornate",
    "dynamic composition",
    "complex background",
    "abstract background",
    "japanese style",
    "chibi",
    "sprite",
    "reference sheet",
    "turnaround",
    "multiple views",
]


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(value: Any) -> str:
    return str(value or "").replace("_", " ").lower()


def prompt_from_record(record: dict[str, Any], metadata_by_name: dict[str, dict[str, Any]]) -> str:
    metadata = metadata_by_name.get(record["file_name"], {})
    description = metadata.get("description")
    if isinstance(description, str) and description.strip():
        return description
    comment_json = metadata.get("comment_json")
    if isinstance(comment_json, dict) and isinstance(comment_json.get("prompt"), str):
        return comment_json["prompt"]
    return ""


def keyword_hits(prompt: str) -> dict[str, list[str]]:
    lower = normalize_text(prompt)
    hits: dict[str, list[str]] = {}
    for group, keywords in STYLE_KEYWORDS.items():
        found = [keyword for keyword in keywords if keyword in lower]
        if found:
            hits[group] = found
    return hits


def extract_style_tags(prompt: str) -> list[str]:
    lower = normalize_text(prompt)
    tags: list[str] = []
    for tag in STYLE_TAG_WHITELIST:
        if tag in lower:
            tags.append(tag)
    artist_tags = re.findall(r"artist\s*:\s*([^,:{}]+)", prompt, flags=re.IGNORECASE)
    for artist in artist_tags[:8]:
        cleaned = re.sub(r"\s+", " ", artist).strip()
        if cleaned:
            tags.append(f"artist:{cleaned}")
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tag)
    return deduped


def classify_lane(category: str, prompt: str) -> str:
    hits = keyword_hits(prompt)
    has_3d = bool(hits.get("3d"))
    has_realistic = bool(hits.get("realistic"))
    has_painterly = bool(hits.get("painterly"))
    if category == "04_sd_chibi_gag_only":
        return "sd_chibi_gag_only"
    if category == "05_pixel_sprite_simplified":
        return "pixel_sprite_simplified"
    if category == "07_character_sheet_reference":
        return "character_sheet_reference"
    if has_3d and has_painterly:
        return "hybrid_3d_painterly_keyvisual"
    if has_3d and has_realistic:
        return "three_d_cg_render"
    if has_painterly:
        return "painterly_ink_atmosphere"
    if category == "02_key_visual_high_fx" or hits.get("high_fx"):
        return "high_fx_keyvisual"
    if category == "03_special_scene_atmosphere":
        return "painterly_ink_atmosphere" if has_painterly else "high_fx_keyvisual"
    return "default_subculture"


def alias_for(filename: str) -> str:
    return Path(filename).stem


def aliases_for(filename: str) -> list[str]:
    stem = Path(filename).stem
    aliases = [f"style_ref_{stem}"]
    if stem.startswith("new_style_"):
        aliases.append("style_ref_" + stem.removeprefix("new_style_"))
    return aliases


def build_manifest(category_manifest: Path, metadata_manifest: Path, selected_dir: Path) -> dict[str, Any]:
    categorized = json.loads(category_manifest.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_manifest.read_text(encoding="utf-8"))
    metadata_by_name = {record["file_name"]: record for record in metadata.get("records", [])}
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    items: list[dict[str, Any]] = []
    lane_counts: dict[str, int] = {}
    for record in sorted(categorized.get("records", []), key=lambda item: item["file_name"]):
        filename = record["file_name"]
        image_path = selected_dir / "images" / filename
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        prompt = prompt_from_record(record, metadata_by_name)
        lane = classify_lane(record["category"], prompt)
        policy = LANE_POLICIES[lane]
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
        profile = {
            "style_lane": lane,
            "style_lane_label_ko": policy["label_ko"],
            "category": record["category"],
            "source_collection": record.get("source_collection"),
            "source_number": record.get("source_number"),
            "style_tags": extract_style_tags(prompt),
            "style_keyword_hits": keyword_hits(prompt),
            "reference_strength_default": policy["reference_strength_default"],
            "reference_information_extracted_default": policy["reference_information_extracted_default"],
            "positive_style_tags": policy["positive_style_tags"],
            "negative_terms_to_suppress": policy["negative_terms_to_suppress"],
            "content_transfer_policy": "style_only; do not copy face, hair color, outfit, pose, background, or NSFW/content/action tags from the reference unless the user explicitly asks.",
            "usage_note_ko": policy["usage_note_ko"],
        }
        items.append(
            {
                "alias": alias_for(filename),
                "aliases": aliases_for(filename),
                "asset_set": "nai_style_candidate_pool_260630",
                "bytes": image_path.stat().st_size,
                "external_ssd_path": str(image_path),
                "filename": filename,
                "sha256": sha256_path(image_path),
                "status": "selected_by_user",
                "usage": "NAI style_reference candidate; style-only unless explicitly overridden",
                "style_profile": profile,
            }
        )

    return {
        "schema": "nai_style_profile_manifest_v1",
        "asset_set": "nai_style_candidate_pool_260630",
        "created_at": now,
        "updated_at": now,
        "count": len(items),
        "asset_root": str(selected_dir),
        "image_dir": str(selected_dir / "images"),
        "source_category_manifest": str(category_manifest),
        "source_metadata_manifest": str(metadata_manifest),
        "policy": {
            "default_mode": "auto profile only when a registered reference alias/path is used with experimental_reference_images=True",
            "style_only": True,
            "do_not_import_content_tags": True,
            "manual_override": "pass reference_style_profile=false, or explicit reference_strength/reference_information_extracted/negative_prompt when needed",
        },
        "lane_counts": lane_counts,
        "lane_policies": LANE_POLICIES,
        "items": items,
    }


def write_csv_summary(manifest: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "alias",
                "filename",
                "category",
                "style_lane",
                "reference_strength_default",
                "reference_information_extracted_default",
                "style_tags",
                "negative_terms_to_suppress",
            ],
        )
        writer.writeheader()
        for item in manifest["items"]:
            profile = item["style_profile"]
            writer.writerow(
                {
                    "alias": item["alias"],
                    "filename": item["filename"],
                    "category": profile["category"],
                    "style_lane": profile["style_lane"],
                    "reference_strength_default": profile["reference_strength_default"],
                    "reference_information_extracted_default": profile["reference_information_extracted_default"],
                    "style_tags": ", ".join(profile["style_tags"]),
                    "negative_terms_to_suppress": ", ".join(profile["negative_terms_to_suppress"]),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category-manifest", type=Path, required=True)
    parser.add_argument("--metadata-manifest", type=Path, required=True)
    parser.add_argument("--selected-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.category_manifest, args.metadata_manifest, args.selected_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv_summary(manifest, args.csv_output)
    print(f"Wrote {manifest['count']} style profiles to {args.output}")
    print("Lane counts:", json.dumps(manifest["lane_counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
