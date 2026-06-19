from pathlib import Path

import gateway.platforms.base as platform_base

from gateway.platforms.base import validate_media_delivery_path, validate_slack_delivery_path
from gateway.platforms.slack import _slack_media_block_reason


def _write_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png-bytes")
    return path


def test_key_visual_challenger_png_path_is_allowed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(platform_base, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),))
    image = _write_png(
        tmp_path
        / "HermesWork"
        / "Image"
        / "key_visual_challenger_round_v1"
        / "hdaRainbowIllusMixV1_v13"
        / "hdaRainbowIllusMixV1_v13_00001_.png"
    )

    assert validate_media_delivery_path(str(image)) == str(image.resolve())
    assert validate_slack_delivery_path(str(image), image_only=True) == str(image.resolve())


def test_fullbody_challenger_png_path_remains_allowed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(platform_base, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),))
    image = _write_png(
        tmp_path
        / "HermesWork"
        / "Image"
        / "fullbody_challenger_round_v1"
        / "hdaRainbowIllusMixV1_v13"
        / "hdaRainbowIllusMixV1_v13_00001_.png"
    )

    assert validate_media_delivery_path(str(image)) == str(image.resolve())
    assert validate_slack_delivery_path(str(image), image_only=True) == str(image.resolve())


def test_sensitive_key_material_paths_remain_blocked(tmp_path: Path):
    private_key = _write_png(tmp_path / "HermesWork" / "secrets" / "private_key.png")
    env_file = _write_png(tmp_path / "HermesWork" / ".env.png")

    assert validate_media_delivery_path(str(private_key)) is None
    assert validate_slack_delivery_path(str(private_key), image_only=True) is None
    assert validate_media_delivery_path(str(env_file)) is None
    assert validate_slack_delivery_path(str(env_file), image_only=True) is None


def test_slack_adapter_allows_v6_uuid_db_publish_root_png(monkeypatch):
    v6_path = (
        "/Volumes/SSD_Hermes/HermesWork/Image/260617_HermesWork_Image/"
        "8f461990-77d3-4b9c-a0e8-8f19362db3ef/windows-remote-comfyui-e2e-512_v1.png"
    )

    assert _slack_media_block_reason(v6_path) is None


def test_slack_adapter_blocks_database_file_by_extension_under_publish_root():
    db_path = "/Volumes/SSD_Hermes/HermesWork/Image/safe_scope/file.db"

    assert _slack_media_block_reason(db_path) == "forbidden_extension:.db"
