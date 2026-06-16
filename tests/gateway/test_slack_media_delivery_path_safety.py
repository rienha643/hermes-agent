from pathlib import Path

import gateway.platforms.base as platform_base

from gateway.platforms.base import validate_media_delivery_path, validate_slack_delivery_path


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
