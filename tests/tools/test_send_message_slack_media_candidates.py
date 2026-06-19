from pathlib import Path

import pytest

import gateway.platforms.base as platform_base
from tools.send_message_tool import _validate_slack_media_candidates


def _write_png(path: Path, payload: bytes = b"png") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _validate(paths: list[Path], *, expected_count: int | None = None):
    return _validate_slack_media_candidates([(str(path), False) for path in paths], expected_count=expected_count)


def test_key_visual_media_candidates_pass_expected_count(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(platform_base, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),))
    files = [
        _write_png(
            tmp_path
            / "HermesWork"
            / "Image"
            / "key_visual_challenger_round_v1"
            / f"checkpoint_{idx}"
            / f"checkpoint_{idx}_00001_.png",
            f"png-{idx}".encode(),
        )
        for idx in range(8)
    ]

    validated, blocked = _validate(files, expected_count=8)

    assert [Path(path).name for path, _ in validated] == [path.name for path in files]
    assert blocked == []


def test_fullbody_media_candidates_still_pass_expected_count(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(platform_base, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),))
    files = [
        _write_png(
            tmp_path
            / "HermesWork"
            / "Image"
            / "fullbody_challenger_round_v1"
            / f"checkpoint_{idx}"
            / f"checkpoint_{idx}_00001_.png",
            f"png-{idx}".encode(),
        )
        for idx in range(8)
    ]

    validated, blocked = _validate(files, expected_count=8)

    assert len(validated) == 8
    assert blocked == []


@pytest.mark.parametrize("name", ["private_key.png", "api_key.png", "secret_key.png", ".env.png", "token.png", "credential.png"])
def test_sensitive_media_candidates_remain_blocked(tmp_path: Path, monkeypatch, name: str):
    monkeypatch.setattr(platform_base, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),))
    image = _write_png(tmp_path / "HermesWork" / "secrets" / name)

    validated, blocked = _validate([image])

    assert validated == []
    assert blocked == [{"path": str(image), "reason": "blocked_sensitive_or_sidecar"}]


def test_expected_count_mismatch_still_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(platform_base, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),))
    image = _write_png(tmp_path / "HermesWork" / "Image" / "fullbody_challenger_round_v1" / "one.png")

    with pytest.raises(ValueError, match="expected 2, got 1"):
        _validate([image], expected_count=2)


def test_uuid_segment_containing_db_does_not_block_publish_root_png(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(platform_base, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),))
    image = _write_png(
        tmp_path
        / "HermesWork"
        / "Image"
        / "8f461990-77d3-4b9c-a0e8-8f19362db3ef"
        / "windows-remote-comfyui-e2e-512_v1.png"
    )

    validated, blocked = _validate([image])

    assert validated == [(str(image), False)]
    assert blocked == []


@pytest.mark.parametrize("name", ["file.db", "state.sqlite", "cache.sqlite3"])
def test_database_files_remain_blocked_even_under_publish_root(tmp_path: Path, monkeypatch, name: str):
    monkeypatch.setattr(platform_base, "_SLACK_ALLOWED_PUBLISH_ROOTS", (str(tmp_path),))
    db_file = tmp_path / "HermesWork" / "Image" / "safe_scope" / name
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_file.write_bytes(b"not an image")

    validated, blocked = _validate([db_file])

    assert validated == []
    assert blocked
