"""Tests for optional-skills/devops/watchers/scripts/_watermark.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[3] / "optional-skills" / "devops" / "watchers" / "scripts" / "_watermark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("watcher_watermark", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_dir_prefers_profile_home(monkeypatch, tmp_path):
    module = _load_module()

    import hermes_constants

    profile_home = tmp_path / "profiles" / "cron-fast"
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: profile_home)
    monkeypatch.delenv("WATCHER_STATE_DIR", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)

    assert module._state_dir() == profile_home / "watcher-state"


def test_state_dir_respects_override(monkeypatch, tmp_path):
    module = _load_module()
    override = tmp_path / "custom-state"
    monkeypatch.setenv("WATCHER_STATE_DIR", str(override))

    assert module._state_dir() == override
