"""Regression tests for canonical HOME + gh HTTPS auth on git push paths."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


CANONICAL_HOME = "/home/ai_agent"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _release_module():
    return _load_module(
        "_release_under_test",
        Path(__file__).resolve().parents[2] / "scripts" / "release.py",
    )


def _backup_module():
    return _load_module(
        "_backup_under_test",
        Path("/home/ai_agent/hermes-profile-backup/cron-fast/scripts/hermes_git_backup.py"),
    )


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_release_setup_git_https_auth_uses_canonical_home(monkeypatch, tmp_path):
    module = _release_module()
    calls: list[tuple[list[str], dict[str, str], str]] = []

    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/gh" if name == "gh" else None)

    def fake_run(cmd, *, capture_output, text, cwd, env):
        calls.append((cmd, env, cwd))
        return _completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    env = module.setup_git_https_auth(cwd=tmp_path)

    assert env["HOME"] == CANONICAL_HOME
    assert calls == [(["/usr/bin/gh", "auth", "setup-git"], env, tmp_path)]


def test_release_push_git_refs_uses_canonical_home(monkeypatch, tmp_path):
    module = _release_module()
    push_env = {"HOME": CANONICAL_HOME, "OTHER": "1"}
    calls: list[tuple[list[str], dict[str, str], str]] = []

    monkeypatch.setattr(module, "setup_git_https_auth", lambda *, cwd=None: push_env)

    def fake_run(cmd, *, capture_output, text, cwd, env):
        calls.append((cmd, env, cwd))
        return _completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result, env = module.push_git_refs(cwd=tmp_path)

    assert result.returncode == 0
    assert env is push_env
    assert calls == [(["git", "push", "origin", "HEAD", "--tags"], push_env, tmp_path)]


def test_backup_setup_git_https_auth_uses_canonical_home(monkeypatch, tmp_path):
    module = _backup_module()
    calls: list[tuple[list[str], dict[str, str], Path]] = []

    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/gh" if name == "gh" else None)

    def fake_run(cmd, *, cwd, text, capture_output, env):
        calls.append((cmd, env, cwd))
        return _completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    env = module.setup_git_https_auth(tmp_path)

    assert env["HOME"] == CANONICAL_HOME
    assert calls == [(["/usr/bin/gh", "auth", "setup-git"], env, tmp_path)]


def test_backup_push_branch_uses_canonical_home(monkeypatch, tmp_path):
    module = _backup_module()
    push_env = {"HOME": CANONICAL_HOME, "OTHER": "1"}
    calls: list[tuple[Path, list[str], dict[str, str] | None]] = []

    monkeypatch.setattr(module, "setup_git_https_auth", lambda repo: push_env)

    def fake_run_git(repo, args, *, check=True, env=None):
        calls.append((repo, args, env))
        return _completed()

    monkeypatch.setattr(module, "run_git", fake_run_git)

    result = module.push_branch(tmp_path, "main")

    assert result.returncode == 0
    assert calls == [(tmp_path, ["push", "origin", "main"], push_env)]
