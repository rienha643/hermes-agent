from __future__ import annotations

import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)


def _init_repo(path: Path) -> None:
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test User")


def test_commit_guard_blocks_staged_govlawlivewip_file(tmp_path):
    guard = Path("/Users/hermes/HermesWork/WIP/GOVLAWLIVEWIP_governance_live_ping_v1/GOVLAWLIVEWIP_commit_guard.sh")
    assert guard.exists()
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "GOVLAWLIVEWIP_manifest.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "add", "GOVLAWLIVEWIP_manifest.json")

    result = subprocess.run([str(guard), str(repo)], text=True, capture_output=True, check=False)

    assert result.returncode == 1
    assert "GOVLAWLIVEWIP" in result.stderr


def test_commit_guard_blocks_governance_related_file_without_wip_name(tmp_path):
    guard = Path("/Users/hermes/HermesWork/WIP/GOVLAWLIVEWIP_governance_live_ping_v1/GOVLAWLIVEWIP_commit_guard.sh")
    assert guard.exists()
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "governance").mkdir()
    (repo / "governance" / "taskrun.py").write_text("print('x')\n", encoding="utf-8")
    _git(repo, "add", "governance/taskrun.py")

    result = subprocess.run([str(guard), str(repo)], text=True, capture_output=True, check=False)

    assert result.returncode == 1
    assert "lacks GOVLAWLIVEWIP" in result.stderr


def test_commit_guard_allows_unrelated_staged_file(tmp_path):
    guard = Path("/Users/hermes/HermesWork/WIP/GOVLAWLIVEWIP_governance_live_ping_v1/GOVLAWLIVEWIP_commit_guard.sh")
    assert guard.exists()
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "README.md").write_text("ok\n", encoding="utf-8")
    _git(repo, "add", "README.md")

    result = subprocess.run([str(guard), str(repo)], text=True, capture_output=True, check=False)

    assert result.returncode == 0
    assert "PASS" in result.stdout
