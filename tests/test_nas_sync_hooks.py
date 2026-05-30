from __future__ import annotations

from pathlib import Path

import pytest

import nas_sync_hooks


def test_queue_nas_sync_hook_launches_and_debounces(monkeypatch, tmp_path):
    script = tmp_path / "hermes_nas_backup.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    launched: list[list[str]] = []

    class DummyPopen:
        def __init__(self, cmd, **kwargs):
            launched.append(cmd)
            self.cmd = cmd
            self.kwargs = kwargs

    monkeypatch.setattr(nas_sync_hooks, "_resolve_nas_hook_script", lambda: script)
    monkeypatch.setattr(nas_sync_hooks.subprocess, "Popen", DummyPopen)
    monkeypatch.setattr(nas_sync_hooks, "_IN_PROCESS_LAST_LAUNCH", {})

    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact_path = source_root / "artifact.png"
    artifact_path.write_bytes(b"x")

    assert nas_sync_hooks.queue_nas_sync_hook(
        category="image",
        scope="task-1",
        artifact_path=artifact_path,
        source_root=source_root,
    )
    assert not nas_sync_hooks.queue_nas_sync_hook(
        category="image",
        scope="task-1",
        artifact_path=artifact_path,
        source_root=source_root,
    )

    assert len(launched) == 1
    assert launched[0][0] == str(Path(nas_sync_hooks.sys.executable))
    assert launched[0][1] == str(script)
    assert launched[0][2:6] == ["--hook", str(source_root), "--category", "image"]
    assert launched[0][6:10] == ["--scope", "task-1", "--artifact-path", str(artifact_path)]


def test_resolve_nas_hook_script_prefers_env_over_common_and_profile(monkeypatch, tmp_path):
    env_script = tmp_path / "env.py"
    env_script.write_text("print('env')\n", encoding="utf-8")

    common_root = tmp_path / "root"
    common_script = common_root / "profiles" / "cron-fast" / "scripts" / "hermes_nas_backup.py"
    common_script.parent.mkdir(parents=True, exist_ok=True)
    common_script.write_text("print('common')\n", encoding="utf-8")

    profile_home = common_root / "profiles" / "artist"
    profile_home.mkdir(parents=True, exist_ok=True)
    profile_script = profile_home / "scripts" / "hermes_nas_backup.py"
    profile_script.parent.mkdir(parents=True, exist_ok=True)
    profile_script.write_text("print('profile')\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_NAS_BACKUP_SCRIPT", str(env_script))
    monkeypatch.setattr(nas_sync_hooks, "get_default_hermes_root", lambda: common_root)
    monkeypatch.setattr(nas_sync_hooks, "get_hermes_home", lambda: profile_home)

    assert nas_sync_hooks._resolve_nas_hook_script() == env_script


def test_resolve_nas_hook_script_uses_configured_script_before_common(monkeypatch, tmp_path):
    common_root = tmp_path / "root"
    common_script = common_root / "profiles" / "cron-fast" / "scripts" / "hermes_nas_backup.py"
    common_script.parent.mkdir(parents=True, exist_ok=True)
    common_script.write_text("print('common')\n", encoding="utf-8")

    profile_home = common_root / "profiles" / "artist"
    profile_home.mkdir(parents=True, exist_ok=True)

    configured_script = tmp_path / "configured.py"
    configured_script.write_text("print('configured')\n", encoding="utf-8")

    monkeypatch.delenv("HERMES_NAS_BACKUP_SCRIPT", raising=False)
    monkeypatch.setattr(nas_sync_hooks, "get_default_hermes_root", lambda: common_root)
    monkeypatch.setattr(nas_sync_hooks, "get_hermes_home", lambda: profile_home)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"nas": {"backup_script": str(configured_script)}},
    )

    assert nas_sync_hooks._resolve_nas_hook_script() == configured_script


def test_resolve_nas_hook_script_uses_common_cron_fast_for_non_cron_profiles(monkeypatch, tmp_path):
    common_root = tmp_path / "root"
    common_script = common_root / "profiles" / "cron-fast" / "scripts" / "hermes_nas_backup.py"
    common_script.parent.mkdir(parents=True, exist_ok=True)
    common_script.write_text("print('common')\n", encoding="utf-8")

    profile_home = common_root / "profiles" / "artist"
    profile_home.mkdir(parents=True, exist_ok=True)

    monkeypatch.delenv("HERMES_NAS_BACKUP_SCRIPT", raising=False)
    monkeypatch.setattr(nas_sync_hooks, "get_default_hermes_root", lambda: common_root)
    monkeypatch.setattr(nas_sync_hooks, "get_hermes_home", lambda: profile_home)
    monkeypatch.setattr(nas_sync_hooks, "_resolve_configured_nas_hook_script", lambda: None)

    assert nas_sync_hooks._resolve_nas_hook_script() == common_script


def test_resolve_nas_hook_script_falls_back_to_profile_script(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "designer"
    profile_script = profile_home / "scripts" / "hermes_nas_backup.py"
    profile_script.parent.mkdir(parents=True, exist_ok=True)
    profile_script.write_text("print('profile')\n", encoding="utf-8")

    monkeypatch.delenv("HERMES_NAS_BACKUP_SCRIPT", raising=False)
    monkeypatch.setattr(nas_sync_hooks, "get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr(nas_sync_hooks, "get_hermes_home", lambda: profile_home)
    monkeypatch.setattr(nas_sync_hooks, "_resolve_configured_nas_hook_script", lambda: None)

    assert nas_sync_hooks._resolve_nas_hook_script() == profile_script


def test_queue_nas_sync_hook_ignores_unsupported_category(monkeypatch, tmp_path):
    monkeypatch.setattr(nas_sync_hooks, "_resolve_nas_hook_script", lambda: tmp_path / "missing.py")
    monkeypatch.setattr(nas_sync_hooks.subprocess, "Popen", pytest.fail)

    assert nas_sync_hooks.queue_nas_sync_hook(
        category="story",
        scope="task-1",
        artifact_path=tmp_path / "artifact.txt",
        source_root=tmp_path,
    ) is False


def test_publishers_trigger_hook(monkeypatch, tmp_path):
    calls: list[tuple[str, str, Path, Path]] = []

    def fake_hook(*, category: str, scope: str, artifact_path: Path, source_root: Path, debounce_seconds: float = 60.0):
        calls.append((category, scope, artifact_path, source_root))
        return True

    monkeypatch.setattr("agent.image_gen_provider.queue_nas_sync_hook", fake_hook)
    monkeypatch.setattr("gateway.delivery.queue_nas_sync_hook", fake_hook)

    def fake_work_dir(*parts: str) -> Path:
        path = tmp_path.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr("hermes_constants.get_hermes_work_dir", fake_work_dir)

    import agent.image_gen_provider as image_mod
    import gateway.delivery as delivery_mod

    monkeypatch.setattr(delivery_mod, "get_hermes_work_dir", fake_work_dir)

    image_source = tmp_path / "cache.png"
    image_source.write_bytes(b"img")
    published_image = image_mod._publish_image_artifact(image_source, prefix="demo")
    assert published_image.exists()

    document_source = tmp_path / "output.md"
    document_source.write_text("hello", encoding="utf-8")
    router = delivery_mod.DeliveryRouter(config=object())  # type: ignore[arg-type]
    published_doc = router._publish_document_artifact(document_source, folder_name="job-1")
    assert published_doc.exists()

    assert calls[0][0] == "image"
    assert calls[0][1] == "demo"
    assert calls[1][0] == "documents"
    assert calls[1][1] == "job-1"


def test_publish_document_artifact_falls_back_when_copy2_metadata_fails(monkeypatch, tmp_path):
    calls: list[tuple[str, str, Path, Path]] = []

    def fake_hook(*, category: str, scope: str, artifact_path: Path, source_root: Path, debounce_seconds: float = 60.0):
        calls.append((category, scope, artifact_path, source_root))
        return True

    monkeypatch.setattr("gateway.delivery.queue_nas_sync_hook", fake_hook)

    def fake_work_dir(*parts: str) -> Path:
        path = tmp_path.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    import gateway.delivery as delivery_mod

    monkeypatch.setattr(delivery_mod, "get_hermes_work_dir", fake_work_dir)

    source = tmp_path / "output.md"
    source.write_text("hello", encoding="utf-8")

    copied: list[tuple[Path, Path]] = []

    def fake_copy2(src: Path, dst: Path):
        raise PermissionError("copystat denied")

    def fake_copyfile(src: Path, dst: Path):
        copied.append((Path(src), Path(dst)))
        Path(dst).write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
        return Path(dst)

    monkeypatch.setattr(delivery_mod.shutil, "copy2", fake_copy2)
    monkeypatch.setattr(delivery_mod.shutil, "copyfile", fake_copyfile)

    router = delivery_mod.DeliveryRouter(config=object())  # type: ignore[arg-type]
    published_doc = router._publish_document_artifact(source, folder_name="job-1")

    assert published_doc.exists()
    assert published_doc.read_text(encoding="utf-8") == "hello"
    assert copied == [(source, published_doc)]
    assert calls == [("documents", "job-1", published_doc, published_doc.parent)]
