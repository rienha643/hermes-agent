"""Regression tests for sudo detection and sudo password handling."""

import json
from pathlib import Path

import tools.terminal_tool as terminal_tool


def setup_function():
    terminal_tool._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool._reset_cached_sudo_passwords()


def test_searching_for_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "rg --line-number --no-heading --with-filename 'sudo' . | head -n 20"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_printf_literal_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "printf '%s\\n' sudo"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_non_command_argument_named_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "grep -n sudo README.md"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_actual_sudo_command_uses_configured_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo apt install -y ripgrep")

    assert transformed == "sudo -S -p '' apt install -y ripgrep"
    assert sudo_stdin == "testpass\n"


def test_actual_sudo_after_leading_env_assignment_is_rewritten(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("DEBUG=1 sudo whoami")

    assert transformed == "DEBUG=1 sudo -S -p '' whoami"
    assert sudo_stdin == "testpass\n"


def test_explicit_empty_sudo_password_tries_empty_without_prompt(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run for explicit empty password")

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"


def test_cached_sudo_password_is_used_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool._set_cached_sudo_password("cached-pass")

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("echo ok && sudo whoami")

    assert transformed == "echo ok && sudo -S -p '' whoami"
    assert sudo_stdin == "cached-pass\n"


def test_cached_sudo_password_isolated_by_session_key(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    terminal_tool._set_cached_sudo_password("alpha-pass")

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-b")
    assert terminal_tool._get_cached_sudo_password() == ""

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    assert terminal_tool._get_cached_sudo_password() == "alpha-pass"


def test_passwordless_sudo_skips_interactive_prompt_and_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError(
            "interactive sudo prompt should not run when sudo -n already works"
        )

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: True, raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo whoami")

    assert transformed == "sudo whoami"
    assert sudo_stdin is None


def test_passwordless_sudo_probe_rechecks_local_terminal(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    calls = []

    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return Result(0 if len(calls) == 1 else 1)

    monkeypatch.setattr(terminal_tool.subprocess, "run", fake_run)

    assert terminal_tool._sudo_nopasswd_works() is True
    assert terminal_tool._sudo_nopasswd_works() is False
    assert len(calls) == 2
    assert calls[0][0] == ["sudo", "-n", "true"]
    assert calls[1][0] == ["sudo", "-n", "true"]


def test_passwordless_sudo_probe_is_disabled_for_nonlocal_terminal_env(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    def _fail_run(*_args, **_kwargs):
        raise AssertionError("host sudo probe must not run for non-local terminal envs")

    monkeypatch.setattr(terminal_tool.subprocess, "run", _fail_run)

    assert terminal_tool._sudo_nopasswd_works() is False


def test_validate_workdir_allows_windows_drive_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project") is None
    assert terminal_tool._validate_workdir("C:/Users/Alice/project") is None


def test_validate_workdir_allows_windows_unc_paths():
    assert terminal_tool._validate_workdir(r"\\server\share\project") is None


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")


def test_build_terminal_approval_metadata_shell_default():
    metadata = terminal_tool._build_terminal_approval_metadata(
        "bash -lc 'python scripts/used_car_briefing.py'",
        task_id="shell-task",
        workdir="/repo/scripts",
        env_type="local",
    )

    assert metadata["purpose"] == "쉘 스크립트 실행 승인"
    assert metadata["work"] == "셸에서 전달된 스크립트를 실행합니다."
    assert metadata["risk_type"] == "shell"
    assert metadata["target"] == "/repo/scripts"
    assert metadata["job_id"] == "shell-task"


def test_build_terminal_approval_metadata_uses_registered_task_context():
    try:
        terminal_tool.register_task_approval_metadata(
            "cron-task",
            {
                "purpose": "중고차 cron 스크립트 수동 실행",
                "work": "추천 매물 출력과 링크를 검증합니다.",
                "risk_type": "cron",
                "target": "scripts/used_car_briefing.py",
                "job_id": "used-car-briefing",
            },
        )
        metadata = terminal_tool._build_terminal_approval_metadata(
            "python scripts/used_car_briefing.py",
            task_id="cron-task",
            workdir="/repo/scripts",
            env_type="local",
        )
    finally:
        terminal_tool.clear_task_approval_metadata("cron-task")

    assert metadata["purpose"] == "중고차 cron 스크립트 수동 실행"
    assert metadata["work"] == "추천 매물 출력과 링크를 검증합니다."
    assert metadata["risk_type"] == "cron"
    assert metadata["target"] == "scripts/used_car_briefing.py"
    assert metadata["job_id"] == "used-car-briefing"


def test_terminal_pending_approval_returns_metadata(monkeypatch):
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "local_persistent": False,
            "cwd": None,
            "timeout": 30,
            "lifetime_seconds": 60,
            "docker_image": None,
            "singularity_image": None,
            "modal_image": None,
            "daytona_image": None,
        },
    )
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda task_id: task_id or "default")
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {}, raising=False)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, approval_metadata=None: {
            "approved": False,
            "status": "pending_approval",
            "command": command,
            "description": "command flagged",
            "pattern_key": "shell_script",
        },
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            command="bash -lc 'python scripts/used_car_briefing.py'",
            task_id="shell-task",
            workdir="/repo/scripts",
        )
    )

    assert result["status"] == "pending_approval"
    assert result["approval_metadata"]["purpose"] == "쉘 스크립트 실행 승인"
    assert result["approval_metadata"]["risk_type"] == "shell"
    assert result["approval_metadata"]["target"] == "/repo/scripts"


def _raise_missing_cwd():
    raise FileNotFoundError(2, "No such file or directory")


def test_get_env_config_uses_pwd_when_getcwd_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", "/does/not/exist")
    monkeypatch.setenv("PWD", str(tmp_path))
    monkeypatch.setattr(terminal_tool.os, "getcwd", _raise_missing_cwd)

    config = terminal_tool._get_env_config()

    assert config["cwd"] == str(tmp_path)


def test_get_env_config_uses_repo_root_when_getcwd_and_env_dirs_are_missing(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", "/does/not/exist")
    monkeypatch.setenv("PWD", "/also/missing")
    monkeypatch.setattr(terminal_tool.os, "getcwd", _raise_missing_cwd)

    config = terminal_tool._get_env_config()

    assert config["cwd"] == str(Path(terminal_tool.__file__).resolve().parents[1])
