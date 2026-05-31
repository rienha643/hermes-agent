from __future__ import annotations

import json
import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    from tools import image_generation_tool

    image_generation_tool._FORGE_LOCAL_SINGLE_PASS_RESULTS.clear()
    image_generation_tool._SINGLE_OUTPUT_TASKS.clear()
    image_generation_tool._SINGLE_OUTPUT_TASK_RESULTS.clear()
    yield
    image_gen_registry._reset_for_tests()
    image_generation_tool._FORGE_LOCAL_SINGLE_PASS_RESULTS.clear()
    image_generation_tool._SINGLE_OUTPUT_TASKS.clear()
    image_generation_tool._SINGLE_OUTPUT_TASK_RESULTS.clear()


class _FakeCodexProvider(ImageGenProvider):
    def __init__(self):
        self.calls = []

    @property
    def name(self) -> str:
        return "codex"

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        self.calls.append({"prompt": prompt, "aspect_ratio": aspect_ratio, **kwargs})
        return {
            "success": True,
            "image": "/tmp/codex-test.png",
            "model": "gpt-5.2-codex",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": "codex",
        }


class TestPluginDispatch:
    def test_dispatch_routes_to_codex_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")
        image_gen_registry.register_provider(_FakeCodexProvider())

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: _FakeCodexProvider() if name == "codex" else None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw cat", "square")
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["image"] == "/tmp/codex-test.png"
        assert payload["aspect_ratio"] == "square"

    def test_dispatch_forwards_project_metadata_to_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")
        provider = _FakeCodexProvider()

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "codex" else None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider(
            "draw cat",
            "square",
            project_name="260601_망각구역",
            artifact_name="scene",
        )
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert provider.calls == [
            {
                "prompt": "draw cat",
                "aspect_ratio": "square",
                "project_name": "260601_망각구역",
                "artifact_name": "scene",
            }
        ]

    def test_dispatch_single_pass_guard_reuses_forge_local_result(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: forge-local\n")
        image_generation_tool._FORGE_LOCAL_SINGLE_PASS_RESULTS.clear()

        class _ForgeLocalProvider:
            name = "forge-local"

            def __init__(self):
                self.calls = 0

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                self.calls += 1
                return {
                    "success": True,
                    "provider": self.name,
                    "image": f"/tmp/forge-local-{self.calls}.png",
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                }

        provider = _ForgeLocalProvider()
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "forge-local")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "forge-local" else None)

        first = image_generation_tool._dispatch_to_plugin_provider("draw a robot", "square", task_id="task-123")
        second = image_generation_tool._dispatch_to_plugin_provider("draw a robot", "square", task_id="task-123")

        assert first is not None
        assert second is not None
        assert json.loads(first)["image"] == "/tmp/forge-local-1.png"
        assert json.loads(second)["image"] == "/tmp/forge-local-1.png"
        assert provider.calls == 1
        assert image_generation_tool._FORGE_LOCAL_SINGLE_PASS_RESULTS["task-123"] == first

    def test_handle_image_generate_infers_project_metadata_from_prompt(self, monkeypatch, tmp_path):
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        captured = {}

        def fake_dispatch(prompt, aspect_ratio, task_id=None, project_name=None, artifact_name=None):
            captured.update(
                {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "task_id": task_id,
                    "project_name": project_name,
                    "artifact_name": artifact_name,
                }
            )
            return json.dumps({"success": True, "image": "/tmp/result.png"})

        monkeypatch.setattr(image_generation_tool, "_dispatch_to_plugin_provider", fake_dispatch)

        result = image_generation_tool._handle_image_generate(
            {
                "prompt": "망각구역 주인공 컨셉 이미지 1장 생성",
                "aspect_ratio": "square",
            },
            task_id="task-2",
        )

        assert json.loads(result)["success"] is True
        assert captured == {
            "prompt": "망각구역 주인공 컨셉 이미지 1장 생성",
            "aspect_ratio": "square",
            "task_id": "task-2",
            "project_name": "망각구역",
            "artifact_name": "주인공",
        }

    def test_handle_image_generate_defaults_artifact_to_image_when_missing(self, monkeypatch, tmp_path):
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        captured = {}

        def fake_dispatch(prompt, aspect_ratio, task_id=None, project_name=None, artifact_name=None):
            captured.update(
                {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "task_id": task_id,
                    "project_name": project_name,
                    "artifact_name": artifact_name,
                }
            )
            return json.dumps({"success": True, "image": "/tmp/result.png"})

        monkeypatch.setattr(image_generation_tool, "_dispatch_to_plugin_provider", fake_dispatch)

        result = image_generation_tool._handle_image_generate(
            {
                "prompt": "망각구역 검증용 이미지 1장 생성",
                "aspect_ratio": "portrait",
            },
            task_id="task-3",
        )

        assert json.loads(result)["success"] is True
        assert captured == {
            "prompt": "망각구역 검증용 이미지 1장 생성",
            "aspect_ratio": "portrait",
            "task_id": "task-3",
            "project_name": "망각구역",
            "artifact_name": "검증용이미지",
        }

    def test_handle_image_generate_uses_task_metadata_registry(self, monkeypatch, tmp_path):
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        captured = {}

        def fake_dispatch(prompt, aspect_ratio, task_id=None, project_name=None, artifact_name=None):
            captured.update(
                {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "task_id": task_id,
                    "project_name": project_name,
                    "artifact_name": artifact_name,
                }
            )
            return json.dumps({"success": True, "image": "/tmp/result.png"})

        monkeypatch.setattr(image_generation_tool, "_dispatch_to_plugin_provider", fake_dispatch)
        image_generation_tool.register_image_task_metadata(
            "child-task-1",
            project_name="망각구역",
            artifact_name="주인공",
        )

        result = image_generation_tool._handle_image_generate(
            {
                "prompt": "주인공 컨셉 이미지 1장 생성",
                "aspect_ratio": "square",
            },
            task_id="child-task-1",
        )

        assert json.loads(result)["success"] is True
        assert captured == {
            "prompt": "주인공 컨셉 이미지 1장 생성",
            "aspect_ratio": "square",
            "task_id": "child-task-1",
            "project_name": "망각구역",
            "artifact_name": "주인공",
        }

    def test_dispatch_single_output_mode_reuses_cached_result_for_non_forge_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")

        class _CountingCodexProvider(_FakeCodexProvider):
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                self.calls += 1
                return {
                    "success": True,
                    "image": f"/tmp/codex-single-{self.calls}.png",
                    "model": "gpt-5.2-codex",
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "provider": "codex",
                }

        provider = _CountingCodexProvider()
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "codex" else None)

        image_generation_tool.enable_single_output_task_mode("task-single")
        try:
            first = image_generation_tool._dispatch_to_plugin_provider("draw one cat", "square", task_id="task-single")
            second = image_generation_tool._dispatch_to_plugin_provider("draw one cat", "square", task_id="task-single")
        finally:
            image_generation_tool.disable_single_output_task_mode("task-single")

        assert first is not None
        assert second is not None
        assert json.loads(first)["image"] == "/tmp/codex-single-1.png"
        assert json.loads(second)["image"] == "/tmp/codex-single-1.png"
        assert provider.calls == 1

    def test_dispatch_reports_missing_registered_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: missing-codex\n")

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "missing-codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda: None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw cat", "landscape")
        payload = json.loads(dispatched)

        assert payload["success"] is False
        assert payload["error_type"] == "provider_not_registered"
        assert "image_gen.provider='missing-codex'" in payload["error"]

    def test_dispatch_force_refreshes_plugins_when_provider_initially_missing(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")

        calls = []
        provider_state = {"provider": None}

        def fake_ensure_plugins_discovered(force=False):
            calls.append(force)
            if force:
                provider_state["provider"] = _FakeCodexProvider()

        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", fake_ensure_plugins_discovered)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider_state["provider"])

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw hammy", "portrait")
        payload = json.loads(dispatched)

        assert calls == [False, True]
        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["aspect_ratio"] == "portrait"
