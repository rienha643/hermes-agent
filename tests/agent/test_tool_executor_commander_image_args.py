import json

from agent import tool_executor


def test_commander_image_args_accept_english_tool_args_hint():
    messages = [
        {
            "role": "user",
            "content": """
[COMMANDER_DISPATCH]
queue_id: q1
[/COMMANDER_DISPATCH]

image_generate tool args:
```json
{"operation": "reference_identity_txt2img", "workflow_key": "character_reference_key_visual_experimental_v1", "reference_image_path": "/tmp/ref.png"}
```
""",
        }
    ]

    merged = tool_executor._merge_commander_image_args("image_generate", {}, messages)

    assert merged["operation"] == "reference_identity_txt2img"
    assert merged["workflow_key"] == "character_reference_key_visual_experimental_v1"
    assert merged["reference_image_path"] == "/tmp/ref.png"


def test_commander_image_args_fall_back_to_queue_command_text(tmp_path, monkeypatch):
    queue_id = "codex_test_reference_identity"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    queue_path = queue_dir / f"{queue_id}.json"
    queue_path.write_text(
        json.dumps(
            {
                "command_text": """
SFW 임시 reference identity 테스트.

image_generate tool args:
```json
{"operation": "reference_identity_txt2img", "workflow_key": "character_reference_key_visual_experimental_v1", "reference_image_path": "/tmp/ref.png", "artifact_name": "a1"}
```
""",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(tool_executor, "_COMMANDER_QUEUE_DIR", queue_dir)
    messages = [
        {
            "role": "user",
            "content": f"""
[COMMANDER_DISPATCH]
queue_id: {queue_id}
[/COMMANDER_DISPATCH]

작업 요청이야.
""",
        }
    ]

    merged = tool_executor._merge_commander_image_args("image_generate", {"prompt": "keep"}, messages)

    assert merged["prompt"] == "keep"
    assert merged["operation"] == "reference_identity_txt2img"
    assert merged["workflow_key"] == "character_reference_key_visual_experimental_v1"
    assert merged["reference_image_path"] == "/tmp/ref.png"
    assert merged["artifact_name"] == "a1"
