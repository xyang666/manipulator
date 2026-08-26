import json

import pytest

from experiments.merge_jsonl_results import merge


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_merge_sorts_and_validates_shards(tmp_path):
    scenes = tmp_path / "scenes.json"
    scenes.write_text(json.dumps([{"scene_id": "a"}, {"scene_id": "b"}]))
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    _write(first, [{"scene_id": "b", "seed": 22}])
    _write(second, [{"scene_id": "a", "seed": 22}])
    assert [row["scene_id"] for row in merge([first, second], scenes)] == ["a", "b"]


def test_merge_rejects_duplicate_scene(tmp_path):
    scenes = tmp_path / "scenes.json"
    scenes.write_text(json.dumps([{"scene_id": "a"}]))
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    _write(first, [{"scene_id": "a", "seed": 22}])
    _write(second, [{"scene_id": "a", "seed": 22}])
    with pytest.raises(ValueError, match="duplicate"):
        merge([first, second], scenes)
