import json

import pytest
import torch

from neurobrain import collate_sequences, load_annotation_sequences


def _record(sequence_id: str, length: int = 2) -> dict:
    dimensions = ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")
    behavior = ("engagement", "avoidance", "expression", "exploration", "initiative", "activation")
    event = {
        "text": "event",
        "current_stimulus": {key: 0.1 for key in dimensions},
        "emotion_target": {key: 0.2 for key in dimensions},
        "behavior_target": {key: 0.5 for key in behavior},
        "delta_seconds": 1.0,
    }
    return {
        "sequence_id": sequence_id,
        "target_character": "test",
        "historical_emotion": {key: 0.0 for key in dimensions},
        "events": [event.copy() for _ in range(length)],
    }


def test_load_and_collate_variable_length_sequences(tmp_path) -> None:
    path = tmp_path / "annotations.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in (_record("a", 2), _record("b", 1))) + "\n",
        encoding="utf-8",
    )
    sequences = load_annotation_sequences(path)
    batch = collate_sequences(sequences)
    assert batch.current_stimulus.shape == (2, 2, 8)
    assert batch.behavior_target.shape == (2, 2, 6)
    assert batch.mask.tolist() == [[True, True], [True, False]]
    assert torch.all(batch.historical_condition == 0)


def test_loader_rejects_protocol_dimension_error(tmp_path) -> None:
    record = _record("bad")
    del record["events"][0]["current_stimulus"]["joy"]
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dimensions do not match"):
        load_annotation_sequences(path)


def test_loader_accepts_event_level_history(tmp_path) -> None:
    record = _record("event-history", 2)
    history = {key: 0.3 for key in ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")}
    del record["historical_emotion"]
    for event in record["events"]:
        event["historical_emotion"] = history
    path = tmp_path / "event-history.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    sequence = load_annotation_sequences(path)[0]
    assert torch.all(sequence.historical_condition == 0.3)
