import pytest

from neurobrain import NeuroBrainModel, NeuroBrainSession


def _model() -> NeuroBrainModel:
    return NeuroBrainModel(encoder_size=4, hidden_size=8)


def test_session_accepts_one_stimulus_and_reads_without_advancing() -> None:
    session = NeuroBrainSession(_model())
    assert session.read() is None
    assert session.read("emotion") is None
    assert session.read("behavior") is None

    first = session.update({"joy": 0.8, "anger": 0.1, "trust": 0.2, "fear": 0.0,
                            "surprise": 0.3, "sadness": 0.1, "disgust": 0.0,
                            "anticipation": 0.5})
    assert first.step == 1
    assert len(first.emotion) == 8
    assert len(first.behavior) == 6
    assert session.read("joy") == first.emotion["joy"]
    assert session.read("avoidance") == first.behavior["avoidance"]
    assert session.read("emotion") == first.emotion
    assert session.read("behavior") == first.behavior
    with pytest.raises(TypeError):
        session.read("emotion")["joy"] = 0.0

    second_read = session.read()
    assert second_read is not None
    assert second_read.as_dict() == first.as_dict()
    assert session.read().step == 1
    for name, value in first.emotion.items():
        assert session.read(name) == value
    for name, value in first.behavior.items():
        assert session.read(name) == value

    second = session.update([0.0] * 8)
    assert second.step == 2
    assert session.read().step == 2


def test_session_normalizes_mapping_order_and_validates_single_input() -> None:
    dimensions = ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")
    by_name = dict(zip(reversed(dimensions), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], strict=True))
    session = NeuroBrainSession(_model())

    result = session.update(by_name)
    assert result.step == 1

    with pytest.raises(ValueError, match="exactly 8"):
        session.update([0.0] * 7)
    with pytest.raises(ValueError, match="missing"):
        session.update({"joy": 0.5})
    with pytest.raises(ValueError, match="between 0 and 1"):
        session.update([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1])
    with pytest.raises(KeyError, match="unknown state dimension"):
        session.read("memory")


def test_session_reset_clears_state_and_readout() -> None:
    session = NeuroBrainSession(_model())
    session.update([0.1] * 8)
    session.reset()

    assert session.read() is None
    assert session.update([0.0] * 8).step == 1


def test_read_rejects_unknown_dimension_before_and_after_first_update() -> None:
    session = NeuroBrainSession(_model())
    with pytest.raises(KeyError, match="unknown state dimension"):
        session.read("memory")

    session.update([0.0] * 8)
    with pytest.raises(KeyError, match="unknown state dimension"):
        session.read("memory")


def test_sessions_keep_separate_state() -> None:
    first = NeuroBrainSession(_model())
    second = NeuroBrainSession(_model())
    first.update([0.0] * 8)

    assert first.read().step == 1
    assert second.read() is None


def test_from_checkpoint_defaults_to_cpu_and_requires_zero_history(tmp_path) -> None:
    import torch

    model = _model()
    path = tmp_path / "model.pt"
    model.save_checkpoint(path)
    session = NeuroBrainSession.from_checkpoint(path)
    assert next(session._model.parameters()).device.type == "cpu"

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["history_mode"] = "provided"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="history_mode must be 'zeros'"):
        NeuroBrainSession.from_checkpoint(path)


def test_session_requires_finite_non_negative_step_interval() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        NeuroBrainSession(_model(), delta_seconds=float("nan"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        NeuroBrainSession(_model(), delta_seconds=-1.0)
