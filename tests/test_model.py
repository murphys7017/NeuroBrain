import torch
import pytest

from neurobrain import NeuroBrainModel


def _inputs(batch: int = 2, steps: int = 3) -> tuple[torch.Tensor, ...]:
    current = torch.rand(batch, steps, 8)
    history = torch.rand(batch, steps, 8)
    delta = torch.ones(batch, steps)
    return current, history, delta


def test_model_sequence_shapes_and_ranges() -> None:
    model = NeuroBrainModel()
    output = model(*_inputs())
    emotion, behavior, hidden = output.emotion, output.behavior, output.final_hidden
    assert emotion.shape == (2, 3, 8)
    assert behavior.shape == (2, 3, 6)
    assert hidden.shape == (2, 512)
    assert torch.all((emotion >= 0) & (emotion <= 1))
    assert torch.all((behavior >= 0) & (behavior <= 1))


def test_hidden_state_changes_step_response() -> None:
    model = NeuroBrainModel()
    current = torch.zeros(8)
    history = torch.zeros(8)
    first = model.step(current, history)
    second = model.step(current, history, hidden_state=first.final_hidden)
    assert second.final_hidden.shape == (512,)
    assert not torch.allclose(first.emotion, second.emotion)


def test_single_frame_forward_keeps_batch_dimension() -> None:
    model = NeuroBrainModel()
    output = model(torch.zeros(2, 8), torch.zeros(2, 8), torch.ones(2))
    assert output.emotion.shape == (2, 1, 8)
    assert output.behavior.shape == (2, 1, 6)
    assert output.final_hidden.shape == (2, 512)


def test_batched_step_shapes() -> None:
    model = NeuroBrainModel()
    output = model.step(torch.zeros(2, 8), torch.zeros(2, 8))
    assert output.emotion.shape == (2, 8)
    assert output.behavior.shape == (2, 6)
    assert output.final_hidden.shape == (2, 512)


def test_forward_supports_gradient_training() -> None:
    model = NeuroBrainModel()
    output = model(*_inputs())
    loss = output.emotion.mean() + output.behavior.mean()
    loss.backward()
    assert model.encoder[0].weight.grad is not None
    assert model.dynamics.rnn_cell.ff1.weight.grad is not None
    assert model.emotion_decoder.weight.grad is not None
    assert model.behavior_decoder.weight.grad is not None


def test_model_rejects_out_of_range_inputs() -> None:
    model = NeuroBrainModel()
    current = torch.zeros(1, 1, 8)
    current[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="between 0 and 1"):
        model(current, torch.zeros_like(current), torch.ones(1, 1))


def test_checkpoint_round_trip(tmp_path) -> None:
    model = NeuroBrainModel()
    current, history, delta = _inputs(batch=1, steps=2)
    expected = model(current, history, delta)
    path = tmp_path / "model.pt"
    model.save_checkpoint(path)
    restored = NeuroBrainModel.load_checkpoint(path)
    actual = restored(current, history, delta)
    assert torch.allclose(expected.emotion, actual.emotion)
    assert torch.allclose(expected.behavior, actual.behavior)


def test_checkpoint_rejects_incompatible_protocol(tmp_path) -> None:
    model = NeuroBrainModel()
    path = tmp_path / "model.pt"
    model.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["protocol_version"] = "io-v0.1"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="protocol version"):
        NeuroBrainModel.load_checkpoint(path)
