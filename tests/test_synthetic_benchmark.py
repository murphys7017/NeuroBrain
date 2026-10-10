import torch

from neurobrain import NeuroBrainModel
from scripts.synthetic_benchmark import evaluate, generate_sequences


def test_evaluate_handles_single_step_sequences_without_direction_samples() -> None:
    model = NeuroBrainModel(encoder_size=4, hidden_size=8)
    metrics = evaluate(
        model,
        generate_sequences(1, seed=123, prefix="test", steps=1),
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert metrics["emotion_direction"] is None
    assert metrics["behavior_direction"] is None
