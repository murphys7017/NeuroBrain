"""Train and evaluate CfC-512 on a deterministic synthetic state teacher.

The benchmark generates independent random episodes in memory, trains a fresh
model, selects an epoch on validation episodes, and reports held-out metrics.
It does not write a checkpoint or modify project data.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch.utils.data import DataLoader

from neurobrain import AnnotationDataset, AnnotationSequence, NeuroBrainModel, collate_sequences
from neurobrain.training import train_one_epoch


def teacher_behavior(emotion: torch.Tensor) -> torch.Tensor:
    """Map synthetic emotion states to six bounded behavior tendencies."""

    joy, trust, fear, surprise, sadness, disgust, anger, anticipation = emotion.unbind(-1)
    values = (
        0.25 + 0.25 * joy + 0.20 * trust + 0.15 * anticipation + 0.15 * surprise,
        0.05 + 0.55 * fear + 0.25 * sadness + 0.15 * disgust,
        0.10 + 0.35 * joy + 0.25 * anger + 0.15 * surprise + 0.15 * trust,
        0.05 + 0.35 * anticipation + 0.30 * surprise + 0.25 * trust,
        0.10 + 0.35 * anticipation + 0.35 * trust + 0.20 * joy,
        0.05 + 0.20 * joy + 0.20 * fear + 0.20 * anger + 0.15 * surprise + 0.20 * anticipation,
    )
    return torch.stack(values, dim=-1).clamp(0.0, 1.0)


def teacher_step(
    state: torch.Tensor,
    stimulus: torch.Tensor,
    delta_seconds: float,
) -> torch.Tensor:
    """Apply a continuous-time leaky state update."""

    alpha = 1.0 - math.exp(-0.55 * delta_seconds)
    return state + (stimulus - state) * alpha


def generate_sequences(
    count: int,
    *,
    seed: int,
    prefix: str,
    steps: int,
) -> list[AnnotationSequence]:
    """Generate independent random episodes with teacher-generated targets."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    sequences: list[AnnotationSequence] = []
    for episode in range(count):
        stimulus = torch.rand((steps, 8), generator=generator)
        keep = (torch.rand((steps, 8), generator=generator) > 0.30).float()
        stimulus *= keep
        delta = 0.3 + 1.7 * torch.rand(steps, generator=generator)

        state = torch.zeros(8)
        emotion_targets: list[torch.Tensor] = []
        for time_index in range(steps):
            state = teacher_step(state, stimulus[time_index], float(delta[time_index]))
            emotion_targets.append(state.clone())
        emotion = torch.stack(emotion_targets)
        behavior = teacher_behavior(emotion)
        sequences.append(
            AnnotationSequence(
                sequence_id=f"{prefix}-{episode}",
                target_character="synthetic-teacher",
                current_stimulus=stimulus,
                historical_condition=torch.zeros_like(stimulus),
                emotion_target=emotion,
                behavior_target=behavior,
                delta_seconds=delta,
                texts=tuple("synthetic random event" for _ in range(steps)),
            )
        )
    return sequences


def make_loader(
    sequences: list[AnnotationSequence],
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    return DataLoader(
        AnnotationDataset(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_sequences,
    )


def evaluate(
    model: NeuroBrainModel,
    sequences: list[AnnotationSequence],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor | float | None]:
    model.eval()
    emotion_predictions: list[torch.Tensor] = []
    emotion_targets: list[torch.Tensor] = []
    behavior_predictions: list[torch.Tensor] = []
    behavior_targets: list[torch.Tensor] = []
    emotion_directions: list[torch.Tensor] = []
    behavior_directions: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in make_loader(sequences, batch_size=batch_size):
            batch = batch.to(device)
            output = model(
                batch.current_stimulus,
                batch.historical_condition,
                batch.delta_seconds,
            )
            for index in range(batch.mask.shape[0]):
                length = int(batch.mask[index].sum())
                emotion_predictions.append(output.emotion[index, :length].cpu())
                emotion_targets.append(batch.emotion_target[index, :length].cpu())
                behavior_predictions.append(output.behavior[index, :length].cpu())
                behavior_targets.append(batch.behavior_target[index, :length].cpu())

                if length > 1:
                    target_delta = batch.emotion_target[index, 1:length] - batch.emotion_target[index, : length - 1]
                    prediction_delta = output.emotion[index, 1:length] - output.emotion[index, : length - 1]
                    changed = target_delta.abs() > 0.03
                    if changed.any():
                        emotion_directions.append((target_delta[changed] * prediction_delta[changed] > 0).float().cpu())

                    behavior_target_delta = batch.behavior_target[index, 1:length] - batch.behavior_target[index, : length - 1]
                    behavior_prediction_delta = output.behavior[index, 1:length] - output.behavior[index, : length - 1]
                    behavior_changed = behavior_target_delta.abs() > 0.03
                    if behavior_changed.any():
                        behavior_directions.append(
                            (behavior_target_delta[behavior_changed] * behavior_prediction_delta[behavior_changed] > 0)
                            .float()
                            .cpu()
                        )

    predicted_emotion = torch.cat(emotion_predictions)
    target_emotion = torch.cat(emotion_targets)
    predicted_behavior = torch.cat(behavior_predictions)
    target_behavior = torch.cat(behavior_targets)

    def direction_mean(values: list[torch.Tensor]) -> float | None:
        if not values:
            return None
        return torch.cat(values).mean().item()

    return {
        "emotion_prediction": predicted_emotion,
        "emotion_target": target_emotion,
        "behavior_prediction": predicted_behavior,
        "behavior_target": target_behavior,
        "emotion_mae": (predicted_emotion - target_emotion).abs().mean().item(),
        "emotion_mse": ((predicted_emotion - target_emotion) ** 2).mean().item(),
        "behavior_mae": (predicted_behavior - target_behavior).abs().mean().item(),
        "behavior_mse": ((predicted_behavior - target_behavior) ** 2).mean().item(),
        "emotion_direction": direction_mean(emotion_directions),
        "behavior_direction": direction_mean(behavior_directions),
    }


def infer_sequence(
    model: NeuroBrainModel,
    stimuli: list[list[float]],
    delta_seconds: list[float],
    *,
    device: torch.device,
) -> torch.Tensor:
    current = torch.tensor(stimuli, dtype=torch.float32, device=device).unsqueeze(0)
    history = torch.zeros_like(current)
    delta = torch.tensor(delta_seconds, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        return model(current, history, delta).emotion[0].cpu()


def teacher_sequence(stimuli: list[list[float]], delta_seconds: list[float]) -> torch.Tensor:
    state = torch.zeros(8)
    states: list[torch.Tensor] = []
    for stimulus, delta in zip(stimuli, delta_seconds, strict=True):
        state = teacher_step(state, torch.tensor(stimulus), delta)
        states.append(state.clone())
    return torch.stack(states)


def run_pulse_probe(model: NeuroBrainModel, *, device: torch.device) -> None:
    zero = [0.0] * 8
    for name, index in (("joy", 0), ("fear", 2)):
        pulse = [0.0] * 8
        pulse[index] = 1.0
        stimuli = [pulse] * 4 + [zero] * 8
        delta = [1.0] * len(stimuli)
        prediction = infer_sequence(model, stimuli, delta, device=device)[:, index]
        target = teacher_sequence(stimuli, delta)[:, index]
        washout_error = (prediction[4:] - target[4:]).abs().mean().item()
        print(f"{name} pulse teacher: " + ", ".join(f"{value:.3f}" for value in target.tolist()))
        print(f"{name} pulse model:   " + ", ".join(f"{value:.3f}" for value in prediction.tolist()))
        print(f"{name} washout MAE (8 no-stimulus steps): {washout_error:.5f}")

    joy = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fear = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    history_joy = infer_sequence(model, [joy] * 4 + [zero], [1.0] * 5, device=device)[-1]
    history_fear = infer_sequence(model, [fear] * 4 + [zero], [1.0] * 5, device=device)[-1]
    teacher_joy = teacher_sequence([joy] * 4 + [zero], [1.0] * 5)[-1]
    teacher_fear = teacher_sequence([fear] * 4 + [zero], [1.0] * 5)[-1]
    print(
        "same neutral input after joy/fear history: "
        f"teacher max delta={(teacher_joy - teacher_fear).abs().max().item():.3f}, "
        f"model max delta={(history_joy - history_fear).abs().max().item():.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=128)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--train-seed", type=int, default=31001)
    parser.add_argument("--validation-seed", type=int, default=31002)
    parser.add_argument("--test-seed", type=int, default=31003)
    parser.add_argument("--model-seed", type=int, default=20261010)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()
    if min(args.train_episodes, args.validation_episodes, args.test_episodes, args.steps, args.epochs, args.batch_size) < 1:
        parser.error("episode counts, steps, epochs, and batch-size must be positive")
    if len({args.train_seed, args.validation_seed, args.test_seed}) != 3:
        parser.error("train, validation, and test seeds must be distinct")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but torch.cuda.is_available() is false")

    train = generate_sequences(args.train_episodes, seed=args.train_seed, prefix="train", steps=args.steps)
    validation = generate_sequences(
        args.validation_episodes,
        seed=args.validation_seed,
        prefix="validation",
        steps=args.steps,
    )
    test = generate_sequences(args.test_episodes, seed=args.test_seed, prefix="test", steps=args.steps)
    print(
        f"device={device}; episodes train/validation/test="
        f"{len(train)}/{len(validation)}/{len(test)}; steps={args.steps}; "
        f"seeds={args.train_seed}/{args.validation_seed}/{args.test_seed}"
    )

    train_emotion = torch.cat([sequence.emotion_target for sequence in train])
    train_behavior = torch.cat([sequence.behavior_target for sequence in train])
    test_emotion = torch.cat([sequence.emotion_target for sequence in test])
    test_behavior = torch.cat([sequence.behavior_target for sequence in test])
    mean_emotion = train_emotion.mean(0)
    mean_behavior = train_behavior.mean(0)
    baseline = {
        "emotion_mae": (test_emotion - mean_emotion).abs().mean().item(),
        "emotion_mse": ((test_emotion - mean_emotion) ** 2).mean().item(),
        "behavior_mae": (test_behavior - mean_behavior).abs().mean().item(),
        "behavior_mse": ((test_behavior - mean_behavior) ** 2).mean().item(),
    }

    torch.manual_seed(args.model_seed)
    model = NeuroBrainModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best_validation_mse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model,
            make_loader(train, batch_size=args.batch_size, shuffle=True, seed=args.model_seed + epoch),
            optimizer,
            device=device,
            max_grad_norm=1.0,
        )
        validation_metrics = evaluate(model, validation, device=device, batch_size=args.batch_size)
        score = float(validation_metrics["emotion_mse"]) + float(validation_metrics["behavior_mse"])
        if score < best_validation_mse:
            best_validation_mse = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch} validation emotion_mse={float(validation_metrics['emotion_mse']):.5f} "
                f"behavior_mse={float(validation_metrics['behavior_mse']):.5f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test, device=device, batch_size=args.batch_size)
    print(f"best_epoch={best_epoch}; training_seconds={time.perf_counter() - start:.1f}")
    for name in ("emotion_mae", "emotion_mse", "behavior_mae", "behavior_mse"):
        print(f"{name}: model={float(test_metrics[name]):.5f}, train_mean_baseline={baseline[name]:.5f}")
    print(
        "MSE skill versus train-mean baseline: "
        f"emotion={1.0 - float(test_metrics['emotion_mse']) / baseline['emotion_mse']:.3f}, "
        f"behavior={1.0 - float(test_metrics['behavior_mse']) / baseline['behavior_mse']:.3f}"
    )
    def format_optional_metric(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    print(
        "direction agreement for adjacent target changes > 0.03: "
        f"emotion={format_optional_metric(test_metrics['emotion_direction'])}, "
        f"behavior={format_optional_metric(test_metrics['behavior_direction'])}"
    )
    run_pulse_probe(model, device=device)


if __name__ == "__main__":
    main()
