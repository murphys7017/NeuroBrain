"""Train the NeuroBrain CfC model on annotated JSONL sequences."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from neurobrain import (
    AnnotationDataset,
    NeuroBrainModel,
    collate_sequences,
    load_annotation_sequences,
)
from neurobrain.protocol import IO_PROTOCOL_VERSION
from neurobrain.training import evaluate, train_one_epoch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return result


def _split_sequences(sequences, val_ratio: float, seed: int):
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if val_ratio == 0.0 or len(sequences) < 2:
        return list(sequences), []
    indices = list(range(len(sequences)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(indices) * val_ratio))
    validation_count = min(validation_count, len(indices) - 1)
    validation_indices = set(indices[:validation_count])
    train = [sequence for index, sequence in enumerate(sequences) if index not in validation_indices]
    validation = [sequence for index, sequence in enumerate(sequences) if index in validation_indices]
    return train, validation


def _loader(sequences, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        AnnotationDataset(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_sequences,
    )


def _apply_history_mode(sequences, mode: str):
    if mode == "provided":
        return list(sequences)
    transformed = []
    for sequence in sequences:
        if mode == "zeros":
            history = torch.zeros_like(sequence.historical_condition)
        elif mode == "fixed":
            history = sequence.historical_condition[0].unsqueeze(0).repeat(sequence.length, 1)
        else:
            raise ValueError(f"unsupported history mode: {mode}")
        transformed.append(replace(sequence, historical_condition=history))
    return transformed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/neurobrain-cfc512.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--emotion-weight", type=float, default=1.0)
    parser.add_argument("--behavior-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--history-mode",
        choices=("zeros", "fixed", "provided"),
        default="zeros",
        help="history input policy; zeros avoids feeding previous target labels as inputs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("learning-rate must be positive and weight-decay must be non-negative")

    _set_seed(args.seed)
    device = _device(args.device)
    sequences = [sequence for path in args.data for sequence in load_annotation_sequences(path)]
    sequences = _apply_history_mode(sequences, args.history_mode)
    train_sequences, validation_sequences = _split_sequences(sequences, args.val_ratio, args.seed)
    if not validation_sequences:
        print("warning: no validation sequence; this is expected when --val-ratio is 0")

    train_loader = _loader(train_sequences, args.batch_size, shuffle=True, seed=args.seed)
    validation_loader = (
        _loader(validation_sequences, args.batch_size, shuffle=False, seed=args.seed)
        if validation_sequences else None
    )

    model = NeuroBrainModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float | int | None]] = []
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"sequences: train={len(train_sequences)} validation={len(validation_sequences)}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            max_grad_norm=args.max_grad_norm,
            emotion_weight=args.emotion_weight,
            behavior_weight=args.behavior_weight,
        )
        validation_metrics = (
            evaluate(
                model,
                validation_loader,
                device=device,
                emotion_weight=args.emotion_weight,
                behavior_weight=args.behavior_weight,
            )
            if validation_loader is not None else None
        )
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["total"],
            "train_emotion_loss": train_metrics["emotion"],
            "train_behavior_loss": train_metrics["behavior"],
            "validation_loss": validation_metrics["total"] if validation_metrics else None,
        }
        history.append(record)
        score = validation_metrics["total"] if validation_metrics else train_metrics["total"]
        if score < best_loss:
            best_loss = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": {
                        "encoder_size": model.encoder_size,
                        "hidden_size": model.hidden_size,
                        "cfc_mode": model.cfc_mode,
                    },
                    "protocol_version": IO_PROTOCOL_VERSION,
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "history": history,
                    "data_paths": [str(path) for path in args.data],
                    "history_mode": args.history_mode,
                    "seed": args.seed,
                },
                args.output,
            )
        print(
            f"epoch {epoch:04d} train={train_metrics['total']:.6f}"
            + (f" validation={validation_metrics['total']:.6f}" if validation_metrics else "")
        )

    print(f"checkpoint: {args.output}")
    print(f"best_loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
