"""Losses and training helpers for supervised NeuroBrain sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer

from .data import SequenceBatch
from .model import NeuroBrainModel


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    emotion: Tensor
    behavior: Tensor


def _masked_smooth_l1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    per_step = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    weights = mask.to(dtype=per_step.dtype)
    return (per_step * weights).sum() / weights.sum().clamp_min(1.0)


def compute_loss(
    model: NeuroBrainModel,
    batch: SequenceBatch,
    *,
    emotion_weight: float = 1.0,
    behavior_weight: float = 1.0,
) -> LossBreakdown:
    """Run a batch and calculate masked robust regression losses."""

    output = model(
        batch.current_stimulus,
        batch.historical_condition,
        batch.delta_seconds,
    )
    emotion = _masked_smooth_l1(output.emotion, batch.emotion_target, batch.mask)
    behavior = _masked_smooth_l1(output.behavior, batch.behavior_target, batch.mask)
    total = emotion_weight * emotion + behavior_weight * behavior
    return LossBreakdown(total=total, emotion=emotion, behavior=behavior)


def train_one_epoch(
    model: NeuroBrainModel,
    batches: Iterable[SequenceBatch],
    optimizer: Optimizer,
    *,
    device: torch.device,
    max_grad_norm: float | None = 1.0,
    emotion_weight: float = 1.0,
    behavior_weight: float = 1.0,
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "emotion": 0.0, "behavior": 0.0}
    count = 0
    for batch in batches:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = compute_loss(
            model,
            batch,
            emotion_weight=emotion_weight,
            behavior_weight=behavior_weight,
        )
        losses.total.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        for name, value in (
            ("total", losses.total),
            ("emotion", losses.emotion),
            ("behavior", losses.behavior),
        ):
            totals[name] += float(value.detach().cpu())
        count += 1
    if count == 0:
        raise ValueError("no batches were provided")
    return {name: value / count for name, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: NeuroBrainModel,
    batches: Iterable[SequenceBatch],
    *,
    device: torch.device,
    emotion_weight: float = 1.0,
    behavior_weight: float = 1.0,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "emotion": 0.0, "behavior": 0.0}
    count = 0
    for batch in batches:
        losses = compute_loss(
            model,
            batch.to(device),
            emotion_weight=emotion_weight,
            behavior_weight=behavior_weight,
        )
        for name, value in (
            ("total", losses.total),
            ("emotion", losses.emotion),
            ("behavior", losses.behavior),
        ):
            totals[name] += float(value.detach().cpu())
        count += 1
    if count == 0:
        raise ValueError("no batches were provided")
    return {name: value / count for name, value in totals.items()}


__all__ = ["LossBreakdown", "compute_loss", "evaluate", "train_one_epoch"]
