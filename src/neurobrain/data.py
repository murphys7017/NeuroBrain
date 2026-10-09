"""Load and batch NeuroBrain JSONL annotation sequences."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .protocol import (
    BEHAVIOR_TENDENCY_DIMENSIONS,
    CURRENT_STIMULUS_DIMENSIONS,
    HISTORICAL_EMOTION_DIMENSIONS,
)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _read_vector(value: Any, dimensions: tuple[str, ...], label: str) -> list[float]:
    mapping = _require_mapping(value, label)
    expected = set(dimensions)
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        suffix = f" ({', '.join(details)})" if details else ""
        raise ValueError(f"{label} dimensions do not match protocol{suffix}")

    result: list[float] = []
    for dimension in dimensions:
        number = mapping[dimension]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"{label}.{dimension} must be a number")
        number = float(number)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{label}.{dimension} must be finite and between 0 and 1")
        result.append(number)
    return result


@dataclass(frozen=True, slots=True)
class AnnotationSequence:
    """One complete, ordered training sequence from an annotation record."""

    sequence_id: str
    target_character: str | None
    current_stimulus: Tensor
    historical_condition: Tensor
    emotion_target: Tensor
    behavior_target: Tensor
    delta_seconds: Tensor
    texts: tuple[str, ...]

    @property
    def length(self) -> int:
        return int(self.current_stimulus.shape[0])


def _parse_record(record: Any, *, source: Path, line_number: int) -> AnnotationSequence:
    prefix = f"{source}:{line_number}"
    record = _require_mapping(record, prefix)
    sequence_id = record.get("sequence_id")
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        raise ValueError(f"{prefix}.sequence_id must be a non-empty string")

    target_character = record.get("target_character")
    if target_character is not None and not isinstance(target_character, str):
        raise ValueError(f"{prefix}.target_character must be a string when provided")

    record_history = record.get("historical_emotion")
    if record_history is not None:
        history = _read_vector(
            record_history, HISTORICAL_EMOTION_DIMENSIONS,
            f"{prefix}.historical_emotion",
        )
    else:
        history = None
    events = record.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError(f"{prefix}.events must be a non-empty array")

    current_values: list[list[float]] = []
    history_values: list[list[float]] = []
    emotion_values: list[list[float]] = []
    behavior_values: list[list[float]] = []
    delta_values: list[float] = []
    texts: list[str] = []

    for event_index, event in enumerate(events):
        event_prefix = f"{prefix}.events[{event_index}]"
        event = _require_mapping(event, event_prefix)
        text = event.get("text", "")
        if not isinstance(text, str):
            raise ValueError(f"{event_prefix}.text must be a string")
        delta = event.get("delta_seconds")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise ValueError(f"{event_prefix}.delta_seconds must be a number")
        delta = float(delta)
        if not math.isfinite(delta) or delta < 0.0:
            raise ValueError(f"{event_prefix}.delta_seconds must be finite and non-negative")

        event_history = event.get("historical_emotion")
        if event_history is not None:
            event_history_values = _read_vector(
                event_history, HISTORICAL_EMOTION_DIMENSIONS,
                f"{event_prefix}.historical_emotion",
            )
        elif history is not None:
            event_history_values = history
        else:
            raise ValueError(
                f"{prefix} must provide historical_emotion at the record or event level"
            )

        current_values.append(_read_vector(
            event.get("current_stimulus"), CURRENT_STIMULUS_DIMENSIONS,
            f"{event_prefix}.current_stimulus",
        ))
        history_values.append(event_history_values.copy())
        emotion_values.append(_read_vector(
            event.get("emotion_target"), CURRENT_STIMULUS_DIMENSIONS,
            f"{event_prefix}.emotion_target",
        ))
        behavior_values.append(_read_vector(
            event.get("behavior_target"), BEHAVIOR_TENDENCY_DIMENSIONS,
            f"{event_prefix}.behavior_target",
        ))
        delta_values.append(delta)
        texts.append(text)

    return AnnotationSequence(
        sequence_id=sequence_id,
        target_character=target_character,
        current_stimulus=torch.tensor(current_values, dtype=torch.float32),
        historical_condition=torch.tensor(history_values, dtype=torch.float32),
        emotion_target=torch.tensor(emotion_values, dtype=torch.float32),
        behavior_target=torch.tensor(behavior_values, dtype=torch.float32),
        delta_seconds=torch.tensor(delta_values, dtype=torch.float32),
        texts=tuple(texts),
    )


def load_annotation_sequences(path: str | Path) -> list[AnnotationSequence]:
    """Load all non-empty JSONL records from ``path`` with protocol validation."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    sequences: list[AnnotationSequence] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number} is not valid JSON: {error.msg}") from error
            sequence = _parse_record(record, source=source, line_number=line_number)
            if sequence.sequence_id in seen_ids:
                raise ValueError(f"duplicate sequence_id: {sequence.sequence_id}")
            seen_ids.add(sequence.sequence_id)
            sequences.append(sequence)
    if not sequences:
        raise ValueError(f"{source} contains no annotation records")
    return sequences


class AnnotationDataset(Dataset[AnnotationSequence]):
    """A PyTorch dataset containing complete sequences without truncation."""

    def __init__(self, sequences: Iterable[AnnotationSequence]) -> None:
        self.sequences = tuple(sequences)
        if not self.sequences:
            raise ValueError("AnnotationDataset requires at least one sequence")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> AnnotationSequence:
        return self.sequences[index]


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    """Padded batch returned by :func:`collate_sequences`."""

    current_stimulus: Tensor
    historical_condition: Tensor
    delta_seconds: Tensor
    emotion_target: Tensor
    behavior_target: Tensor
    mask: Tensor
    sequence_ids: tuple[str, ...]

    def to(self, device: torch.device | str) -> "SequenceBatch":
        return SequenceBatch(
            current_stimulus=self.current_stimulus.to(device),
            historical_condition=self.historical_condition.to(device),
            delta_seconds=self.delta_seconds.to(device),
            emotion_target=self.emotion_target.to(device),
            behavior_target=self.behavior_target.to(device),
            mask=self.mask.to(device),
            sequence_ids=self.sequence_ids,
        )


def collate_sequences(sequences: list[AnnotationSequence]) -> SequenceBatch:
    """Pad variable-length sequences and create a loss mask for valid steps."""

    if not sequences:
        raise ValueError("cannot collate an empty sequence list")
    batch_size = len(sequences)
    max_length = max(sequence.length for sequence in sequences)
    current = torch.zeros(batch_size, max_length, len(CURRENT_STIMULUS_DIMENSIONS))
    history = torch.zeros_like(current)
    delta = torch.zeros(batch_size, max_length)
    emotion = torch.zeros_like(current)
    behavior = torch.zeros(batch_size, max_length, len(BEHAVIOR_TENDENCY_DIMENSIONS))
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for index, sequence in enumerate(sequences):
        length = sequence.length
        current[index, :length] = sequence.current_stimulus
        history[index, :length] = sequence.historical_condition
        delta[index, :length] = sequence.delta_seconds
        emotion[index, :length] = sequence.emotion_target
        behavior[index, :length] = sequence.behavior_target
        mask[index, :length] = True

    return SequenceBatch(
        current_stimulus=current,
        historical_condition=history,
        delta_seconds=delta,
        emotion_target=emotion,
        behavior_target=behavior,
        mask=mask,
        sequence_ids=tuple(sequence.sequence_id for sequence in sequences),
    )


__all__ = [
    "AnnotationDataset",
    "AnnotationSequence",
    "SequenceBatch",
    "collate_sequences",
    "load_annotation_sequences",
]
