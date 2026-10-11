from __future__ import annotations

from dataclasses import dataclass

from .api import NeuroBrainSession, StateSnapshot
from .adapters.astrbot import (
    ASTRBOT_TENDENCY_DIMENSIONS,
    convert_astrbot_tendency_to_stimulus,
)
from .data import (
    AnnotationDataset,
    AnnotationSequence,
    SequenceBatch,
    collate_sequences,
    load_annotation_sequences,
)
from .model import ModelOutput, NeuroBrainModel
from .protocol import (
    BEHAVIOR_TENDENCY_DIMENSIONS,
    CURRENT_STIMULUS_DIMENSIONS,
    HISTORICAL_EMOTION_DIMENSIONS,
    INPUT_DIMENSIONS,
    IO_PROTOCOL_VERSION,
    PSYCHOLOGICAL_STATE_DIMENSIONS,
)


@dataclass(frozen=True, slots=True)
class InputFrame:
    """第一阶段输入协议：当前刺激与历史情绪条件保持语义独立。"""

    current_stimulus: tuple[float, ...]
    historical_emotion: tuple[float, ...]
    delta_seconds: float = 1.0

    def __post_init__(self) -> None:
        if len(self.current_stimulus) != len(CURRENT_STIMULUS_DIMENSIONS):
            raise ValueError("current stimulus must contain 8 values")
        if len(self.historical_emotion) != len(HISTORICAL_EMOTION_DIMENSIONS):
            raise ValueError("historical emotion must contain 8 values")
        if self.delta_seconds < 0.0:
            raise ValueError("delta_seconds must be non-negative")
        values = (*self.current_stimulus, *self.historical_emotion)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("input values must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class NeuroState:
    """第一阶段的状态协议占位类型。算法实现将在后续研究阶段加入。"""

    psychological: tuple[float, ...]
    behavioral: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.psychological) != len(PSYCHOLOGICAL_STATE_DIMENSIONS):
            raise ValueError("psychological state must contain 8 values")
        if len(self.behavioral) != len(BEHAVIOR_TENDENCY_DIMENSIONS):
            raise ValueError("behavioral state must contain 6 values")
        values = (*self.psychological, *self.behavioral)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("state values must be between 0 and 1")


def main() -> None:
    """Run a small smoke check for the initialized package."""

    print("NeuroBrain CfC-512 package is ready.")


__all__ = [
    "BEHAVIOR_TENDENCY_DIMENSIONS",
    "AnnotationDataset",
    "AnnotationSequence",
    "ASTRBOT_TENDENCY_DIMENSIONS",
    "CURRENT_STIMULUS_DIMENSIONS",
    "HISTORICAL_EMOTION_DIMENSIONS",
    "INPUT_DIMENSIONS",
    "IO_PROTOCOL_VERSION",
    "PSYCHOLOGICAL_STATE_DIMENSIONS",
    "InputFrame",
    "ModelOutput",
    "NeuroBrainModel",
    "NeuroBrainSession",
    "NeuroState",
    "StateSnapshot",
    "SequenceBatch",
    "collate_sequences",
    "convert_astrbot_tendency_to_stimulus",
    "load_annotation_sequences",
]
