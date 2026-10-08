from __future__ import annotations

from dataclasses import dataclass
from typing import Final

INPUT_DIMENSIONS: Final[tuple[str, ...]] = (
    "joy",
    "sadness",
    "tension",
    "curiosity",
    "surprise",
)

PSYCHOLOGICAL_STATE_DIMENSIONS: Final[tuple[str, ...]] = INPUT_DIMENSIONS

BEHAVIOR_TENDENCY_DIMENSIONS: Final[tuple[str, ...]] = (
    "engagement",
    "avoidance",
    "expression",
    "exploration",
    "initiative",
    "activation",
)


@dataclass(frozen=True, slots=True)
class NeuroState:
    """第一阶段的状态协议占位类型。算法实现将在后续研究阶段加入。"""

    psychological: tuple[float, ...]
    behavioral: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.psychological) != len(PSYCHOLOGICAL_STATE_DIMENSIONS):
            raise ValueError("psychological state must contain 5 values")
        if len(self.behavioral) != len(BEHAVIOR_TENDENCY_DIMENSIONS):
            raise ValueError("behavioral state must contain 6 values")
        values = (*self.psychological, *self.behavioral)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("state values must be between 0 and 1")


def main() -> None:
    """Run a small smoke check for the initialized package."""

    print("NeuroBrain package initialized; core dynamics are not implemented yet.")
