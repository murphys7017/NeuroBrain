from __future__ import annotations

from typing import Final

IO_PROTOCOL_VERSION: Final[str] = "io-v0.2"

CURRENT_STIMULUS_DIMENSIONS: Final[tuple[str, ...]] = (
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
)

HISTORICAL_EMOTION_DIMENSIONS: Final[tuple[str, ...]] = CURRENT_STIMULUS_DIMENSIONS
PSYCHOLOGICAL_STATE_DIMENSIONS: Final[tuple[str, ...]] = CURRENT_STIMULUS_DIMENSIONS
# Compatibility alias for callers that used the original input name.
INPUT_DIMENSIONS: Final[tuple[str, ...]] = CURRENT_STIMULUS_DIMENSIONS

BEHAVIOR_TENDENCY_DIMENSIONS: Final[tuple[str, ...]] = (
    "engagement",
    "avoidance",
    "expression",
    "exploration",
    "initiative",
    "activation",
)
