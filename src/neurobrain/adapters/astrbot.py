"""Convert AstrBot Persona tendency values to NeuroBrain stimuli.

This module deliberately has no AstrBot runtime dependency.  AstrBot owns the
source contract (eight named integer intensities from 0 through 10); NeuroBrain
only receives the normalized, lowercase stimulus mapping produced here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final


ASTRBOT_TENDENCY_DIMENSIONS: Final[tuple[tuple[str, str], ...]] = (
    ("Joy", "joy"),
    ("Trust", "trust"),
    ("Fear", "fear"),
    ("Surprise", "surprise"),
    ("Sadness", "sadness"),
    ("Disgust", "disgust"),
    ("Anger", "anger"),
    ("Anticipation", "anticipation"),
)

_ASTRBOT_KEYS: Final[frozenset[str]] = frozenset(
    source_name for source_name, _ in ASTRBOT_TENDENCY_DIMENSIONS
)


def convert_astrbot_tendency_to_stimulus(
    tendency: Mapping[str, object],
) -> dict[str, float]:
    """Normalize an AstrBot 0..10 tendency mapping to NeuroBrain's 0..1 input.

    The source mapping must contain exactly the eight case-sensitive AstrBot
    Persona keys. Values are integer intensities; booleans, floats, missing
    keys, and extra keys are rejected. Integer values below 0 or above 10 are
    clamped to the nearest boundary before normalization.
    """

    if not isinstance(tendency, Mapping):
        raise TypeError("AstrBot tendency must be a mapping")

    received_keys = set(tendency)
    missing = _ASTRBOT_KEYS - received_keys
    extra = received_keys - _ASTRBOT_KEYS
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {sorted(missing)}")
        if extra:
            details.append(f"unknown: {sorted(extra, key=str)}")
        raise ValueError(
            "AstrBot tendency dimensions do not match the eight-dimensional protocol ("
            + "; ".join(details)
            + ")"
        )

    stimulus: dict[str, float] = {}
    for source_name, target_name in ASTRBOT_TENDENCY_DIMENSIONS:
        raw_value = tendency[source_name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise TypeError(f"AstrBot tendency.{source_name} must be an integer")
        stimulus[target_name] = max(0, min(10, raw_value)) / 10.0

    return stimulus


__all__ = [
    "ASTRBOT_TENDENCY_DIMENSIONS",
    "convert_astrbot_tendency_to_stimulus",
]
