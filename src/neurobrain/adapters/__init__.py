"""Host-application adapters for NeuroBrain input protocols."""

from .astrbot import (
    ASTRBOT_TENDENCY_DIMENSIONS,
    convert_astrbot_tendency_to_stimulus,
)

__all__ = [
    "ASTRBOT_TENDENCY_DIMENSIONS",
    "convert_astrbot_tendency_to_stimulus",
]
