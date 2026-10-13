"""Small stateful inference API for application integrations.

An integration sends one current-stimulus vector per event. Historical
conditions default to zero and each event advances one configured time step.
The latest decoded state can be read repeatedly without advancing the model.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from types import MappingProxyType
from typing import Any, Literal, overload

import torch

from .model import NeuroBrainModel
from .protocol import (
    BEHAVIOR_TENDENCY_DIMENSIONS,
    CURRENT_STIMULUS_DIMENSIONS,
    IO_PROTOCOL_VERSION,
)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Immutable, JSON-friendly decoded state from one model update."""

    step: int
    emotion: Mapping[str, float]
    behavior: Mapping[str, float]
    protocol_version: str = IO_PROTOCOL_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Return a detached dictionary suitable for JSON serialization."""

        return {
            "protocol_version": self.protocol_version,
            "step": self.step,
            "emotion": dict(self.emotion),
            "behavior": dict(self.behavior),
        }


class NeuroBrainSession:
    """Own one continuous CfC state for one character or conversation.

    The public update accepts only the current 8-dimensional stimulus. The
    v0.2 historical-condition channel is fixed to zeros, matching the default
    training/inference path, and ``delta_seconds`` is a session-level setting
    (one second by default). ``read`` is observational and never updates the
    hidden state. Create a separate session for each independently evolving
    character.
    """

    def __init__(
        self,
        model: NeuroBrainModel,
        *,
        delta_seconds: float = 1.0,
    ) -> None:
        if not isinstance(model, NeuroBrainModel):
            raise TypeError("model must be a NeuroBrainModel")
        if isinstance(delta_seconds, bool) or not isinstance(delta_seconds, (int, float)):
            raise TypeError("delta_seconds must be a number")
        if not math.isfinite(float(delta_seconds)) or delta_seconds < 0:
            raise ValueError("delta_seconds must be finite and non-negative")

        self._model = model
        self._model.eval()
        self._delta_seconds = float(delta_seconds)
        self._hidden_state: torch.Tensor | None = None
        self._latest: StateSnapshot | None = None
        self._step = 0
        self._lock = threading.RLock()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | PathLike[str],
        *,
        device: str | torch.device = "cpu",
        delta_seconds: float = 1.0,
    ) -> "NeuroBrainSession":
        """Load a trained model and start a fresh zero-hidden-state session."""

        model = NeuroBrainModel.load_checkpoint(
            path,
            map_location="cpu",
            expected_history_mode="zeros",
        ).to(torch.device(device))
        return cls(model, delta_seconds=delta_seconds)

    def update(self, stimulus: Mapping[str, float] | Sequence[float]) -> StateSnapshot:
        """Advance once with a named mapping or ordered 8-value sequence."""

        return self.update_many([stimulus])[-1]

    def update_many(
        self,
        stimuli: Sequence[Mapping[str, float] | Sequence[float]],
    ) -> tuple[StateSnapshot, ...]:
        """Advance a sequence atomically, rolling back if any model step fails."""

        if isinstance(stimuli, (str, bytes, bytearray)) or not isinstance(
            stimuli, Sequence
        ):
            raise TypeError("stimuli must be a non-empty sequence")
        if not stimuli:
            raise ValueError("stimuli must be a non-empty sequence")
        normalized = [self._normalize_stimulus(stimulus) for stimulus in stimuli]

        with self._lock, torch.inference_mode():
            previous_hidden_state = self._hidden_state
            previous_latest = self._latest
            previous_step = self._step
            snapshots: list[StateSnapshot] = []
            try:
                parameter = next(self._model.parameters())
                for values in normalized:
                    current = torch.tensor(
                        values,
                        dtype=parameter.dtype,
                        device=parameter.device,
                    )
                    history = torch.zeros_like(current)
                    output = self._model.step(
                        current,
                        history,
                        delta_seconds=self._delta_seconds,
                        hidden_state=self._hidden_state,
                    )
                    self._hidden_state = output.final_hidden.detach()
                    self._step += 1
                    self._latest = StateSnapshot(
                        step=self._step,
                        emotion=MappingProxyType(
                            {
                                name: float(value)
                                for name, value in zip(
                                    CURRENT_STIMULUS_DIMENSIONS,
                                    output.emotion.detach().cpu().tolist(),
                                    strict=True,
                                )
                            }
                        ),
                        behavior=MappingProxyType(
                            {
                                name: float(value)
                                for name, value in zip(
                                    BEHAVIOR_TENDENCY_DIMENSIONS,
                                    output.behavior.detach().cpu().tolist(),
                                    strict=True,
                                )
                            }
                        ),
                    )
                    snapshots.append(self._latest)
            except Exception:
                self._hidden_state = previous_hidden_state
                self._latest = previous_latest
                self._step = previous_step
                raise
            return tuple(snapshots)

    @overload
    def read(self) -> StateSnapshot | None: ...

    @overload
    def read(self, name: Literal["emotion", "behavior"]) -> Mapping[str, float] | None: ...

    @overload
    def read(self, name: str) -> float | None: ...

    def read(
        self,
        name: str | None = None,
    ) -> StateSnapshot | Mapping[str, float] | float | None:
        """Read the latest output, or one named dimension, without updating.

        Returns ``None`` before the first update. ``name`` can be an output
        group (``"emotion"`` or ``"behavior"``) or any named dimension, such
        as ``"joy"`` or ``"avoidance"``.
        """

        if name is not None and name not in (
            "emotion",
            "behavior",
            *CURRENT_STIMULUS_DIMENSIONS,
            *BEHAVIOR_TENDENCY_DIMENSIONS,
        ):
            raise KeyError(f"unknown state dimension: {name!r}")

        with self._lock:
            if self._latest is None:
                return None
            if name is None:
                # Return a fresh immutable snapshot, so no caller can mutate
                # the session's retained result through a reference.
                return StateSnapshot(
                    step=self._latest.step,
                    emotion=MappingProxyType(dict(self._latest.emotion)),
                    behavior=MappingProxyType(dict(self._latest.behavior)),
                )
            if name == "emotion":
                return self._latest.emotion
            if name == "behavior":
                return self._latest.behavior
            if name in self._latest.emotion:
                return self._latest.emotion[name]
            if name in self._latest.behavior:
                return self._latest.behavior[name]
            # The dimension was validated above; this indicates an internal
            # inconsistency if a snapshot stops matching the protocol.
            raise RuntimeError(f"state snapshot is missing dimension: {name!r}")

    def reset(self) -> None:
        """Start a new sequence by clearing hidden state and cached output."""

        with self._lock:
            self._hidden_state = None
            self._latest = None
            self._step = 0

    @staticmethod
    def _normalize_stimulus(
        stimulus: Mapping[str, float] | Sequence[float],
    ) -> tuple[float, ...]:
        if isinstance(stimulus, Mapping):
            expected = set(CURRENT_STIMULUS_DIMENSIONS)
            received = set(stimulus)
            missing = expected - received
            extra = received - expected
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing: {sorted(missing)}")
                if extra:
                    details.append(f"unknown: {sorted(extra)}")
                raise ValueError("stimulus dimensions do not match protocol (" + "; ".join(details) + ")")
            raw_values = [stimulus[name] for name in CURRENT_STIMULUS_DIMENSIONS]
        elif isinstance(stimulus, Sequence) and not isinstance(stimulus, (str, bytes, bytearray)):
            raw_values = list(stimulus)
            if len(raw_values) != len(CURRENT_STIMULUS_DIMENSIONS):
                raise ValueError("stimulus must contain exactly 8 values")
        else:
            raise TypeError("stimulus must be a dimension mapping or an 8-value sequence")

        values: list[float] = []
        for name, raw_value in zip(CURRENT_STIMULUS_DIMENSIONS, raw_values, strict=True):
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError(f"stimulus.{name} must be a number")
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"stimulus.{name} must be finite and between 0 and 1")
            values.append(value)
        return tuple(values)


__all__ = ["NeuroBrainSession", "StateSnapshot"]
