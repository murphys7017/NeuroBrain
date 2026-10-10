from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import torch
from ncps.torch import CfC
from torch import Tensor, nn

from .protocol import (
    BEHAVIOR_TENDENCY_DIMENSIONS,
    CURRENT_STIMULUS_DIMENSIONS,
    HISTORICAL_EMOTION_DIMENSIONS,
    IO_PROTOCOL_VERSION,
)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Outputs produced by the shared CfC state representation."""

    emotion: Tensor
    behavior: Tensor
    final_hidden: Tensor


class NeuroBrainModel(nn.Module):
    """The first NeuroBrain CfC-512 architecture.

    Inputs use ``[batch, time, features]`` for sequence calls.  ``step``
    accepts one frame per batch and returns the corresponding two readouts.
    The current stimulus and historical condition stay as separate protocol
    fields until they are concatenated by the input encoder.
    """

    input_size = len(CURRENT_STIMULUS_DIMENSIONS) + len(HISTORICAL_EMOTION_DIMENSIONS)
    emotion_size = len(CURRENT_STIMULUS_DIMENSIONS)
    behavior_size = len(BEHAVIOR_TENDENCY_DIMENSIONS)

    def __init__(
        self,
        *,
        encoder_size: int = 32,
        hidden_size: int = 512,
        cfc_mode: str = "default",
    ) -> None:
        super().__init__()
        self.encoder_size = encoder_size
        self.hidden_size = hidden_size
        self.cfc_mode = cfc_mode

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, encoder_size),
            nn.Tanh(),
        )
        self.dynamics = CfC(
            input_size=encoder_size,
            units=hidden_size,
            mode=cfc_mode,
            return_sequences=True,
            batch_first=True,
        )
        self.emotion_decoder = nn.Linear(hidden_size, self.emotion_size)
        self.behavior_decoder = nn.Linear(hidden_size, self.behavior_size)

    def reset_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Create a zero CfC hidden state for a new sequence."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.hidden_size,
            device=device if device is not None else parameter.device,
            dtype=dtype if dtype is not None else parameter.dtype,
        )

    def forward(
        self,
        current_stimulus: Tensor,
        historical_condition: Tensor,
        delta_seconds: Tensor,
        hidden_state: Tensor | None = None,
    ) -> ModelOutput:
        """Process a complete sequence and return the final hidden state."""

        current, history, delta, was_unbatched = self._normalise_inputs(
            current_stimulus, historical_condition, delta_seconds
        )
        encoded = self.encoder(torch.cat((current, history), dim=-1))
        if hidden_state is not None:
            if hidden_state.dim() == 1:
                hidden_state = hidden_state.unsqueeze(0)
            expected_shape = (current.shape[0], self.hidden_size)
            if hidden_state.shape != expected_shape:
                raise ValueError(f"hidden_state must have shape {expected_shape}")
        hidden_sequence, final_hidden = self.dynamics(
            encoded,
            hx=hidden_state,
            # ncps 1.0.1 broadcasts the time tensor elementwise against the
            # hidden vector; expand the scalar per sample to [B, T, H].
            timespans=delta.unsqueeze(-1).expand(-1, -1, self.hidden_size),
        )
        output = ModelOutput(
            emotion=torch.sigmoid(self.emotion_decoder(hidden_sequence)),
            behavior=torch.sigmoid(self.behavior_decoder(hidden_sequence)),
            final_hidden=final_hidden,
        )
        if was_unbatched:
            return ModelOutput(
                emotion=output.emotion.squeeze(0),
                behavior=output.behavior.squeeze(0),
                final_hidden=output.final_hidden.squeeze(0),
            )
        return output

    def step(
        self,
        current_stimulus: Tensor,
        historical_condition: Tensor,
        delta_seconds: Tensor | float = 1.0,
        hidden_state: Tensor | None = None,
    ) -> ModelOutput:
        """Process one time step and return a new hidden state.

        A one-frame call keeps the batch dimension in the returned readouts:
        ``[batch, dimensions]``.  A one-frame unbatched call returns
        ``[dimensions]``.
        """

        if current_stimulus.dim() not in (1, 2):
            raise ValueError("step inputs must have shape [8] or [B, 8]")
        if historical_condition.shape != current_stimulus.shape:
            raise ValueError("historical_condition must have the same shape as current_stimulus")
        current_was_unbatched = current_stimulus.dim() == 1
        current = (
            current_stimulus.unsqueeze(0).unsqueeze(1)
            if current_was_unbatched
            else current_stimulus.unsqueeze(1)
        )
        history = (
            historical_condition.unsqueeze(0).unsqueeze(1)
            if historical_condition.dim() == 1
            else historical_condition.unsqueeze(1)
        )
        delta = torch.as_tensor(delta_seconds, device=current.device, dtype=current.dtype)
        if delta.dim() == 0:
            delta = delta.expand(current.shape[0])
        elif delta.dim() == 2 and delta.shape[-1] == 1:
            delta = delta.squeeze(-1)
        result = self.forward(current, history, delta, hidden_state)
        if current_was_unbatched:
            return ModelOutput(
                emotion=result.emotion[0, 0],
                behavior=result.behavior[0, 0],
                final_hidden=result.final_hidden[0],
            )
        return ModelOutput(
            emotion=result.emotion[:, 0],
            behavior=result.behavior[:, 0],
            final_hidden=result.final_hidden,
        )

    def save_checkpoint(
        self,
        path: str | PathLike[str],
    ) -> None:
        """Save model weights together with architecture metadata."""

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_config": {
                    "encoder_size": self.encoder_size,
                    "hidden_size": self.hidden_size,
                    "cfc_mode": self.cfc_mode,
                },
                "protocol_version": IO_PROTOCOL_VERSION,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | PathLike[str],
        *,
        map_location: str | torch.device | None = None,
        protocol_version: str = IO_PROTOCOL_VERSION,
        expected_history_mode: str | None = None,
    ) -> "NeuroBrainModel":
        """Load a checkpoint created by :meth:`save_checkpoint`."""

        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if checkpoint.get("protocol_version") != protocol_version:
            raise ValueError(f"checkpoint protocol version must be {protocol_version!r}")
        checkpoint_history_mode = checkpoint.get("history_mode")
        if (
            expected_history_mode is not None
            and checkpoint_history_mode is not None
            and checkpoint_history_mode != expected_history_mode
        ):
            raise ValueError(
                "checkpoint history_mode must be "
                f"{expected_history_mode!r}, got {checkpoint_history_mode!r}"
            )
        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _normalise_inputs(
        self,
        current_stimulus: Tensor,
        historical_condition: Tensor,
        delta_seconds: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, bool]:
        if current_stimulus.dim() not in (1, 2, 3):
            raise ValueError("current_stimulus must have shape [8], [B, 8], or [B, T, 8]")
        if historical_condition.shape != current_stimulus.shape:
            raise ValueError("historical_condition must have the same shape as current_stimulus")
        if current_stimulus.shape[-1] != self.input_size // 2:
            raise ValueError("emotion inputs must contain 8 values")

        was_unbatched = current_stimulus.dim() == 1
        if was_unbatched:
            current = current_stimulus.unsqueeze(0).unsqueeze(1)
            history = historical_condition.unsqueeze(0).unsqueeze(1)
        elif current_stimulus.dim() == 2:
            # [B, 8] is a batch containing one frame per sample.
            current = current_stimulus.unsqueeze(1)
            history = historical_condition.unsqueeze(1)
        else:
            current = current_stimulus
            history = historical_condition

        for name, values in (
            ("current_stimulus", current),
            ("historical_condition", history),
        ):
            if not values.is_floating_point():
                raise ValueError(f"{name} must use a floating-point dtype")
            if not torch.isfinite(values).all():
                raise ValueError(f"{name} must contain only finite values")
            if torch.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"{name} values must be between 0 and 1")

        delta = torch.as_tensor(delta_seconds, device=current.device, dtype=current.dtype)
        if delta.dim() == 0:
            delta = delta.expand(current.shape[0], current.shape[1])
        elif delta.dim() == 1:
            if delta.shape[0] == current.shape[1]:
                delta = delta.unsqueeze(0).expand(current.shape[0], -1)
            elif delta.shape[0] == current.shape[0] and current.shape[1] == 1:
                delta = delta.unsqueeze(1)
            else:
                raise ValueError("delta_seconds has incompatible shape")
        elif delta.dim() == 3 and delta.shape[-1] == 1:
            delta = delta.squeeze(-1)
        if delta.shape != current.shape[:2]:
            raise ValueError("delta_seconds must have shape [B, T] or [B, T, 1]")
        if not torch.isfinite(delta).all():
            raise ValueError("delta_seconds must contain only finite values")
        if torch.any(delta < 0):
            raise ValueError("delta_seconds must be non-negative")
        return current, history, delta, was_unbatched
