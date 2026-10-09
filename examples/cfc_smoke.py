"""Run one short CfC-512 sequence and print its protocol shapes."""

from __future__ import annotations

import torch

from neurobrain import NeuroBrainModel


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroBrainModel().to(device)
    current = torch.zeros((1, 4, 8), device=device)
    current[:, 0, 0] = 1.0  # one joy stimulus at the first time step
    history = torch.zeros_like(current)
    delta_seconds = torch.ones((1, 4), device=device)
    output = model(current, history, delta_seconds)
    print(f"device: {device}")
    print(f"emotion: {tuple(output.emotion.shape)}")
    print(f"behavior: {tuple(output.behavior.shape)}")
    print(f"final hidden: {tuple(output.final_hidden.shape)}")


if __name__ == "__main__":
    main()
