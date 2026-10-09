"""Check the local PyTorch, CUDA, and ncps installation."""

from __future__ import annotations

import torch
from ncps.torch import CfC


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {device.type == 'cuda'}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        x = torch.ones((2, 2), device="cuda")
        print(f"CUDA matmul: {(x @ x).tolist()}")
    cfc = CfC(input_size=32, units=512, batch_first=True).to(device)
    y, h = cfc(torch.zeros((2, 1, 32), device=device))
    print(f"CfC output: {tuple(y.shape)}")
    print(f"CfC hidden: {tuple(h.shape)}")


if __name__ == "__main__":
    main()
