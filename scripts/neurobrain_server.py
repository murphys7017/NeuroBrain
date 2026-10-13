#!/usr/bin/env python3
"""Run the isolated NeuroBrain WebSocket state service."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from neurobrain.ws_server import NeuroBrainWebSocketServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/kurisu-two-volumes-baseline-cfc512.pt"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--delta-seconds", type=float, default=1.0)
    parser.add_argument("--max-sessions", type=int, default=128)
    parser.add_argument(
        "--read-token",
        default=os.environ.get("NEUROBRAIN_READ_TOKEN", ""),
    )
    parser.add_argument(
        "--write-token",
        default=os.environ.get("NEUROBRAIN_WRITE_TOKEN", ""),
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    server = NeuroBrainWebSocketServer.from_checkpoint(
        args.checkpoint,
        device=args.device,
        delta_seconds=args.delta_seconds,
        max_sessions=args.max_sessions,
        host=args.host,
        port=args.port,
        read_token=args.read_token,
        write_token=args.write_token,
    )
    print(
        f"NeuroBrain WebSocket listening on ws://{args.host}:{args.port} "
        f"checkpoint={args.checkpoint} device={args.device}",
        flush=True,
    )
    await server.serve_forever()


def main() -> int:
    args = build_parser().parse_args()
    if args.delta_seconds < 0:
        raise SystemExit("--delta-seconds must be non-negative")
    if args.max_sessions <= 0:
        raise SystemExit("--max-sessions must be positive")
    if not args.write_token.strip():
        raise SystemExit(
            "Set --write-token or NEUROBRAIN_WRITE_TOKEN; anonymous writer access is disabled."
        )
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
