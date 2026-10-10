#!/usr/bin/env python3
"""Run a human-readable roleplay -> NeuroBrain -> mindset test loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from neurobrain.character_loop import (
    CharacterLoop,
    MiniMaxClient,
    MiniMaxConfig,
    OllamaClient,
    OllamaConfig,
)


def _print_result(result: dict) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _format_dimensions(values: dict[str, float], labels: dict[str, str]) -> str:
    return "、".join(
        f"{labels.get(name, name)} {value:.2f}"
        for name, value in values.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/kurisu-two-volumes-baseline-cfc512.pt"),
    )
    parser.add_argument("--provider", choices=("ollama", "minimax"), default="ollama")
    parser.add_argument("--model", help="model name; defaults to qwen3.5:2b or MiniMax-M2.7")
    parser.add_argument("--endpoint", help="chat API endpoint; provider-specific default if omitted")
    parser.add_argument(
        "--api-key-env",
        default="MINIMAX_API_KEY",
        help="MiniMax credential variable; Token Plan users should use their Subscription Key",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, help="sampling temperature (provider default if omitted)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--history-mode",
        choices=("zeros", "previous_output"),
        default="zeros",
        help="history condition passed to CfC; zeros matches the current checkpoint training mode",
    )
    parser.add_argument("--delta-seconds", type=float, default=1.0)
    parser.add_argument(
        "--conversation-turns",
        type=int,
        default=8,
        help="保留给角色模型的最近对话轮数；0 表示只使用 NeuroBrain 状态",
    )
    parser.add_argument("--message", help="run one turn and exit")
    parser.add_argument("--output", type=Path, help="write the final JSON result to this file")
    parser.add_argument("--show-json", action="store_true", help="also print the full result and diagnostics")
    args = parser.parse_args()

    if args.delta_seconds < 0.0:
        parser.error("--delta-seconds must be non-negative")
    if args.conversation_turns < 0:
        parser.error("--conversation-turns must be non-negative")
    if args.provider == "minimax":
        config = MiniMaxConfig(
            model=args.model or "MiniMax-M2.7",
            endpoint=args.endpoint or "https://api.minimax.cn/v1/chat/completions",
            timeout_seconds=args.timeout,
            temperature=args.temperature if args.temperature is not None else 0.35,
            api_key_env=args.api_key_env,
        )
        client = MiniMaxClient(config)
        model_name = args.model or "MiniMax-M2.7"
        endpoint = config.endpoint
    else:
        config = OllamaConfig(
            model=args.model or "qwen3.5:2b",
            endpoint=args.endpoint or "http://127.0.0.1:11434/api/chat",
            timeout_seconds=args.timeout,
            temperature=args.temperature if args.temperature is not None else 0.25,
        )
        client = OllamaClient(config)
        model_name = args.model or "qwen3.5:2b"
        endpoint = config.endpoint
    loop = CharacterLoop(
        str(args.checkpoint),
        client=client,
        device=args.device,
        history_mode=args.history_mode,
        conversation_turns=args.conversation_turns,
    )

    def run(message: str) -> None:
        result = loop.turn(message, delta_seconds=args.delta_seconds).as_dict()
        response = result["response"]
        psychology = result["current_psychology"]
        diagnostics = result["diagnostics"]
        print(f"你：{result['user_input']}")
        print(f"红莉栖：{response['text']}")
        actions = [action for segment in response["segments"] for action in segment["actions"]]
        if actions:
            print(f"动作：{'；'.join(actions)}")
        print("当前心理活动（模型估计）：")
        print(
            "  情绪："
            + _format_dimensions(
                psychology["emotion_estimate"],
                {
                    "joy": "愉悦",
                    "trust": "信任",
                    "fear": "恐惧",
                    "surprise": "惊讶",
                    "sadness": "悲伤",
                    "disgust": "厌恶",
                    "anger": "愤怒",
                    "anticipation": "期待",
                },
            )
        )
        print(
            "  行为倾向："
            + _format_dimensions(
                psychology["behavior_estimate"],
                {
                    "engagement": "投入",
                    "avoidance": "回避",
                    "expression": "表达",
                    "exploration": "探索",
                    "initiative": "主动",
                    "activation": "活跃",
                },
            )
        )
        print(f"  意图：{psychology['intent']}")
        print(f"  心态解读：{psychology['mindset']}")
        timing = diagnostics["timing"]
        print(
            "耗时："
            f"2.7 前段 {timing['draft_model_seconds']:.2f}s；"
            f"CfC {timing['cfc_seconds']:.2f}s；"
            f"2.7 后段 {timing['final_model_seconds']:.2f}s；"
            f"合计 {timing['total_seconds']:.2f}s"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"完整结果已保存：{args.output}")
        if args.show_json:
            _print_result(result)

    if args.message:
        try:
            run(args.message)
        except Exception as error:  # noqa: BLE001 - show a concise one-shot CLI error
            print(f"本轮失败：{type(error).__name__}: {error}", file=sys.stderr)
            return 1
        return 0

    print("NeuroBrain 角色测试：/reset 重置内部状态，/quit 退出。")
    print(
        f"角色模型：{args.provider}/{model_name}；API：{endpoint}；"
        f"CfC 设备：{args.device}；history-mode：{args.history_mode}"
    )
    while True:
        try:
            message = input("用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n测试结束。")
            return 0
        if not message:
            continue
        if message.lower() in {"/quit", "/exit", "q"}:
            print("测试结束。")
            return 0
        if message.lower() == "/reset":
            loop.reset()
            print("已重置 CfC 隐状态和历史条件。")
            continue
        try:
            run(message)
        except Exception as error:  # noqa: BLE001 - CLI should keep interactive loop alive
            print(f"本轮失败：{type(error).__name__}: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
