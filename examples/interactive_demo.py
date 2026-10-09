"""Interactively inspect CfC state updates before training data is added."""

from __future__ import annotations

import math

import torch

from neurobrain import (
    BEHAVIOR_TENDENCY_DIMENSIONS,
    CURRENT_STIMULUS_DIMENSIONS,
    NeuroBrainModel,
)


def _parse_values(raw: str, *, count: int, label: str) -> list[float]:
    values = [float(part) for part in raw.replace(",", " ").split()]
    if len(values) != count:
        raise ValueError(f"{label}需要输入 {count} 个数值")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"{label}的每个值必须在 0 到 1 之间")
    return values


def _format_values(names: tuple[str, ...], values: torch.Tensor) -> str:
    return "  ".join(
        f"{name}={value:.3f}"
        for name, value in zip(names, values.tolist(), strict=True)
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroBrainModel().to(device).eval()
    history = torch.zeros(8, device=device)
    hidden_state: torch.Tensor | None = None

    print(f"NeuroBrain CfC-512 交互试玩（设备：{device}）")
    print("提示：模型尚未训练，数值只用于检查状态传递，不代表真实心理反应。")
    print("输入 8 个 0～1 数值（逗号或空格分隔），可追加第 9 个数值作为 delta_seconds。")
    print("命令：/history 后跟 8 个历史条件值；/reset 重置隐状态；/quit 退出。")
    print("示例：0.8, 0, 0, 0, 0, 0, 0, 0, 1")

    while True:
        try:
            raw = input("刺激> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n试玩结束。")
            return

        if not raw:
            continue
        if raw.lower() in {"/quit", "/exit", "q"}:
            print("试玩结束。")
            return
        if raw.lower() == "/reset":
            hidden_state = None
            print("已重置 CfC 隐状态。")
            continue
        if raw.lower().startswith("/history"):
            try:
                values_text = raw[len("/history") :].strip()
                history = torch.tensor(
                    _parse_values(
                        values_text,
                        count=8,
                        label="历史条件",
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                print(f"历史条件已设置：{_format_values(CURRENT_STIMULUS_DIMENSIONS, history)}")
            except ValueError as error:
                print(f"输入有误：{error}")
            continue

        try:
            parts = raw.replace(",", " ").split()
            if len(parts) not in (8, 9):
                raise ValueError("请输 8 个情绪值，可选地再输入第 9 个时间间隔")
            stimulus = torch.tensor(
                _parse_values(" ".join(parts[:8]), count=8, label="当前刺激"),
                dtype=torch.float32,
                device=device,
            )
            delta_seconds = float(parts[8]) if len(parts) == 9 else 1.0
            if not math.isfinite(delta_seconds) or delta_seconds < 0.0:
                raise ValueError("delta_seconds 必须是有限的非负数")
        except ValueError as error:
            print(f"输入有误：{error}")
            continue

        with torch.inference_mode():
            output = model.step(
                stimulus,
                history,
                delta_seconds=delta_seconds,
                hidden_state=hidden_state,
            )
        hidden_state = output.final_hidden
        print(f"情绪：{_format_values(CURRENT_STIMULUS_DIMENSIONS, output.emotion)}")
        print(f"行为：{_format_values(BEHAVIOR_TENDENCY_DIMENSIONS, output.behavior)}")
        print(f"已更新内部隐状态（维度：{hidden_state.shape[-1]}，delta={delta_seconds:g}s）")


if __name__ == "__main__":
    main()
