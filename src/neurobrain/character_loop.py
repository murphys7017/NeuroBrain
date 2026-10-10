"""Roleplay bridge for exercising NeuroBrain end to end.

Each turn uses a two-pass model flow: the roleplay model drafts a response and
emotional stimulus, NeuroBrain updates its continuous state, then the model
uses that state to produce the final response and a human-readable explanation.
Ollama and MiniMax share a small, dependency-free HTTP client interface.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import torch

from .model import NeuroBrainModel
from .protocol import (
    BEHAVIOR_TENDENCY_DIMENSIONS,
    CURRENT_STIMULUS_DIMENSIONS,
)


EMOTION_NAMES = CURRENT_STIMULUS_DIMENSIONS
BEHAVIOR_NAMES = BEHAVIOR_TENDENCY_DIMENSIONS


ROLEPLAY_SYSTEM_PROMPT = """你是一个小型角色扮演模型，扮演《Steins;Gate》的牧濑红莉栖。
保持她的科研能力、嘴硬、敏感、理性和偶尔露出的害羞。根据用户最新消息和当前状态
生成一小段角色回复。不要替用户做决定，不要把内部状态写成论文式说明。

只输出一个 JSON 对象，不要 Markdown，不要代码围栏：
{
  "segments": [
    {
      "speech": "角色实际说的话",
      "actions": ["可见动作"],
      "thought": "角色此刻的简短内心想法",
      "tendency": {
        "Joy": 0,
        "Trust": 0,
        "Fear": 0,
        "Surprise": 0,
        "Sadness": 0,
        "Disgust": 0,
        "Anger": 0,
        "Anticipation": 0
      }
    }
  ],
  "final_result": "把所有 speech 合并后的最终回复",
  "effect_calls": []
}

tendency 是给 NeuroBrain 的本轮情绪输入，不是角色模型的自评，也不是最终状态。
每个值使用 0 到 10 的整数或小数；程序会将它除以 10 后作为 NeuroBrain 的 0 到 1
输入。它应表达这一段内容对角色当前反应的情绪倾向，不能直接填写 NeuroBrain 的最终
输出。effect_calls 没有动作时使用空数组。"""


FINALIZE_SYSTEM_PROMPT = """你是一个小型角色扮演模型，扮演《Steins;Gate》的牧濑红莉栖。
你会收到用户消息、角色回复草稿，以及 NeuroBrain 根据草稿 tendency 和连续历史隐状态
计算出的情绪与行为估计。请先参考 NeuroBrain 输出调整草稿，生成角色本轮真正说出的最终
回复；之后再简短分析这段最终回复可能表达的意图和心态。

最终回复必须实际使用 NeuroBrain 的 emotion 和 behavior 来调整语气、措辞或动作，不能只
在分析段落里提到它们，也不要机械照抄数值。保持草稿的主题、人物关系和事实，不凭空改变
对话内容。NeuroBrain 是状态估计，不是客观心理真值；不要在角色台词中提到 NeuroBrain、
维度名或数值。每个输入 segment 对应输出一个 segment，并保留其原有 tendency，不要重算。

只输出一个 JSON 对象，不要 Markdown，不要代码围栏：
{
  "segments": [
    {"speech": "最终台词", "actions": ["可见动作"], "thought": "简短的角色内心想法"}
  ],
  "final_result": "把所有 speech 合并后的最终回复",
  "effect_calls": [],
  "mindset_analysis": {
    "intent": "这次回复最可能的意图",
    "mindset": "当前心态的中文概括",
    "emotion_interpretation": "CfC 情绪估计如何影响最终回复",
    "behavior_interpretation": "CfC 行为估计如何影响动作或语气",
    "confidence": "high|medium|low",
    "caveat": "这是模型估计而非心理真值"
  }
}"""


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    model: str = "qwen3.5:2b"
    endpoint: str = "http://127.0.0.1:11434/api/chat"
    timeout_seconds: float = 180.0
    temperature: float = 0.25


@dataclass(frozen=True, slots=True)
class MiniMaxConfig:
    model: str = "MiniMax-M2.7"
    endpoint: str = "https://api.minimax.cn/v1/chat/completions"
    timeout_seconds: float = 180.0
    temperature: float = 0.35
    api_key_env: str = "MINIMAX_API_KEY"


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, system: str) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Serializable result returned by one end-to-end test turn."""

    user_input: str
    roleplay: dict[str, Any]
    roleplay_draft: dict[str, Any]
    neurobrain_input: dict[str, Any]
    neurobrain_output: dict[str, Any]
    mindset_analysis: dict[str, Any]
    timing: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_input": self.user_input,
            "response": {
                "segments": [
                    {"speech": segment["speech"], "actions": segment["actions"]}
                    for segment in self.roleplay["segments"]
                ],
                "text": self.roleplay["final_result"],
                "effect_calls": self.roleplay["effect_calls"],
            },
            "current_psychology": {
                "emotion_estimate": self.neurobrain_output["emotion"],
                "behavior_estimate": self.neurobrain_output["behavior"],
                "top_emotions": self.neurobrain_output["top_emotions"],
                "top_behaviors": self.neurobrain_output["top_behaviors"],
                **self.mindset_analysis,
            },
            "diagnostics": {
                "roleplay_candidate": self.roleplay_draft,
                "neurobrain_input": self.neurobrain_input,
                "neurobrain_output": self.neurobrain_output,
                "timing": self.timing,
            },
        }


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for index, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                break
            except json.JSONDecodeError:
                continue
        if value is None:
            raise ValueError("模型没有返回可解析的 JSON 对象")
    if not isinstance(value, dict):
        raise ValueError("模型返回的 JSON 根节点必须是对象")
    return value


def _number(value: Any, *, minimum: float, maximum: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return result


def validate_roleplay(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the roleplay model's JSON response."""

    segments = value.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments 必须是非空数组")
    normalized_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"segments[{index}] 必须是对象")
        speech = segment.get("speech", "")
        thought = segment.get("thought", "")
        actions = segment.get("actions", [])
        if not isinstance(speech, str) or not speech.strip():
            raise ValueError(f"segments[{index}].speech 不能为空")
        if not isinstance(thought, str):
            raise ValueError(f"segments[{index}].thought 必须是字符串")
        if not isinstance(actions, list) or any(not isinstance(item, str) for item in actions):
            raise ValueError(f"segments[{index}].actions 必须是字符串数组")
        tendency = segment.get("tendency", {})
        if not isinstance(tendency, Mapping):
            raise ValueError(f"segments[{index}].tendency 必须是对象")
        normalized_tendency = {
            name: _number(
                tendency.get(name, 0.0),
                minimum=0.0,
                maximum=10.0,
                name=f"segments[{index}].tendency.{name}",
            )
            for name in ("Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation")
        }
        normalized_segments.append(
            {
                "speech": speech,
                "actions": actions,
                "thought": thought,
                "tendency": normalized_tendency,
            }
        )

    final_result = value.get("final_result")
    if not isinstance(final_result, str) or not final_result.strip():
        final_result = "".join(segment["speech"] for segment in normalized_segments)
    effect_calls = value.get("effect_calls", [])
    if not isinstance(effect_calls, list) or any(not isinstance(item, Mapping) for item in effect_calls):
        raise ValueError("effect_calls 必须是对象数组")
    normalized_effects = [dict(item) for item in effect_calls]
    return {
        "segments": normalized_segments,
        "final_result": final_result,
        "effect_calls": normalized_effects,
    }


def validate_finalization(
    value: Mapping[str, Any], draft: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the post-CfC reply while keeping the draft's input tendency."""

    draft_segments = draft["segments"]
    segments = value.get("segments")
    if not isinstance(segments, list) or len(segments) != len(draft_segments):
        raise ValueError("最终回复的 segments 数量必须与角色草稿一致")
    normalized_segments: list[dict[str, Any]] = []
    for index, (segment, draft_segment) in enumerate(zip(segments, draft_segments, strict=True)):
        if not isinstance(segment, Mapping):
            raise ValueError(f"最终回复 segments[{index}] 必须是对象")
        speech = segment.get("speech")
        thought = segment.get("thought", "")
        actions = segment.get("actions", [])
        if not isinstance(speech, str) or not speech.strip():
            raise ValueError(f"最终回复 segments[{index}].speech 不能为空")
        if not isinstance(thought, str):
            raise ValueError(f"最终回复 segments[{index}].thought 必须是字符串")
        if not isinstance(actions, list) or any(not isinstance(item, str) for item in actions):
            raise ValueError(f"最终回复 segments[{index}].actions 必须是字符串数组")
        normalized_segments.append(
            {
                "speech": speech,
                "actions": actions,
                "thought": thought,
                "tendency": draft_segment["tendency"],
            }
        )

    final_result = value.get("final_result")
    if not isinstance(final_result, str) or not final_result.strip():
        final_result = "".join(segment["speech"] for segment in normalized_segments)
    effect_calls = value.get("effect_calls", [])
    if not isinstance(effect_calls, list) or any(not isinstance(item, Mapping) for item in effect_calls):
        raise ValueError("最终回复 effect_calls 必须是对象数组")

    analysis = value.get("mindset_analysis")
    required = (
        "intent",
        "mindset",
        "emotion_interpretation",
        "behavior_interpretation",
        "confidence",
        "caveat",
    )
    if not isinstance(analysis, Mapping):
        raise ValueError("mindset_analysis 必须是对象")
    normalized_analysis = {key: analysis.get(key, "") for key in required}
    if any(not isinstance(normalized_analysis[key], str) for key in required):
        raise ValueError("mindset_analysis 的字段必须是字符串")
    if normalized_analysis["confidence"] not in {"high", "medium", "low"}:
        normalized_analysis["confidence"] = "low"

    return (
        {
            "segments": normalized_segments,
            "final_result": final_result,
            "effect_calls": [dict(item) for item in effect_calls],
        },
        normalized_analysis,
    )


def _pretty_vector(names: tuple[str, ...], values: torch.Tensor) -> dict[str, float]:
    return {name: round(float(value), 4) for name, value in zip(names, values.tolist(), strict=True)}


def _top_dimensions(values: Mapping[str, float], count: int = 3) -> list[dict[str, Any]]:
    return [
        {"name": name, "value": round(float(value), 4)}
        for name, value in sorted(values.items(), key=lambda item: item[1], reverse=True)[:count]
    ]


class OllamaClient:
    """Minimal client for Ollama's local ``/api/chat`` endpoint."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()

    def chat(self, messages: list[dict[str, str]], *, system: str) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": self.config.temperature},
            "keep_alive": "10m",
        }
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"无法连接 Ollama {self.config.endpoint}；请确认 Ollama 正在运行。"
            ) from error
        content = payload.get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama 返回中没有 message.content")
        return _json_object(content)


class MiniMaxClient:
    """Minimal client for MiniMax's OpenAI-compatible chat endpoint."""

    def __init__(self, config: MiniMaxConfig | None = None) -> None:
        self.config = config or MiniMaxConfig()

    def chat(self, messages: list[dict[str, str]], *, system: str) -> dict[str, Any]:
        api_key = os.environ.get(self.config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"未找到环境变量 {self.config.api_key_env}；请先设置 MiniMax Subscription Key 或 API key。"
            )
        body = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "temperature": self.config.temperature,
            "max_completion_tokens": 4096,
        }
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = ""
            request_id = ""
            try:
                error_payload = json.loads(error.read().decode("utf-8", errors="replace"))
                if isinstance(error_payload, Mapping):
                    api_error = error_payload.get("error")
                    if isinstance(api_error, Mapping):
                        details = str(api_error.get("message", ""))
                    elif api_error:
                        details = str(api_error)
                    if not details and isinstance(error_payload.get("message"), str):
                        details = error_payload["message"]
                    base_response = error_payload.get("base_resp")
                    if not details and isinstance(base_response, Mapping):
                        details = str(
                            base_response.get("status_msg", base_response.get("msg", ""))
                        )
                    request_id = str(error_payload.get("request_id", ""))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            if api_key and api_key in details:
                details = details.replace(api_key, "[redacted]")
            suffix = f" 服务端信息：{details}" if details else ""
            if error.code == 401:
                suffix += (
                    " 请确认环境变量中放的是 Token Plan 的 Subscription Key"
                    "（通常以 sk-cp 开头），而不是按量 API Key；同时确认接口地址与密钥所属区域匹配。"
                )
            if request_id:
                suffix += f" request_id={request_id}"
            raise RuntimeError(f"MiniMax API 请求失败（HTTP {error.code}）。{suffix}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"无法连接 MiniMax API {self.config.endpoint}。") from error
        if not isinstance(payload, Mapping):
            raise RuntimeError("MiniMax 返回的响应根节点不是 JSON 对象")
        api_error = payload.get("error")
        if api_error:
            if isinstance(api_error, Mapping):
                message = api_error.get("message", "未知错误")
            else:
                message = str(api_error)
            raise RuntimeError(f"MiniMax API 返回错误：{message}")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise RuntimeError("MiniMax 返回中没有 choices[0]")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("MiniMax 返回中没有 choices[0].message")
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, Mapping) and isinstance(item.get("text", ""), str)
            )
        if not isinstance(content, str) or not content.strip():
            finish_reason = choice.get("finish_reason")
            raise RuntimeError(
                f"MiniMax 没有返回文本 content（finish_reason={finish_reason!r}）。"
            )
        return _json_object(content)


class CharacterLoop:
    """Run roleplay, CfC state update and mindset analysis for each turn."""

    def __init__(
        self,
        checkpoint: str,
        *,
        client: ChatClient | None = None,
        ollama: OllamaClient | None = None,
        device: str | torch.device = "cpu",
        history_mode: str = "zeros",
        conversation_turns: int = 8,
    ) -> None:
        if history_mode not in {"zeros", "previous_output"}:
            raise ValueError("history_mode 必须是 zeros 或 previous_output")
        if conversation_turns < 0:
            raise ValueError("conversation_turns 必须是非负整数")
        self.device = torch.device(device)
        self.model = NeuroBrainModel.load_checkpoint(checkpoint, map_location=self.device).to(self.device).eval()
        if client is not None and ollama is not None:
            raise ValueError("client 和 ollama 只能指定一个")
        self.client = client or ollama or OllamaClient()
        self.history_mode = history_mode
        self.hidden_state: torch.Tensor | None = None
        self.history = torch.zeros(len(EMOTION_NAMES), device=self.device)
        self.last_output: dict[str, Any] | None = None
        self.conversation_turns = conversation_turns
        self.conversation_history: list[dict[str, str]] = []

    def reset(self) -> None:
        self.hidden_state = None
        self.history.zero_()
        self.last_output = None
        self.conversation_history.clear()

    def _state_context(self) -> str:
        if self.last_output is None:
            return "这是本次对话的第一轮，当前没有上一轮 NeuroBrain 状态。"
        output = self.last_output["diagnostics"]["neurobrain_output"]
        return json.dumps(output, ensure_ascii=False, separators=(",", ":"))

    def _roleplay(self, user_message: str) -> dict[str, Any]:
        context = (
            f"用户消息：{user_message}\n\n"
            f"上一轮 NeuroBrain 状态摘要：{self._state_context()}\n\n"
            "请生成这一轮角色回复。"
        )
        messages = [*self.conversation_history, {"role": "user", "content": context}]
        value = self.client.chat(
            messages,
            system=ROLEPLAY_SYSTEM_PROMPT,
        )
        roleplay = validate_roleplay(value)
        return roleplay

    def _finalize(
        self,
        user_message: str,
        draft: dict[str, Any],
        neurobrain_input: dict[str, Any],
        neurobrain_output: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        context = {
            "user_message": user_message,
            "roleplay_draft": draft,
            "neurobrain_input": neurobrain_input,
            "neurobrain_output": neurobrain_output,
            "instruction": "参考 NeuroBrain 输出，生成真正发给用户的最终角色回复，并解释其意图和心态。",
        }
        value = self.client.chat(
            [
                *self.conversation_history,
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            system=FINALIZE_SYSTEM_PROMPT,
        )
        return validate_finalization(value, draft)

    def turn(self, user_message: str, *, delta_seconds: float = 1.0) -> TurnResult:
        turn_started = time.perf_counter()
        draft_started = time.perf_counter()
        roleplay_draft = self._roleplay(user_message)
        draft_seconds = time.perf_counter() - draft_started
        input_segments: list[dict[str, Any]] = []
        output_segments: list[dict[str, Any]] = []
        hidden_state = self.hidden_state
        history = self.history.detach().clone()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        cfc_started = time.perf_counter()
        for segment in roleplay_draft["segments"]:
            tendency = segment["tendency"]
            tendency_values = torch.tensor(
                [tendency[name] / 10.0 for name in ("Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation")],
                dtype=torch.float32,
                device=self.device,
            )
            input_history = history.detach().clone()
            with torch.inference_mode():
                output = self.model.step(
                    tendency_values,
                    input_history,
                    delta_seconds=delta_seconds,
                    hidden_state=hidden_state,
                )
            hidden_state = output.final_hidden.detach()
            emotion = _pretty_vector(EMOTION_NAMES, output.emotion.detach().cpu())
            behavior = _pretty_vector(BEHAVIOR_NAMES, output.behavior.detach().cpu())
            input_segments.append(
                {
                    "tendency": tendency,
                    "current_stimulus": _pretty_vector(EMOTION_NAMES, tendency_values.detach().cpu()),
                    "historical_emotion": _pretty_vector(EMOTION_NAMES, input_history.detach().cpu()),
                    "delta_seconds": float(delta_seconds),
                }
            )
            output_segments.append(
                {
                    "emotion": emotion,
                    "behavior": behavior,
                    "top_emotions": _top_dimensions(emotion),
                    "top_behaviors": _top_dimensions(behavior),
                }
            )
            if self.history_mode == "previous_output":
                history = output.emotion.detach()
            else:
                history.zero_()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        cfc_seconds = time.perf_counter() - cfc_started

        final_output = output_segments[-1]
        neurobrain_output = {
            "segments": output_segments,
            "emotion": final_output["emotion"],
            "behavior": final_output["behavior"],
            "top_emotions": final_output["top_emotions"],
            "top_behaviors": final_output["top_behaviors"],
            "delta_seconds": float(delta_seconds),
            "history_mode": self.history_mode,
        }
        neurobrain_input = {"segments": input_segments}
        final_started = time.perf_counter()
        roleplay, analysis = self._finalize(
            user_message,
            roleplay_draft,
            neurobrain_input,
            neurobrain_output,
        )
        final_seconds = time.perf_counter() - final_started
        timing = {
            "draft_model_seconds": round(draft_seconds, 4),
            "cfc_seconds": round(cfc_seconds, 4),
            "final_model_seconds": round(final_seconds, 4),
            "total_seconds": round(time.perf_counter() - turn_started, 4),
        }
        result = TurnResult(
            user_input=user_message,
            roleplay=roleplay,
            roleplay_draft=roleplay_draft,
            neurobrain_input=neurobrain_input,
            neurobrain_output=neurobrain_output,
            mindset_analysis=analysis,
            timing=timing,
        )
        self.hidden_state = hidden_state
        self.history = history
        if self.conversation_turns:
            self.conversation_history.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": roleplay["final_result"]},
                ]
            )
            self.conversation_history = self.conversation_history[-2 * self.conversation_turns :]
        self.last_output = result.as_dict()
        return result


__all__ = [
    "BEHAVIOR_NAMES",
    "ChatClient",
    "EMOTION_NAMES",
    "FINALIZE_SYSTEM_PROMPT",
    "MiniMaxClient",
    "MiniMaxConfig",
    "OllamaClient",
    "OllamaConfig",
    "ROLEPLAY_SYSTEM_PROMPT",
    "CharacterLoop",
    "TurnResult",
    "validate_roleplay",
    "validate_finalization",
]
