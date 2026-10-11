import io
import json
import urllib.error

import pytest

from neurobrain.character_loop import (
    FINALIZE_SYSTEM_PROMPT,
    ROLEPLAY_SYSTEM_PROMPT,
    CharacterLoop,
    MiniMaxClient,
    MiniMaxConfig,
    validate_finalization,
    validate_roleplay,
)
from scripts.character_test import _format_dimensions


def _draft(speech: str = "草稿回复") -> dict:
    return {
        "segments": [
            {
                "speech": speech,
                "actions": ["停顿"],
                "thought": "斟酌措辞",
                "tendency": {"Joy": 5, "Anger": 1},
            }
        ],
        "final_result": speech,
        "effect_calls": [],
    }


def _finalization(speech: str = "参考状态后的最终回复") -> dict:
    return {
        "segments": [{"speech": speech, "actions": ["微微点头"], "thought": "平静地回应"}],
        "final_result": speech,
        "effect_calls": [],
        "mindset_analysis": {
            "intent": "继续交流",
            "mindset": "较为平静",
            "emotion_interpretation": "情绪估计影响了语气",
            "behavior_interpretation": "行为估计使回应更主动",
            "confidence": "medium",
            "caveat": "这是模型估计",
        },
    }


def test_validate_roleplay_normalizes_tendency_and_defaults_final_result() -> None:
    result = validate_roleplay(
        {
            "segments": [
                {
                    "speech": "你好。",
                    "actions": ["挥手"],
                    "thought": "先观察一下。",
                    "tendency": {"Joy": 5},
                }
            ],
            "effect_calls": [],
        }
    )
    assert result["final_result"] == "你好。"
    assert result["segments"][0]["tendency"]["Joy"] == 5.0
    assert result["segments"][0]["tendency"]["Fear"] == 0.0


def test_validate_roleplay_clamps_out_of_range_tendency() -> None:
    result = validate_roleplay(
        {
            "segments": [
                {
                    "speech": "你好",
                    "actions": [],
                    "thought": "",
                    "tendency": {"Joy": -1.0, "Anger": 11.0},
                }
            ],
            "effect_calls": [],
        }
    )

    assert result["segments"][0]["tendency"]["Joy"] == 0.0
    assert result["segments"][0]["tendency"]["Anger"] == 10.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_roleplay_rejects_non_finite_tendency(value: float) -> None:
    with pytest.raises(ValueError, match="有限数字"):
        validate_roleplay(
            {
                "segments": [
                    {
                        "speech": "你好",
                        "actions": [],
                        "thought": "",
                        "tendency": {"Joy": value},
                    }
                ],
                "effect_calls": [],
            }
        )


def test_finalization_preserves_tendency_from_cfc_input_draft() -> None:
    roleplay, analysis = validate_finalization(_finalization(), _draft())
    assert roleplay["final_result"] == "参考状态后的最终回复"
    assert roleplay["segments"][0]["tendency"]["Joy"] == 5.0
    assert analysis["intent"] == "继续交流"


def test_finalization_requires_one_output_segment_per_draft_segment() -> None:
    with pytest.raises(ValueError, match="segments 数量"):
        validate_finalization(
            {**_finalization(), "segments": []},
            _draft(),
        )


def test_character_loop_feeds_cfc_output_to_final_reply_and_keeps_final_history() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[list[dict[str, str]], str]] = []

        def chat(self, messages, *, system):
            self.calls.append((list(messages), system))
            if system == ROLEPLAY_SYSTEM_PROMPT:
                return _draft("草稿内容")
            assert system == FINALIZE_SYSTEM_PROMPT
            context = json.loads(messages[-1]["content"])
            assert context["roleplay_draft"]["final_result"] == "草稿内容"
            assert "emotion" in context["neurobrain_output"]
            assert "behavior" in context["neurobrain_output"]
            assert "current_stimulus" in context["neurobrain_input"]["segments"][0]
            return _finalization("CfC 反馈后的最终答复")

    fake = FakeClient()
    loop = CharacterLoop(
        "artifacts/kurisu-two-volumes-baseline-cfc512.pt",
        client=fake,
        conversation_turns=8,
    )
    result = loop.turn("第一句")
    assert len(fake.calls) == 2
    assert result.roleplay["final_result"] == "CfC 反馈后的最终答复"
    result_data = result.as_dict()
    assert result_data["user_input"] == "第一句"
    assert result_data["response"]["text"] == "CfC 反馈后的最终答复"
    assert result_data["diagnostics"]["roleplay_candidate"]["final_result"] == "草稿内容"
    assert "emotion_estimate" in result_data["current_psychology"]
    assert len(result_data["current_psychology"]["emotion_estimate"]) == 8
    assert len(result_data["current_psychology"]["behavior_estimate"]) == 6
    timing = result_data["diagnostics"]["timing"]
    assert set(timing) == {
        "draft_model_seconds",
        "cfc_seconds",
        "final_model_seconds",
        "total_seconds",
    }
    assert all(value >= 0.0 for value in timing.values())

    loop.turn("第二句")
    second_roleplay_messages = fake.calls[2][0]
    assert [message["content"] for message in second_roleplay_messages[:2]] == [
        "第一句",
        "CfC 反馈后的最终答复",
    ]
    loop.reset()
    assert loop.conversation_history == []
    assert loop.hidden_state is None


def test_cli_dimension_formatter_displays_every_dimension() -> None:
    dimensions = {
        "joy": 0.1,
        "trust": 0.2,
        "fear": 0.3,
        "surprise": 0.4,
        "sadness": 0.5,
        "disgust": 0.6,
        "anger": 0.7,
        "anticipation": 0.8,
    }

    formatted = _format_dimensions(dimensions, {name: name for name in dimensions})

    assert formatted.count("、") == 7
    assert "anticipation 0.80" in formatted
    assert "joy 0.10" in formatted


def test_minimax_client_requires_configured_credential(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MINIMAX_KEY", raising=False)
    client = MiniMaxClient(MiniMaxConfig(api_key_env="TEST_MINIMAX_KEY"))
    with pytest.raises(RuntimeError, match="TEST_MINIMAX_KEY"):
        client.chat([], system="JSON only")


def test_minimax_default_endpoint_targets_china_token_plan() -> None:
    assert MiniMaxConfig().endpoint == "https://api.minimax.cn/v1/chat/completions"


def test_minimax_client_parses_openai_compatible_json_response(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MINIMAX_KEY", "sk-cp-test-secret")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '{"ok": true}'}}]}
            ).encode()

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = MiniMaxClient(
        MiniMaxConfig(model="MiniMax-M2.7", api_key_env="TEST_MINIMAX_KEY")
    )
    assert client.chat([], system="JSON only") == {"ok": True}
    assert captured["request"].get_header("Authorization") == "Bearer sk-cp-test-secret"
    body = json.loads(captured["request"].data)
    assert body["model"] == "MiniMax-M2.7"
    assert body["messages"][0] == {"role": "system", "content": "JSON only"}


def test_minimax_client_explains_401_without_echoing_subscription_key(monkeypatch) -> None:
    subscription_key = "sk-cp-private-test"
    monkeypatch.setenv("TEST_MINIMAX_KEY", subscription_key)

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(
                json.dumps(
                    {"error": {"message": "invalid subscription credential"}, "request_id": "req-test"}
                ).encode(),
            ),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = MiniMaxClient(MiniMaxConfig(api_key_env="TEST_MINIMAX_KEY"))
    with pytest.raises(RuntimeError) as error:
        client.chat([], system="JSON only")

    message = str(error.value)
    assert "HTTP 401" in message
    assert "invalid subscription credential" in message
    assert "Subscription Key" in message
    assert "req-test" in message
    assert subscription_key not in message
