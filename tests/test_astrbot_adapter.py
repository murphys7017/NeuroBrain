import pytest

from neurobrain import convert_astrbot_tendency_to_stimulus


def _tendency() -> dict[str, int]:
    return {
        "Joy": 0,
        "Trust": 1,
        "Fear": 2,
        "Surprise": 3,
        "Sadness": 4,
        "Disgust": 5,
        "Anger": 6,
        "Anticipation": 10,
    }


def test_converts_astrbot_tendency_by_dimension_name() -> None:
    tendency = _tendency()

    stimulus = convert_astrbot_tendency_to_stimulus(tendency)

    assert stimulus == {
        "joy": 0.0,
        "trust": 0.1,
        "fear": 0.2,
        "surprise": 0.3,
        "sadness": 0.4,
        "disgust": 0.5,
        "anger": 0.6,
        "anticipation": 1.0,
    }


def test_clamps_out_of_range_astrbot_tendency_values() -> None:
    tendency = {**_tendency(), "Joy": -3, "Anger": 17}

    stimulus = convert_astrbot_tendency_to_stimulus(tendency)

    assert stimulus["joy"] == 0.0
    assert stimulus["anger"] == 1.0


@pytest.mark.parametrize(
    "tendency",
    [
        {**_tendency(), "Joy": True},
        {**_tendency(), "Joy": 1.5},
        {key: value for key, value in _tendency().items() if key != "Joy"},
        {**_tendency(), "Curiosity": 1},
    ],
)
def test_rejects_invalid_astrbot_tendency(tendency) -> None:
    with pytest.raises((TypeError, ValueError)):
        convert_astrbot_tendency_to_stimulus(tendency)


def test_rejects_non_mapping_input() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        convert_astrbot_tendency_to_stimulus([0] * 8)
