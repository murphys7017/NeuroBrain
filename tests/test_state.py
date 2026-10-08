from neurobrain import InputFrame, NeuroState


def test_input_frame_keeps_stimulus_and_history_separate() -> None:
    frame = InputFrame(
        current_stimulus=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        historical_emotion=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1),
        delta_seconds=2.0,
    )

    assert frame.current_stimulus != frame.historical_emotion
    assert frame.delta_seconds == 2.0


def test_neuro_state_accepts_first_phase_dimensions() -> None:
    state = NeuroState(
        psychological=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        behavioral=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )

    assert len(state.psychological) == 8
    assert len(state.behavioral) == 6


def test_neuro_state_rejects_out_of_range_values() -> None:
    try:
        NeuroState(psychological=(0.0,) * 7 + (1.1,), behavioral=(0.0,) * 6)
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("expected ValueError")
