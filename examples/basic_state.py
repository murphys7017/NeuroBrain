"""Run a minimal first-phase protocol example."""

from neurobrain import InputFrame, NeuroState

inputs = InputFrame(
    current_stimulus=(0.2, 0.0, 0.1, 0.4, 0.0, 0.1, 0.0, 0.2),
    historical_emotion=(0.1, 0.2, 0.0, 0.3, 0.0, 0.1, 0.0, 0.0),
)

state = NeuroState(
    psychological=(0.2, 0.0, 0.1, 0.4, 0.0, 0.1, 0.0, 0.2),
    behavioral=(0.6, 0.2, 0.5, 0.7, 0.4, 0.5),
)

print(state)
print(inputs)
