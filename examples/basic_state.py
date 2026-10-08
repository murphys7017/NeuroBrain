"""Run a minimal first-phase protocol example."""

from neurobrain import NeuroState

state = NeuroState(
    psychological=(0.2, 0.0, 0.1, 0.4, 0.0),
    behavioral=(0.6, 0.2, 0.5, 0.7, 0.4, 0.5),
)

print(state)
