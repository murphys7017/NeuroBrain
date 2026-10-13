import asyncio

import pytest

from neurobrain import NeuroBrainModel
from neurobrain.ws_server import NeuroBrainStateStore, NeuroBrainWebSocketServer


def _tendency(value: int) -> dict[str, int]:
    return {
        "Joy": value,
        "Trust": 2,
        "Fear": 0,
        "Surprise": 1,
        "Sadness": 0,
        "Disgust": 0,
        "Anger": 0,
        "Anticipation": 3,
    }


def test_state_store_updates_multiple_segments_and_deduplicates() -> None:
    store = NeuroBrainStateStore(model=NeuroBrainModel(encoder_size=4, hidden_size=8))

    async def run() -> None:
        first = await store.update(
            "persona:Alice pro",
            event_id="turn-1:result-1",
            segments=[
                {"tendency": _tendency(2)},
                {"tendency": _tendency(7)},
            ],
        )
        assert first["duplicate"] is False
        assert len(first["snapshots"]) == 2
        assert first["snapshot"]["step"] == 2
        assert store.read("persona:Alice pro")["step"] == 2

        duplicate = await store.update(
            "persona:Alice pro",
            event_id="turn-1:result-1",
            segments=[{"tendency": _tendency(0)}],
        )
        assert duplicate["duplicate"] is True
        assert duplicate["snapshot"]["step"] == 2

    asyncio.run(run())


def test_state_store_reset_removes_readable_snapshot() -> None:
    store = NeuroBrainStateStore(model=NeuroBrainModel(encoder_size=4, hidden_size=8))

    async def run() -> None:
        await store.update(
            "global",
            event_id="turn-1",
            segments=[{"tendency": _tendency(5)}],
        )
        result = await store.reset("global")
        assert result == {"state_key": "global", "step": 0, "reset": True}
        assert store.read("global") is None

    asyncio.run(run())


def test_reset_releases_a_session_slot() -> None:
    store = NeuroBrainStateStore(
        model=NeuroBrainModel(encoder_size=4, hidden_size=8),
        max_sessions=1,
    )

    async def run() -> None:
        await store.update(
            "conversation:first",
            event_id="first-turn",
            segments=[{"tendency": _tendency(4)}],
        )
        await store.reset("conversation:first")
        second = await store.update(
            "conversation:second",
            event_id="second-turn",
            segments=[{"tendency": _tendency(7)}],
        )
        assert second["snapshot"]["step"] == 1

    asyncio.run(run())


def test_websocket_writer_requires_a_token_and_readers_remain_read_only() -> None:
    store = NeuroBrainStateStore(model=NeuroBrainModel(encoder_size=4, hidden_size=8))
    with pytest.raises(ValueError, match="write_token is required"):
        NeuroBrainWebSocketServer(store)

    server = NeuroBrainWebSocketServer(
        store,
        read_token="reader-secret",
        write_token="writer-secret",
    )
    assert server._authorize("writer", "writer-secret")
    assert not server._authorize("writer", "reader-secret")
    assert not server._authorize("writer", "")
    assert server._authorize("reader", "reader-secret")
    assert server._authorize("reader", "writer-secret")
    assert not server._authorize("reader", "")

    open_reader = NeuroBrainWebSocketServer(store, write_token="writer-secret")
    assert open_reader._authorize("reader", "")
