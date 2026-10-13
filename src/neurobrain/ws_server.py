"""WebSocket transport for the stateful NeuroBrain inference facade.

The transport is deliberately kept outside :class:`NeuroBrainSession`: the
model process owns sessions and clients only submit structured AstrBot
stimulus data or read immutable state snapshots.  A writer connection is
required for state updates; reader connections can only query and subscribe.
"""

from __future__ import annotations

import asyncio
import copy
import hmac
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

from .adapters.astrbot import convert_astrbot_tendency_to_stimulus
from .api import NeuroBrainSession, StateSnapshot
from .model import NeuroBrainModel

WS_PROTOCOL_VERSION = "neurobrain-ws-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _json_message(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@dataclass(slots=True)
class _StateRecord:
    state_key: str
    snapshot: dict[str, Any]
    last_tendency: dict[str, int]
    updated_at: str
    last_event_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            **copy.deepcopy(self.snapshot),
            "last_tendency": dict(self.last_tendency),
            "updated_at": self.updated_at,
            "last_event_id": self.last_event_id,
        }


class NeuroBrainStateStore:
    """Own sessions and apply idempotent, ordered state updates."""

    def __init__(
        self,
        checkpoint: str | PathLike[str] | None = None,
        *,
        device: str = "cpu",
        delta_seconds: float = 1.0,
        max_sessions: int = 128,
        model: NeuroBrainModel | None = None,
        session_factory: Any | None = None,
        processed_event_limit: int = 4096,
    ) -> None:
        if isinstance(max_sessions, bool) or max_sessions <= 0:
            raise ValueError("max_sessions must be a positive integer")
        if isinstance(processed_event_limit, bool) or processed_event_limit <= 0:
            raise ValueError("processed_event_limit must be a positive integer")

        if model is None and session_factory is None:
            if checkpoint is None:
                raise ValueError("checkpoint is required when model is not supplied")
            model = NeuroBrainModel.load_checkpoint(
                Path(checkpoint),
                map_location="cpu",
                expected_history_mode="zeros",
            ).to(device)
            model.eval()

        self._model = model
        self._session_factory = session_factory
        self._delta_seconds = float(delta_seconds)
        self._max_sessions = max_sessions
        self._processed_event_limit = processed_event_limit
        self._sessions: dict[str, NeuroBrainSession] = {}
        self._states: dict[str, _StateRecord] = {}
        self._processed_events: dict[str, OrderedDict[str, None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._model_lock = asyncio.Lock()

    def _get_lock(self, state_key: str) -> asyncio.Lock:
        lock = self._locks.get(state_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[state_key] = lock
        return lock

    def _get_session(self, state_key: str) -> NeuroBrainSession:
        session = self._sessions.get(state_key)
        if session is not None:
            return session
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError("maximum NeuroBrain session count reached")
        if self._session_factory is not None:
            session = self._session_factory()
        else:
            if self._model is None:
                raise RuntimeError("NeuroBrain model is not initialized")
            session = NeuroBrainSession(
                self._model,
                delta_seconds=self._delta_seconds,
            )
        self._sessions[state_key] = session
        return session

    async def _update_session(
        self,
        session: NeuroBrainSession,
        stimuli: Sequence[Mapping[str, float]],
    ) -> tuple[StateSnapshot, ...]:
        # The CfC model is read-only after loading, but serializing forwards
        # keeps third-party model implementations safe as well.
        async with self._model_lock:
            return await asyncio.to_thread(session.update_many, stimuli)

    def _remember_event(self, state_key: str, event_id: str) -> None:
        events = self._processed_events.setdefault(state_key, OrderedDict())
        events.pop(event_id, None)
        events[event_id] = None
        while len(events) > self._processed_event_limit:
            events.popitem(last=False)

    def _is_duplicate(self, state_key: str, event_id: str) -> bool:
        return event_id in self._processed_events.get(state_key, {})

    async def update(
        self,
        state_key: str,
        *,
        event_id: str,
        segments: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        state_key = _non_empty_string(state_key, "state_key")
        event_id = _non_empty_string(event_id, "event_id")
        if not segments:
            raise ValueError("segments must contain at least one item")
        normalized_segments = [
            dict(segment) for segment in segments if isinstance(segment, Mapping)
        ]
        if len(normalized_segments) != len(segments):
            raise TypeError("each segment must be an object")

        async with self._get_lock(state_key):
            if self._is_duplicate(state_key, event_id):
                record = self._states.get(state_key)
                if record is None:
                    raise RuntimeError("duplicate event has no retained state")
                return {
                    "state_key": state_key,
                    "event_id": event_id,
                    "duplicate": True,
                    "snapshots": [],
                    "snapshot": record.as_dict(),
                }

            last_tendency = normalized_segments[-1].get(
                "tendency", normalized_segments[-1]
            )
            if not isinstance(last_tendency, Mapping):
                raise TypeError("segment tendency must be an object")
            stimuli = [
                convert_astrbot_tendency_to_stimulus(segment.get("tendency", segment))
                for segment in normalized_segments
            ]
            session = self._get_session(state_key)
            state_snapshots = await self._update_session(session, stimuli)
            snapshots = [snapshot.as_dict() for snapshot in state_snapshots]
            normalized_tendency = {
                str(key): int(value) for key, value in last_tendency.items()
            }
            record = _StateRecord(
                state_key=state_key,
                snapshot=snapshots[-1],
                last_tendency=normalized_tendency,
                updated_at=_utc_now(),
                last_event_id=event_id,
            )
            self._states[state_key] = record
            self._remember_event(state_key, event_id)
            return {
                "state_key": state_key,
                "event_id": event_id,
                "duplicate": False,
                "snapshots": snapshots,
                "snapshot": record.as_dict(),
            }

    def read(self, state_key: str) -> dict[str, Any] | None:
        state_key = _non_empty_string(state_key, "state_key")
        record = self._states.get(state_key)
        return record.as_dict() if record is not None else None

    def list_states(self) -> list[dict[str, Any]]:
        return [self._states[key].as_dict() for key in sorted(self._states)]

    async def reset(self, state_key: str) -> dict[str, Any]:
        state_key = _non_empty_string(state_key, "state_key")
        async with self._get_lock(state_key):
            session = self._sessions.get(state_key)
            if session is not None:
                await asyncio.to_thread(session.reset)
                self._sessions.pop(state_key, None)
            self._states.pop(state_key, None)
            self._processed_events.pop(state_key, None)
            return {"state_key": state_key, "step": 0, "reset": True}


@dataclass(slots=True)
class _Connection:
    websocket: Any
    role: str
    client_id: str
    subscriptions: set[str] = field(default_factory=set)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class NeuroBrainWebSocketServer:
    """Serve the NeuroBrain state store over a small JSON WebSocket protocol."""

    def __init__(
        self,
        store: NeuroBrainStateStore,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        read_token: str = "",
        write_token: str = "",
    ) -> None:
        self.store = store
        self.host = host
        self.port = port
        self.read_token = read_token.strip()
        self.write_token = write_token.strip()
        if not self.write_token:
            raise ValueError("write_token is required; writer access cannot be anonymous")
        self._connections: list[_Connection] = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | PathLike[str],
        *,
        device: str = "cpu",
        delta_seconds: float = 1.0,
        max_sessions: int = 128,
        host: str = "127.0.0.1",
        port: int = 8765,
        read_token: str = "",
        write_token: str = "",
    ) -> "NeuroBrainWebSocketServer":
        return cls(
            NeuroBrainStateStore(
                checkpoint,
                device=device,
                delta_seconds=delta_seconds,
                max_sessions=max_sessions,
            ),
            host=host,
            port=port,
            read_token=read_token,
            write_token=write_token,
        )

    async def _send(self, connection: _Connection, payload: Mapping[str, Any]) -> None:
        async with connection.send_lock:
            await connection.websocket.send(_json_message(payload))

    async def _error(
        self,
        connection: _Connection,
        code: str,
        message: str,
        request_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "error",
            "protocol_version": WS_PROTOCOL_VERSION,
            "code": code,
            "message": message,
        }
        if request_id:
            payload["request_id"] = request_id
        await self._send(connection, payload)

    def _authorize(self, role: str, token: str) -> bool:
        if role == "writer":
            return bool(token) and hmac.compare_digest(token, self.write_token)
        if role == "reader":
            if token and hmac.compare_digest(token, self.write_token):
                return True
            if self.read_token:
                return bool(token) and hmac.compare_digest(token, self.read_token)
            return not token
        return False

    async def _broadcast_state(self, snapshot: Mapping[str, Any]) -> None:
        state_key = str(snapshot.get("state_key") or "")
        payload = {
            "type": "state",
            "protocol_version": WS_PROTOCOL_VERSION,
            "state_key": state_key,
            "snapshot": copy.deepcopy(dict(snapshot)),
        }
        stale: list[_Connection] = []
        for connection in tuple(self._connections):
            if (
                state_key not in connection.subscriptions
                and "*" not in connection.subscriptions
            ):
                continue
            try:
                await self._send(connection, payload)
            except Exception:
                stale.append(connection)
        for connection in stale:
            if connection in self._connections:
                self._connections.remove(connection)

    async def _broadcast_reset(self, state_key: str) -> None:
        payload = {
            "type": "state_reset",
            "protocol_version": WS_PROTOCOL_VERSION,
            "state_key": state_key,
            "step": 0,
        }
        for connection in tuple(self._connections):
            if (
                state_key not in connection.subscriptions
                and "*" not in connection.subscriptions
            ):
                continue
            try:
                await self._send(connection, payload)
            except Exception:
                if connection in self._connections:
                    self._connections.remove(connection)

    @staticmethod
    def _segments_from_payload(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        raw_segments = payload.get("segments")
        if raw_segments is None:
            tendency = payload.get("tendency")
            if not isinstance(tendency, Mapping):
                raise TypeError("stimulus requires tendency or segments")
            return [tendency]
        if not isinstance(raw_segments, list) or not raw_segments:
            raise TypeError("segments must be a non-empty array")
        return raw_segments

    async def _handle_command(
        self,
        connection: _Connection,
        payload: Mapping[str, Any],
    ) -> None:
        message_type = _non_empty_string(payload.get("type"), "type")
        request_id = payload.get("request_id")
        request_id = request_id.strip() if isinstance(request_id, str) else None

        if message_type == "get_state":
            state_key = _non_empty_string(payload.get("state_key"), "state_key")
            await self._send(
                connection,
                {
                    "type": "state",
                    "protocol_version": WS_PROTOCOL_VERSION,
                    "request_id": request_id,
                    "state_key": state_key,
                    "snapshot": self.store.read(state_key),
                },
            )
            return

        if message_type == "subscribe":
            state_key = _non_empty_string(payload.get("state_key"), "state_key")
            connection.subscriptions.add(state_key)
            await self._send(
                connection,
                {
                    "type": "subscribed",
                    "protocol_version": WS_PROTOCOL_VERSION,
                    "request_id": request_id,
                    "state_key": state_key,
                },
            )
            if state_key == "*":
                for snapshot in self.store.list_states():
                    await self._send(
                        connection,
                        {
                            "type": "state",
                            "protocol_version": WS_PROTOCOL_VERSION,
                            "state_key": snapshot["state_key"],
                            "snapshot": snapshot,
                        },
                    )
            else:
                snapshot = self.store.read(state_key)
                if snapshot is not None:
                    await self._send(
                        connection,
                        {
                            "type": "state",
                            "protocol_version": WS_PROTOCOL_VERSION,
                            "state_key": state_key,
                            "snapshot": snapshot,
                        },
                    )
            return

        if message_type == "unsubscribe":
            state_key = _non_empty_string(payload.get("state_key"), "state_key")
            connection.subscriptions.discard(state_key)
            await self._send(
                connection,
                {
                    "type": "unsubscribed",
                    "protocol_version": WS_PROTOCOL_VERSION,
                    "request_id": request_id,
                    "state_key": state_key,
                },
            )
            return

        if message_type == "stimulus":
            if connection.role != "writer":
                raise PermissionError("writer role is required for stimulus updates")
            state_key = _non_empty_string(payload.get("state_key"), "state_key")
            event_id = _non_empty_string(payload.get("event_id"), "event_id")
            result = await self.store.update(
                state_key,
                event_id=event_id,
                segments=self._segments_from_payload(payload),
            )
            await self._send(
                connection,
                {
                    "type": "stimulus_ack",
                    "protocol_version": WS_PROTOCOL_VERSION,
                    "request_id": request_id,
                    **result,
                },
            )
            if not result["duplicate"]:
                await self._broadcast_state(result["snapshot"])
            return

        if message_type == "reset":
            if connection.role != "writer":
                raise PermissionError("writer role is required for reset")
            state_key = _non_empty_string(payload.get("state_key"), "state_key")
            result = await self.store.reset(state_key)
            await self._send(
                connection,
                {
                    "type": "reset_ack",
                    "protocol_version": WS_PROTOCOL_VERSION,
                    "request_id": request_id,
                    **result,
                },
            )
            await self._broadcast_reset(state_key)
            return

        raise ValueError(f"unknown message type: {message_type}")

    async def _handle_connection(self, websocket: Any) -> None:
        connection: _Connection | None = None
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            payload = json.loads(raw)
            if not isinstance(payload, Mapping) or payload.get("type") != "hello":
                raise ValueError("first message must be hello")
            if payload.get("protocol_version") != WS_PROTOCOL_VERSION:
                raise ValueError("unsupported protocol_version")
            role = _non_empty_string(payload.get("role"), "role")
            token = payload.get("token", "")
            if not isinstance(token, str) or not self._authorize(role, token):
                await websocket.send(
                    _json_message(
                        {
                            "type": "error",
                            "protocol_version": WS_PROTOCOL_VERSION,
                            "code": "unauthorized",
                            "message": "invalid role or token",
                        }
                    )
                )
                return
            connection = _Connection(
                websocket=websocket,
                role=role,
                client_id=str(payload.get("client_id") or "anonymous"),
            )
            self._connections.append(connection)
            await self._send(
                connection,
                {
                    "type": "hello_ack",
                    "protocol_version": WS_PROTOCOL_VERSION,
                    "role": role,
                    "client_id": connection.client_id,
                    "capabilities": {
                        "read": True,
                        "subscribe": True,
                        "write": role == "writer",
                    },
                    "states": self.store.list_states(),
                },
            )
            async for raw_message in websocket:
                command: Any = None
                try:
                    command = json.loads(raw_message)
                    if not isinstance(command, Mapping):
                        raise TypeError("message must be a JSON object")
                    await self._handle_command(connection, command)
                except PermissionError as exc:
                    await self._error(
                        connection,
                        "forbidden",
                        str(exc),
                        command.get("request_id")
                        if isinstance(command, Mapping)
                        else None,
                    )
                except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                    await self._error(
                        connection,
                        "invalid_request",
                        str(exc),
                        command.get("request_id")
                        if isinstance(command, Mapping)
                        else None,
                    )
                except Exception as exc:  # noqa: BLE001
                    await self._error(
                        connection,
                        "internal_error",
                        str(exc),
                        command.get("request_id")
                        if isinstance(command, Mapping)
                        else None,
                    )
        except Exception:
            if connection is not None:
                try:
                    await self._error(
                        connection, "invalid_handshake", "invalid WebSocket handshake"
                    )
                except Exception:
                    pass
        finally:
            if connection is not None and connection in self._connections:
                self._connections.remove(connection)

    async def serve_forever(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:  # pragma: no cover - compatibility with websockets < 14
            from websockets.server import serve

        async with serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
        ):
            await asyncio.Future()


__all__ = [
    "WS_PROTOCOL_VERSION",
    "NeuroBrainStateStore",
    "NeuroBrainWebSocketServer",
]
