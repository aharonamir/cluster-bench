"""WebSocketHub — fan-out + bounded history replay (FR-21/FR-22).

A bounded ring buffer retains the last N events so a mid-run connect can replay
them (AC-8). New connections receive the full snapshot, then continue to receive
live events until they disconnect.

The hub implements the EventEmitter protocol from `orchestrator` — same
`async emit(event_type, payload)` signature — so the orchestrator's events flow
straight into the WebSocket fan-out without an adapter.

Duck-typed `WebSocketLike` (accept / send_text / close) is the only contract the
hub needs from a connection. starlette.WebSocket satisfies it; tests can pass a
fake that records `sent` to assert ordering and content without an HTTP layer.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any, Protocol, runtime_checkable

# Bounded replay buffer size. ~5000 events covers a SOAK that emits many
# per-task outcomes without unbounded memory growth across runs.
DEFAULT_HISTORY_MAXLEN = 5000


@runtime_checkable
class WebSocketLike(Protocol):
    """Subset of starlette.WebSocket the hub actually calls. Duck-typed so
    tests can pass a fake with just these three methods."""

    async def accept(self) -> None: ...

    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = 1000) -> None: ...


def _event_to_text(event_type: str, payload: dict[str, Any]) -> str:
    """Serialize an event for the wire. Format: {"type": ..., "payload": ...}."""
    return json.dumps({"type": event_type, "payload": payload})


class WebSocketHub:
    """Fan-out emitter with bounded history replay (FR-21/FR-22).

    Each `emit()` appends to a bounded deque under a lock, then broadcasts
    outside the lock to all subscribers. Subscribers whose `send_text` raises
    (broken pipe, client gone) are pruned silently rather than tearing down the
    whole fan-out.

    `subscribe()` performs the handshake: accept → snapshot history under lock
    → register → replay snapshot outside lock. The snapshot is taken before
    registration so concurrent emits can't be lost between snapshot and join;
    any emit that lands during the handshake is delivered live after replay.
    """

    def __init__(self, *, history_maxlen: int = DEFAULT_HISTORY_MAXLEN) -> None:
        self._subscribers: set[WebSocketLike] = set()
        self._history: deque[str] = deque(maxlen=history_maxlen)
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def n_subscribers(self) -> int:
        return len(self._subscribers)

    @property
    def closed(self) -> bool:
        return self._closed

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """EventEmitter protocol: record + fan-out."""
        if self._closed:
            return
        text = _event_to_text(event_type, payload)
        async with self._lock:
            self._history.append(text)
            subscribers = list(self._subscribers)
        # Broadcast outside the lock so a slow client can't stall other emits.
        await self._broadcast(subscribers, text)

    async def _broadcast(
        self, subscribers: list[WebSocketLike], text: str
    ) -> None:
        """Send `text` to each subscriber; prune any that fail."""
        if not subscribers:
            return
        dead: list[WebSocketLike] = []
        results = await asyncio.gather(
            *(sub.send_text(text) for sub in subscribers), return_exceptions=True
        )
        for sub, result in zip(subscribers, results):
            if isinstance(result, Exception):
                dead.append(sub)
        if dead:
            async with self._lock:
                for sub in dead:
                    self._subscribers.discard(sub)

    async def subscribe(self, ws: WebSocketLike) -> None:
        """Accept a new connection, replay history, and register for live
        updates. Safe to call inside a FastAPI WS route."""
        if self._closed:
            await ws.accept()
            await ws.close(code=1011)
            return
        await ws.accept()
        async with self._lock:
            snapshot = list(self._history)
            self._subscribers.add(ws)
        try:
            for text in snapshot:
                await ws.send_text(text)
        except Exception:
            await self.unsubscribe(ws)

    async def unsubscribe(self, ws: WebSocketLike) -> None:
        """Drop a subscriber. Idempotent — safe to call twice or for a
        connection that was never registered (e.g. pruned earlier)."""
        async with self._lock:
            self._subscribers.discard(ws)

    async def history(self) -> list[str]:
        """Return a copy of the bounded history (serialized events)."""
        async with self._lock:
            return list(self._history)

    async def close(self) -> None:
        """Tear down: close every connection and refuse further emits.
        Idempotent."""
        if self._closed:
            return
        self._closed = True
        async with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        # Best-effort close; ignore failures.
        await asyncio.gather(
            *(sub.close() for sub in subscribers), return_exceptions=True
        )


__all__ = ["WebSocketHub", "WebSocketLike", "DEFAULT_HISTORY_MAXLEN"]
