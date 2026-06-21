"""Tests for clusterbench.web.hub — fan-out, bounded replay, pruning.

Covers AC-8 (mid-run connect replays history) and the resilience contract:
broken subscribers are pruned silently, the hub refuses further emits after
close, and history is bounded.
"""
import asyncio
import json
from typing import Any

import pytest

from clusterbench.web.hub import DEFAULT_HISTORY_MAXLEN, WebSocketHub


class _FakeWebSocket:
    """Records every send_text in order so tests can assert on the stream.
    Optional `fail_after` simulates a broken pipe mid-stream."""

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.sent: list[str] = []
        self.accepted = False
        self.closed: int | None = None
        self._fail_after = fail_after
        self._sent_count = 0

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, data: str) -> None:
        if self._fail_after is not None and self._sent_count >= self._fail_after:
            raise RuntimeError("simulated broken pipe")
        self._sent_count += 1
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed = code


def _event(text: str) -> dict[str, Any]:
    parsed = json.loads(text)
    return {"type": parsed["type"], "payload": parsed["payload"]}


# ---------------------------------------------------------------------------
# emit / fan-out
# ---------------------------------------------------------------------------


def test_emit_with_no_subscribers_still_records_history():
    hub = WebSocketHub()

    async def go():
        await hub.emit("run_start", {"run_id": "abc"})

    asyncio.run(go())
    history = asyncio.run(hub.history())
    assert len(history) == 1
    assert _event(history[0]) == {"type": "run_start", "payload": {"run_id": "abc"}}


def test_emit_broadcasts_to_all_subscribers():
    hub = WebSocketHub()
    a, b = _FakeWebSocket(), _FakeWebSocket()

    async def go():
        await hub.subscribe(a)
        await hub.subscribe(b)
        await hub.emit("level_start", {"level": 1})

    asyncio.run(go())
    # Both subscribers got the level_start event (after the empty replay).
    assert any(_event(s)["type"] == "level_start" for s in a.sent)
    assert any(_event(s)["type"] == "level_start" for s in b.sent)


# ---------------------------------------------------------------------------
# subscribe / replay
# ---------------------------------------------------------------------------


def test_subscribe_replays_history_then_lives_on():
    """AC-8: a mid-run connect receives everything it missed, then live events."""
    hub = WebSocketHub()

    async def go():
        # Two events before any client connects.
        await hub.emit("run_start", {"run_id": "abc"})
        await hub.emit("level_start", {"level": 1})
        ws = _FakeWebSocket()
        await hub.subscribe(ws)
        await hub.emit("level_done", {"level": 1})
        return ws

    ws = asyncio.run(go())
    types = [_event(s)["type"] for s in ws.sent]
    # History first (in order), then the live event.
    assert types == ["run_start", "level_start", "level_done"]


def test_subscribe_replay_snapshot_taken_before_registration():
    """Subtle: an emit that lands between the snapshot and the registration
    must still be delivered live. We can't easily reproduce this race exactly,
    but we can assert that all events the client saw were unique and in order."""
    hub = WebSocketHub()

    async def go():
        await hub.emit("a", {"x": 1})
        ws = _FakeWebSocket()
        # Interleave: subscribe, then immediately emit before yielding.
        await hub.subscribe(ws)
        await hub.emit("b", {"x": 2})
        return ws

    ws = asyncio.run(go())
    types = [_event(s)["type"] for s in ws.sent]
    assert types == ["a", "b"]


def test_subscribe_accepts_then_sends():
    """Accept happens before any replay — so the client is in CONNECTED state
    when its first frame arrives."""
    hub = WebSocketHub()
    ws = _FakeWebSocket()

    async def go():
        await hub.emit("x", {"y": 1})
        await hub.subscribe(ws)

    asyncio.run(go())
    assert ws.accepted is True
    assert len(ws.sent) == 1


# ---------------------------------------------------------------------------
# Broken subscriber pruning
# ---------------------------------------------------------------------------


def test_broken_subscriber_is_pruned_silently():
    """A subscriber whose send_text raises is removed from the fan-out; the
    emit still completes successfully for other subscribers."""
    hub = WebSocketHub()
    broken = _FakeWebSocket(fail_after=0)  # always fails
    healthy = _FakeWebSocket()

    async def go():
        await hub.subscribe(broken)
        await hub.subscribe(healthy)
        # broken fails on its first live send (its accept replay history is
        # empty here); healthy should still get the event.
        await hub.emit("ping", {"n": 1})
        # The next emit should NOT go to broken (it's been pruned).
        await hub.emit("ping", {"n": 2})

    asyncio.run(go())
    healthy_types = [_event(s)["type"] for s in healthy.sent]
    assert healthy_types == ["ping", "ping"]
    assert hub.n_subscribers == 1


# ---------------------------------------------------------------------------
# Bounded history
# ---------------------------------------------------------------------------


def test_history_is_bounded_by_maxlen():
    """Long runs must not grow memory unbounded. The deque drops oldest."""
    hub = WebSocketHub(history_maxlen=3)

    async def go():
        for i in range(5):
            await hub.emit("tick", {"i": i})

    asyncio.run(go())
    history = asyncio.run(hub.history())
    assert len(history) == 3
    # The first two were dropped; the last three are kept in order.
    payloads = [_event(h)["payload"]["i"] for h in history]
    assert payloads == [2, 3, 4]


def test_history_maxlen_default_is_set():
    """Sanity check: the production default is the documented 5000."""
    assert DEFAULT_HISTORY_MAXLEN == 5000


# ---------------------------------------------------------------------------
# close / lifecycle
# ---------------------------------------------------------------------------


def test_close_drops_subscribers_and_refuses_further_emits():
    hub = WebSocketHub()
    ws = _FakeWebSocket()

    async def go():
        await hub.subscribe(ws)
        await hub.close()
        # After close, emits are no-ops and don't appear in history.
        await hub.emit("late", {"x": 1})

    asyncio.run(go())
    assert ws.closed is not None
    history = asyncio.run(hub.history())
    assert all(_event(h)["type"] != "late" for h in history)
    assert hub.closed is True


def test_close_is_idempotent():
    hub = WebSocketHub()

    async def go():
        await hub.close()
        await hub.close()  # second close is a no-op

    asyncio.run(go())
    assert hub.closed is True


def test_close_closes_all_live_subscribers():
    hub = WebSocketHub()
    a, b = _FakeWebSocket(), _FakeWebSocket()

    async def go():
        await hub.subscribe(a)
        await hub.subscribe(b)
        await hub.close()

    asyncio.run(go())
    assert a.closed is not None
    assert b.closed is not None


# ---------------------------------------------------------------------------
# unsubscribe
# ---------------------------------------------------------------------------


def test_unsubscribe_stops_further_emits_to_that_ws():
    hub = WebSocketHub()
    ws = _FakeWebSocket()

    async def go():
        await hub.subscribe(ws)
        await hub.unsubscribe(ws)
        await hub.emit("after", {"x": 1})

    asyncio.run(go())
    types = [_event(s)["type"] for s in ws.sent]
    assert "after" not in types


def test_unsubscribe_is_idempotent():
    hub = WebSocketHub()
    ws = _FakeWebSocket()

    async def go():
        await hub.subscribe(ws)
        await hub.unsubscribe(ws)
        await hub.unsubscribe(ws)  # safe to call twice

    asyncio.run(go())
    assert hub.n_subscribers == 0


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


def test_event_serialization_format():
    """Wire frame is the documented {"type":..., "payload":...} JSON."""
    hub = WebSocketHub()

    async def go():
        await hub.emit("knee", {"level": 4, "reason": "p99"})

    asyncio.run(go())
    history = asyncio.run(hub.history())
    assert len(history) == 1
    frame = json.loads(history[0])
    assert frame == {"type": "knee", "payload": {"level": 4, "reason": "p99"}}
