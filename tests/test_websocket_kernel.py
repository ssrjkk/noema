"""Tests for noema.modules.websocket.kernel — WebSocket module core."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from noema.modules.websocket.kernel import (
    ConnectionManager,
    EventBus,
    RoomManager,
    WebSocketModule,
    WSClient,
    WSEventType,
    WSMessage,
)

# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class TestEventBusUnsubscribe:
    """Lines 74-75: unsubscribe removes a handler when event exists."""

    def test_unsubscribe_removes_handler(self):
        bus = EventBus()
        handler = MagicMock()
        bus.subscribe("msg", handler)
        assert handler in bus._handlers["msg"]

        bus.unsubscribe("msg", handler)
        assert handler not in bus._handlers["msg"]

    def test_unsubscribe_keeps_other_handlers(self):
        bus = EventBus()
        h1 = MagicMock()
        h2 = MagicMock()
        bus.subscribe("msg", h1)
        bus.subscribe("msg", h2)

        bus.unsubscribe("msg", h1)
        assert h1 not in bus._handlers["msg"]
        assert h2 in bus._handlers["msg"]

    def test_unsubscribe_noop_when_event_missing(self):
        bus = EventBus()
        kept = MagicMock()
        bus.subscribe("known", kept)
        bus.unsubscribe("nonexistent", MagicMock())
        assert "nonexistent" not in bus._handlers
        assert bus._handlers["known"] == [kept]


class TestEventBusPublishHistoryTrim:
    """Line 81: history is trimmed when it exceeds _max_history."""

    @pytest.mark.asyncio
    async def test_history_trimmed_at_max(self):
        bus = EventBus()
        bus._max_history = 5

        for i in range(10):
            await bus.publish("evt", data=i)

        assert len(bus._history) == 5
        # Should keep the most recent 5
        assert bus._history[0].data == 5
        assert bus._history[-1].data == 9


class TestEventBusPublishAsyncHandler:
    """Line 88: async handlers are awaited."""

    @pytest.mark.asyncio
    async def test_async_handler_is_awaited(self):
        bus = EventBus()
        received = []

        async def async_handler(msg):
            received.append(msg)

        bus.subscribe("evt", async_handler)
        count = await bus.publish("evt", data="hello")

        assert count == 1
        assert len(received) == 1
        assert received[0].data == "hello"


class TestEventBusPublishHandlerException:
    """Lines 92-93: handler exceptions are caught and logged."""

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_crash(self):
        bus = EventBus()

        def bad_handler(msg):
            raise RuntimeError("boom")

        bus.subscribe("evt", bad_handler)
        # Should not raise
        count = await bus.publish("evt", data="test")
        assert count == 0

    @pytest.mark.asyncio
    async def test_handler_exception_still_counts_others(self):
        bus = EventBus()
        good_received = []

        def bad_handler(msg):
            raise ValueError("fail")

        def good_handler(msg):
            good_received.append(msg)

        bus.subscribe("evt", bad_handler)
        bus.subscribe("evt", good_handler)
        count = await bus.publish("evt", data="test")

        assert count == 1
        assert len(good_received) == 1


class TestEventBusGetHistory:
    """Lines 97-100: get_history with event filter and limit."""

    @pytest.mark.asyncio
    async def test_get_history_no_filter(self):
        bus = EventBus()
        await bus.publish("a", data=1)
        await bus.publish("b", data=2)
        await bus.publish("a", data=3)

        msgs = bus.get_history()
        assert len(msgs) == 3

    @pytest.mark.asyncio
    async def test_get_history_with_event_filter(self):
        bus = EventBus()
        await bus.publish("a", data=1)
        await bus.publish("b", data=2)
        await bus.publish("a", data=3)

        msgs = bus.get_history(event="a")
        assert len(msgs) == 2
        assert all(m.event == "a" for m in msgs)

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self):
        bus = EventBus()
        for i in range(10):
            await bus.publish("evt", data=i)

        msgs = bus.get_history(limit=3)
        assert len(msgs) == 3
        # Should return the last 3
        assert msgs[0].data == 7
        assert msgs[-1].data == 9

    @pytest.mark.asyncio
    async def test_get_history_filter_and_limit(self):
        bus = EventBus()
        for i in range(10):
            evt = "a" if i % 2 == 0 else "b"
            await bus.publish(evt, data=i)

        msgs = bus.get_history(event="a", limit=2)
        assert len(msgs) == 2


# ---------------------------------------------------------------------------
# RoomManager
# ---------------------------------------------------------------------------


class TestRoomManagerJoinFull:
    """Line 117: join returns False when room is at max_size."""

    def test_join_rejected_when_room_full(self):
        rm = RoomManager()
        rm.create_room("small", max_size=2)
        assert rm.join("small", "c1") is True
        assert rm.join("small", "c2") is True
        assert rm.join("small", "c3") is False

    def test_join_unlimited_room(self):
        rm = RoomManager()
        rm.create_room("big", max_size=0)
        for i in range(100):
            assert rm.join("big", f"c{i}") is True


class TestRoomManagerLeave:
    """Lines 122-128: leave removes client, deletes empty room, handles missing room."""

    def test_leave_existing_room(self):
        rm = RoomManager()
        rm.join("lobby", "c1")
        rm.join("lobby", "c2")

        result = rm.leave("lobby", "c1")
        assert result is True
        assert "c1" not in rm.rooms["lobby"].clients
        assert "c2" in rm.rooms["lobby"].clients

    def test_leave_deletes_empty_room(self):
        rm = RoomManager()
        rm.join("temp", "c1")
        assert "temp" in rm.rooms

        rm.leave("temp", "c1")
        assert "temp" not in rm.rooms

    def test_leave_nonexistent_room(self):
        rm = RoomManager()
        result = rm.leave("ghost", "c1")
        assert result is False


class TestRoomManagerGetClientRooms:
    """Line 135: get_client_rooms returns rooms a client belongs to."""

    def test_get_client_rooms(self):
        rm = RoomManager()
        rm.join("r1", "c1")
        rm.join("r2", "c1")
        rm.join("r2", "c2")

        rooms = rm.get_client_rooms("c1")
        assert set(rooms) == {"r1", "r2"}

    def test_get_client_rooms_unknown_client(self):
        rm = RoomManager()
        rm.join("r1", "c1")
        rooms = rm.get_client_rooms("unknown")
        assert rooms == []


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------


class TestConnectionManagerOn:
    """Line 164: on() registers event hooks."""

    def test_on_registers_hook(self):
        cm = ConnectionManager()
        handler = MagicMock()
        cm.on("custom_event", handler)
        assert handler in cm._hooks["custom_event"]

    def test_on_creates_new_hook_list(self):
        cm = ConnectionManager()
        handler = MagicMock()
        cm.on("new_event", handler)
        assert "new_event" in cm._hooks
        assert cm._hooks["new_event"] == [handler]


class TestConnectionManagerConnect:
    """Lines 167-169: connect creates and stores a client."""

    def test_connect_creates_client(self):
        cm = ConnectionManager()
        client = cm.connect("c1", user_id="u1")
        assert isinstance(client, WSClient)
        assert client.id == "c1"
        assert client.user_id == "u1"
        assert cm.clients["c1"] is client

    def test_connect_default_user_id(self):
        cm = ConnectionManager()
        client = cm.connect("c1")
        assert client.user_id == ""


class TestConnectionManagerDisconnect:
    """Line 172: disconnect removes a client."""

    def test_disconnect_removes_client(self):
        cm = ConnectionManager()
        cm.connect("c1")
        assert "c1" in cm.clients

        cm.disconnect("c1")
        assert "c1" not in cm.clients

    def test_disconnect_nonexistent_is_noop(self):
        cm = ConnectionManager()
        cm.connect("kept")
        cm.disconnect("ghost")
        assert list(cm.clients) == ["kept"]


class TestConnectionManagerGetClient:
    """Line 175: get_client retrieves a client or returns None."""

    def test_get_client_existing(self):
        cm = ConnectionManager()
        cm.connect("c1")
        client = cm.get_client("c1")
        assert client is not None
        assert client.id == "c1"

    def test_get_client_missing(self):
        cm = ConnectionManager()
        assert cm.get_client("ghost") is None


class TestConnectionManagerEnqueue:
    """Lines 178-180: _enqueue appends and trims overflow."""

    def test_enqueue_normal(self):
        cm = ConnectionManager()
        client = cm.connect("c1")
        msg = WSMessage(data="hello")
        cm._enqueue(client, msg)
        assert len(client.outbox) == 1
        assert client.outbox[0] is msg

    def test_enqueue_trims_overflow(self):
        cm = ConnectionManager()
        client = cm.connect("c1")
        cm.MAX_OUTBOX = 3  # Lower for testing

        for i in range(5):
            cm._enqueue(client, WSMessage(data=i))

        assert len(client.outbox) == 3
        # Should keep the last 3
        assert client.outbox[0].data == 2
        assert client.outbox[-1].data == 4


class TestConnectionManagerBroadcast:
    """Lines 183-190: broadcast sends to all clients, optionally excluding one."""

    def test_broadcast_to_all(self):
        cm = ConnectionManager()
        cm.connect("c1")
        cm.connect("c2")
        cm.connect("c3")

        count = cm.broadcast("hello everyone")
        assert count == 3
        for cid in ["c1", "c2", "c3"]:
            assert len(cm.clients[cid].outbox) == 1
            assert cm.clients[cid].outbox[0].data == "hello everyone"

    def test_broadcast_with_exclude(self):
        cm = ConnectionManager()
        cm.connect("c1")
        cm.connect("c2")
        cm.connect("c3")

        count = cm.broadcast("hello", exclude="c2")
        assert count == 2
        assert len(cm.clients["c2"].outbox) == 0
        assert len(cm.clients["c1"].outbox) == 1
        assert len(cm.clients["c3"].outbox) == 1

    def test_broadcast_empty(self):
        cm = ConnectionManager()
        count = cm.broadcast("nobody here")
        assert count == 0


class TestConnectionManagerSend:
    """Lines 193-200: send delivers to a specific client, returns False if missing."""

    def test_send_to_existing_client(self):
        cm = ConnectionManager()
        cm.connect("c1")

        result = cm.send("c1", "private msg")
        assert result is True
        assert len(cm.clients["c1"].outbox) == 1
        msg = cm.clients["c1"].outbox[0]
        assert msg.data == "private msg"
        assert msg.target == "c1"
        assert msg.event == WSEventType.MESSAGE.value

    def test_send_to_missing_client(self):
        cm = ConnectionManager()
        result = cm.send("ghost", "msg")
        assert result is False


class TestConnectionManagerDrain:
    """Lines 203-207: drain returns and clears outbox, handles missing client."""

    def test_drain_returns_and_clears(self):
        cm = ConnectionManager()
        cm.connect("c1")
        cm.send("c1", "msg1")
        cm.send("c1", "msg2")

        pending = cm.drain("c1")
        assert len(pending) == 2
        assert cm.clients["c1"].outbox == []

    def test_drain_empty_outbox(self):
        cm = ConnectionManager()
        cm.connect("c1")
        pending = cm.drain("c1")
        assert pending == []

    def test_drain_missing_client(self):
        cm = ConnectionManager()
        pending = cm.drain("ghost")
        assert pending == []


# ---------------------------------------------------------------------------
# WebSocketModule.execute
# ---------------------------------------------------------------------------


class TestWebSocketModuleExecute:
    """Lines 231-232: execute adds redis_adapter/sticky_sessions for scale/distributed tags."""

    def test_execute_with_scale_tag(self):
        mod = WebSocketModule()
        task = MagicMock()
        task.tags = ["scale"]

        result = mod.execute(task)
        assert "redis_adapter" in result["features"]
        assert "sticky_sessions" in result["features"]
        assert result["type"] == "websocket"
        assert result["transport"] == "ws"

    def test_execute_with_distributed_tag(self):
        mod = WebSocketModule()
        task = MagicMock()
        task.tags = ["distributed"]

        result = mod.execute(task)
        assert "redis_adapter" in result["features"]
        assert "sticky_sessions" in result["features"]

    def test_execute_without_scale_tags(self):
        mod = WebSocketModule()
        task = MagicMock()
        task.tags = []

        result = mod.execute(task)
        assert "redis_adapter" not in result["features"]
        assert "sticky_sessions" not in result["features"]
        assert len(result["features"]) == 4  # base features only

    def test_execute_with_secure_tag(self):
        mod = WebSocketModule()
        task = MagicMock()
        task.tags = ["secure"]

        result = mod.execute(task)
        assert result["transport"] == "wss"

    def test_execute_includes_stats(self):
        mod = WebSocketModule()
        mod.connections.connect("c1")
        mod.rooms.join("lobby", "c1")

        task = MagicMock()
        task.tags = []

        result = mod.execute(task)
        assert result["connection_stats"]["connected"] == 1
        assert result["room_stats"]["rooms"] == 1
        assert result["_confidence"] == 0.85
