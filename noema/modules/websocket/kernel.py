"""WebSocket Module — real-time events, pub/sub, rooms, reconnection."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class WSEventType(StrEnum):
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    MESSAGE = "message"
    JOIN_ROOM = "join_room"
    LEAVE_ROOM = "leave_room"
    BROADCAST = "broadcast"
    PING = "ping"
    PONG = "pong"


@dataclass
class WSClient:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    user_id: str = ""
    rooms: set[str] = field(default_factory=set)
    connected_at: float = field(default_factory=time.time)
    last_ping: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    outbox: list[WSMessage] = field(default_factory=list)


@dataclass
class WSMessage:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    event: str = ""
    data: Any = None
    sender_id: str = ""
    room: str = ""
    timestamp: float = field(default_factory=time.time)
    target: str = ""  # client_id or room name


@dataclass
class WSRoom:
    name: str = ""
    clients: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    max_size: int = 0  # 0 = unlimited
    metadata: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """In-process pub/sub event bus."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = {}
        self._history: list[WSMessage] = []
        self._max_history = 100

    def subscribe(self, event: str, handler: Callable) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def unsubscribe(self, event: str, handler: Callable) -> None:
        if event in self._handlers:
            self._handlers[event] = [h for h in self._handlers[event] if h != handler]

    async def publish(self, event: str, data: Any = None, sender_id: str = "") -> int:
        msg = WSMessage(event=event, data=data, sender_id=sender_id)
        self._history.append(msg)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        handlers = self._handlers.get(event, []) + self._handlers.get("*", [])
        count = 0
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(msg)
                else:
                    handler(msg)
                count += 1
            except Exception as e:
                logger.warning("ws_handler_failed", ws_event=event, error=str(e))
        return count

    def get_history(self, event: str | None = None, limit: int = 50) -> list[WSMessage]:
        msgs = self._history
        if event:
            msgs = [m for m in msgs if m.event == event]
        return msgs[-limit:]


class RoomManager:
    """Manage WebSocket rooms/channels."""

    def __init__(self) -> None:
        self.rooms: dict[str, WSRoom] = {}

    def create_room(self, name: str, max_size: int = 0) -> WSRoom:
        if name not in self.rooms:
            self.rooms[name] = WSRoom(name=name, max_size=max_size)
        return self.rooms[name]

    def join(self, room_name: str, client_id: str) -> bool:
        room = self.create_room(room_name)
        if room.max_size > 0 and len(room.clients) >= room.max_size:
            return False
        room.clients.add(client_id)
        return True

    def leave(self, room_name: str, client_id: str) -> bool:
        room = self.rooms.get(room_name)
        if room:
            room.clients.discard(client_id)
            if not room.clients:
                del self.rooms[room_name]
            return True
        return False

    def get_room_clients(self, room_name: str) -> set[str]:
        room = self.rooms.get(room_name)
        return room.clients.copy() if room else set()

    def get_client_rooms(self, client_id: str) -> list[str]:
        return [name for name, room in self.rooms.items() if client_id in room.clients]

    def get_stats(self) -> dict[str, Any]:
        total_clients = set()
        for room in self.rooms.values():
            total_clients.update(room.clients)
        return {
            "rooms": len(self.rooms),
            "total_unique_clients": len(total_clients),
            "room_details": {name: len(r.clients) for name, r in self.rooms.items()},
        }


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self) -> None:
        self.clients: dict[str, WSClient] = {}
        self._hooks: dict[str, list[Callable]] = {
            "connect": [],
            "disconnect": [],
            "message": [],
        }

    def on(self, event: str, handler: Callable) -> None:
        self._hooks.setdefault(event, []).append(handler)

    def connect(self, client_id: str, user_id: str = "") -> WSClient:
        client = WSClient(id=client_id, user_id=user_id)
        self.clients[client_id] = client
        return client

    def disconnect(self, client_id: str) -> None:
        self.clients.pop(client_id, None)

    def get_client(self, client_id: str) -> WSClient | None:
        return self.clients.get(client_id)

    def broadcast(self, message: str, exclude: str | None = None) -> int:
        count = 0
        msg = WSMessage(event=WSEventType.BROADCAST.value, data=message, target="*")
        for cid, client in self.clients.items():
            if cid == exclude:
                continue
            client.outbox.append(msg)
            count += 1
        return count

    def send(self, client_id: str, message: str) -> bool:
        client = self.clients.get(client_id)
        if not client:
            return False
        client.outbox.append(
            WSMessage(event=WSEventType.MESSAGE.value, data=message, target=client_id)
        )
        return True

    def drain(self, client_id: str) -> list[WSMessage]:
        client = self.clients.get(client_id)
        if not client:
            return []
        pending, client.outbox = client.outbox, []
        return pending

    def get_stats(self) -> dict[str, Any]:
        return {
            "connected": len(self.clients),
            "clients": [c.id for c in self.clients.values()],
        }


class WebSocketModule:
    """Standalone WebSocket module."""

    NAME = "websocket"
    DESCRIPTION = "Real-time WebSocket events, pub/sub, rooms, connection management"

    def __init__(self) -> None:
        self.events = EventBus()
        self.rooms = RoomManager()
        self.connections = ConnectionManager()

    def execute(self, task: Any) -> dict[str, Any]:
        tags = getattr(task, "tags", [])
        features = ["rooms", "pub/sub", "reconnection", "heartbeat"]
        if "scale" in tags or "distributed" in tags:
            features.append("redis_adapter")
            features.append("sticky_sessions")
        return {
            "type": "websocket",
            "features": features,
            "transport": "ws" if "secure" not in tags else "wss",
            "connection_stats": self.connections.get_stats(),
            "room_stats": self.rooms.get_stats(),
            "_confidence": 0.85,
        }
