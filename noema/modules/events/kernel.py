"""Event-driven architecture module — event sourcing, CQRS, event bus, saga pattern."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class Event:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    aggregate_id: str = ""
    version: int = 1


@dataclass
class SagaStep:
    name: str
    action: str
    compensation: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SagaConfig:
    name: str
    steps: list[SagaStep] = field(default_factory=list)
    trigger_event: str = ""
    result_event: str = ""


class EventStore:
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._subscribers: dict[str, list[Callable[[Event], Any]]] = {}
        self._snapshots: dict[str, list[Event]] = {}

    def append(self, event: Event) -> Event:
        self._events.append(event)
        if event.aggregate_id not in self._snapshots:
            self._snapshots[event.aggregate_id] = []
        self._snapshots[event.aggregate_id].append(event)

        if event.type in self._subscribers:
            for handler in self._subscribers[event.type]:
                handler(event)
        if "*" in self._subscribers:
            for handler in self._subscribers["*"]:
                handler(event)

        return event

    def get_events(
        self, aggregate_id: str | None = None, event_type: str | None = None, from_version: int = 0
    ) -> list[Event]:
        results = self._events
        if aggregate_id:
            results = [e for e in results if e.aggregate_id == aggregate_id]
        if event_type:
            results = [e for e in results if e.type == event_type]
        if from_version > 0:
            results = [e for e in results if e.version >= from_version]
        return results

    def replay(self, aggregate_id: str, from_version: int = 0) -> list[Event]:
        events = self.get_events(aggregate_id=aggregate_id, from_version=from_version)
        return events

    def subscribe(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def get_aggregate_version(self, aggregate_id: str) -> int:
        events = self.get_events(aggregate_id=aggregate_id)
        if events:
            return max(e.version for e in events)
        return 0

    def create_event(self, event_type: str, aggregate_id: str, payload: dict[str, Any]) -> Event:
        version = self.get_aggregate_version(aggregate_id) + 1
        event = Event(
            type=event_type,
            aggregate_id=aggregate_id,
            payload=payload,
            version=version,
        )
        return self.append(event)


class CommandBus:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register(self, command_type: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        self._handlers[command_type] = handler

    def dispatch(self, command_type: str, payload: dict[str, Any]) -> Any:
        if command_type not in self._handlers:
            raise ValueError(f"No handler registered for command: {command_type}")
        return self._handlers[command_type](payload)

    def has_handler(self, command_type: str) -> bool:
        return command_type in self._handlers

    def list_handlers(self) -> list[str]:
        return list(self._handlers.keys())


class QueryBus:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register(self, query_type: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        self._handlers[query_type] = handler

    def execute(self, query_type: str, params: dict[str, Any]) -> Any:
        if query_type not in self._handlers:
            raise ValueError(f"No handler registered for query: {query_type}")
        return self._handlers[query_type](params)

    def has_handler(self, query_type: str) -> bool:
        return query_type in self._handlers

    def list_handlers(self) -> list[str]:
        return list(self._handlers.keys())


class Saga:
    def __init__(self, config: SagaConfig) -> None:
        self.config = config
        self._executed_steps: list[str] = []
        self._compensated_steps: list[str] = []
        self._status = "pending"
        self._results: dict[str, Any] = {}

    @property
    def status(self) -> str:
        return self._status

    @property
    def executed_steps(self) -> list[str]:
        return list(self._executed_steps)

    @property
    def results(self) -> dict[str, Any]:
        return dict(self._results)

    def execute_step(self, step_index: int, result: Any = None) -> dict[str, Any]:
        if step_index >= len(self.config.steps):
            return {"error": "Step index out of range"}

        step = self.config.steps[step_index]
        self._executed_steps.append(step.name)
        self._results[step.name] = result
        self._status = "in_progress"

        return {
            "step": step.name,
            "action": step.action,
            "payload": step.payload,
            "status": "completed",
        }

    def compensate(self, from_step: int = -1) -> list[dict[str, Any]]:
        if from_step == -1:
            from_step = len(self._executed_steps) - 1

        compensations = []
        step_map = {s.name: s for s in self.config.steps}

        for i in range(from_step, -1, -1):
            step_name = self._executed_steps[i] if i < len(self._executed_steps) else None
            if step_name and step_name in step_map:
                step = step_map[step_name]
                compensations.append(
                    {
                        "step": step.name,
                        "compensation": step.compensation,
                        "status": "compensated",
                    }
                )
                self._compensated_steps.append(step_name)

        self._status = "compensated" if compensations else self._status
        return compensations

    def complete(self) -> None:
        self._status = "completed"

    def fail(self) -> None:
        self._status = "failed"
        self.compensate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "status": self._status,
            "trigger_event": self.config.trigger_event,
            "result_event": self.config.result_event,
            "steps": [
                {
                    "name": s.name,
                    "action": s.action,
                    "compensation": s.compensation,
                    "executed": s.name in self._executed_steps,
                    "compensated": s.name in self._compensated_steps,
                }
                for s in self.config.steps
            ],
        }


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Event], Any]]] = {}
        self._event_log: list[Event] = []

    def publish(self, event: Event) -> None:
        self._event_log.append(event)
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                handler(event)

    def subscribe(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def get_log(self) -> list[Event]:
        return list(self._event_log)


class EventsModule:
    NAME = "events"
    DESCRIPTION = "Event-driven architecture: event sourcing, CQRS, event bus, saga pattern"

    def __init__(self) -> None:
        self.event_store = EventStore()
        self.command_bus = CommandBus()
        self.query_bus = QueryBus()
        self.event_bus = EventBus()

    def execute(self, task: Any) -> dict[str, Any]:
        task_title = getattr(task, "title", "")
        task_tags = getattr(task, "tags", [])

        action = "sourcing"
        if "cqrs" in str(task_title).lower() or "command" in task_tags or "query" in task_tags:
            action = "cqrs"
        elif "saga" in str(task_title).lower() or "saga" in task_tags:
            action = "saga"
        elif "event" in str(task_title).lower() and "bus" in str(task_title).lower():
            action = "bus"

        metadata = {}
        if hasattr(task, "metadata"):
            metadata = task.metadata

        if action == "sourcing":
            events_config = metadata.get(
                "events",
                [
                    {
                        "type": "OrderCreated",
                        "payload": {"order_id": "123", "items": ["item1"], "total": 99.99},
                    },
                    {
                        "type": "PaymentProcessed",
                        "payload": {"order_id": "123", "amount": 99.99, "method": "credit_card"},
                    },
                    {
                        "type": "OrderShipped",
                        "payload": {"order_id": "123", "tracking": "TRACK-001"},
                    },
                ],
            )
            created_events = []
            for ev_cfg in events_config:
                event = self.event_store.create_event(
                    event_type=ev_cfg.get("type", "UnknownEvent"),
                    aggregate_id=ev_cfg.get(
                        "aggregate_id", ev_cfg.get("payload", {}).get("order_id", "default")
                    ),
                    payload=ev_cfg.get("payload", {}),
                )
                created_events.append(
                    {
                        "id": event.id,
                        "type": event.type,
                        "aggregate_id": event.aggregate_id,
                        "version": event.version,
                        "timestamp": event.timestamp,
                    }
                )

            return {
                "action": "event_sourcing",
                "events_stored": len(created_events),
                "events": created_events,
                "total_events_in_store": len(self.event_store.get_events()),
                "_confidence": 0.85,
            }

        elif action == "cqrs":
            commands = metadata.get(
                "commands",
                [
                    {"type": "CreateOrder", "handler": "create_order_handler"},
                    {"type": "ProcessPayment", "handler": "process_payment_handler"},
                    {"type": "ShipOrder", "handler": "ship_order_handler"},
                ],
            )
            queries = metadata.get(
                "queries",
                [
                    {"type": "GetOrder", "handler": "get_order_handler"},
                    {"type": "ListOrders", "handler": "list_orders_handler"},
                    {"type": "GetOrderStatus", "handler": "get_order_status_handler"},
                ],
            )

            for cmd in commands:
                self.command_bus.register(
                    cmd["type"], lambda p: {"status": "executed", "command": p}
                )
            for q in queries:
                self.query_bus.register(q["type"], lambda p: {"result": "data", "query": p})

            return {
                "action": "cqrs",
                "commands_registered": self.command_bus.list_handlers(),
                "queries_registered": self.query_bus.list_handlers(),
                "_confidence": 0.80,
            }

        elif action == "saga":
            steps_config = metadata.get(
                "steps",
                [
                    {
                        "name": "ReserveInventory",
                        "action": "reserve_inventory",
                        "compensation": "release_inventory",
                    },
                    {
                        "name": "ProcessPayment",
                        "action": "charge_payment",
                        "compensation": "refund_payment",
                    },
                    {
                        "name": "ShipOrder",
                        "action": "create_shipment",
                        "compensation": "cancel_shipment",
                    },
                    {
                        "name": "SendConfirmation",
                        "action": "send_email",
                        "compensation": "send_cancellation_email",
                    },
                ],
            )

            saga_steps = [
                SagaStep(
                    name=s["name"],
                    action=s["action"],
                    compensation=s["compensation"],
                    payload=s.get("payload", {}),
                )
                for s in steps_config
            ]

            saga_config = SagaConfig(
                name=metadata.get("saga_name", "OrderProcessingSaga"),
                steps=saga_steps,
                trigger_event=metadata.get("trigger_event", "OrderCreated"),
                result_event=metadata.get("result_event", "OrderCompleted"),
            )

            saga = Saga(saga_config)
            for i in range(len(saga_steps)):
                saga.execute_step(i, result={"status": "success"})
            saga.complete()

            return {
                "action": "saga",
                "saga": saga.to_dict(),
                "_confidence": 0.80,
            }

        elif action == "bus":
            events = metadata.get(
                "events",
                [
                    {
                        "type": "UserRegistered",
                        "payload": {"user_id": "u1", "email": "user@example.com"},
                    },
                ],
            )
            published = []
            for ev in events:
                event = Event(type=ev["type"], payload=ev.get("payload", {}))
                self.event_bus.publish(event)
                published.append({"type": event.type, "id": event.id})

            return {
                "action": "event_bus",
                "events_published": published,
                "log_size": len(self.event_bus.get_log()),
                "_confidence": 0.80,
            }

        return {
            "action": "sourcing",
            "message": "Defaulting to event sourcing demonstration",
            "_confidence": 0.50,
        }
