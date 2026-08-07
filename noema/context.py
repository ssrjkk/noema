"""Context variables for tenant/session isolation in asyncio."""

from __future__ import annotations

import contextvars

tenant_context: contextvars.ContextVar[str] = contextvars.ContextVar("tenant_id", default="default")
session_context: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")
request_id_context: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def get_tenant_id() -> str:
    return tenant_context.get()


def set_tenant_id(tenant_id: str) -> contextvars.Token:
    return tenant_context.set(tenant_id)


def reset_tenant_id(token: contextvars.Token) -> None:
    tenant_context.reset(token)


def get_session_id() -> str:
    return session_context.get()


def set_session_id(session_id: str) -> contextvars.Token:
    return session_context.set(session_id)


def get_request_id() -> str:
    return request_id_context.get()


def set_request_id(request_id: str) -> contextvars.Token:
    return request_id_context.set(request_id)
