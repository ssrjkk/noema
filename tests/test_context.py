"""Tests for context variables module."""

from __future__ import annotations

from noema.context import (
    get_request_id,
    get_session_id,
    get_tenant_id,
    reset_tenant_id,
    set_request_id,
    set_session_id,
    set_tenant_id,
)


def test_get_tenant_id_default():
    assert get_tenant_id() == "default"


def test_set_and_get_tenant_id():
    token = set_tenant_id("test-tenant")
    try:
        assert get_tenant_id() == "test-tenant"
    finally:
        reset_tenant_id(token)


def test_reset_tenant_id():
    token = set_tenant_id("temp-tenant")
    assert get_tenant_id() == "temp-tenant"
    reset_tenant_id(token)
    assert get_tenant_id() == "default"


def test_get_session_id_default():
    assert get_session_id() == ""


def test_set_and_get_session_id():
    set_session_id("session-123")
    try:
        assert get_session_id() == "session-123"
    finally:
        set_session_id("")


def test_get_request_id_default():
    assert get_request_id() == ""


def test_set_and_get_request_id():
    set_request_id("req-456")
    try:
        assert get_request_id() == "req-456"
    finally:
        set_request_id("")
