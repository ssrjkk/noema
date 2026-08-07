"""RFC 7807 Problem Details for HTTP API errors."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


class ProblemResponse(JSONResponse):
    """RFC 7807 problem detail response."""

    media_type = "application/problem+json"

    def __init__(
        self,
        status: int,
        title: str,
        detail: str = "",
        instance: str = "",
        type_: str = "about:blank",
        extra: dict[str, Any] | None = None,
    ) -> None:
        body: dict[str, Any] = {
            "type": type_,
            "title": title,
            "status": status,
            "detail": detail,
            "instance": instance,
        }
        if extra:
            body.update(extra)
        super().__init__(content=body, status_code=status)


def problem_response(
    status: int,
    title: str,
    detail: str = "",
    instance: str = "",
    type_: str = "about:blank",
    extra: dict[str, Any] | None = None,
) -> ProblemResponse:
    """Create RFC 7807 problem response."""
    return ProblemResponse(
        status=status,
        title=title,
        detail=detail,
        instance=instance,
        type_=type_,
        extra=extra,
    )
