"""Unified design system for the Noema CLI.

Centralizes brand colors, typography, and reusable Rich primitives so every
command renders with a consistent identity: violet brand accents, cyan data
labels, and semantic status colors (green/yellow/red).
"""

from __future__ import annotations

import io
import sys
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

BRAND = "#a78bfa"
ACCENT = "#22d3ee"
OK = "#34d399"
WARN = "#fbbf24"
ERR = "#f87171"
PATH = "#f472b6"
TEXT = "#e2e8f0"
DIM = "#64748b"

THEME = Theme(
    {
        "brand": f"bold {BRAND}",
        "brand.dim": f"dim {BRAND}",
        "accent": ACCENT,
        "ok": f"bold {OK}",
        "ok.dim": DIM,
        "warn": f"bold {WARN}",
        "err": f"bold {ERR}",
        "dim": DIM,
        "path": PATH,
        "key": f"bold {ACCENT}",
        "val": TEXT,
    }
)


def _encodable(ch: str) -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        ch.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


EMDASH = "—" if _encodable("—") else "--"
ELLIPSIS = "…" if _encodable("…") else "..."
MIDDOT = "·" if _encodable("·") else "-"
ARROW = "→" if _encodable("→") else "->"
BULLET = "•" if _encodable("•") else "*"
STATUS_OK = "✓" if _encodable("✓") else "OK"
STATUS_ERR = "✗" if _encodable("✗") else "FAIL"
STATUS_WARN = "⚠" if _encodable("⚠") else "!!"
STATUS_DOT = "●" if _encodable("●") else "*"

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is None:
        continue
    with suppress(ValueError, io.UnsupportedOperation):
        reconfigure(encoding="utf-8", errors="replace")

console = Console(theme=THEME)

LOGO = """\
   n   n   oooo   eeee   m   m    aaa
   nn  n  o    o  e    mm mm   a   a
   n n n  o    o  eee   m m m  aaaaa
   n  nn  o    o  e     m   m  a   a
   n   n   oooo   eeee  m   m  a   a"""

TAGLINE = "Генерация мощных технических решений на любом стеке"


def print_banner() -> None:
    console.print(Text(LOGO, style="brand"))
    console.print(Text(f"  {TAGLINE}", style="brand.dim"))
    console.print()


def ok(message: str) -> None:
    console.print(f"[ok]{STATUS_OK}[/ok] {message}")


def err(message: str) -> None:
    console.print(f"[err]{STATUS_ERR}[/err] {message}")


def warn(message: str) -> None:
    console.print(f"[warn]{STATUS_WARN}[/warn] {message}")


def info(message: str) -> None:
    console.print(f"[accent]{BULLET}[/accent] {message}")


def panel(
    body: str,
    title: str,
    border: str = "brand",
    expand: bool = False,
) -> None:
    console.print(
        Panel(
            body,
            title=Text(title, style=f"bold {BRAND}"),
            border_style=border,
            title_align="left",
            expand=expand,
        )
    )


def kv_panel(title: str, rows: list[tuple[str, str]], border: str = "brand") -> None:
    lines = "\n".join(f"[key]{k}:[/key] [val]{v}[/val]" for k, v in rows)
    panel(lines, title=title, border=border)


def data_table(
    title: str,
    columns: list[str],
    rows: list[list[Any]],
    border: str = "dim",
) -> None:
    table = Table(
        title=Text(title, style=f"bold {BRAND}"),
        title_justify="left",
        header_style=f"bold {ACCENT}",
        border_style=border,
        pad_edge=False,
        expand=False,
    )
    for idx, col in enumerate(columns):
        table.add_column(col, style="accent" if idx == 0 else "val", no_wrap=False)
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    console.print(table)


def section(title: str) -> None:
    console.print(Rule(Text(title, style=f"bold {BRAND}"), style="brand"))


def fmt_duration(ms: float) -> str:
    if ms < 1_000:
        return f"{ms:.0f} ms"
    if ms < 60_000:
        return f"{ms / 1_000:.1f} s"
    return f"{ms / 60_000:.1f} min"


def fmt_path(path: str) -> str:
    return f"[path]{path}[/path]"


@asynccontextmanager
async def spinner(message: str) -> AsyncIterator[None]:
    with console.status(f"[brand]{message}[/brand]", spinner="dots"):
        yield


def task_progress(description: str) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[brand]{task.description}[/brand]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def human_rows(data: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(k).replace("_", " ").title(), str(v))
        for k, v in data.items()
        if not isinstance(v, (dict, list))
    ]


def run_async(coro_factory: Callable[[], Any]) -> None:
    import asyncio

    asyncio.run(coro_factory())
