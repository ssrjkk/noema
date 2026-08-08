"""Repair reversible mojibake across repository text files.

Applies the ``cp1251 -> utf-8`` round trip ONLY to lines whose restored text
contains plausible content (Cyrillic, box-drawing, arrows, dashes, quotes).
Clean lines — including legitimate Cyrillic — are never touched because the
round trip raises for them.

This script is intentionally conservative; lossy corruptions (C1 control
characters such as U+0098) are NOT auto-fixed and must be repaired by hand.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

SKIP_PARTS = {
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".benchmarks",
    "node_modules",
    "__pycache__",
    ".venv",
    ".noema",
}

TEXT_EXTS = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".proto",
    ".txt",
    ".ini",
    ".cfg",
    ".env",
    ".xml",
    ".html",
    ".sh",
    ".rst",
    ".jinja",
    ".tpl",
    ".values",
    ".dockerignore",
    ".gitignore",
}
TEXT_NAMES = {"Dockerfile", "Makefile", "compose.env", "compose.yaml"}

RESTORED_OK_RE = re.compile(
    "[\u0400-\u04ff\u2500-\u257f\u2190-\u21ff\u2013\u2014\u2018\u2019\u201c\u201d]"
)

#: Lines containing lossy corruption are reported but left untouched.
LOSSY_RE = re.compile("[\u0080-\u009f]")


def _roundtrip(line: str) -> str | None:
    try:
        restored = line.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if restored == line or not RESTORED_OK_RE.search(restored):
        return None
    return restored


def _text_files() -> list[pathlib.Path]:
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        if p.suffix.lower() in TEXT_EXTS or p.name in TEXT_NAMES:
            files.append(p)
    return files


def main() -> int:
    fixed = 0
    lossy = []
    for p in _text_files():
        data = p.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        lines = text.splitlines(keepends=True)
        changed = False
        for i, line in enumerate(lines):
            if LOSSY_RE.search(line):
                lossy.append(f"{p.relative_to(ROOT)}:{i + 1}")
                continue
            restored = _roundtrip(line.rstrip("\r\n"))
            if restored is not None:
                ending = "\n" if line.endswith("\n") else ""
                lines[i] = restored + ending
                changed = True
        if changed:
            p.write_text("".join(lines), encoding="utf-8")
            fixed += 1
            print(f"fixed {p.relative_to(ROOT)}")
    print(f"\n{fixed} file(s) fixed.")
    if lossy:
        print(f"{len(lossy)} lossy line(s) left for manual repair (first 40):")
        for msg in lossy[:40]:
            print(f"  {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
