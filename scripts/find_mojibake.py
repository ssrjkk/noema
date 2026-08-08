"""Detect all mojibake / encoding-artifact lines in the repository.

Catches the classes that ``check_encoding.py`` misses:
  1. cp1251 round-trip mojibake whose restored text is NOT Cyrillic
     (box-drawing, arrows, em-dashes, quotes) - e.g. ``в“Ђв“Ђ`` for ``──``.
  2. Lossy corruption embedding cp1251 control/undefined code points
     (e.g. ``U+0098``) produced by a second corruption pass.

Report only; never edits files.
"""

from __future__ import annotations

import pathlib
import re
import sys

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

#: cp1251 C1 control / undefined code points (lossy second-pass corruption).
LOSSY_RE = re.compile("[\u0080-\u009f]")

#: Plausible restored content for round-trip mojibake: Cyrillic, box-drawing,
#: arrows, dashes, curly quotes. If the restored text contains only ASCII,
#: the original line was probably clean and the round trip is coincidental.
RESTORED_OK_RE = re.compile(
    "[\u0400-\u04ff\u2500-\u257f\u2190-\u21ff\u2013\u2014\u2018\u2019\u201c\u201d]"
)


def _roundtrip(line: str) -> str | None:
    try:
        restored = line.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if restored == line:
        return None
    if not RESTORED_OK_RE.search(restored):
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
    problems = []
    for p in _text_files():
        data = p.read_bytes()
        rel = p.relative_to(ROOT)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{rel}: NOT VALID UTF-8")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if LOSSY_RE.search(line):
                problems.append(
                    f"{rel}:{lineno}: lossy control char {[hex(ord(c)) for c in line if LOSSY_RE.search(c)][:6]}"
                )
                continue
            restored = _roundtrip(line)
            if restored is not None:
                problems.append(f"{rel}:{lineno}: mojibake -> {restored!r}")
    if problems:
        print(f"Found {len(problems)} problem line(s):")
        for msg in problems:
            print(f"  {msg}")
        return 1
    print("No mojibake found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
