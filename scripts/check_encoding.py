"""Check repository text files for encoding corruption.

Detects two classes of problems:
  1. Files that are not valid UTF-8 (wrong encoding or binary data).
  2. Mojibake: UTF-8 text that was decoded as cp1251 at some point and
     re-saved, leaving Cyrillic replaced by Serbian/Macedonian-looking
     characters (e.g. ``РџСЂРѕРµРєС‚`` instead of ``Проект``). These are
     reversible via the ``cp1251 -> utf-8`` round trip.

Exit code is 1 if any problem is found, 0 otherwise.
"""

from __future__ import annotations

import pathlib
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

REPLACEMENT = "\ufffd"


def _roundtrip_fix(line: str) -> str | None:
    """Return restored text if ``line`` is UTF-8-read-as-cp1251 mojibake."""
    try:
        restored = line.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if restored == line or not any("\u0400" <= c <= "\u04ff" for c in restored):
        return None
    return restored


def _text_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        if p.suffix.lower() in TEXT_EXTS or p.name in TEXT_NAMES:
            files.append(p)
    return files


def check() -> int:
    problems: list[str] = []
    for p in _text_files():
        data = p.read_bytes()
        rel = p.relative_to(ROOT)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{rel}: not valid UTF-8")
            continue
        if REPLACEMENT in text:
            problems.append(f"{rel}: contains U+FFFD (lossy corruption)")
        for lineno, line in enumerate(text.splitlines(), 1):
            if _roundtrip_fix(line) is not None:
                problems.append(f"{rel}:{lineno}: mojibake (utf-8 decoded as cp1251)")
    if problems:
        print(f"Encoding check failed: {len(problems)} problem(s)")
        for msg in problems:
            print(f"  {msg}")
        return 1
    print("Encoding check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(check())
