"""Second-stage repair for lossy mojibake (C1 control characters).

A prior cp1251 round-trip was blocked wherever the original byte was 0x98
(U+0098 is undefined in cp1251). This script maps C1 characters to their
raw byte via surrogateescape, then decodes the result as UTF-8, which
recovers the true Cyrillic text (e.g. ``Р\\x98СЃ...`` -> ``Используйте``).

Only lines whose restored text is plausible are replaced; everything else
is reported for manual repair.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

C1_RE = re.compile("[\u0080-\u009f]")
PLAUSIBLE_RE = re.compile(
    "[\u0400-\u04ff\u2500-\u257f\u2190-\u21ff\u2013\u2014\u2018\u2019\u201c\u201d]"
)
BAD_AFTER_RE = re.compile("[\u0080-\u009f\u00ff\ufffd]")

TARGETS = [
    pathlib.Path("noema_knowledge.json"),
    pathlib.Path("noema"),
]


def _restore(line: str) -> str | None:
    text = C1_RE.sub(lambda m: chr(0xDC00 + ord(m.group(0))), line)
    try:
        raw = text.encode("cp1251", errors="surrogateescape")
    except UnicodeEncodeError:
        return None
    if not raw or not any(0x80 <= b <= 0x9F for b in raw):
        return None
    try:
        restored = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if BAD_AFTER_RE.search(restored) or not PLAUSIBLE_RE.search(restored):
        return None
    return restored


def main() -> int:
    files = []
    for base in TARGETS:
        p = ROOT / base
        if p.is_dir():
            files.extend(p.rglob("*.py"))
        elif p.is_file():
            files.append(p)

    remaining = []
    total = 0
    for p in files:
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        changed = False
        for i, line in enumerate(lines):
            if not C1_RE.search(line):
                continue
            body = line.rstrip("\r\n")
            restored = _restore(body)
            if restored is not None:
                ending = "\n" if line.endswith("\n") else ""
                lines[i] = restored + ending
                changed = True
                total += 1
            else:
                remaining.append(f"{p.relative_to(ROOT)}:{i + 1} {line.rstrip()!r}")
        if changed:
            p.write_text("".join(lines), encoding="utf-8")
            print(f"fixed {p.relative_to(ROOT)}")

    print(f"\n{total} lossy line(s) repaired.")
    if remaining:
        print(f"{len(remaining)} line(s) left for manual repair:")
        for msg in remaining:
            print(f"  {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
