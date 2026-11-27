"""Fail-loud guards for the static storefront under ``docs/``.

Each assertion corresponds to a defect that was actually shipped and only showed up when the page
was opened in a browser: an SRI pin without ``crossorigin`` makes the browser refuse the script
(the 3D connectome replaced itself with a fallback message), a redirect stub built on
``meta refresh`` alone drops the query and fragment of a deep link, and a tab icon that is
referenced but not committed is a 404 nobody sees.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = ("index.html", "atlas.html")

_VERSIONED = re.compile(r"/r\d+/|@\d+\.\d+\.\d+")


def _page(name: str) -> str:
    path = DOCS / name
    assert path.is_file(), f"{name} is missing from docs/"
    return path.read_text(encoding="utf-8")


def _script_tags(html: str) -> list[str]:
    """Opening ``<script>`` tags, however the page spells the name."""
    return re.findall(r"<script\b[^>]*>", html, re.IGNORECASE)


def test_script_tags_match_markup_of_any_case() -> None:
    assert _script_tags("<script src=a>") == ["<script src=a>"]
    assert _script_tags("<SCRIPT SRC=a>") == ["<SCRIPT SRC=a>"]
    assert _script_tags("<ScRiPt INTEGRITY=x>") == ["<ScRiPt INTEGRITY=x>"]
    assert _script_tags("<scripts></scripts>") == []


def test_pinned_scripts_are_cors_enabled() -> None:
    """An integrity attribute on a no-CORS subresource is a hard block, not a warning."""
    for name in ("index.html", "atlas.html"):
        for tag in _script_tags(_page(name)):
            attrs = tag.lower()
            if "integrity=" not in attrs:
                continue
            assert "crossorigin=" in attrs, f"{name}: pinned script without crossorigin: {tag}"


def test_pinned_cdns_are_versioned() -> None:
    """A pin is only safe on an immutable URL; an unversioned payload changes under the hash, and
    an immutable one shipped unpinned is a pin waiting to be forgotten."""
    for name in ("index.html", "atlas.html"):
        html = _page(name)
        pinned: list[str] = []
        unpinned: list[str] = []
        for tag in _script_tags(html):
            src = re.search(r'src="(https://[^"]+)"', tag, re.IGNORECASE)
            if src is None:
                continue
            (pinned if "integrity=" in tag.lower() else unpinned).append(src.group(1))
        for url in pinned:
            assert _VERSIONED.search(url), f"{name}: pin on an unversioned URL {url}"
        for url in unpinned:
            assert not _VERSIONED.search(url), f"{name}: immutable URL shipped without a pin: {url}"


def test_tab_icons_exist_and_are_linked() -> None:
    for filename in ("favicon.ico", "apple-touch-icon.png"):
        icon = DOCS / filename
        assert icon.is_file(), f"docs/{filename} is referenced but not committed"
        assert icon.stat().st_size > 0, f"docs/{filename} is empty"
    for name in PAGES:
        html = _page(name)
        assert re.search(r'rel="icon"[^>]+href="(\.\./)?favicon\.ico"', html), name
        assert re.search(
            r'rel="apple-touch-icon"[^>]+href="(\.\./)?apple-touch-icon\.png"', html
        ), name


def test_redirect_stub_preserves_query_and_fragment() -> None:
    stub = _page("noema.html")
    assert "location.replace(" in stub, "stub must replace, so the back button skips it"
    assert "window.location.search" in stub and "window.location.hash" in stub, (
        "stub drops the deep-link query/fragment"
    )
    assert 'http-equiv="refresh"' in stub, "stub needs a no-JS path to the storefront"


def test_sitemap_lists_published_pages() -> None:
    sitemap = _page("sitemap.xml")
    robots = _page("robots.txt")
    for loc in (
        "https://ssrjkk.github.io/noema/",
        "https://ssrjkk.github.io/noema/atlas.html",
    ):
        assert f"<loc>{loc}</loc>" in sitemap, f"{loc} missing from docs/sitemap.xml"
    assert "https://ssrjkk.github.io/noema/sitemap.xml" in robots


def test_brain_has_a_dependency_free_fallback() -> None:
    html = _page("index.html")
    assert 'id="brainFallback"' in html
    assert 'id="brainUnavailableNotice"' in html
    assert "function drawBrainFallback()" in html
    assert "fallbackCanvas.hidden = true" in html
    assert "fallbackCanvas.hidden = false" in html
    assert "container.textContent = message" not in html
