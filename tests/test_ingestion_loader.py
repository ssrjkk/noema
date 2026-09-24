"""Tests for noema.ingestion.loader — knowledge ingestion and the URL SSRF guard."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from noema.ingestion.loader import KnowledgeLoader

# 65 chars: clears the 30-char general-knowledge threshold and contains no
# prescriptive marker, so extraction outcomes are deterministic.
NEUTRAL = "The ingest pipeline batches documents before hashing them for dedupe"
MUST_FACT = "Retry budgets must be lower than the caller timeout for sagas"
ALWAYS_FACT = "Always pin dependency versions inside a lockfile before deploying"

PUBLIC_IP = "93.184.216.34"


class RecordingStore:
    """Stands in for KnowledgeStore, recording what the loader learns."""

    def __init__(self) -> None:
        self.facts: list[dict] = []
        self.save_calls = 0

    def learn_fact(self, **kwargs):
        self.facts.append(kwargs)

    def save(self):
        self.save_calls += 1


class Url:
    """Fluent stand-in for ``aiohttp.ClientSession`` driven by scripted bodies."""

    def __init__(self, body: str = "<html></html>", error: Exception | None = None) -> None:
        self._body = body
        self._error = error
        self.get = MagicMock(side_effect=self._next)
        self.sessions_created = 0
        self.timeouts: list = []
        self.closed = 0

    def _next(self, url, **kwargs):
        self.last_get_kwargs = kwargs
        if self._error is not None:
            raise self._error
        return _Response(self._body)

    def factory(self, timeout=None, **kwargs):
        self.sessions_created += 1
        self.timeouts.append(timeout)
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed += 1
        return False


class _Response:
    def __init__(self, text: str) -> None:
        self._text = text

    async def text(self, encoding="utf-8", errors="ignore"):
        assert encoding == "utf-8" and errors == "ignore"
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def dns(monkeypatch):
    """Control the resolver the guard uses; returns the object it hands back."""

    class Dns:
        def __init__(self) -> None:
            self.addresses = [PUBLIC_IP]
            self.error: Exception | None = None
            self.queries: list[str] = []

        async def getaddrinfo(self, host, port, **kwargs):
            self.queries.append(host)
            if self.error is not None:
                raise self.error
            return [(0, 1, 6, "", (ip, 0)) for ip in self.addresses]

    dns_ = Dns()
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", dns_.getaddrinfo)
    return dns_


@pytest.fixture
def session(monkeypatch):
    """Install a fake ``aiohttp.ClientSession`` and hand back its recording object."""
    holder: dict[str, Url] = {}

    def create(timeout=None, **kwargs):
        return holder["url"].factory(timeout=timeout, **kwargs)

    monkeypatch.setattr("aiohttp.ClientSession", create)
    return holder


def _use(session_holder: dict, url: Url) -> Url:
    session_holder["url"] = url
    return url


class TestIngestFile:
    @pytest.mark.asyncio
    async def test_missing_file_reports_error(self, tmp_path: Path):
        missing = tmp_path / "absent.md"
        result = await KnowledgeLoader().ingest_file(missing)
        assert result.source_type == "file"
        assert result.source == str(missing)
        assert result.errors == [f"File not found: {missing}"]
        assert result.entries_ingested == 0

    @pytest.mark.asyncio
    async def test_read_error_is_reported_not_raised(self, tmp_path: Path, monkeypatch):
        path = tmp_path / "note.md"
        path.write_text(MUST_FACT, encoding="utf-8")
        monkeypatch.setattr(Path, "read_text", MagicMock(side_effect=OSError("disk gone")))

        result = await KnowledgeLoader().ingest_file(path)

        assert result.errors == ["Read error: disk gone"]
        assert result.entries_ingested == 0

    @pytest.mark.asyncio
    async def test_extracts_prescriptive_facts(self, tmp_path: Path):
        path = tmp_path / "notes.md"
        path.write_text(f"{MUST_FACT}. {ALWAYS_FACT}.", encoding="utf-8")

        result = await KnowledgeLoader().ingest_file(path)

        assert result.entries_ingested == 2
        assert result.topics_extracted == ["notes"]
        assert (result.errors, result.entries_skipped) == ([], 0)

    @pytest.mark.asyncio
    async def test_learns_into_store_and_saves_once(self, tmp_path: Path):
        store = RecordingStore()
        path = tmp_path / "notes.md"
        path.write_text(f"{MUST_FACT}. {ALWAYS_FACT}.", encoding="utf-8")

        result = await KnowledgeLoader(knowledge_store=store).ingest_file(path, tags=["docs"])

        assert result.entries_ingested == 2
        assert store.save_calls == 1
        assert [f["source"] for f in store.facts] == ["file:notes.md", "file:notes.md"]
        assert {f["fact"] for f in store.facts} == {MUST_FACT, ALWAYS_FACT}
        assert [f["confidence"] for f in store.facts] == [0.7, 0.7]
        assert [f["topic"] for f in store.facts] == ["notes", "notes"]
        assert all(f["tags"][0] == "docs" for f in store.facts)

    @pytest.mark.asyncio
    async def test_without_store_content_is_still_counted(self, tmp_path: Path):
        path = tmp_path / "notes.md"
        path.write_text(f"{MUST_FACT}.", encoding="utf-8")
        result = await KnowledgeLoader(knowledge_store=None).ingest_file(path)
        assert result.entries_ingested == 1

    @pytest.mark.asyncio
    async def test_duplicate_content_is_skipped(self, tmp_path: Path):
        loader = KnowledgeLoader()
        path = tmp_path / "notes.md"
        path.write_text(f"{MUST_FACT}.", encoding="utf-8")

        first = await loader.ingest_file(path)
        second = await loader.ingest_file(path)

        assert (first.entries_ingested, first.entries_skipped) == (1, 0)
        assert (second.entries_ingested, second.entries_skipped) == (0, 1)
        assert second.errors == []

    @pytest.mark.asyncio
    async def test_accepts_str_path(self, tmp_path: Path):
        path = tmp_path / "notes.md"
        path.write_text(f"{MUST_FACT}.", encoding="utf-8")
        result = await KnowledgeLoader().ingest_file(str(path))
        assert result.entries_ingested == 1

    @pytest.mark.asyncio
    async def test_hash_registry_prevents_reingest_across_methods(self, tmp_path: Path):
        loader = KnowledgeLoader()
        path = tmp_path / "notes.md"
        text = f"{MUST_FACT}."
        path.write_text(text, encoding="utf-8")

        await loader.ingest_file(path)
        via_text = await loader.ingest_text(text)

        assert (via_text.entries_ingested, via_text.entries_skipped) == (0, 1)


class TestIngestDirectory:
    @pytest.mark.asyncio
    async def test_missing_directory_reports_error(self, tmp_path: Path):
        absent = tmp_path / "nope"
        result = await KnowledgeLoader().ingest_directory(absent)
        assert result.source_type == "directory"
        assert result.errors == [f"Directory not found: {absent}"]

    @pytest.mark.asyncio
    async def test_regular_file_is_rejected(self, tmp_path: Path):
        a_file = tmp_path / "notes.md"
        a_file.write_text(NEUTRAL, encoding="utf-8")
        result = await KnowledgeLoader().ingest_directory(a_file)
        assert result.errors == [f"Directory not found: {a_file}"]

    @pytest.mark.asyncio
    async def test_aggregates_recursive_matches(self, tmp_path: Path):
        (tmp_path / "a.md").write_text(f"{MUST_FACT}.", encoding="utf-8")
        nested = tmp_path / "sub" / "deeper"
        nested.mkdir(parents=True)
        (nested / "b.txt").write_text(f"{ALWAYS_FACT}.", encoding="utf-8")

        result = await KnowledgeLoader().ingest_directory(tmp_path, patterns=["*.md", "*.txt"])

        assert result.entries_ingested == 2
        assert (result.entries_skipped, result.errors) == (0, [])

    @pytest.mark.asyncio
    async def test_default_patterns_only_match_known_extensions(self, tmp_path: Path):
        (tmp_path / "notes.md").write_text(f"{MUST_FACT}.", encoding="utf-8")
        (tmp_path / "payload.exe").write_text(f"{ALWAYS_FACT}.", encoding="utf-8")

        result = await KnowledgeLoader().ingest_directory(tmp_path)

        assert result.entries_ingested == 1

    @pytest.mark.asyncio
    async def test_oversized_files_are_skipped(self, tmp_path: Path):
        big = tmp_path / "big.md"
        big.write_text(f"{MUST_FACT}. " * 20_000, encoding="utf-8")
        assert big.stat().st_size >= 1_000_000

        result = await KnowledgeLoader().ingest_directory(tmp_path, patterns=["*.md"])

        assert (result.entries_ingested, result.errors) == (0, [])

    @pytest.mark.asyncio
    async def test_child_errors_propagate(self, tmp_path: Path, monkeypatch):
        (tmp_path / "a.md").write_text(MUST_FACT, encoding="utf-8")
        monkeypatch.setattr(Path, "read_text", MagicMock(side_effect=OSError("nope")))
        result = await KnowledgeLoader().ingest_directory(tmp_path, patterns=["*.md"])
        assert result.errors == ["Read error: nope"]

    @pytest.mark.asyncio
    async def test_topics_are_deduplicated(self, tmp_path: Path):
        (tmp_path / "same.md").write_text(f"{MUST_FACT}.", encoding="utf-8")
        (tmp_path / "other.md").write_text(f"{ALWAYS_FACT}.", encoding="utf-8")
        result = await KnowledgeLoader().ingest_directory(tmp_path, patterns=["*.md"])
        assert sorted(result.topics_extracted) == ["other", "same"]

    @pytest.mark.asyncio
    async def test_empty_directory_yields_nothing(self, tmp_path: Path):
        result = await KnowledgeLoader().ingest_directory(tmp_path)
        assert (result.entries_ingested, result.entries_skipped, result.errors) == (0, 0, [])


class TestIngestText:
    @pytest.mark.asyncio
    async def test_records_source_and_type(self):
        result = await KnowledgeLoader().ingest_text(f"{MUST_FACT}.", source_name="meeting notes")
        assert (result.source_type, result.source) == ("text", "meeting notes")
        assert result.entries_ingested == 1

    @pytest.mark.asyncio
    async def test_default_source_name(self):
        assert (await KnowledgeLoader().ingest_text(f"{MUST_FACT}.")).source == "direct_input"

    @pytest.mark.asyncio
    async def test_duplicate_text_is_skipped(self):
        loader = KnowledgeLoader()
        await loader.ingest_text(f"{MUST_FACT}.")
        second = await loader.ingest_text(f"{MUST_FACT}.")
        assert (second.entries_ingested, second.entries_skipped) == (0, 1)

    @pytest.mark.asyncio
    async def test_empty_text_yields_no_items(self):
        result = await KnowledgeLoader().ingest_text("")
        assert (result.entries_ingested, result.entries_skipped) == (0, 0)
        assert result.topics_extracted == []

    @pytest.mark.asyncio
    async def test_learns_with_text_source_prefix(self):
        store = RecordingStore()
        result = await KnowledgeLoader(knowledge_store=store).ingest_text(
            f"{MUST_FACT}.", source_name="standup", tags=["team"]
        )
        assert result.entries_ingested == 1
        assert store.facts[0]["source"] == "text:standup"
        assert store.facts[0]["tags"][0] == "team"
        assert store.save_calls == 1


class TestExtractFromText:
    def test_only_prescriptive_markers_qualify(self):
        text = f"{MUST_FACT}. The weather in Lisbon is mild in autumn season. {ALWAYS_FACT}."
        items = KnowledgeLoader()._extract_from_text(text, "guide", ["tag"])
        assert {i["fact"] for i in items} == {MUST_FACT, ALWAYS_FACT}
        assert all(i["confidence"] == 0.7 for i in items)
        assert all(i["topic"] == "guide" for i in items)

    @pytest.mark.parametrize(
        "marker",
        ["should", "must", "always", "never", "important", "best practice", "recommended"],
    )
    def test_every_marker_is_recognised(self, marker):
        sentence = f"Rotate the signing keys {marker} quarterly without exception"
        items = KnowledgeLoader()._extract_from_text(sentence + ".", "doc", [])
        assert [i["fact"] for i in items] == [sentence]

    def test_marker_matching_is_case_insensitive(self):
        items = KnowledgeLoader()._extract_from_text(
            "Credentials MUST be rotated quarterly by the on-call", "doc", []
        )
        assert len(items) == 1

    def test_falls_back_to_general_knowledge(self):
        items = KnowledgeLoader()._extract_from_text(NEUTRAL + ".", "doc", [])
        assert [i["fact"] for i in items] == [NEUTRAL]
        assert items[0]["confidence"] == 0.5

    def test_general_fallback_requires_over_thirty_chars(self):
        short = "Batch the documents first."
        assert 20 < len(short.rstrip(".")) <= 30
        assert KnowledgeLoader()._extract_from_text(short, "doc", []) == []

    def test_sentences_under_twenty_chars_are_dropped(self):
        assert KnowledgeLoader()._extract_from_text("Tiny.", "doc", []) == []
        items = KnowledgeLoader()._extract_from_text(f"Tiny. {NEUTRAL}.", "doc", [])
        assert [i["fact"] for i in items] == [NEUTRAL]

    def test_prescriptive_extraction_caps_at_fifty(self):
        text = ". ".join(f"{NEUTRAL[:45]} {i} must hold" for i in range(60))
        assert len(KnowledgeLoader()._extract_from_text(text, "doc", [])) == 50

    def test_general_fallback_caps_at_ten(self):
        text = ". ".join(f"{NEUTRAL[:45]} {i} is a statement" for i in range(30))
        assert len(KnowledgeLoader()._extract_from_text(text, "doc", [])) == 10

    def test_tags_lead_with_explicit_tags_then_keywords(self):
        text = f"Kafka {NEUTRAL} must be rebalanced before shutdown " + "Kafka " * 5
        items = KnowledgeLoader()._extract_from_text(text, "doc", ["base"])
        assert items[0]["tags"][0] == "base"
        assert "kafka" in items[0]["tags"]

    def test_topic_derived_from_source_stem(self):
        items = KnowledgeLoader()._extract_from_text(f"{NEUTRAL}.", "/tmp/notes.md", [])
        assert [i["topic"] for i in items] == ["notes"]

    def test_topic_is_whole_source_without_path_separators(self):
        items = KnowledgeLoader()._extract_from_text(f"{NEUTRAL}.", "standup", [])
        assert [i["topic"] for i in items] == ["standup"]


class TestExtractKeywords:
    def test_ranks_by_frequency_and_caps_at_ten(self):
        # Fillers must be letters-only to be matched by the ``\b[a-zA-Z]{4,}\b``
        # pattern; tokens such as ``w0xyz`` are skipped by design.
        fillers = " ".join(f"extra{chr(ord('a') + i)}" for i in range(20))
        text = "alpha " * 5 + "beta " * 4 + "gamma " * 3 + fillers
        keywords = KnowledgeLoader()._extract_keywords(text)
        assert keywords[:3] == ["alpha", "beta", "gamma"]
        assert len(keywords) == 10

    def test_tokens_containing_digits_are_ignored(self):
        assert KnowledgeLoader()._extract_keywords("alpha w0xyz beta1") == ["alpha"]

    def test_words_shorter_than_four_chars_are_ignored(self):
        assert KnowledgeLoader()._extract_keywords("go now up abc xyz abcd") == ["abcd"]

    def test_stopwords_are_excluded(self):
        assert KnowledgeLoader()._extract_keywords("this that with from which about would") == []

    def test_matching_is_case_insensitive(self):
        assert KnowledgeLoader()._extract_keywords("Kafka KAFKA kafka") == ["kafka"]


class TestIngestUrlGuard:
    """The guard must refuse non-http(s) schemes and any non-public address."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        ["file:///etc/passwd", "ftp://host/x", "gopher://host:70/x", "http://", "not a url"],
    )
    async def test_rejects_unusable_urls(self, url, dns, session):
        _use(session, Url())
        result = await KnowledgeLoader().ingest_url(url)
        assert (result.source, result.source_type) == (url, "url")
        assert result.errors == [f"Unsupported URL: {url!r}"]
        assert dns.queries == []
        session["url"].get.assert_not_called()

    @pytest.mark.asyncio
    async def test_dns_failure_is_reported(self, dns, session):
        dns.error = OSError("dns exploded")
        _use(session, Url())

        result = await KnowledgeLoader().ingest_url("http://nope.invalid/page")

        assert result.errors == ["DNS resolution failed: dns exploded"]
        assert dns.queries == ["nope.invalid"]
        session["url"].get.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "10.0.0.5",
            "172.16.0.9",
            "192.168.1.1",
            "169.254.169.254",
            "224.0.0.1",
            "240.0.0.1",
            "::1",
            "fe80::1",
            "fc00::1",
        ],
    )
    async def test_refuses_non_public_addresses(self, ip, dns, session):
        dns.addresses = [ip]
        _use(session, Url())

        result = await KnowledgeLoader().ingest_url("http://internal.example/page")

        assert result.errors == [f"Refusing non-public address: {ip}"]
        assert result.entries_ingested == 0
        session["url"].get.assert_not_called()

    @pytest.mark.asyncio
    async def test_refuses_when_any_resolved_address_is_private(self, dns, session):
        dns.addresses = [PUBLIC_IP, "10.0.0.5"]
        _use(session, Url())

        result = await KnowledgeLoader().ingest_url("http://multi.example/page")

        assert result.errors == ["Refusing non-public address: 10.0.0.5"]
        session["url"].get.assert_not_called()

    @pytest.mark.asyncio
    async def test_public_address_passes_the_guard(self, dns, session):
        _use(session, Url(f"<p>{MUST_FACT}</p>"))
        result = await KnowledgeLoader().ingest_url("http://example.com/page")
        assert [e for e in result.errors if e.startswith("Refusing")] == []
        assert dns.queries == ["example.com"]


class TestIngestUrlFetch:
    @pytest.mark.asyncio
    async def test_extracts_visible_text_and_drops_script_and_style(self, dns, session):
        html = (
            "<html><head><style>body{color:red}</style>"
            "<script>var hidden = 'never-ingest-this';</script></head>"
            f"<body><h1>Guide</h1><p>{MUST_FACT}</p></body></html>"
        )
        _use(session, Url(html))

        result = await KnowledgeLoader().ingest_url("http://example.com/guide", tags=["web"])

        assert result.errors == []
        assert result.entries_ingested == 1
        assert result.source == "http://example.com/guide"
        assert result.source_type == "url"

    @pytest.mark.asyncio
    async def test_script_content_never_reaches_the_store(self, dns, session):
        store = RecordingStore()
        html = f"<html><script>var k='{MUST_FACT}'</script><body>{ALWAYS_FACT}</body></html>"
        _use(session, Url(html))

        await KnowledgeLoader(knowledge_store=store).ingest_url("http://example.com/guide")

        assert store.facts
        assert all(
            "never-ingest" not in f["fact"] and "var k=" not in f["fact"] for f in store.facts
        )

    @pytest.mark.asyncio
    async def test_sends_user_agent_and_timeout(self, dns, session):
        url = _use(session, Url(f"<p>{MUST_FACT}</p>"))

        await KnowledgeLoader().ingest_url("http://example.com/guide")

        url.get.assert_called_once_with(
            "http://example.com/guide", headers={"User-Agent": "Noema/1.0"}
        )
        assert url.timeouts[0].total == 15

    @pytest.mark.asyncio
    async def test_blank_pages_report_no_content(self, dns, session):
        _use(session, Url("   "))
        result = await KnowledgeLoader().ingest_url("http://example.com/empty")
        assert result.errors == ["No text content extracted from URL"]
        assert result.entries_ingested == 0

    @pytest.mark.asyncio
    async def test_fetch_errors_are_captured(self, dns, session):
        _use(session, Url(error=RuntimeError("connection reset")))
        result = await KnowledgeLoader().ingest_url("http://example.com/boom")
        assert result.errors == ["URL fetch error: connection reset"]
        assert result.entries_ingested == 0

    @pytest.mark.asyncio
    async def test_ingests_into_store_when_provided(self, dns, session):
        store = RecordingStore()
        _use(session, Url(f"<p>{MUST_FACT}</p>"))

        result = await KnowledgeLoader(knowledge_store=store).ingest_url("http://example.com/g")

        assert result.entries_ingested == 1
        assert store.save_calls == 1
        assert store.facts[0]["source"] == "text:http://example.com/g"

    @pytest.mark.asyncio
    async def test_page_text_is_deduplicated_against_prior_ingests(self, dns, session):
        body = f"<p>{MUST_FACT}</p>"
        loader = KnowledgeLoader()
        first = await loader.ingest_text(MUST_FACT)
        _use(session, Url(body))

        second = await loader.ingest_url("http://example.com/guide")

        assert (first.entries_ingested, second.entries_ingested, second.entries_skipped) == (
            1,
            0,
            1,
        )
