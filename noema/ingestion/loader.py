from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field


class IngestionResult(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = ""
    source_type: str = ""  # file, directory, url, git
    entries_ingested: int = 0
    entries_skipped: int = 0
    errors: list[str] = Field(default_factory=list)
    topics_extracted: list[str] = Field(default_factory=list)


class KnowledgeLoader:
    """Ingest knowledge from files, directories, URLs, and repos into the knowledge store."""

    def __init__(self, knowledge_store: Any = None) -> None:
        self.store = knowledge_store
        self._ingested_hashes: set[str] = set()

    async def ingest_file(self, path: str | Path, tags: list[str] | None = None) -> IngestionResult:
        """Ingest a single file as knowledge."""
        path = Path(path)
        result = IngestionResult(source=str(path), source_type="file")

        if not path.exists():
            result.errors.append(f"File not found: {path}")
            return result

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            result.errors.append(f"Read error: {e}")
            return result

        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self._ingested_hashes:
            result.entries_skipped = 1
            return result
        self._ingested_hashes.add(content_hash)

        # Extract knowledge from the file
        knowledge_items = self._extract_from_text(content, path.name, tags or [])
        result.entries_ingested = len(knowledge_items)
        result.topics_extracted = list({k["topic"] for k in knowledge_items if k.get("topic")})

        if self.store:
            for item in knowledge_items:
                self.store.learn_fact(
                    topic=item.get("topic", path.stem),
                    fact=item.get("fact", ""),
                    confidence=item.get("confidence", 0.7),
                    source=f"file:{path.name}",
                    tags=item.get("tags", []),
                )
            self.store.save()

        return result

    async def ingest_directory(
        self, path: str | Path, patterns: list[str] | None = None, tags: list[str] | None = None
    ) -> IngestionResult:
        """Ingest all matching files in a directory."""
        path = Path(path)
        result = IngestionResult(source=str(path), source_type="directory")

        if not path.exists() or not path.is_dir():
            result.errors.append(f"Directory not found: {path}")
            return result

        patterns = patterns or ["*.py", "*.md", "*.txt", "*.rst", "*.yaml", "*.yml", "*.json"]

        for pattern in patterns:
            for file_path in path.rglob(pattern):
                if file_path.is_file() and file_path.stat().st_size < 1_000_000:
                    file_result = await self.ingest_file(file_path, tags)
                    result.entries_ingested += file_result.entries_ingested
                    result.entries_skipped += file_result.entries_skipped
                    result.errors.extend(file_result.errors)
                    result.topics_extracted.extend(file_result.topics_extracted)

        result.topics_extracted = list(set(result.topics_extracted))
        return result

    async def ingest_text(
        self, text: str, source_name: str = "direct_input", tags: list[str] | None = None
    ) -> IngestionResult:
        """Ingest raw text as knowledge."""
        result = IngestionResult(source=source_name, source_type="text")

        content_hash = hashlib.md5(text.encode()).hexdigest()
        if content_hash in self._ingested_hashes:
            result.entries_skipped = 1
            return result
        self._ingested_hashes.add(content_hash)

        knowledge_items = self._extract_from_text(text, source_name, tags or [])
        result.entries_ingested = len(knowledge_items)
        result.topics_extracted = list({k["topic"] for k in knowledge_items if k.get("topic")})

        if self.store:
            for item in knowledge_items:
                self.store.learn_fact(
                    topic=item.get("topic", source_name),
                    fact=item.get("fact", ""),
                    confidence=item.get("confidence", 0.7),
                    source=f"text:{source_name}",
                    tags=item.get("tags", []),
                )
            self.store.save()

        return result

    async def ingest_url(self, url: str, tags: list[str] | None = None) -> IngestionResult:
        """Ingest content from a URL.

        SSRF-guarded: only ``http``/``https`` targets resolving to public
        addresses are fetched; private/loopback/link-local ranges are refused.
        The fetch is async (never blocks the event loop) with a hard timeout.
        """
        result = IngestionResult(source=url, source_type="url")

        try:
            import asyncio
            import ipaddress
            from html.parser import HTMLParser

            import aiohttp

            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                result.errors.append(f"Unsupported URL: {url!r}")
                return result

            host = parsed.hostname or ""
            try:
                loop = asyncio.get_running_loop()
                infos = await loop.getaddrinfo(host, None)
            except OSError as e:
                result.errors.append(f"DNS resolution failed: {e}")
                return result
            for info in infos:
                ip = ipaddress.ip_address(info[4][0])
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                    or ip.is_multicast
                ):
                    result.errors.append(f"Refusing non-public address: {ip}")
                    return result

            class TextExtractor(HTMLParser):
                def __init__(self) -> None:
                    super().__init__()
                    self.text_parts: list[str] = []
                    self._skip = False

                def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                    if tag in ("script", "style"):
                        self._skip = True

                def handle_endtag(self, tag: str) -> None:
                    if tag in ("script", "style"):
                        self._skip = False

                def handle_data(self, data: str) -> None:
                    if not self._skip:
                        self.text_parts.append(data.strip())

            timeout = aiohttp.ClientTimeout(total=15)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(url, headers={"User-Agent": "Noema/1.0"}) as resp,
            ):
                html = await resp.text(encoding="utf-8", errors="ignore")

            extractor = TextExtractor()
            extractor.feed(html)
            text = " ".join(extractor.text_parts)

            if text:
                text_result = await self.ingest_text(text, source_name=url, tags=tags or [])
                text_result.source = url
                text_result.source_type = "url"
                return text_result
            else:
                result.errors.append("No text content extracted from URL")

        except Exception as e:
            result.errors.append(f"URL fetch error: {e}")

        return result

    def _extract_from_text(self, text: str, source: str, tags: list[str]) -> list[dict[str, Any]]:
        """Extract knowledge items from text using heuristics."""
        items: list[dict[str, Any]] = []
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # extract topic from source name
        topic = Path(source).stem if "/" in source or "\\" in source else source

        # extract key facts
        keywords = self._extract_keywords(text)

        for sentence in sentences[:50]:  # limit
            if any(
                kw in sentence.lower()
                for kw in [
                    "should",
                    "must",
                    "always",
                    "never",
                    "important",
                    "best practice",
                    "recommended",
                ]
            ):
                items.append(
                    {
                        "topic": topic,
                        "fact": sentence,
                        "confidence": 0.7,
                        "tags": tags + keywords[:3],
                    }
                )

        # if no prescriptive statements, extract general knowledge
        if not items and sentences:
            for sentence in sentences[:10]:
                if len(sentence) > 30:
                    items.append(
                        {
                            "topic": topic,
                            "fact": sentence,
                            "confidence": 0.5,
                            "tags": tags + keywords[:2],
                        }
                    )

        return items

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract potential keywords from text."""
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        freq: dict[str, int] = {}
        for w in words:
            if w not in {
                "this",
                "that",
                "with",
                "from",
                "have",
                "been",
                "were",
                "they",
                "their",
                "which",
                "about",
                "would",
                "could",
                "should",
                "will",
                "into",
                "also",
                "when",
                "what",
                "some",
                "than",
                "only",
                "other",
                "more",
                "very",
                "just",
                "over",
                "such",
            }:
                freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:10]]
