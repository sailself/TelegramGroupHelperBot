"""Jina Reader and Search helpers."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

from bot.config import JINA_AI_API_KEY, JINA_READER_ENDPOINT, JINA_SEARCH_ENDPOINT

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 30.0
DEFAULT_WRITE_TIMEOUT = 10.0
DEFAULT_POOL_TIMEOUT = 10.0

TITLE_PATTERN = re.compile(r"\[(\d+)\]\s+Title:\s*(.+)")
URL_PATTERN = re.compile(r"\[(\d+)\]\s+URL Source:\s*(.+)")
SNIPPET_PATTERN = re.compile(r"\[(\d+)\]\s+(Description|Snippet):\s*(.+)")


@dataclass
class JinaSearchResult:
    """Structured representation of a Jina search hit."""

    title: str
    url: str
    snippet: str


@dataclass
class JinaSearchResponse:
    """Top-level payload returned by the Jina search helper."""

    query: str
    results: List[JinaSearchResult]


def _authorized_client(timeout: Optional[float] = None) -> httpx.Client:
    """Create an HTTP client with Bearer auth if an API key is configured."""

    headers: Dict[str, str] = {}
    if JINA_AI_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_AI_API_KEY}"
    return httpx.Client(
        headers=headers,
        timeout=httpx.Timeout(
            timeout or DEFAULT_READ_TIMEOUT,
            connect=DEFAULT_CONNECT_TIMEOUT,
            read=timeout or DEFAULT_READ_TIMEOUT,
            write=DEFAULT_WRITE_TIMEOUT,
            pool=DEFAULT_POOL_TIMEOUT,
        ),
    )


def _parse_search_text(payload: str, *, max_results: int) -> List[JinaSearchResult]:
    """Parse the plain-text Jina search response into structured results."""

    results: Dict[int, JinaSearchResult] = {}

    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!["):
            continue

        title_match = TITLE_PATTERN.match(line)
        if title_match:
            idx = int(title_match.group(1))
            entry = results.setdefault(idx, JinaSearchResult(title="", url="", snippet=""))
            entry.title = title_match.group(2).strip()
            continue

        url_match = URL_PATTERN.match(line)
        if url_match:
            idx = int(url_match.group(1))
            entry = results.setdefault(idx, JinaSearchResult(title="", url="", snippet=""))
            entry.url = url_match.group(2).strip()
            continue

        snippet_match = SNIPPET_PATTERN.match(line)
        if snippet_match:
            idx = int(snippet_match.group(1))
            entry = results.setdefault(idx, JinaSearchResult(title="", url="", snippet=""))
            entry.snippet = snippet_match.group(3).strip()
            continue

    ordered = [results[idx] for idx in sorted(results.keys()) if results[idx].url]
    return ordered[:max_results]


def search_jina_web(query: str, *, max_results: int = 5) -> JinaSearchResponse:
    """Query the Jina search endpoint and return structured results."""

    if not query.strip():
        raise ValueError("query must not be empty")

    payload = {"q": query}
    logger.info("Calling Jina search endpoint %s with query: %s", JINA_SEARCH_ENDPOINT, query)
    with _authorized_client() as client:
        response = client.post(JINA_SEARCH_ENDPOINT, json=payload)
        response.raise_for_status()
        parsed = _parse_search_text(response.text, max_results=max_results)
        return JinaSearchResponse(query=query, results=parsed)


def fetch_jina_reader(url: str) -> str:
    """Retrieve markdown content from the Jina Reader endpoint."""

    if not url.strip():
        raise ValueError("url must not be empty")

    target = JINA_READER_ENDPOINT.rstrip("/") + "/" + url.lstrip("/")
    logger.info("Calling Jina reader endpoint %s", target)
    with _authorized_client(timeout=60.0) as client:
        response = client.get(target)
        response.raise_for_status()
        return response.text


def format_search_results_markdown(results: JinaSearchResponse) -> str:
    """Convert a search response into markdown suitable for LLM consumption."""

    if not results.results:
        return f"No results found for query: {results.query}"

    lines = [f"Search results for **{results.query}**:"]
    for idx, item in enumerate(results.results, start=1):
        snippet = item.snippet.replace("\n", " ")
        snippet = snippet[:200] + "…" if len(snippet) > 200 else snippet
        title = item.title or item.url
        lines.append(f"{idx}. [{title}]({item.url})\n   {snippet}")
    return "\n".join(lines)


def jina_search_tool(query: str, *, max_results: int = 5) -> Dict[str, str]:
    """Synchronous wrapper that returns dict payload for function calling."""

    results = search_jina_web(query, max_results=max_results)
    return {
        "query": results.query,
        "markdown": format_search_results_markdown(results),
    }
