"""Exa search helper utilities for OpenRouter tool calling."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

from bot.config import EXA_API_KEY, EXA_SEARCH_ENDPOINT

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_DEFAULT_RESULTS = 5


class ExaSearchError(RuntimeError):
    """Raised when an Exa search request fails."""


def _authorized_client(timeout: Optional[float] = None) -> httpx.Client:
    """Return an HTTP client configured for Exa requests."""

    headers: Dict[str, str] = {}
    if EXA_API_KEY:
        headers["x-api-key"] = EXA_API_KEY
    return httpx.Client(
        headers=headers,
        timeout=httpx.Timeout(
            timeout or DEFAULT_TIMEOUT_SECONDS,
            connect=10.0,
            read=timeout or DEFAULT_TIMEOUT_SECONDS,
            write=10.0,
            pool=10.0,
        ),
    )


def _normalise_snippet(value: Optional[str]) -> str:
    """Return a compact, single-line snippet."""

    if not value:
        return ""
    snippet = value.replace("\n", " ").strip()
    return snippet[:240] + ("…" if len(snippet) > 240 else "")


def _extract_results(payload: Dict[str, object]) -> List[Dict[str, str]]:
    """Extract structured results from the Exa response."""

    results: List[Dict[str, str]] = []
    for item in payload.get("results", []) or []:
        if not isinstance(item, dict):
            continue

        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = (
            item.get("highlight")
            or item.get("snippet")
            or item.get("text")
            or item.get("summary")
            or ""
        )

        if not url:
            continue

        results.append(
            {
                "title": title or url,
                "url": url,
                "snippet": _normalise_snippet(snippet if isinstance(snippet, str) else ""),
            }
        )
    return results


def exa_search(query: str, *, max_results: int = MAX_DEFAULT_RESULTS) -> List[Dict[str, str]]:
    """Execute an Exa search API call and return structured results."""

    if not EXA_API_KEY:
        raise ExaSearchError("EXA_API_KEY is not configured.")
    if not query or not query.strip():
        raise ValueError("query must not be empty.")

    payload = {
        "query": query,
        "numResults": max(1, min(max_results, 10)),
        "type": "auto"
    }

    logger.info("Calling Exa search endpoint %s with query: %s", EXA_SEARCH_ENDPOINT, query)
    try:
        with _authorized_client() as client:
            response = client.post(EXA_SEARCH_ENDPOINT, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        logger.error("Exa search request failed: %s", exc, exc_info=True)
        raise ExaSearchError(f"Exa search request failed: {exc}") from exc

    return _extract_results(data)


def format_results_markdown(query: str, results: List[Dict[str, str]]) -> str:
    """Convert Exa search results into Markdown suitable for LLM consumption."""

    if not results:
        return f"No web results found for query: {query}"

    lines = [f"Search results for **{query}**:"]
    for idx, item in enumerate(results, start=1):
        title = item["title"]
        snippet = item["snippet"]
        lines.append(f"{idx}. [{title}]({item['url']})")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def exa_search_tool(query: str, *, max_results: int = MAX_DEFAULT_RESULTS) -> Dict[str, str]:
    """Wrapper that returns a dict payload for function calling."""

    results = exa_search(query=query, max_results=max_results)
    return {
        "query": query,
        "markdown": format_results_markdown(query, results),
    }
