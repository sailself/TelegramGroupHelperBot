"""Shared asynchronous HTTP session helpers."""

from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp

_session_lock = asyncio.Lock()
_session: Optional[aiohttp.ClientSession] = None

_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=30)


async def get_http_session() -> aiohttp.ClientSession:
    """Return a shared aiohttp.ClientSession instance."""
    global _session  # noqa: PLW0603
    if _session is None or _session.closed:
        async with _session_lock:
            if _session is None or _session.closed:
                _session = aiohttp.ClientSession(timeout=_DEFAULT_TIMEOUT)
    return _session


async def close_http_session() -> None:
    """Close the shared aiohttp session if it exists."""
    global _session  # noqa: PLW0603
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None

