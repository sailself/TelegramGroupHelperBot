"""Media related helpers for LLM integrations."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from aiohttp import ClientError

from bot.utils.http import get_http_session

logger = logging.getLogger(__name__)


def detect_mime_type(image_data: bytes) -> str:
    """Detect the MIME type of image data.

    Args:
        image_data: The image data as bytes.

    Returns:
        The MIME type as a string.
    """
    # First few bytes of the file can identify the format
    if image_data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
        return "image/webp"
    elif image_data.startswith(b"\x00\x00\x00") and image_data[4:8] in (
        b"ftyp",
        b"heic",
        b"heix",
        b"hevc",
        b"hevx",
    ):
        # HEIC/HEIF formats
        if b"heic" in image_data[4:20] or b"heis" in image_data[4:20]:
            return "image/heic"
        else:
            return "image/heif"

    # Default to JPEG if the format couldn't be determined
    logger.warning("Could not determine image MIME type from data, defaulting to JPEG")
    return "image/jpeg"

async def download_media(media_url: str) -> Optional[bytes]:
    """Download a media file (image or video) from a URL.

    Args:
        media_url: The URL of the media to download.

    Returns:
        The media data as bytes, or None if the download failed.
    """
    session = await get_http_session()
    retries = 2
    delay = 1.5
    retry_statuses = {408, 429, 500, 502, 503, 504}

    for attempt in range(1, retries + 2):
        try:
            async with session.get(media_url) as response:
                if response.status == 200:
                    return await response.read()
                if response.status in retry_statuses and attempt <= retries:
                    logger.warning(
                        "Retrying media download (%d/%d) for %s due to status %s",
                        attempt,
                        retries + 1,
                        media_url,
                        response.status,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                logger.error(
                    "Failed to download media: %s from %s",
                    response.status,
                    media_url,
                )
                return None
        except (asyncio.TimeoutError, ClientError, OSError) as exc:
            if attempt > retries:
                logger.error(
                    "HTTP error downloading media from %s: %s",
                    media_url,
                    exc,
                    exc_info=True,
                )
                return None
            logger.warning(
                "Retrying media download (%d/%d) for %s due to error: %s",
                attempt,
                retries + 1,
                media_url,
                exc,
            )
            await asyncio.sleep(delay)
            delay *= 2
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected error downloading media from %s: %s",
                media_url,
                exc,
                exc_info=True,
            )
            return None
    return None
