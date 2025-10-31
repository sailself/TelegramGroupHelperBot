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
    try:
        async with session.get(media_url) as response:
            if response.status == 200:
                return await response.read()
            logger.error(
                "Failed to download media: %s from %s",
                response.status,
                media_url,
            )
            return None
    except asyncio.TimeoutError as exc:
        logger.error("Timeout downloading media from %s: %s", media_url, exc)
        return None
    except ClientError as exc:
        logger.error(
            "HTTP error downloading media from %s: %s", media_url, exc, exc_info=True
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected error downloading media from %s: %s",
            media_url,
            exc,
            exc_info=True,
        )
        return None
