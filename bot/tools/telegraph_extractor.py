"""Telegraph extraction helpers."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

from aiohttp import ClientError

from bot.utils.http import get_http_session

logger = logging.getLogger(__name__)


async def extract_telegraph_content(url: str) -> dict:
    """
    Extract text content, image URLs, and video URLs from a Telegraph page.

    Args:
        url: The Telegraph page URL.

    Returns:
        A dictionary with keys 'text_content', 'image_urls', and 'video_urls'.

    Raises:
        Exception: If the page cannot be fetched or parsed.
    """
    logger.info("Starting extraction for url: %s", url)
    parsed_url = urlparse(url)
    path = parsed_url.path.lstrip("/")

    if not path:
        logger.error("Invalid Telegraph URL: Missing path component.")
        raise ValueError("Invalid Telegraph URL: Missing path component.")

    api_url = f"https://api.telegra.ph/getPage/{path}?return_content=true"
    logger.debug("Telegraph API URL: %s", api_url)

    session = await get_http_session()
    try:
        async with session.get(api_url, timeout=15) as response:
            response.raise_for_status()
            data = await response.json()
    except asyncio.TimeoutError as exc:
        logger.error("Timeout fetching Telegraph page %s", api_url)
        raise Exception(f"Error fetching Telegraph page: {exc}") from exc
    except ClientError as exc:
        logger.error("Error fetching Telegraph page %s: %s", api_url, exc)
        raise Exception(f"Error fetching Telegraph page: {exc}") from exc

    if not data.get("ok"):
        error_message = data.get("error", "Unknown error from Telegraph API")
        logger.error("Telegraph API error: %s", error_message)
        raise Exception(f"Telegraph API error: {error_message}")

    content_nodes = data.get("result", {}).get("content", [])

    text_content = ""
    image_urls: list[str] = []
    video_urls: list[str] = []
    image_counter = 0
    video_counter = 0

    def process_node_children(nodes):
        nonlocal text_content, image_urls, video_urls, image_counter, video_counter
        current_text = ""
        for node in nodes:
            if isinstance(node, str):
                current_text += node
            elif isinstance(node, dict):
                tag = node.get("tag")
                children = node.get("children", [])

                if tag == "img":
                    image_counter += 1
                    src = node.get("attrs", {}).get("src")
                    if src:
                        if src.startswith("/"):
                            src = "https://telegra.ph" + src
                        image_urls.append(src)
                        current_text += f"[image_{image_counter}]"
                elif tag == "video" or (
                    tag == "figure"
                    and any(
                        c.get("tag") == "video"
                        for c in children
                        if isinstance(c, dict)
                    )
                ):
                    video_counter += 1
                    video_node = None
                    if tag == "video":
                        video_node = node
                    else:
                        for child_node in children:
                            if (
                                isinstance(child_node, dict)
                                and child_node.get("tag") == "video"
                            ):
                                video_node = child_node
                                break
                    if video_node:
                        src = video_node.get("attrs", {}).get("src")
                        if src:
                            if src.startswith("/"):
                                src = "https://telegra.ph" + src
                            video_urls.append(src)
                            current_text += f"[video_{video_counter}]"
                elif tag == "iframe":
                    video_counter += 1
                    src = node.get("attrs", {}).get("src")
                    if src:
                        if src.startswith("/embed/youtube"):
                            src = "https://www.youtube.com" + src
                        elif src.startswith("/embed/vimeo"):
                            src = "https://player.vimeo.com" + src
                        elif src.startswith("/"):
                            src = "https://telegra.ph" + src
                        video_urls.append(src)
                        current_text += f"[video_{video_counter}]"
                elif tag in [
                    "p",
                    "a",
                    "li",
                    "h3",
                    "h4",
                    "em",
                    "strong",
                    "figcaption",
                    "blockquote",
                    "code",
                    "span",
                ]:
                    current_text += process_node_children(children)
                    if tag in ["p", "h3", "h4", "li", "blockquote"]:
                        current_text += "\n"
                elif tag == "figure":
                    current_text += process_node_children(children)
                    current_text += "\n"
                elif tag == "br":
                    current_text += "\n"
                elif tag == "hr":
                    current_text += "\n---\n"
                elif tag in ["ul", "ol"]:
                    for child in children:
                        if isinstance(child, dict) and child.get("tag") == "li":
                            current_text += (
                                "- "
                                + process_node_children(child.get("children", []))
                                + "\n"
                            )
                        else:
                            current_text += process_node_children([child])
                    current_text += "\n"
                elif tag == "pre":
                    code_block_text = ""
                    for child in children:
                        if isinstance(child, dict) and child.get("tag") == "code":
                            code_block_text += process_node_children(
                                child.get("children", [])
                            )
                        else:
                            code_block_text += process_node_children([child])
                    current_text += f"\n```\n{code_block_text.strip()}\n```\n"
                elif children:
                    current_text += process_node_children(children)
        return current_text

    text_content = process_node_children(content_nodes)

    result = {
        "text_content": text_content.strip(),
        "image_urls": image_urls,
        "video_urls": video_urls,
    }
    logger.debug(
        "Extraction result: text length=%d, images=%d, videos=%d",
        len(result["text_content"]),
        len(result["image_urls"]),
        len(result["video_urls"]),
    )
    return result

