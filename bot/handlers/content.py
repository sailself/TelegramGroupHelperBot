"""Content extraction helpers for TelegramGroupHelperBot."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import markdown
import requests
from bs4 import BeautifulSoup
from html2text import html2text

from bot.config import (
    TELEGRAPH_ACCESS_TOKEN,
    TELEGRAPH_AUTHOR_NAME,
    TELEGRAPH_AUTHOR_URL,
)
from bot.llm.media import download_media
from bot.tools.telegraph_extractor import extract_telegraph_content
from bot.tools.twitter_extractor import extract_twitter_content

logger = logging.getLogger(__name__)


TWITTER_HOST_ALLOWLIST = {
    'x.com',
    'www.x.com',
    'twitter.com',
    'www.twitter.com',
    'mobile.twitter.com',
    'm.twitter.com',
    'fxtwitter.com',
    'www.fxtwitter.com',
    'vxtwitter.com',
    'www.vxtwitter.com',
    'fixupx.com',
    'www.fixupx.com',
    'fixvx.com',
    'www.fixvx.com',
    'twittpr.com',
    'www.twittpr.com',
    'pxtwitter.com',
    'www.pxtwitter.com',
    'tweetpik.com',
    'www.tweetpik.com',
}

TWITTER_URL_PATTERN = re.compile(
    r"(https?://(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com|"
    r"fxtwitter\.com|vxtwitter\.com|fixupx\.com|fixvx\.com|twittpr\.com|pxtwitter\.com|"
    r"tweetpik\.com)/[^\s]+)",
    re.IGNORECASE,
)

def markdown_to_telegraph_nodes(md_content: str) -> List[Dict]:
    """Convert markdown content to Telegraph node format.

    Args:
        md_content: Content in markdown format

    Returns:
        List of Telegraph node objects
    """
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)

    # Parse HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Convert to Telegraph nodes
    return html_to_telegraph_nodes(soup)


def html_to_telegraph_nodes(element) -> List[Dict]:
    """Convert HTML elements to Telegraph node format recursively.

    Args:
        element: BeautifulSoup element

    Returns:
        List of Telegraph node objects
    """
    nodes = []

    # Process all child elements
    for child in element.children:
        # Text node
        if child.name is None:
            text = child.string.strip()
            if text:
                nodes.append(text)
        # Element node
        else:
            tag_name = child.name
            node_content = {}

            # Handle table specifically
            if tag_name == "table":
                table_html_str = str(child)
                # Remove newlines from table_html_str to prevent html2text from adding too many blank lines
                table_html_str = table_html_str.replace("\n", "")
                table_text = html2text(table_html_str)
                # Strip leading/trailing whitespace, but preserve internal formatting
                stripped_table_text = table_text.strip()
                if stripped_table_text:  # Only add if there's content
                    nodes.append({"tag": "pre", "children": [stripped_table_text]})
                continue

            # Supported tags list (excluding table, handled above)
            supported_tags = [
                "p",
                "aside",
                "ul",
                "ol",
                "li",
                "blockquote",
                "pre",
                "code",
                "a",
                "b",
                "strong",
                "i",
                "em",
                "u",
                "s",
                "br",
                "hr",
                "img",
                "video",
                "figcaption",
                "figure",
            ]
            header_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]

            if tag_name in header_tags:
                node_content["tag"] = "h4"  # Map all h1-h6 to h4
            elif tag_name in supported_tags:
                node_content["tag"] = tag_name
            else:
                # Unsupported tag (that is not a table): recursively process children
                nodes.extend(html_to_telegraph_nodes(child))
                continue  # Skip creating a node for the unsupported tag itself

            # Add attributes if needed for supported tags
            if tag_name == "a" and child.get("href"):
                node_content["attrs"] = {"href": child["href"]}
            elif tag_name == "img" and child.get("src"):
                node_content["attrs"] = {"src": child["src"]}
                if child.get("alt"):
                    node_content["attrs"]["alt"] = child["alt"]
            # Note: video, figure, figcaption attributes might be needed too if they use them.

            # Process children recursively for the current supported tag
            processed_children = html_to_telegraph_nodes(child)
            if processed_children:
                node_content["children"] = processed_children

            nodes.append(node_content)

    return nodes


async def create_telegraph_page(title: str, content: str) -> Optional[str]:
    """Create a Telegraph page with the provided markdown content.

    Args:
        title: The title of the page.
        content: The content of the page in markdown format.

    Returns:
        The URL of the created page, or None if creation failed.
    """
    try:
        # Convert markdown to Telegraph nodes
        nodes = markdown_to_telegraph_nodes(content)

        # Create the page
        response = requests.post(
            "https://api.telegra.ph/createPage",
            data={
                "access_token": TELEGRAPH_ACCESS_TOKEN,
                "author_name": TELEGRAPH_AUTHOR_NAME,
                "author_url": TELEGRAPH_AUTHOR_URL,
                "title": title,
                "content": json.dumps(nodes, ensure_ascii=False),
                "return_content": "false",
            },
            timeout=10,
        )

        response_data = response.json()

        if response_data.get("ok"):
            return response_data["result"]["url"]
        else:
            # This is an error from the API, not a Python exception in this block.
            # If we wanted to log the response_data itself for debugging, that's different.
            # For now, not adding exc_info=True as 'e' is not in this scope.
            logger.error(
                "Failed to create Telegraph page: %s", response_data.get("error")
            )
            return None

    except Exception as e:  # noqa: BLE001
        logger.error("Error creating Telegraph page: %s", e, exc_info=True)
        return None


def extract_youtube_urls(text: str, max_urls: int = 10):
    """Extract up to max_urls YouTube video URLs (including shorts) from the given text and replace them with a marker.
    Returns (modified_text, list_of_urls).
    """
    if not text:
        return text, []
    # Regex for YouTube URLs (full, short, and shorts)
    pattern = r"((?:https?://)?(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([\w-]{11})(?:[\?&][^\s]*)?)"
    matches = list(re.finditer(pattern, text))
    urls = []
    new_text = text
    count = 0
    for match in reversed(matches):  # reversed so replacement doesn't affect indices
        if count >= max_urls:
            break
        vid_id = match.group(2)
        url = f"https://www.youtube.com/watch?v={vid_id}"
        urls.insert(0, url)  # maintain order
        # Replace the entire matched URL in text
        start, end = match.span(0)
        new_text = new_text[:start] + f"YouTube_{vid_id}" + new_text[end:]
        count += 1
    return new_text, urls


async def extract_telegraph_urls_and_content(
    text: str, message_entities=None, max_urls: int = 5
):
    """Extract up to max_urls Telegraph URLs from the given text and message entities, replace them with their content.
    Returns (modified_text, list_of_extracted_content).

    Args:
        text: The message text
        message_entities: Telegram message entities (for embedded links)
        max_urls: Maximum number of URLs to process
    """
    if not text:
        return text, []

    extracted_contents = []
    new_text = text
    count = 0
    telegraph_urls = []

    # First, extract URLs from message entities (embedded links)
    if message_entities:
        for entity in message_entities:
            if count >= max_urls:
                break

            # Check for URL entities or text_link entities
            if entity.type in ["url", "text_link"]:
                url = None
                if entity.type == "url":
                    # Extract URL from the text using entity offset and length
                    url = text[entity.offset : entity.offset + entity.length]
                elif entity.type == "text_link":
                    # Use the URL from the entity
                    url = entity.url

                # Check if it's a Telegraph URL
                if url and "telegra.ph" in url:
                    telegraph_urls.append(
                        {
                            "url": url,
                            "offset": entity.offset,
                            "length": entity.length,
                            "is_entity": True,
                        }
                    )
                    count += 1

    # Then, extract plain text URLs using regex
    pattern = r"(https?://telegra\.ph/[^\s]+)"
    matches = list(re.finditer(pattern, text))

    for match in matches:
        if count >= max_urls:
            break

        telegraph_url = match.group(1)
        # Check if this URL is already found in entities to avoid duplicates
        if not any(existing["url"] == telegraph_url for existing in telegraph_urls):
            telegraph_urls.append(
                {
                    "url": telegraph_url,
                    "offset": match.start(),
                    "length": match.end() - match.start(),
                    "is_entity": False,
                }
            )
            count += 1

    # Process Telegraph URLs (in reverse order to maintain text indices)
    for telegraph_info in reversed(sorted(telegraph_urls, key=lambda x: x["offset"])):
        telegraph_url = telegraph_info["url"]
        try:
            logger.info("Extracting content from Telegraph URL: %s", telegraph_url)
            content_data = extract_telegraph_content(telegraph_url)

            # Format the extracted content
            content_text = content_data.get("text_content", "")
            image_urls = content_data.get("image_urls", [])
            video_urls = content_data.get("video_urls", [])

            formatted_content = f"\n\n--- Telegraph Content ---\n{content_text}"
            if image_urls:
                formatted_content += (
                    f"\n\nImages in Telegraph page: {len(image_urls)} images"
                )
                # Note: We could download these images too, but for now just mention them
            if video_urls:
                formatted_content += (
                    f"\nVideos in Telegraph page: {len(video_urls)} videos"
                )
            formatted_content += "\n--- End Telegraph Content ---\n\n"

            extracted_contents.insert(
                0,
                {
                    "url": telegraph_url,
                    "text_content": content_text,
                    "image_urls": image_urls,
                    "video_urls": video_urls,
                    "formatted_content": formatted_content,
                },
            )

            # Replace the URL in text with the content
            start = telegraph_info["offset"]
            end = start + telegraph_info["length"]
            new_text = new_text[:start] + formatted_content + new_text[end:]

        except Exception as e:
            logger.error(
                "Error extracting Telegraph content from %s: %s",
                telegraph_url,
                e,
            )
            # Keep the original URL if extraction fails
            extracted_contents.insert(
                0,
                {
                    "url": telegraph_url,
                    "text_content": f"[Telegraph content extraction failed for {telegraph_url}]",
                    "image_urls": [],
                    "video_urls": [],
                    "formatted_content": f"\n[Telegraph content extraction failed for {telegraph_url}]\n",
                },
            )

    return new_text, extracted_contents


async def download_telegraph_media(
    telegraph_contents: List[dict], max_images: int = 5, max_videos: int = 2
) -> Tuple[List[bytes], Optional[bytes], Optional[str]]:
    """Download images and videos from Telegraph content.

    Args:
        telegraph_contents: List of Telegraph content dictionaries
        max_images: Maximum number of images to download
        max_videos: Maximum number of videos to download

    Returns:
        Tuple of (image_data_list, video_data, video_mime_type)
    """
    image_data_list = []
    video_data = None
    video_mime_type = None

    image_count = 0
    video_count = 0

    for content in telegraph_contents:
        # Download images
        if image_count < max_images:
            for img_url in content.get("image_urls", []):
                if image_count >= max_images:
                    break
                try:
                    logger.info("Downloading Telegraph image: %s", img_url)
                    img_bytes = await download_media(img_url)
                    if img_bytes:
                        image_data_list.append(img_bytes)
                        image_count += 1
                        logger.info(
                            "Successfully downloaded Telegraph image %s",
                            image_count,
                        )
                    else:
                        logger.warning("Failed to download Telegraph image: %s", img_url)
                except Exception as e:
                    logger.error("Error downloading Telegraph image %s: %s", img_url, e)

        # Download first video only (Gemini typically handles one video at a time)
        if video_count < max_videos and not video_data:
            for vid_url in content.get("video_urls", []):
                if video_data:  # Only download the first video
                    break
                try:
                    logger.info("Downloading Telegraph video: %s", vid_url)
                    vid_bytes = await download_media(vid_url)
                    if vid_bytes:
                        video_data = vid_bytes
                        # Try to determine MIME type from URL
                        if vid_url.endswith(".mp4"):
                            video_mime_type = "video/mp4"
                        elif vid_url.endswith(".webm"):
                            video_mime_type = "video/webm"
                        elif vid_url.endswith(".mov"):
                            video_mime_type = "video/quicktime"
                        else:
                            video_mime_type = "video/mp4"  # Default

                        video_count += 1
                        logger.info(
                            "Successfully downloaded Telegraph video with MIME type: %s",
                            video_mime_type,
                        )
                        break
                    else:
                        logger.warning("Failed to download Telegraph video: %s", vid_url)
                except Exception as e:
                    logger.error("Error downloading Telegraph video %s: %s", vid_url, e)

    logger.info(
        "Downloaded %s images and %s video from Telegraph content",
        len(image_data_list),
        1 if video_data else 0,
    )
    return image_data_list, video_data, video_mime_type


def _is_twitter_url(url: str) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return False

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host in TWITTER_HOST_ALLOWLIST:
        return True
    return host.endswith(".x.com") or host.endswith(".twitter.com")


async def extract_twitter_urls_and_content(
    text: str,
    message_entities=None,
    max_urls: int = 5,
    source_text: Optional[str] = None,
):
    """Extract Twitter/X URLs, inline their content, and collect metadata."""
    if not text:
        return text, []

    base_text = source_text if source_text is not None else text
    replacements: List[Dict[str, str]] = []
    count = 0

    if message_entities:
        for entity in message_entities:
            if count >= max_urls:
                break
            if entity.type not in {"url", "text_link"}:
                continue

            if entity.type == "url":
                substring = base_text[entity.offset : entity.offset + entity.length]
                url_candidate = substring
            else:
                substring = base_text[entity.offset : entity.offset + entity.length]
                url_candidate = entity.url

            if not url_candidate or not _is_twitter_url(url_candidate):
                continue

            replacements.append(
                {
                    "url": url_candidate,
                    "offset": entity.offset,
                    "substring": substring or "",
                }
            )
            count += 1

    if count < max_urls:
        for match in TWITTER_URL_PATTERN.finditer(base_text):
            if count >= max_urls:
                break
            url_candidate = match.group(1)
            if not _is_twitter_url(url_candidate):
                continue
            if any(
                existing["offset"] == match.start() and existing["url"] == url_candidate
                for existing in replacements
            ):
                continue
            replacements.append(
                {
                    "url": url_candidate,
                    "offset": match.start(),
                    "substring": match.group(0),
                }
            )
            count += 1

    if not replacements:
        return text, []

    new_text = text
    extracted_contents: List[Dict] = []

    for info in reversed(sorted(replacements, key=lambda item: item["offset"])):
        url = info["url"]
        substring = info["substring"]
        try:
            content_data = extract_twitter_content(url)
            formatted_content = content_data.get("formatted_content", "")
            if not formatted_content:
                formatted_content = f"\n[Twitter content extracted from {url}]\n"

            if not substring:
                new_text = f"{new_text}\n{formatted_content}"
            else:
                position = new_text.rfind(substring)
                if position == -1:
                    logger.debug(
                        "Unable to locate Twitter substring '%s' in text; appending content.",
                        substring,
                    )
                    new_text = f"{new_text}\n{formatted_content}"
                else:
                    new_text = (
                        f"{new_text[:position]}{formatted_content}{new_text[position + len(substring):]}"
                    )

            extracted_contents.insert(0, content_data)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error extracting Twitter content from %s: %s", url, exc)
            fallback = f"\n[Twitter content extraction failed for {url}]\n"
            if not substring:
                new_text = f"{new_text}{fallback}"
            else:
                position = new_text.rfind(substring)
                if position == -1:
                    new_text = f"{new_text}{fallback}"
                else:
                    new_text = (
                        f"{new_text[:position]}{fallback}{new_text[position + len(substring):]}"
                    )
            extracted_contents.insert(
                0,
                {
                    "url": url,
                    "text_content": f"[Twitter content extraction failed for {url}]",
                    "image_urls": [],
                    "video_urls": [],
                    "formatted_content": fallback,
                },
            )

    return new_text, extracted_contents


async def download_twitter_media(
    twitter_contents: List[dict],
    max_images: int = 5,
    max_videos: int = 1,
) -> Tuple[List[bytes], Optional[bytes], Optional[str]]:
    """Download images and videos referenced in extracted Twitter content."""
    image_data_list: List[bytes] = []
    video_data: Optional[bytes] = None
    video_mime_type: Optional[str] = None

    image_count = 0
    video_count = 0

    for content in twitter_contents:
        if image_count < max_images:
            for img_url in content.get("image_urls", []):
                if image_count >= max_images:
                    break
                try:
                    logger.info("Downloading Twitter image: %s", img_url)
                    img_bytes = await download_media(img_url)
                    if img_bytes:
                        image_data_list.append(img_bytes)
                        image_count += 1
                    else:
                        logger.warning("Failed to download Twitter image: %s", img_url)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error downloading Twitter image %s: %s", img_url, exc)

        if video_data or video_count >= max_videos:
            continue

        for vid_url in content.get("video_urls", []):
            if video_data:
                break
            try:
                logger.info("Downloading Twitter video: %s", vid_url)
                vid_bytes = await download_media(vid_url)
                if vid_bytes:
                    video_data = vid_bytes
                    lowered = vid_url.lower()
                    if lowered.endswith(".m3u8"):
                        video_mime_type = "application/x-mpegURL"
                    elif lowered.endswith(".mpd"):
                        video_mime_type = "application/dash+xml"
                    elif lowered.endswith(".webm"):
                        video_mime_type = "video/webm"
                    else:
                        video_mime_type = "video/mp4"
                    video_count += 1
                else:
                    logger.warning("Failed to download Twitter video: %s", vid_url)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error downloading Twitter video %s: %s", vid_url, exc)

    logger.info(
        "Downloaded %s Twitter image(s) and %s video(s)",
        len(image_data_list),
        1 if video_data else 0,
    )
    return image_data_list, video_data, video_mime_type
