import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

logger = logging.getLogger(__name__)

_SUPPORTED_HOST_TOKENS = {
    "x.com",
    "twitter.com",
    "fxtwitter.com",
    "vxtwitter.com",
    "fixupx.com",
    "fixvx.com",
    "twittpr.com",
    "pxtwitter.com",
    "tweetpik.com",
}

_STOP_MARKERS = {
    "new to x?",
    "join x today",
    "sign up now to get your own personalized timeline!",
    "sign up",
    "log in",
    "tweet your reply",
    "trending now",
    "what's happening",
    "terms of service",
    "privacy policy",
    "cookie policy",
    "accessibility",
    "ads info",
}

_STOP_PREFIXES = (
    "watch on",
    "show more",
    "related",
    "more replies",
    "explore",
    "tweet your reply",
)

_MEDIA_PATTERN = re.compile(r"!\[[^\]]*?\]\((https?://[^\)]+)\)", re.IGNORECASE)
_LINK_PATTERN = re.compile(r"\[(?!\s*!\[)([^\]]*?)\]\((https?://[^\)]+)\)", re.IGNORECASE)
_EMPTY_LINK_PATTERN = re.compile(r"\[\s*\]\((https?://[^\)]+)\)", re.IGNORECASE)
_TIMESTAMP_PATTERN = re.compile(r"\d{1,2}:\d{2}\s?[AP]M", re.IGNORECASE)
_MONTH_TOKENS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
)
_PUNCT_PREFIXES = (".", ",", ";", ":", ")", "]", "}", "!", "?")
_PROFILE_MEDIA_TOKENS = (
    "profile_images",
    "profile_banners",
    "semantic_core_img",
    "/emoji/",
)
_VIDEO_EXTENSIONS = (".mp4", ".m3u8", ".mpd")
_REQUEST_TIMEOUT = 20
_USER_AGENT = "TelegramGroupHelperBot/0.1 (+https://github.com/sailself/TelegramGroupHelperBot)"


def _is_supported_host(host: str) -> bool:
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.endswith("x.com"):
        return True
    return any(host.endswith(token) for token in _SUPPORTED_HOST_TOKENS)


def _normalize_status_url(raw_url: str) -> str:
    if not raw_url:
        raise ValueError("Empty URL provided for Twitter extraction")

    raw_url = raw_url.strip()
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        raw_url = f"https://{raw_url}"

    parsed = urlparse(raw_url)
    host = parsed.netloc.lower()
    if not _is_supported_host(host):
        raise ValueError(f"Unsupported Twitter/X host: {host}")

    path = parsed.path or ""
    if not path:
        raise ValueError("Twitter/X URL is missing a path component")
    if "/status/" not in path:
        raise ValueError("Twitter/X URL does not reference a status update")

    canonical_url = urlunparse((
        "https",
        "x.com",
        path,
        "",
        parsed.query,
        "",
    ))
    return canonical_url


def _build_proxy_url(normalized_url: str) -> str:
    stripped = normalized_url.replace("https://", "", 1)
    return f"https://r.jina.ai/https://{stripped}"


def _append_unique(target: List[str], item: str) -> None:
    if item and item not in target:
        target.append(item)


def _normalize_media_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    path = parsed.path
    query = parsed.query

    if not netloc and path:
        # Handle URLs like //pbs.twimg.com/...
        if path.startswith("//"):
            netloc = path[2:].split("/", 1)[0]
            path = "/" + path[2 + len(netloc):]
    if netloc.endswith("twimg.com"):
        query_map = parse_qs(query, keep_blank_values=True)
        if "name" in query_map:
            query_map["name"] = ["orig"]
        query = urlencode(query_map, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def _is_profile_media(url: str) -> bool:
    lowered = url.lower()
    return any(token in lowered for token in _PROFILE_MEDIA_TOKENS)


def _is_video_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith(_VIDEO_EXTENSIONS) or "video.twimg.com" in lowered


def _looks_like_timestamp(text: str) -> bool:
    if _TIMESTAMP_PATTERN.search(text):
        return True
    lowered = text.lower()
    if "am" in lowered or "pm" in lowered:
        return any(month in lowered for month in _MONTH_TOKENS)
    return False


def _collect_relevant_lines(markdown_block: str) -> List[str]:
    lines = markdown_block.splitlines()
    relevant: List[str] = []
    collecting = False
    seen_content = False

    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()

        if not collecting:
            if lowered == "conversation":
                collecting = True
            continue

        if not seen_content:
            if not stripped or set(stripped) <= {"-", "—", "–"}:
                continue
            seen_content = True

        if not stripped:
            if relevant and relevant[-1] != "":
                relevant.append("")
            continue

        if lowered in _STOP_MARKERS or any(lowered.startswith(prefix) for prefix in _STOP_PREFIXES):
            break

        relevant.append(line)

    if not relevant:
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered in _STOP_MARKERS or any(lowered.startswith(prefix) for prefix in _STOP_PREFIXES):
                break
            relevant.append(line)
    return relevant


def _clean_lines_and_media(lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
    cleaned: List[str] = []
    image_urls: List[str] = []
    video_urls: List[str] = []

    for line in lines:
        working = line
        for match in _MEDIA_PATTERN.finditer(working):
            media_url = _normalize_media_url(match.group(1))
            if _is_profile_media(media_url):
                continue
            if _is_video_url(media_url):
                _append_unique(video_urls, media_url)
            else:
                _append_unique(image_urls, media_url)
        working = _MEDIA_PATTERN.sub("", working)
        working = _EMPTY_LINK_PATTERN.sub("", working)

        def link_replacer(match: re.Match) -> str:
            text = match.group(1).strip()
            if not text:
                return ""
            return text

        working = _LINK_PATTERN.sub(link_replacer, working)
        working = re.sub(r"\s+", " ", working).strip()
        if not working:
            continue

        if (
            cleaned
            and not cleaned[-1].endswith((".", "!", "?", ":"))
            and working.startswith(("@", "#"))
        ):
            cleaned[-1] = f"{cleaned[-1]} {working}"
            continue

        if cleaned and working[0] in _PUNCT_PREFIXES:
            cleaned[-1] = f"{cleaned[-1]}{working}"
            continue

        cleaned.append(working)

    return cleaned, image_urls, video_urls


def _extract_metadata(cleaned_lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    display_name: Optional[str] = None
    handle: Optional[str] = None
    handle_index: Optional[int] = None

    for idx, line in enumerate(cleaned_lines[:6]):
        if line.startswith("@") and " " not in line:
            prev = next(
                (cleaned_lines[j] for j in range(idx - 1, -1, -1) if cleaned_lines[j]),
                None,
            )
            handle = line
            handle_index = idx
            if prev:
                display_name = prev
            break
    return display_name, handle, handle_index


def _strip_indices(cleaned_lines: List[str], indexes: List[Optional[int]]) -> List[str]:
    skip = {idx for idx in indexes if idx is not None}
    return [line for idx, line in enumerate(cleaned_lines) if idx not in skip]


def extract_twitter_content(url: str) -> Dict[str, Any]:
    """Extract text, image URLs, and video URLs from a Twitter/X status link."""
    try:
        normalized_url = _normalize_status_url(url)
    except ValueError as exc:
        logger.error("Twitter extraction rejected URL %s: %s", url, exc)
        raise

    proxy_url = _build_proxy_url(normalized_url)
    logger.info("Fetching Twitter/X content via proxy: %s", proxy_url)

    try:
        response = requests.get(
            proxy_url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Error fetching Twitter/X content from %s: %s", proxy_url, exc)
        raise Exception(f"Error fetching Twitter/X content: {exc}") from exc

    raw_text = response.text
    marker = "Markdown Content:\n"
    marker_idx = raw_text.find(marker)
    if marker_idx == -1:
        raise Exception("Unable to locate Twitter/X markdown content in response")

    markdown_block = raw_text[marker_idx + len(marker) :]
    relevant_lines = _collect_relevant_lines(markdown_block)
    cleaned_lines, image_urls, video_urls = _clean_lines_and_media(relevant_lines)

    if not cleaned_lines and not image_urls and not video_urls:
        raise Exception("No content extracted from Twitter/X response")

    display_name, handle, handle_idx = _extract_metadata(cleaned_lines)
    timestamp_idx = None
    timestamp_text = None
    for idx, line in enumerate(cleaned_lines):
        if _looks_like_timestamp(line):
            timestamp_idx = idx
            timestamp_text = line
            break

    body_lines = _strip_indices(cleaned_lines, [handle_idx, cleaned_lines.index(display_name) if display_name in cleaned_lines else None, timestamp_idx])

    body_text = "\n".join(body_lines).strip()

    header_parts: List[str] = []
    if display_name:
        header_parts.append(display_name)
    if handle and handle not in header_parts:
        header_parts.append(handle)
    header_text = ""
    if header_parts:
        header_text = f"Tweet by {' '.join(header_parts)}"
    if timestamp_text:
        header_text = f"{header_text} at {timestamp_text}" if header_text else f"Tweet at {timestamp_text}"

    sections = [section for section in (header_text.strip(), body_text, f"Original link: {normalized_url}") if section]
    text_content = "\n\n".join(sections)

    formatted_content = "\n\n--- Twitter Content ---\n" + text_content
    if image_urls:
        formatted_content += f"\n\nImages attached: {len(image_urls)} image(s)"
    if video_urls:
        formatted_content += f"\nVideos attached: {len(video_urls)} video(s)"
    formatted_content += "\n--- End Twitter Content ---\n\n"

    return {
        "url": normalized_url,
        "text_content": text_content,
        "image_urls": image_urls,
        "video_urls": video_urls,
        "formatted_content": formatted_content,
        "metadata": {
            "display_name": display_name,
            "handle": handle,
            "timestamp": timestamp_text,
        },
    }
