import logging
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

"""
Telegraph content extractor tool for MCP server.
Extracts text, image URLs, and video URLs from a Telegraph page.
"""

def extract_telegraph_content(url: str) -> dict:
    """
    Extracts text content, image URLs, and video URLs from a Telegraph page.

    Args:
        url (str): The Telegraph page URL.
    Returns:
        dict: A dictionary with keys 'text_content', 'image_urls', and 'video_urls'.
    Raises:
        Exception: If the page cannot be fetched or parsed.
    """
    logger.info(f"Starting extraction for url: {url}")
    parsed_url = urlparse(url)
    path = parsed_url.path.lstrip('/')  # Remove leading slash

    if not path:
        logger.error("Invalid Telegraph URL: Missing path component.")
        raise ValueError("Invalid Telegraph URL: Missing path component.")

    api_url = f"https://api.telegra.ph/getPage/{path}?return_content=true"
    logger.debug(f"Telegraph API URL: {api_url}")

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Telegraph page: {e}")
        raise Exception(f"Error fetching Telegraph page: {e}") from e

    try:
        data = response.json()
    except ValueError as e:
        logger.error("Error parsing JSON response from Telegraph API.")
        raise Exception("Error parsing JSON response from Telegraph API.") from e

    if not data.get("ok"):
        error_message = data.get("error", "Unknown error from Telegraph API")
        logger.error(f"Telegraph API error: {error_message}")
        raise Exception(f"Telegraph API error: {error_message}")

    content_nodes = data.get("result", {}).get("content", [])

    text_content = ""
    image_urls = []
    video_urls = []
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
                elif tag == "video" or (tag == "figure" and any(c.get("tag") == "video" for c in children if isinstance(c,dict))):
                    video_counter += 1
                    video_node = None
                    if tag == "video":
                        video_node = node
                    else: # figure contains video
                        for child_node in children:
                            if isinstance(child_node, dict) and child_node.get("tag") == "video":
                                video_node = child_node
                                break
                    if video_node:
                        src = video_node.get("attrs", {}).get("src")
                        if src:
                            if src.startswith("/"):
                                src = "https://telegra.ph" + src
                            video_urls.append(src)
                            current_text += f"[video_{video_counter}]"
                elif tag == "iframe": # Often used for embedding videos
                    video_counter += 1
                    src = node.get("attrs", {}).get("src")
                    if src:
                        # Fix for common relative iframe URLs from telegra.ph
                        if src.startswith("/embed/youtube"): # Example: YouTube
                            src = "https://www.youtube.com" + src
                        elif src.startswith("/embed/vimeo"): # Example: Vimeo
                            src = "https://player.vimeo.com" + src
                        elif src.startswith("/"):
                            src = "https://telegra.ph" + src
                        video_urls.append(src)
                        current_text += f"[video_{video_counter}]"
                elif tag in ["p", "a", "li", "h3", "h4", "em", "strong", "figcaption", "blockquote", "code", "span"]:
                    current_text += process_node_children(children)
                    if tag in ["p", "h3", "h4", "li", "blockquote"]:
                        current_text += "\n"
                elif tag == "figure": # Handle figure if it's not a video (e.g. image with figcaption)
                    current_text += process_node_children(children) # Process children like img, figcaption
                    current_text += "\n"
                elif tag == "br":
                    current_text += "\n"
                elif tag == "hr":
                    current_text += "\n---\n"
                elif tag in ["ul", "ol"]:
                    for child in children: # Iterate over li
                        if isinstance(child, dict) and child.get("tag") == "li":
                            current_text += "- " + process_node_children(child.get("children", [])) + "\n"
                        else: # Handle case where li might be nested deeper or direct text
                            current_text += process_node_children([child])
                    current_text += "\n"
                elif tag == "pre": # Often contains a code block
                    code_block_text = ""
                    for child in children:
                        if isinstance(child, dict) and child.get("tag") == "code":
                            code_block_text += process_node_children(child.get("children", []))
                        else:
                            code_block_text += process_node_children([child])
                    current_text += f"\n```\n{code_block_text.strip()}\n```\n"
                # Potentially handle other tags: aside, details, summary, etc.
                # For now, we just recursively process their children if any
                elif children:
                    current_text += process_node_children(children)
        return current_text

    text_content = process_node_children(content_nodes)

    result = {
        "text_content": text_content.strip(),
        "image_urls": image_urls,
        "video_urls": video_urls,
    }
    logger.debug(f"Extraction result: text length={len(result['text_content'])}, images={len(result['image_urls'])}, videos={len(result['video_urls'])}")
    return result

if __name__ == '__main__':
    # Example Usage (for testing only)
    # Replace with a valid Telegraph URL
    test_url = "https://telegra.ph/Gifs-FAQ" # Page describing the API, likely has diverse content

    print(f"Testing with URL: {test_url}")

    try:
        content = extract_telegraph_content(test_url)
        print("\nExtracted Content:")
        print("------------------")
        print("Text Content:")
        print(content["text_content"])
        print("\nImage URLs:")
        for img_url in content["image_urls"]:
            print(img_url)
        print("\nVideo URLs:")
        for vid_url in content["video_urls"]:
            print(vid_url)
    except Exception as e:
        print(f"\nError during extraction: {e}")

    # Test with an invalid URL
    print("\nTesting with an invalid URL:")
    try:
        extract_telegraph_content("https://example.com/not-a-telegraph-page")
    except Exception as e:
        print(f"Error (expected): {e}")

    print("\nTesting with a non-existent Telegraph page:")
    try:
        extract_telegraph_content("https://telegra.ph/NonExistentPage-12-31")
    except Exception as e:
        print(f"Error (expected for non-existent page): {e}")

    print("\nTesting with a URL with no path:")
    try:
        extract_telegraph_content("https://telegra.ph/")
    except Exception as e:
        print(f"Error (expected for URL with no path): {e}")
