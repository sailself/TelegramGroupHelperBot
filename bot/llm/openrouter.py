"""OpenRouter helpers and response parsing utilities."""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from openai import BadRequestError, NotFoundError, OpenAIError, RateLimitError

from bot.config import (
    ENABLE_JINA_MCP,
    GPT_MODEL,
    JINA_AI_API_KEY,
    OPENROUTER_TEMPERATURE,
    OPENROUTER_TOP_K,
    OPENROUTER_TOP_P,
    QWEN_MODEL,
)
from .clients import get_openrouter_client, get_openrouter_responses_client
from .jina_search import jina_search_tool

logger = logging.getLogger(__name__)

_MCP_SERVER_URL = "https://mcp.jina.ai/sse"
_MCP_SERVER_LABEL = "jina-mcp"
_MCP_SERVER_DESCRIPTION = (
    "Jina AI MCP server providing Reader, web search, and reranker tools."
)
MAX_TOOL_CALL_ITERATIONS = 3
TOOL_LIMIT_SYSTEM_PROMPT = (
    "Tool call limit reached. Provide the best possible answer using the available information without requesting more tool calls."
)




def parse_gpt_content(content: str) -> dict[str, str]:
    """Split OpenRouter GPT reasoning output into analysis and final sections."""
    # Find the last <|message|> tag
    last_message_pos = content.rfind("<|message|>")

    if last_message_pos == -1:
        # No <|message|> tag found, treat entire content as final
        analysis = ""
        final = content
    else:
        # Split at the last <|message|> tag
        analysis = content[:last_message_pos]
        final = content[last_message_pos + len("<|message|>"):]

    # Clean up tags from both parts
    analysis = re.sub(r"<\|.*?\|>", "", analysis).strip()
    final = re.sub(r"<\|.*?\|>", "", final).strip()

    # Replace assistantSearch with Search in both parts
    analysis = analysis.replace("assistantSearch", "Search")
    final = final.replace("assistantSearch", "Search")

    return {
        "analysis": analysis,
        "final": final,
    }


def parse_qwen_content(content: str) -> dict[str, str]:
    """Split Qwen output into thought and final sections using <think> tags."""
    match = re.search(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
    if match:
        analysis = match.group(1).strip()
        final = match.group(2).strip()
    else:
        analysis = ""
        final = content.strip()
    return {"analysis": analysis, "final": final}


def _parse_openrouter_response(model_name: str, content: str) -> Union[dict[str, str], str]:
    """Return parsed response based on the model type."""
    if model_name == GPT_MODEL:
        return parse_gpt_content(content)
    if model_name == QWEN_MODEL:
        return parse_qwen_content(content)
    return content


def _build_mcp_tool() -> Dict[str, Any]:
    """Construct the Jina MCP tool descriptor for the Responses API."""
    headers: Dict[str, str] = {}
    if JINA_AI_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_AI_API_KEY}"

    tool: Dict[str, Any] = {
        "type": "mcp",
        "server_label": _MCP_SERVER_LABEL,
        "server_url": _MCP_SERVER_URL,
        "server_description": _MCP_SERVER_DESCRIPTION,
    }
    if headers:
        tool["headers"] = headers
    return tool



def _build_function_tools() -> List[Dict[str, Any]]:
    """Define function tools exposed to the model via chat completions."""
    if not ENABLE_JINA_MCP:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": "jina_web_search",
                "description": "Perform a web search using the Jina AI search API and return summarised results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to look up.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default 5).",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def _message_to_dict(message: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {"role": message.role}
    if message.content is not None:
        if isinstance(message.content, list):
            data["content"] = [
                part.model_dump() if hasattr(part, "model_dump") else part
                for part in message.content
            ]
        else:
            data["content"] = message.content
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        data["tool_calls"] = [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
    return data


def _execute_function_tool(call: Any) -> str:
    if getattr(call, "type", None) != "function":
        return "Unsupported tool call type."

    name = getattr(call.function, "name", "") or ""
    raw_arguments = getattr(call.function, "arguments", "") or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        arguments = {}

    if name == "jina_web_search":
        query = arguments.get("query")
        max_results = arguments.get("max_results", 5)
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5

        if not query:
            return "Search failed: missing 'query' parameter."

        lowered = query.lower() if isinstance(query, str) else ""
        if lowered:
            logger.debug("Jina search requested with query: %s", query)
            location_keywords = ("where", "where's", "location", "identify the location", "what city")
            photo_keywords = ("photo", "picture", "image", "this photo", "this picture")
            if any(keyword in lowered for keyword in location_keywords) and any(keyword in lowered for keyword in photo_keywords):
                logger.debug("Skipping Jina search for location-from-photo query: %s", query)
                return "Search skipped: analyze the provided imagery and respond without additional web searches."

        try:
            payload = jina_search_tool(query, max_results=max_results)
            return payload.get("markdown", f"No results returned for {query}.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Jina search tool execution failed: %s", exc, exc_info=True)
            return f"Search failed: {exc}"

    return f"Unsupported tool call: {name}"


async def _chat_completion_with_tools(
    client: Any,
    *,
    base_messages: List[Dict[str, Any]],
    model_name: str,
    tools: List[Dict[str, Any]],
) -> Any:
    """Execute chat completions with optional tool-calling loop."""

    messages: List[Dict[str, Any]] = [message.copy() for message in base_messages]
    active_tools = list(tools) if tools else []
    tool_iterations = 0
    seen_calls: set[str] = set()

    while True:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=OPENROUTER_TEMPERATURE,
            top_p=OPENROUTER_TOP_P,
            extra_body={"top_k": OPENROUTER_TOP_K},
            tools=active_tools if active_tools else None,
            tool_choice="auto" if active_tools else None,
        )
        choice = completion.choices[0]
        message = choice.message
        messages.append(_message_to_dict(message))

        tool_calls = getattr(message, "tool_calls", None) or []
        if not active_tools or not tool_calls:
            return completion

        tool_iterations += 1
        if tool_iterations > MAX_TOOL_CALL_ITERATIONS:
            messages.append({"role": "system", "content": TOOL_LIMIT_SYSTEM_PROMPT})
            active_tools = []
            continue

        for call in tool_calls:
            signature = f"{getattr(call.function, 'name', '')}:{getattr(call.function, 'arguments', '')}"
            if signature in seen_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "Search skipped: this request was already executed. Provide the best answer with the available information.",
                    }
                )
                continue
            seen_calls.add(signature)
            tool_response = _execute_function_tool(call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_response,
                }
            )


def _should_use_responses(has_images: bool, has_video: bool, has_audio: bool) -> bool:
    """Determine whether to call the Responses API for better tool support."""
    # The OpenRouter Responses API alpha does not yet accept MCP tool descriptors.
    # Skip for now and rely on chat completions with function calling.
    return False


def _extract_responses_output(response: Any, model_name: str) -> Optional[Union[dict[str, str], str]]:
    """Normalize a Responses API payload into bot-friendly data."""
    if response is None:
        return None

    output_items = getattr(response, "output", None) or []
    analysis_parts: List[str] = []
    final_parts: List[str] = []

    for item in output_items:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            for summary in getattr(item, "summary", []) or []:
                summary_text = getattr(summary, "text", "").strip()
                if summary_text:
                    analysis_parts.append(summary_text)
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", "").strip()
                if text:
                    analysis_parts.append(text)
        elif item_type == "message":
            for content in getattr(item, "content", []) or []:
                content_type = getattr(content, "type", None)
                if content_type == "output_text":
                    text = getattr(content, "text", "").strip()
                    if text:
                        final_parts.append(text)
                elif content_type == "refusal":
                    text = getattr(content, "refusal", "").strip()
                    if text:
                        final_parts.append(text)

    final_text = "\n\n".join(part for part in final_parts if part).strip()
    analysis_text = "\n\n".join(part for part in analysis_parts if part).strip()

    if not final_text:
        direct_text = getattr(response, "output_text", None)
        if direct_text:
            final_text = str(direct_text).strip()

    parsed: Optional[Union[dict[str, str], str]]
    parsed = _parse_openrouter_response(model_name, final_text) if final_text else None

    if analysis_text:
        if isinstance(parsed, dict):
            existing = parsed.get("analysis")
            # Avoid item assignment if parsed is not a mutable mapping
            if hasattr(parsed, "copy"):
                parsed = parsed.copy()
            parsed.update({
                "analysis": f"{existing}\n\n{analysis_text}".strip() if existing else analysis_text
            })
        else:
            parsed = {"analysis": analysis_text, "final": parsed or ""}

    return parsed if parsed else None


async def _call_openrouter_with_responses(
    *,
    system_prompt: str,
    user_content: str,
    model_name: str,
) -> Any:
    """Invoke OpenRouter Responses API with Jina MCP enabled."""
    client = get_openrouter_responses_client()
    tools = [_build_mcp_tool()]
    return await client.responses.create(
        model=model_name,
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_content},
                ],
            }
        ],
        temperature=OPENROUTER_TEMPERATURE,
        top_p=OPENROUTER_TOP_P,
        extra_body={"top_k": OPENROUTER_TOP_K},
        tools=tools,
    )


async def call_openrouter(
    *,
    system_prompt: str,
    user_content: str,
    image_data_list: Optional[List[bytes]] = None,
    video_data: Optional[bytes] = None,
    video_mime_type: Optional[str] = None,
    use_pro_model: bool = False,
    youtube_urls: Optional[List[str]] = None,
    audio_data: Optional[bytes] = None,
    audio_mime_type: Optional[str] = None,
    model_name: str,
) -> Optional[Union[dict[str, str], str]]:
    """Call an OpenRouter model and return parsed output."""

    client = get_openrouter_client()

    augmented_user_content = user_content
    if youtube_urls:
        youtube_listing = "\n".join(
            f"YouTube URL: {url}" for url in youtube_urls
        )
        augmented_user_content += "\n" + youtube_listing
    has_images = bool(image_data_list)
    has_video = video_data is not None
    has_audio = audio_data is not None

    if _should_use_responses(has_images, has_video, has_audio):
        try:
            response = await _call_openrouter_with_responses(
                system_prompt=system_prompt,
                user_content=augmented_user_content,
                model_name=model_name,
            )
            parsed = _extract_responses_output(response, model_name)
            if parsed is not None:
                return parsed
        except (BadRequestError, NotFoundError) as e:
            logger.warning(
                "Responses API unavailable for model %s, falling back to Chat Completions: %s",
                model_name,
                e,
            )
        except OpenAIError as e:
            logger.error(
                "Responses API call failed for model %s: %s", model_name, e, exc_info=True
            )

    def build_messages(include_media: bool = True):
        user_parts = [{"type": "text", "text": augmented_user_content}]
        if include_media:
            if image_data_list:
                for img in image_data_list:
                    b64 = base64.b64encode(img).decode("utf-8")
                    user_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        }
                    )
            if video_data:
                b64 = base64.b64encode(video_data).decode("utf-8")
                mime_type = video_mime_type or "video/mp4"
                user_parts.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:{mime_type};base64,{b64}"},
                    }
                )
            if audio_data:
                b64 = base64.b64encode(audio_data).decode("utf-8")
                mime_type = audio_mime_type or "audio/mp3"
                user_parts.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:{mime_type};base64,{b64}"},
                    }
                )

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_parts},
        ]

    tools = _build_function_tools()

    async def invoke_chat(include_media: bool) -> Any:
        base_messages = build_messages(include_media=include_media)
        return await _chat_completion_with_tools(
            client,
            base_messages=base_messages,
            model_name=model_name,
            tools=tools,
        )

    try:
        completion = await invoke_chat(include_media=True)
        return _parse_openrouter_response(
            model_name, completion.choices[0].message.content
        )
    except RateLimitError as e:  # pragma: no cover - best effort
        logger.error("Rate limit error calling OpenRouter model %s: %s", model_name, e)
        return {
            "final": f"Model {model_name} is currently rate-limited. Please try again later.",
        }
    except (BadRequestError, NotFoundError) as e:
        status = getattr(e, "status_code", None) if hasattr(e, "status_code") else None
        if status in {400, 415} or "media" in str(e).lower():
            logger.warning(
                "Model %s does not support provided media, retrying without media. Error: %s",
                model_name,
                e,
            )
            try:
                completion = await invoke_chat(include_media=False)
                return _parse_openrouter_response(
                    model_name, completion.choices[0].message.content
                )
            except OpenAIError as inner_e:  # pragma: no cover - best effort
                logger.error(
                    "Retry without media failed for model %s: %s", model_name, inner_e, exc_info=True
                )
                return None
        logger.error("OpenRouter request failed for model %s: %s", model_name, e, exc_info=True)
        return None
    except OpenAIError as e:  # pragma: no cover - best effort
        logger.error("Error calling OpenRouter model %s: %s", model_name, e, exc_info=True)
        return None
