"""OpenRouter helpers and response parsing utilities."""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from openai import BadRequestError, NotFoundError, OpenAIError, RateLimitError

from bot.config import (
    ENABLE_EXA_SEARCH,
    EXA_API_KEY,
    GPT_MODEL,
    OPENROUTER_TEMPERATURE,
    OPENROUTER_TOP_K,
    OPENROUTER_TOP_P,
    QWEN_MODEL,
)
from .clients import get_openrouter_client
from .exa_search import exa_search_tool, ExaSearchError

logger = logging.getLogger(__name__)

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


def _build_function_tools() -> List[Dict[str, Any]]:
    """Define function tools exposed to the model via chat completions."""

    if not ENABLE_EXA_SEARCH or not EXA_API_KEY:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": "exa_web_search",
                "description": "Search the web using Exa AI and return a concise Markdown summary of the results.",
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


def _coerce_text(value: Any) -> str:
    """Convert arbitrary message content structures into plain text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_coerce_text(part) for part in value]
        return "\n".join(part for part in parts if part.strip())
    if isinstance(value, dict):
        if "text" in value:
            return _coerce_text(value.get("text"))
        if {"type", "text"} <= value.keys() and value.get("type") == "output_text":
            return _coerce_text(value.get("text"))
        # Fall back to first textual field
        for key in ("content", "value", "message"):
            if key in value:
                return _coerce_text(value.get(key))
        return ""
    text = getattr(value, "text", None)
    if text is not None:
        return _coerce_text(text)
    if hasattr(value, "model_dump"):
        return _coerce_text(value.model_dump())
    return ""


def _extract_reasoning_text(payload: Any) -> Optional[str]:
    """Recursively search for reasoning text within model_extra payloads."""

    if payload is None:
        return None
    if isinstance(payload, str):
        stripped = payload.strip()
        return stripped or None
    if isinstance(payload, dict):
        for key in ("reasoning", "thoughts", "thinking", "explanation", "text", "output_text"):
            if key in payload:
                candidate = _extract_reasoning_text(payload.get(key))
                if candidate:
                    return candidate
        for value in payload.values():
            candidate = _extract_reasoning_text(value)
            if candidate:
                return candidate
        return None
    if isinstance(payload, list):
        for item in payload:
            candidate = _extract_reasoning_text(item)
            if candidate:
                return candidate
        return None
    if hasattr(payload, "model_dump"):
        return _extract_reasoning_text(payload.model_dump())
    return None


def _message_content_to_text(message: Any) -> str:
    """Extract the best-effort text response from a chat completion message."""

    content = _coerce_text(getattr(message, "content", ""))
    if content.strip():
        return content

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        return content

    extra = getattr(message, "model_extra", None)
    if not extra and hasattr(message, "additional_kwargs"):
        extra = getattr(message, "additional_kwargs", {}).get("model_extra")
    reasoning = _extract_reasoning_text(extra)
    if reasoning:
        logger.debug("Falling back to reasoning text from model_extra for final response.")
        return reasoning

    return content


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

    if name == "exa_web_search":
        query = arguments.get("query")
        max_results = arguments.get("max_results", 5)
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5

        if not query:
            return "Search failed: missing 'query' parameter."

        try:
            payload = exa_search_tool(query, max_results=max_results)
            return payload.get("markdown", f"No results returned for {query}.")
        except ExaSearchError as exc:
            logger.error("Exa search tool execution failed: %s", exc, exc_info=True)
            return f"Search failed: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error during Exa search execution: %s", exc, exc_info=True)
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
    supports_tools: bool = True,
) -> Optional[Union[dict[str, str], str]]:
    """Call an OpenRouter model and return parsed output."""

    client = get_openrouter_client()

    augmented_user_content = user_content
    if youtube_urls:
        youtube_listing = "\n".join(
            f"YouTube URL: {url}" for url in youtube_urls
        )
        augmented_user_content += "\n" + youtube_listing
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

    tools = _build_function_tools() if supports_tools else []

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
        message_text = _message_content_to_text(completion.choices[0].message)
        return _parse_openrouter_response(model_name, message_text)
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
                message_text = _message_content_to_text(completion.choices[0].message)
                return _parse_openrouter_response(model_name, message_text)
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
