"""Gemini AI helper module for the TelegramGroupHelperBot."""

import asyncio
import os
from typing import Dict, List, Optional

import google.generativeai as genai

from bot.config import (
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_TOP_K,
    GEMINI_TOP_P,
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Generation configuration
_generation_cfg = {
    "temperature": GEMINI_TEMPERATURE,
    "top_p": GEMINI_TOP_P,
    "top_k": GEMINI_TOP_K,
    "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
    "response_mime_type": "text/plain",
}

# Initialize model
_MODEL = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    generation_config=_generation_cfg,
)


async def call_gemini(
    system_prompt: str,
    user_content: str,
    response_language: Optional[str] = None,
) -> str:
    """Call the Gemini API with the given prompts.
    
    Args:
        system_prompt: The system prompt.
        user_content: The user content.
        response_language: The language to respond in, if specified.
        
    Returns:
        The model's response.
    """
    msgs: List[Dict] = [
        {"role": "system", "parts": [system_prompt]},
        {"role": "user", "parts": [user_content]},
    ]
    
    if response_language:
        msgs.append({"role": "user", "parts": [f"Please reply in {response_language}."]})

    # Use thread pool to avoid blocking the event loop
    resp = await asyncio.to_thread(_MODEL.generate_content, msgs)
    return resp.text.strip()


async def stream_gemini(
    system_prompt: str,
    user_content: str,
    response_language: Optional[str] = None,
) -> asyncio.Queue:
    """Stream the Gemini API response.
    
    Args:
        system_prompt: The system prompt.
        user_content: The user content.
        response_language: The language to respond in, if specified.
        
    Returns:
        A queue of response chunks.
    """
    msgs: List[Dict] = [
        {"role": "system", "parts": [system_prompt]},
        {"role": "user", "parts": [user_content]},
    ]
    
    if response_language:
        msgs.append({"role": "user", "parts": [f"Please reply in {response_language}."]})

    # Create a queue to store response chunks
    queue: asyncio.Queue[str] = asyncio.Queue()
    
    # Use threading to avoid blocking
    async def stream_worker() -> None:
        full_text = ""
        response = _MODEL.generate_content(msgs, stream=True)
        
        for chunk in response:
            if not chunk.text:
                continue
                
            full_text += chunk.text
            await queue.put(full_text)
        
        # Signal end of stream
        await queue.put(None)
    
    # Start the worker
    asyncio.create_task(stream_worker())
    return queue 