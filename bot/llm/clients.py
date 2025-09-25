"""Client factories for Gemini, Vertex, and OpenRouter."""
from __future__ import annotations

import logging
from typing import Optional

import google.genai as genai
from openai import AsyncOpenAI

from bot.config import (
    GEMINI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_ALPHA_BASE_URL,
    USE_VERTEX_IMAGE,
    USE_VERTEX_VIDEO,
    VERTEX_LOCATION,
    VERTEX_PROJECT_ID,
)

logger = logging.getLogger(__name__)

_global_gemini_client: Optional[genai.Client] = None
_global_openrouter_client: Optional[AsyncOpenAI] = None
_global_openrouter_responses_client: Optional[AsyncOpenAI] = None
_global_vertex_client: Optional[genai.Client] = None

def get_gemini_client():
    """Lazily initializes and returns the global Gemini client."""
    global _global_gemini_client  # noqa: PLW0603
    if _global_gemini_client is None:
        api_key = GEMINI_API_KEY
        if not api_key:
            logger.error(
                "GEMINI_API_KEY environment variable not set during client initialization."
            )
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        logger.info("Initializing global Gemini client with API key.")
        _global_gemini_client = genai.Client(api_key=api_key)
    return _global_gemini_client

def get_openrouter_client() -> AsyncOpenAI:
    """Lazily initializes and returns the global OpenRouter client."""
    global _global_openrouter_client  # noqa: PLW0603
    if _global_openrouter_client is None:
        if not OPENROUTER_API_KEY:
            logger.error("OPENROUTER_API_KEY environment variable not set during client initialization.")
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        logger.info("Initializing global OpenRouter client with API key.")
        _global_openrouter_client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            max_retries=0,
        )
    return _global_openrouter_client

def get_openrouter_responses_client() -> AsyncOpenAI:
    """Return an OpenRouter client configured for the alpha Responses API."""
    global _global_openrouter_responses_client  # noqa: PLW0603
    if _global_openrouter_responses_client is None:
        if not OPENROUTER_API_KEY:
            logger.error("OPENROUTER_API_KEY environment variable not set during responses client initialization.")
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        logger.info("Initializing OpenRouter responses client with alpha base URL.")
        _global_openrouter_responses_client = AsyncOpenAI(
            base_url=OPENROUTER_ALPHA_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            max_retries=0,
        )
    return _global_openrouter_responses_client

def get_vertex_client():
    """Lazily initializes and returns the global Vertex client."""
    global _global_vertex_client  # noqa: PLW0603
    if USE_VERTEX_IMAGE or USE_VERTEX_VIDEO:
        if _global_vertex_client is None:
            logger.info("Initializing global Vertex client.")
            _global_vertex_client = genai.Client(
                vertexai=True, project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION
            )
    return _global_vertex_client
