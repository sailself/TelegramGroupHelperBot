"""Gemini AI helper module for the TelegramGroupHelperBot."""

import asyncio
import logging
import os
from typing import Dict, List, Optional

from google import genai

from bot.config import (
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_TOP_K,
    GEMINI_TOP_P,
)

# Set up logging
logger = logging.getLogger(__name__)

# Initialize Gemini API client
client = genai.Client(api_key=GEMINI_API_KEY)

# Safety settings for the models
_safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH", 
        "threshold": "BLOCK_ONLY_HIGH"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH"
    }
]

logger.info(f"Using Gemini model: {GEMINI_MODEL}")


async def call_gemini(
    system_prompt: str,
    user_content: str,
    response_language: Optional[str] = None,
    use_search_grounding: bool = True,
) -> str:
    """Call the Gemini API with the given prompts.
    
    Args:
        system_prompt: The system prompt.
        user_content: The user content.
        response_language: The language to respond in, if specified.
        use_search_grounding: Whether to use Google Search Grounding.
        
    Returns:
        The model's response.
    """
    # Format the user content with language if specified
    if response_language:
        user_content += f"\n\nPlease reply in {response_language}."
    
    # Log the complete prompts
    logger.info(f"System prompt: {system_prompt}")
    logger.info(f"User content: {user_content}")
    logger.info(f"Using search grounding: {use_search_grounding}")
    
    # Combine system prompt and user content
    combined_prompt = f"{system_prompt}\n\n{user_content}"
    
    try:
        # Generation configuration
        config = {
            "temperature": GEMINI_TEMPERATURE,
            "top_p": GEMINI_TOP_P,
            "top_k": GEMINI_TOP_K,
            "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
            "safety_settings": _safety_settings,
        }
        
        # Add search tool if enabled
        if use_search_grounding:
            config["tools"] = [{"google_search": {}}]
            logger.info("Using Google Search Grounding for this request")
        
        # Make the API call
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=combined_prompt,
            config=config
        )
        
        # Log the response details
        logger.info(f"Response received. Has text: {hasattr(response, 'text')}")
        if hasattr(response, 'text') and response.text:
            logger.info(f"Response text (first 100 chars): {response.text[:100]}...")
        
        # Extract the text from the response
        return response.text.strip() if response.text else "No response generated."
        
    except Exception as e:
        logger.error(f"Error calling Gemini: {e}", exc_info=True)
        # Fall back to non-grounding call if grounding fails
        if use_search_grounding:
            logger.info("Falling back to non-grounding call")
            return await call_gemini(system_prompt, user_content, response_language, False)
        else:
            raise  # Re-raise the exception if not using grounding


async def stream_gemini(
    system_prompt: str,
    user_content: str,
    response_language: Optional[str] = None,
    use_search_grounding: bool = True,
) -> asyncio.Queue:
    """Stream the Gemini API response.
    
    Args:
        system_prompt: The system prompt.
        user_content: The user content.
        response_language: The language to respond in, if specified.
        use_search_grounding: Whether to use Google Search Grounding.
        
    Returns:
        A queue of response chunks.
    """
    # Format the user content with language if specified
    if response_language:
        user_content += f"\n\nPlease reply in {response_language}."
    
    # Log the complete prompts
    logger.info(f"Stream - System prompt: {system_prompt}")
    logger.info(f"Stream - User content: {user_content}")
    logger.info(f"Stream - Using search grounding: {use_search_grounding}")
    
    # Combine system prompt and user content
    combined_prompt = f"{system_prompt}\n\n{user_content}"
    
    # Create a queue to store response chunks
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    
    # Use threading to avoid blocking
    async def stream_worker() -> None:
        try:
            # Generation configuration
            config = {
                "temperature": GEMINI_TEMPERATURE,
                "top_p": GEMINI_TOP_P,
                "top_k": GEMINI_TOP_K,
                "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
                "safety_settings": _safety_settings,
            }
            
            # Add search tool if enabled
            if use_search_grounding:
                config["tools"] = [{"google_search": {}}]
                logger.info("Stream - Using Google Search Grounding")
            
            logger.info(f"Stream - Sending message to model: {GEMINI_MODEL}")
            
            # For streaming, we need to handle the generate_content_stream method
            stream_response = await asyncio.to_thread(
                client.models.generate_content_stream,
                model=GEMINI_MODEL,
                contents=combined_prompt,
                config=config
            )
            
            # Process the streaming response
            full_text = ""
            # Convert the generator to a list and process it normally
            for chunk in stream_response:
                if hasattr(chunk, 'text') and chunk.text:
                    full_text += chunk.text
                    await queue.put(full_text)
                    logger.info(f"Stream - Chunk received, length so far: {len(full_text)}")
            
            if not full_text:
                logger.warning("No text received from streaming response")
                await queue.put("⚠️ No text received from Gemini.")
            
            # Signal end of stream
            await queue.put(None)
            logger.info("Stream - Completed streaming")
            
        except Exception as e:
            logger.error(f"Error in stream_worker: {e}", exc_info=True)
            # Fall back to non-grounding streaming if grounding fails
            if use_search_grounding:
                logger.info("Falling back to non-grounding streaming")
                fallback_queue = await stream_gemini(system_prompt, user_content, response_language, False)
                
                # Forward all messages from the fallback queue
                while True:
                    item = await fallback_queue.get()
                    await queue.put(item)
                    if item is None:  # End of stream
                        break
            else:
                # Put error message and end stream
                await queue.put(f"⚠️ Error getting response from Gemini: {str(e)}")
                await queue.put(None)
    
    # Start the worker
    asyncio.create_task(stream_worker())
    return queue 