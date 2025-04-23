"""Gemini AI helper module for the TelegramGroupHelperBot."""

import asyncio
import base64
import logging
import os
import mimetypes
from io import BytesIO
from typing import Dict, List, Optional, Union, Any

import aiohttp
from google import genai
from google.genai import types

from bot.config import (
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_PRO_MODEL,
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
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH", 
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
        "threshold": "OFF"
    }
]

logger.info(f"Using Gemini model: {GEMINI_MODEL}")

# Define a multimodal model for image processing
# Use one of the recommended models that support image understanding
# GEMINI_VISION_MODEL = "gemini-2.0-flash"  # Supports images and is optimized for faster responses
logger.info(f"Using vision model: {GEMINI_MODEL}")


def detect_mime_type(image_data: bytes) -> str:
    """Detect the MIME type of image data.
    
    Args:
        image_data: The image data as bytes.
        
    Returns:
        The MIME type as a string.
    """
    # First few bytes of the file can identify the format
    if image_data.startswith(b'\xff\xd8\xff'):
        return "image/jpeg"
    elif image_data.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png"
    elif image_data.startswith(b'RIFF') and image_data[8:12] == b'WEBP':
        return "image/webp"
    elif image_data.startswith(b'\x00\x00\x00') and image_data[4:8] in (b'ftyp', b'heic', b'heix', b'hevc', b'hevx'):
        # HEIC/HEIF formats
        if b'heic' in image_data[4:20] or b'heis' in image_data[4:20]:
            return "image/heic"
        else:
            return "image/heif"
    
    # Default to JPEG if the format couldn't be determined
    logger.warning("Could not determine image MIME type from data, defaulting to JPEG")
    return "image/jpeg"


async def download_image(image_url: str) -> Optional[bytes]:
    """Download an image from a URL.
    
    Args:
        image_url: The URL of the image to download.
        
    Returns:
        The image data as bytes, or None if the download failed.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Failed to download image: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return None


async def call_gemini(
    system_prompt: str,
    user_content: str,
    response_language: Optional[str] = None,
    use_search_grounding: bool = True,
    image_url: Optional[str] = None,
    use_pro_model: bool = False,
) -> str:
    """Call the Gemini API with the given prompts.
    
    Args:
        system_prompt: The system prompt.
        user_content: The user content.
        response_language: The language to respond in, if specified.
        use_search_grounding: Whether to use Google Search Grounding.
        image_url: Optional URL to an image to include in the query.
        use_pro_model: Whether to use Gemini Pro model.
        
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
        # If an image URL is provided, use the vision model
        if image_url:
            logger.info(f"Processing with image: {image_url}")
            
            # Download the image
            image_data = await download_image(image_url)
            if not image_data:
                logger.error("Failed to download image, proceeding with text only")
                # Fall back to text-only if image download fails
                image_url = None
            else:
                # Process with the vision model
                return await call_gemini_vision(
                    system_prompt=system_prompt, 
                    user_content=user_content, 
                    image_data=image_data,
                    use_search_grounding=use_search_grounding,
                    response_language=response_language,
                    use_pro_model=use_pro_model
                )
        
        # Configure generation parameters
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
        
        model = GEMINI_MODEL
        if use_pro_model:
            model = GEMINI_PRO_MODEL
            logger.info("Using Pro model for Gemini")
        else:
            logger.info("Using Standard model for Gemini")
        # Make the API call
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
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
        if use_search_grounding and not image_url:
            logger.info("Falling back to non-grounding call")
            return await call_gemini(system_prompt, user_content, response_language, False, False)
        else:
            raise  # Re-raise the exception


async def call_gemini_vision(
    system_prompt: str,
    user_content: str,
    image_data: bytes,
    use_search_grounding: bool = True,
    response_language: Optional[str] = None,
    use_pro_model: bool = False,
) -> str:
    """Call the Gemini Vision API with text and image.
    
    Args:
        system_prompt: The system prompt.
        user_content: The user content.
        image_data: The image data as bytes.
        use_search_grounding: Whether to use Google Search Grounding.
        response_language: The language to respond in, if specified.
        use_pro_model: Whether to use Gemini Pro model.
        
    Returns:
        The model's response.
    """
    try:
        # Format the user content with language if specified
        if response_language:
            user_content += f"\n\nPlease reply in {response_language}."
        
        # Determine MIME type from the image data
        mime_type = detect_mime_type(image_data)
        logger.info(f"Detected image MIME type: {mime_type}")
        
        # Combine system prompt and user content
        combined_prompt = f"{system_prompt}\n\n{user_content}"
        
        # Configure generation parameters
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
        
        # Create the content parts for the request according to the API documentation
        # First the text prompt, then the image
        contents = [
            combined_prompt,
            types.Part.from_bytes(data=image_data, mime_type=mime_type)
        ]

        model = GEMINI_MODEL
        if use_pro_model:
            model = GEMINI_PRO_MODEL
            logger.info("Using Pro model for Gemini Vision with image and text")
        else:
            logger.info("Using Standard model for Gemini Vision with image and text")
        # Make the API call with both text and image using the client.models method
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=config
        )
        
        # Log the response details
        logger.info(f"Vision response received. Has text: {hasattr(response, 'text')}")
        if hasattr(response, 'text') and response.text:
            logger.info(f"Vision response text (first 100 chars): {response.text[:100]}...")
        
        # Extract the text from the response
        return response.text.strip() if response.text else "No response generated from image analysis."
        
    except Exception as e:
        logger.error(f"Error calling Gemini Vision: {e}", exc_info=True)
        return f"Error processing image: {str(e)}"


async def stream_gemini(
    system_prompt: str,
    user_content: str,
    response_language: Optional[str] = None,
    use_search_grounding: bool = True,
    image_url: Optional[str] = None,
    use_pro_model: bool = False,
) -> asyncio.Queue:
    """Stream the Gemini API response.
    
    Args:
        system_prompt: The system prompt.
        user_content: The user content.
        response_language: The language to respond in, if specified.
        use_search_grounding: Whether to use Google Search Grounding.
        image_url: Optional URL to an image to include in the query.
        use_pro_model: Whether to use Gemini Pro model.
        
    Returns:
        A queue of response chunks.
    """
    # Create a queue to store response chunks
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    
    # If we have an image, we can't stream the response
    # so we'll get the full response and put it in the queue
    if image_url:
        try:
            # Get full response with image
            response = await call_gemini(
                system_prompt=system_prompt,
                user_content=user_content,
                response_language=response_language, 
                use_search_grounding=use_search_grounding,
                image_url=image_url,
                use_pro_model=use_pro_model
            )
            
            # Put the full response in the queue
            await queue.put(response)
            
            # Signal end of stream
            await queue.put(None)
            
            return queue
            
        except Exception as e:
            logger.error(f"Error in stream with image: {e}", exc_info=True)
            await queue.put(f"Error processing image: {str(e)}")
            await queue.put(None)
            return queue
    
    # Format the user content with language if specified
    if response_language:
        user_content += f"\n\nPlease reply in {response_language}."
    
    # Log the complete prompts
    logger.info(f"Stream - System prompt: {system_prompt}")
    logger.info(f"Stream - User content: {user_content}")
    logger.info(f"Stream - Using search grounding: {use_search_grounding}")
    
    # Combine system prompt and user content
    combined_prompt = f"{system_prompt}\n\n{user_content}"
    
    # Use threading to avoid blocking
    async def stream_worker() -> None:
        try:
            # Configure generation parameters
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
            
            model = GEMINI_MODEL
            if use_pro_model:
                model = GEMINI_PRO_MODEL
            logger.info(f"Stream - Sending message to model: {model}")
            
            # For streaming, we need to handle the generate_content_stream method
            stream_response = await asyncio.to_thread(
                client.models.generate_content_stream,
                model=model,
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
                
                # Transfer items from the fallback queue to our queue
                while True:
                    item = await fallback_queue.get()
                    await queue.put(item)
                    
                    if item is None:
                        break
            else:
                # Put the error message in the queue
                await queue.put(f"⚠️ Error: {str(e)}")
                # Signal end of stream
                await queue.put(None)
    
    # Start the worker
    asyncio.create_task(stream_worker())
    
    # Return the queue immediately
    return queue


async def test_gemini_vision(image_url: str) -> None:
    """Test the Gemini Vision API with a sample image.
    
    This function is for testing purposes only.
    
    Args:
        image_url: The URL of an image to test with.
    """
    try:
        logger.info(f"Testing Gemini Vision API with image: {image_url}")
        
        # Basic prompt for image analysis
        system_prompt = "You are a helpful assistant that can analyze images."
        user_content = "Please describe this image in detail."
        
        # Get response from Gemini with the image
        response = await call_gemini(
            system_prompt=system_prompt,
            user_content=user_content,
            image_url=image_url
        )
        
        # Log and print the response
        logger.info(f"Test response: {response}")
        print(f"Test response: {response}")
        
        return response
    except Exception as e:
        logger.error(f"Error in test_gemini_vision: {e}", exc_info=True)
        print(f"Error testing Gemini Vision: {str(e)}")
        return f"Error: {str(e)}"


# Example usage of the test function (uncomment to run)
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         image_url = sys.argv[1]
#         asyncio.run(test_gemini_vision(image_url))
#     else:
#         print("Please provide an image URL as an argument") 