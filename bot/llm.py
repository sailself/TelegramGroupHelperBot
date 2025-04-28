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
from PIL import Image

from bot.config import (
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_PRO_MODEL,
    GEMINI_IMAGE_MODEL,
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

logger.info(f"Using Image model: {GEMINI_IMAGE_MODEL}")


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
        
        # Log the response details for debugging
        logger.info(f"Vision response received. Has text: {hasattr(response, 'text')}")
        if hasattr(response, 'text') and response.text:
            # Log the first part of the response for debugging
            text_sample = response.text[:100] if len(response.text) > 100 else response.text
            logger.info(f"Vision response text sample: {text_sample}")
            
            # Check if the response looks like a base64 string
            if all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in text_sample):
                logger.info("Response appears to be a valid base64 string")
            else:
                logger.warning("Response does not appear to be a base64 string, may contain explanation text")
        
        # Extract the text from the response
        return response.text.strip() if response.text else "No response generated from image analysis."
        
    except Exception as e:
        logger.error(f"Error calling Gemini Vision: {e}", exc_info=True)
        return f"Error processing image: {str(e)}"


def extract_and_process_image_from_text(text_response: str) -> Optional[bytes]:
    """Extract and process a base64 encoded image from text response.
    
    Args:
        text_response: The text response potentially containing a base64 encoded image.
        
    Returns:
        The processed image as bytes, or None if extraction failed.
    """
    try:
        # Log response information
        logger.info(f"Response length: {len(text_response)} characters")
        text_preview = text_response[:50] + "..." if len(text_response) > 50 else text_response
        logger.info(f"Response preview: {text_preview}")
        
        # Check if the response is likely a text explanation rather than base64 data
        # Common indicators include: "I'm sorry", "cannot", "unable", etc.
        if any(phrase in text_response.lower() for phrase in ["sorry", "cannot", "unable", "don't", "doesn't", "can't", "error"]):
            logger.warning(f"Response appears to be an explanation rather than image data: {text_response[:200]}...")
            return None
        
        # Strip any markdown code block markers that might be present
        base64_data = text_response.strip()
        
        # Check if the response contains text explanation instead of just a base64 image
        if len(base64_data) > 10000 or "```" in base64_data or "base64," in base64_data:
            # Extract content between code blocks if present
            import re
            
            # Check for markdown code blocks
            code_match = re.search(r'```(?:.*?)\n(.*?)```', base64_data, re.DOTALL)
            if code_match:
                base64_data = code_match.group(1).strip()
                logger.info("Extracted base64 data from code block")
            
            # Check for data URI format (common in image responses)
            data_uri_match = re.search(r'data:image/[^;]+;base64,([^"\'\\s]+)', base64_data)
            if data_uri_match:
                base64_data = data_uri_match.group(1).strip()
                logger.info("Extracted base64 data from data URI format")
                
            # Check for JSON response with base64 field
            json_match = re.search(r'"image":\s*"([A-Za-z0-9+/=]+)"', base64_data)
            if json_match:
                base64_data = json_match.group(1).strip()
                logger.info("Extracted base64 data from JSON response")
        
        # Log statistics about the data quality
        if base64_data:
            # Basic validation - check character distribution for signs of text vs base64
            valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
            invalid_chars = set(c for c in base64_data if c not in valid_chars)
            
            if invalid_chars:
                logger.warning(f"Found invalid base64 characters: {invalid_chars}")
                base64_data = ''.join(c for c in base64_data if c in valid_chars)
            
            # Log statistics
            logger.info(f"Base64 data length: {len(base64_data)}")
            logger.info(f"Base64 data looks valid: {all(c in valid_chars for c in base64_data)}")
                
        # Remove any non-base64 characters
        base64_data = ''.join(c for c in base64_data if c in 
                             'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        
        # Clean the base64 data - remove any leading/trailing non-base64 content
        # Find the longest valid base64 substring
        base64_pattern = re.compile(r'[A-Za-z0-9+/=]+')
        matches = base64_pattern.findall(base64_data)
        if matches:
            # Use the longest match
            base64_data = max(matches, key=len)
        
        # Fix padding issues - ensure the length is a multiple of 4
        remainder = len(base64_data) % 4
        if remainder > 0:
            # Add padding to make length multiple of 4
            base64_data += '=' * (4 - remainder)
        
        # Try to decode the base64 data to binary
        try:
            image_bytes = base64.b64decode(base64_data)
        except Exception as decode_error:
            logger.error(f"Base64 decode error: {decode_error}. Trying with padding stripped.")
            try:
                # Try again with any padding stripped, then add back correct padding
                base64_data = base64_data.rstrip('=')
                remainder = len(base64_data) % 4
                if remainder > 0:
                    base64_data += '=' * (4 - remainder)
                image_bytes = base64.b64decode(base64_data)
            except Exception as e:
                logger.error(f"Failed to decode base64 data after retry: {e}")
                return None
        
        # Verify it's a valid image using PIL
        try:
            from PIL import Image
            from io import BytesIO
            
            # Log the first few bytes to help with debugging
            logger.info(f"First 20 bytes of decoded data: {image_bytes[:20]}")
            
            # Check if it's likely text rather than image data
            is_probably_text = all(byte < 128 for byte in image_bytes[:100])
            if is_probably_text:
                text_sample = image_bytes[:100].decode('utf-8', errors='replace')
                logger.error(f"Data appears to be text, not an image: {text_sample}...")
                return None
            
            # Try to open the image with PIL to validate it
            input_buffer = BytesIO(image_bytes)
            img = Image.open(input_buffer)
            
            # Get the image format for logging
            img_format = img.format
            img_mode = img.mode
            img_size = img.size
            logger.info(f"Valid image detected: Format={img_format}, Mode={img_mode}, Size={img_size}")
            
            # Convert to RGB mode if it's not already (required for JPEG)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG to a new BytesIO object
            output = BytesIO()
            img.save(output, format='JPEG', quality=95)
            output.seek(0)
            
            # Return the JPEG bytes
            return output.getvalue()
            
        except Exception as img_error:
            logger.error(f"Invalid image data: {img_error}")
            
            # Try to decode the data as text to see if the model returned an explanation
            try:
                if len(image_bytes) > 20:
                    text_response = image_bytes.decode('utf-8', errors='replace')
                    logger.info(f"Model returned text instead of image: {text_response[:200]}...")
            except:
                pass
            
            return None
    except Exception as e:
        logger.error(f"Error extracting image from text: {e}", exc_info=True)
        return None


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
                # Make sure to use the correct parameter order
                fallback_queue = await stream_gemini(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    response_language=response_language,
                    use_search_grounding=False
                )
                
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


async def generate_image_with_gemini(
    system_prompt: str,
    prompt: str,
    input_image_url: Optional[str] = None,
) -> Optional[bytes]:
    """Generate or edit an image using Gemini.
    
    Args:
        system_prompt: The system prompt for the model.
        prompt: The user's description of the desired image.
        input_image_url: Optional URL to an image to edit.
        
    Returns:
        The generated image as bytes, or None if generation failed.
    """
    logger.info(f"Generating image with prompt: {prompt[:100]}...")
    
    try:
        # Keep the prompt simple as suggested
        image_generation_prompt = prompt
        
        # If there's an input image, include it in the request
        if input_image_url:
            logger.info(f"Editing existing image from URL: {input_image_url}")
            image_data = await download_image(input_image_url)
            
            if not image_data:
                logger.error("Failed to download input image, proceeding with text-only generation")
                input_image_url = None
            else:
                # Configure generation parameters
                model = GEMINI_IMAGE_MODEL
                config = types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    max_output_tokens=65535,
                    safety_settings=_safety_settings
                )
                
                # Determine MIME type from the image data
                mime_type = detect_mime_type(image_data)
                
                # Create multipart model request with image and prompt
                contents = [
                    f"Edit this image: {prompt}",
                    types.Part.from_bytes(data=image_data, mime_type=mime_type)
                ]
                
                # Make the API call with both text and image
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config
                )
        else:
            # Text-only image generation
            model = GEMINI_IMAGE_MODEL  # Use the specialized image model
            config = types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                max_output_tokens=65535,
                safety_settings=_safety_settings
            )
            
            # Log the model being used
            logger.info(f"Using model {model} for image generation")
            
            # Make the API call
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=image_generation_prompt,
                config=config
            )
        
        # Extract the image data from the response parts
        logger.info(f"Response received. Candidates count: {len(response.candidates)}")
        
        for candidate in response.candidates:
            if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
                continue
                
            for part in candidate.content.parts:
                # Check if the part has inline_data (image data)
                if hasattr(part, 'inline_data') and part.inline_data is not None:
                    logger.info(f"Found inline image data with mime type: {part.inline_data.mime_type}")
                    
                    # Convert the image data to bytes
                    image_bytes = part.inline_data.data
                    
                    # Process the image with PIL to ensure it's valid and in the right format
                    try:
                        from PIL import Image
                        from io import BytesIO
                        
                        # Open the image with PIL
                        img = Image.open(BytesIO(image_bytes))
                        
                        # Convert to RGB mode if it's not already (required for JPEG)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save as JPEG to a new BytesIO object
                        output = BytesIO()
                        img.save(output, format='JPEG', quality=95)
                        output.seek(0)
                        
                        # Return the JPEG bytes
                        return output.getvalue()
                    except Exception as img_error:
                        logger.error(f"Error processing inline image data: {img_error}")
                
                # Check if part has text (might contain error messages)
                if hasattr(part, 'text') and part.text:
                    logger.info(f"Part has text: {part.text[:100]}")
        
        logger.error("No valid image data found in response")
        return None
            
    except Exception as e:
        logger.error(f"Error generating image with Gemini: {e}", exc_info=True)
        
        # Check for specific API errors
        error_message = str(e).lower()
        if "not supported" in error_message or "unavailable" in error_message or "feature" in error_message:
            logger.error("Gemini API does not support image generation capability")
            # You might want to use a fallback service or inform the user more specifically
        
        return None


# Example usage of the test function (uncomment to run)
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         image_url = sys.argv[1]
#         asyncio.run(test_gemini_vision(image_url))
#     else:
#         print("Please provide an image URL as an argument") 