"""Gemini AI helper module for the TelegramGroupHelperBot."""

import asyncio
import base64
import logging
import os
import mimetypes
from io import BytesIO
from typing import List, Optional, Union, Any

import aiohttp
import google.genai as genai
from google.genai import types
from PIL import Image

from bot.config import (
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_PRO_MODEL,
    GEMINI_IMAGE_MODEL,
    GEMINI_VIDEO_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_TOP_K,
    GEMINI_TOP_P,
    VERTEX_PROJECT_ID,
    VERTEX_LOCATION,
    USE_VERTEX_VIDEO,
    VERTEX_VIDEO_MODEL,
    USE_VERTEX_IMAGE,
    VERTEX_IMAGE_MODEL,
)
import time # Added for polling
# from google.cloud import aiplatform # Removed as per refactoring task
# from google.cloud.aiplatform_v1beta1.types import content as gapic_content_types # May not be needed

# Set up logging
logger = logging.getLogger(__name__)

# Global client variable, initialized as None
_global_client = None

def get_gemini_client():
    """Lazily initializes and returns the global Gemini client."""
    global _global_client
    if _global_client is None:
        api_key = GEMINI_API_KEY
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set during client initialization.")
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        logger.info("Initializing global Gemini client with API key.")
        _global_client = genai.Client(api_key=api_key)
    return _global_client

_global_vertex_client = None

def get_vertex_client():
    """Lazily initializes and returns the global Vertex client."""
    global _global_vertex_client
    if USE_VERTEX_IMAGE or USE_VERTEX_VIDEO:
        if _global_vertex_client is None:
            logger.info("Initializing global Vertex client.")
            # Assuming genai.Client can handle Vertex AI image models too.
            # If not, this might need to be aiplatform.PredictionServiceClient or similar
            _global_vertex_client = genai.Client(vertexai=True, project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
    return _global_vertex_client

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
logger.info(f"Using Gemini Pro model: {GEMINI_PRO_MODEL}")
logger.info(f"Using Image model: {GEMINI_IMAGE_MODEL}")
logger.info(f"Using Video model: {GEMINI_VIDEO_MODEL}")


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


async def download_media(media_url: str) -> Optional[bytes]:
    """Download a media file (image or video) from a URL.
    
    Args:
        media_url: The URL of the media to download.
        
    Returns:
        The media data as bytes, or None if the download failed.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(media_url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Failed to download media: {response.status} from {media_url}")
                    return None
    except Exception as e:
        logger.error(f"Error downloading media from {media_url}: {e}", exc_info=True)
        return None


async def call_gemini(
    system_prompt: str,
    user_content: str,
    response_language: Optional[str] = None,
    use_search_grounding: bool = True,
    use_url_context: bool = False,
    image_url: Optional[str] = None,
    use_pro_model: bool = False,
    image_data_list: Optional[List[bytes]] = None,
    video_data: Optional[bytes] = None,
    video_mime_type: Optional[str] = None,
) -> str:
    """Call the Gemini API with the given prompts.
    
    Args:
        system_prompt: The system prompt.
        user_content: The user content.
        response_language: The language to respond in, if specified.
        use_search_grounding: Whether to use Google Search Grounding.
        use_url_context: Whether to use URL Context.
        image_url: Optional URL to an image to include in the query.
        use_pro_model: Whether to use Gemini Pro model.
        image_data_list: Optional list of image data as bytes.
        video_data: Optional video data as bytes.
        video_mime_type: Optional MIME type for the video data.
        
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
        if video_data and video_mime_type:
            logger.info(f"Processing with provided video data (MIME: {video_mime_type}).")
            return await call_gemini_vision(
                system_prompt=system_prompt,
                user_content=user_content,
                video_data=video_data,
                video_mime_type=video_mime_type,
                image_data_list=None, # Ensure images are not processed if video is present
                use_search_grounding=use_search_grounding,
                use_url_context=use_url_context,
                response_language=response_language,
                use_pro_model=use_pro_model
            )
        elif image_data_list:
            logger.info(f"Processing with {len(image_data_list)} provided image(s).")
            return await call_gemini_vision(
                system_prompt=system_prompt,
                user_content=user_content,
                image_data_list=image_data_list,
                video_data=None, # Ensure video is not processed
                video_mime_type=None,
                use_search_grounding=use_search_grounding,
                use_url_context=use_url_context,
                response_language=response_language,
                use_pro_model=use_pro_model
            )
        elif image_url: 
            logger.info(f"Processing with single image URL: {image_url}")
            media_data = await download_media(image_url)
            if not media_data:
                logger.error("Failed to download media from URL, proceeding with text only")
                 # Proceed to text-only call by falling through
            else:
                # Process with the vision model, passing the single image as a list
                return await call_gemini_vision(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    image_data_list=[media_data],
                    video_data=None, # Ensure video is not processed
                    video_mime_type=None,
                    use_search_grounding=use_search_grounding,
                    use_url_context=use_url_context,
                    response_language=response_language,
                    use_pro_model=use_pro_model
                )

        # If no media (video, image_data_list, or downloadable image_url), proceed with text-only call
        logger.info("Proceeding with text-only Gemini call.")
        config = {
            "temperature": GEMINI_TEMPERATURE,
            "top_p": GEMINI_TOP_P,
            "top_k": GEMINI_TOP_K,
            "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
            "safety_settings": _safety_settings,
        }
        
        # Add search tool if enabled
        # Build the tools list based on enabled features
        tools = []
        if use_search_grounding:
            tools.append({"google_search": {}})
            logger.info("Using Google Search Grounding for this request")
        if use_url_context:
            tools.append({"url_context": {}})
            logger.info("Using URL Context for this request")
        if tools:
            config["tools"] = tools
        
        model = GEMINI_MODEL
        if use_pro_model:
            model = GEMINI_PRO_MODEL
            logger.info("Using Pro model for Gemini")
        else:
            logger.info("Using Standard model for Gemini")
        # Make the API call
        response = await asyncio.to_thread(
            get_gemini_client().models.generate_content,
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
        if use_search_grounding and not (video_data or image_data_list or image_url): # Fallback for text-only
            logger.info("Falling back to non-grounding call for text-only request.")
            return await call_gemini(
                system_prompt=system_prompt, 
                user_content=user_content, 
                response_language=response_language, 
                use_search_grounding=False, 
                use_url_context=use_url_context,
                use_pro_model=use_pro_model, # Retain pro_model choice
                video_data=None, video_mime_type=None, image_data_list=None, image_url=None # Ensure no media in fallback
            )
        elif use_search_grounding and (video_data or image_data_list or image_url):
            logger.warning("Grounding failed with media. Re-raising the exception as fallback for media with no grounding is not defined.")
            raise
        else: # Non-grounding call failed, or other error
            raise  # Re-raise the exception


async def call_gemini_vision(
    system_prompt: str,
    user_content: str,
    image_data_list: Optional[List[bytes]] = None,
    use_search_grounding: bool = True,
    use_url_context: bool = True,
    response_language: Optional[str] = None,
    use_pro_model: bool = False,
    video_data: Optional[bytes] = None,
    video_mime_type: Optional[str] = None,
) -> str:
    """Call the Gemini Vision API with text, and optionally image(s) or video.
    
    Args:
        system_prompt: The system prompt.
        user_content: The user content.
        image_data_list: Optional list of image data as bytes.
        use_search_grounding: Whether to use Google Search Grounding.
        use_url_context: Whether to use URL Context.
        response_language: The language to respond in, if specified.
        use_pro_model: Whether to use Gemini Pro model.
        video_data: Optional video data as bytes.
        video_mime_type: Optional MIME type for the video data.
        
    Returns:
        The model's response.
    """
    try:
        # Format the user content with language if specified
        if response_language:
            user_content += f"\n\nPlease reply in {response_language}."
        
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
        tools = []
        if use_search_grounding:
            tools.append({"google_search": {}})
            logger.info("Using Google Search Grounding for this request")
        if use_url_context:
            tools.append({"url_context": {}})
            logger.info("Using URL Context for this request")
        if tools:
            config["tools"] = tools
        
        # Create the content parts for the request
        contents = [combined_prompt]
        log_media_info = "no media"

        if video_data and video_mime_type:
            logger.info(f"Processing with video data, MIME type: {video_mime_type}.")
            if image_data_list and any(image_data_list): # check if list is not empty
                logger.warning("Video data provided; image_data_list will be ignored as video takes precedence.")
            contents.append(types.Part(inline_data=types.Blob(data=video_data, mime_type=video_mime_type))) # Reverted
            log_media_info = f"video ({video_mime_type})"
        elif image_data_list and any(image_data_list): # check if list is not empty
            logger.info(f"Processing with {len(image_data_list)} image(s).")
            for img_data in image_data_list:
                mime_type = detect_mime_type(img_data) 
                logger.info(f"Detected image MIME type: {mime_type} for one of the images.")
                contents.append(types.Part.from_bytes(data=img_data, mime_type=mime_type)) # Reverted
            log_media_info = f"{len(image_data_list)} image(s)"
        
        model_to_use = GEMINI_PRO_MODEL if use_pro_model else GEMINI_MODEL
        
        logger.info(f"Using {'Pro' if use_pro_model else 'Standard'} model ({model_to_use}) for Gemini Vision with {log_media_info} and text")
        
        # Make the API call with text and media using the client.models method
        response = await asyncio.to_thread(
            get_gemini_client().models.generate_content,
            model=model_to_use,
            contents=contents,
            config=config
        )
        
        # Log the response details for debugging
        logger.info(f"Vision response received. Has text: {hasattr(response, 'text')}")
        if hasattr(response, 'text') and response.text:
            text_sample = response.text[:100] if len(response.text) > 100 else response.text
            logger.info(f"Vision response text sample: {text_sample}")
            # Base64 check might be less relevant if typical responses are textual explanations for media.
        
        # Extract the text from the response
        return response.text.strip() if response.text else "No response generated from media analysis."
        
    except Exception as e:
        logger.error(f"Error calling Gemini Vision with media: {e}", exc_info=True)
        return f"Error processing media (image/video): {str(e)}"


async def extract_and_process_image_from_text(text_response: str) -> Optional[bytes]:
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
            logger.error(f"Base64 decode error: {decode_error}. Trying with padding stripped.", exc_info=True)
            try:
                # Try again with any padding stripped, then add back correct padding
                base64_data = base64_data.rstrip('=')
                remainder = len(base64_data) % 4
                if remainder > 0:
                    base64_data += '=' * (4 - remainder)
                image_bytes = base64.b64decode(base64_data)
            except Exception as e:
                logger.error(f"Failed to decode base64 data after retry: {e}", exc_info=True)
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
            img = await asyncio.to_thread(Image.open, input_buffer)
            
            # Get the image format for logging
            img_format = img.format
            img_mode = img.mode
            img_size = img.size
            logger.info(f"Valid image detected: Format={img_format}, Mode={img_mode}, Size={img_size}")
            
            # Convert to RGB mode if it's not already (required for JPEG)
            if img.mode != 'RGB':
                img = await asyncio.to_thread(img.convert, 'RGB')
            
            # Save as JPEG to a new BytesIO object
            output = BytesIO()
            await asyncio.to_thread(img.save, output, format='JPEG', quality=95)
            output.seek(0)
            
            # Return the JPEG bytes
            return output.getvalue()
            
        except Exception as img_error:
            logger.error(f"Invalid image data: {img_error}", exc_info=True)
            
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
    use_url_context: bool = True,
    image_url: Optional[str] = None,
    use_pro_model: bool = False,
    image_data_list: Optional[List[bytes]] = None,
    video_data: Optional[bytes] = None,
    video_mime_type: Optional[str] = None,
) -> asyncio.Queue:
    """Stream the Gemini API response.
    
    Args:
        system_prompt: The system prompt.
        user_content: The user content.
        response_language: The language to respond in, if specified.
        use_search_grounding: Whether to use Google Search Grounding.
        use_url_context: Whether to use URL Context.
        image_url: Optional URL to an image to include in the query.
        use_pro_model: Whether to use Gemini Pro model.
        image_data_list: Optional list of image data as bytes.
        video_data: Optional video data as bytes.
        video_mime_type: Optional MIME type for the video data.
        
    Returns:
        A queue of response chunks.
    """
    # Create a queue to store response chunks
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    
    # If we have video, image_data_list, or an image_url, we can't truly stream the vision model's response.
    # We'll get the full response from call_gemini_vision (or call_gemini for URL) and put it in the queue.
    if video_data or image_data_list or image_url:
        try:
            if video_data and video_mime_type:
                logger.info(f"Stream mode: Processing with provided video data (MIME: {video_mime_type}).")
                response = await call_gemini_vision(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    video_data=video_data,
                    video_mime_type=video_mime_type,
                    image_data_list=None, # Prioritize video
                    use_search_grounding=use_search_grounding,
                    use_url_context=use_url_context,
                    response_language=response_language,
                    use_pro_model=use_pro_model
                )
            elif image_data_list:
                logger.info(f"Stream mode: Processing with {len(image_data_list)} provided image(s).")
                response = await call_gemini_vision(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    image_data_list=image_data_list,
                    video_data=None, # No video
                    video_mime_type=None,
                    use_search_grounding=use_search_grounding,
                    use_url_context=use_url_context,
                    response_language=response_language,
                    use_pro_model=use_pro_model
                )
            elif image_url: 
                logger.info(f"Stream mode: Processing with single image URL: {image_url}")
                response = await call_gemini( 
                    system_prompt=system_prompt,
                    user_content=user_content,
                    response_language=response_language,
                    use_search_grounding=use_search_grounding,
                    use_url_context=use_url_context,
                    image_url=image_url, 
                    use_pro_model=use_pro_model,
                    # Pass None for direct media data as call_gemini handles URL download
                    image_data_list=None, 
                    video_data=None,
                    video_mime_type=None
                )
            else: # Should ideally not be reached if the outer 'if' is true.
                 response = "Error: stream_gemini logic error in media handling."


            await queue.put(response)
            await queue.put(None) # Signal end of stream
            return queue
            
        except Exception as e:
            logger.error(f"Error in stream_gemini with media (video/image): {e}", exc_info=True)
            error_msg = f"Error processing media: {str(e)}"
            if video_data:
                error_msg = f"Error processing video: {str(e)}"
            elif image_data_list:
                error_msg = f"Error processing {len(image_data_list)} image(s): {str(e)}"
            elif image_url:
                 error_msg = f"Error processing media from URL: {str(e)}"
            await queue.put(error_msg)
            await queue.put(None)
            return queue
    
    # Text-only streaming logic starts here
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
            # Refactored tool selection logic
            tools = []
            if use_search_grounding:
                tools.append({"google_search": {}})
                logger.info("Stream - Using Google Search Grounding")
            if use_url_context:
                tools.append({"url_context": {}})
                logger.info("Stream - Using URL Context")
            if tools:
                config["tools"] = tools
            
            model = GEMINI_MODEL
            if use_pro_model:
                model = GEMINI_PRO_MODEL
            logger.info(f"Stream - Sending message to model: {model}")
            
            # For streaming, we need to handle the generate_content_stream method
            stream_response = await asyncio.to_thread(
                get_gemini_client().models.generate_content_stream,
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
            # Fall back to non-grounding streaming if grounding fails (text-only scenario)
            if use_search_grounding: # This implies no media due to logic above
                logger.info("Falling back to non-grounding text-only streaming")
                fallback_queue = await stream_gemini(
                    system_prompt=system_prompt,
                    user_content=user_content, 
                    response_language=response_language,
                    use_search_grounding=False, 
                    use_url_context=use_url_context,
                    use_pro_model=use_pro_model,
                    # Ensure no media parameters are passed in fallback text-only call
                    image_url=None,
                    image_data_list=None,
                    video_data=None,
                    video_mime_type=None
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
            image_data = await download_media(input_image_url) # Changed to download_media
            
            if not image_data: 
                logger.error("Failed to download input media for image generation/edit, proceeding with text-only generation if applicable")
                input_image_url = None # Ensure no further attempt to use it
            else:
                # Configure generation parameters
                model = GEMINI_IMAGE_MODEL # This is specific to image generation
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
                    get_gemini_client().models.generate_content,
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
                get_gemini_client().models.generate_content,
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
                        img = await asyncio.to_thread(Image.open, BytesIO(image_bytes))
                        
                        # Convert to RGB mode if it's not already (required for JPEG)
                        if img.mode != 'RGB':
                            img = await asyncio.to_thread(img.convert, 'RGB')
                        
                        # Save as JPEG to a new BytesIO object
                        output = BytesIO()
                        await asyncio.to_thread(img.save, output, format='JPEG', quality=95)
                        output.seek(0)
                        
                        # Return the JPEG bytes
                        return output.getvalue()
                    except Exception as img_error:
                        logger.error(f"Error processing inline image data: {img_error}", exc_info=True)
                
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


async def generate_video_with_veo(
    system_prompt: str, 
    user_prompt: str, 
    image_data: Optional[bytes] = None
) -> tuple[Optional[bytes], Optional[str]]:
    """Generate video using Gemini VEO model.

    Args:
        system_prompt: The system prompt for the model.
        user_prompt: The user's description of the desired video.
        image_data: Optional image data to include in the generation.

    Returns:
        A tuple containing the video bytes and its MIME type, or (None, None) if generation failed.
    """
    logger.info(f"Generating video with VEO model: {GEMINI_VIDEO_MODEL}")
    logger.info(f"Initiating VEO video generation with model: {GEMINI_VIDEO_MODEL}")
    logger.info(f"System prompt (prepended): {system_prompt[:100]}...")
    logger.info(f"User prompt: {user_prompt[:100]}...")
    
    # combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    combined_prompt = user_prompt

    if image_data:
        logger.info(f"Image data provided: {len(image_data)} bytes")
    else:
        logger.info("No image data provided for video generation.")

    def _sync_generate_video():
        # Initialize client within the thread for safety
        if USE_VERTEX_VIDEO:
            sync_client = get_vertex_client()
        else:
            api_key_for_video = GEMINI_API_KEY
            if not api_key_for_video:
                logger.error("GEMINI_API_KEY not found in environment for _sync_generate_video.")
                raise ValueError("GEMINI_API_KEY not found for video generation client.")
            sync_client = genai.Client(api_key=api_key_for_video)
            logger.info("Synchronous Gemini client initialized for VEO generation.")

        image_part = None
        if image_data:
            try:
                img_mime_type = detect_mime_type(image_data)
                if not img_mime_type.startswith("image/"):
                    logger.error(f"Invalid MIME type for image_data: {img_mime_type}. Skipping image.")
                else: # Reverted
                    logger.info(f"Image data received with MIME type: {img_mime_type}")
            except Exception as e:
                logger.error(f"Error processing image_data: {e}. Skipping image.", exc_info=True)
                image_part = None # Ensure it's None if processing failed
        
        video_config = types.GenerateVideosConfig( # Reverted
            person_generation="allow_adult", # As per example
            aspect_ratio="16:9",          # As per example
            number_of_videos=1            # As per example
        )
        logger.info(f"Video generation config: {video_config}")

        try:
            logger.info(f"Calling client.models.generate_videos with model '{GEMINI_VIDEO_MODEL}'")
            operation = sync_client.models.generate_videos(
                model=GEMINI_VIDEO_MODEL if not USE_VERTEX_VIDEO else VERTEX_VIDEO_MODEL,
                prompt=combined_prompt,
                image=image_data if image_part else None,
                config=video_config
            )
            logger.info(f"Initial operation received: {operation.name}, Done: {operation.done}")

            polling_interval = 20  # seconds
            max_polling_attempts = 30 # Max 10 minutes (30 * 20s)
            attempts = 0

            while not operation.done and attempts < max_polling_attempts:
                logger.info(f"Polling operation '{operation.name}', attempt {attempts + 1}. Sleeping for {polling_interval}s.")
                time.sleep(polling_interval)
                operation = sync_client.operations.get(operation)
                logger.info(f"Operation status: Done={operation.done}")
                attempts += 1
            
            if not operation.done:
                logger.error(f"Video generation operation timed out after {attempts * polling_interval} seconds.")
                return None, None

            if operation.error:
                logger.error(f"Video generation operation failed with error: {operation.error}")
                return None, None

            if operation.response and operation.response.generated_videos:
                logger.info(f"Video generation successful. Found {len(operation.response.generated_videos)} video(s).")
                first_video = operation.response.generated_videos[0]
                
                # Assuming first_video.video is a resource that client.files can handle
                # and that it has a 'name' attribute (e.g., "files/file-id-string")
                if hasattr(first_video, 'video'):                    
                    # Get file metadata (includes mime_type)
                    mime_type = first_video.video.mime_type
                    logger.info(f"Video MIME type from metadata: {mime_type}")

                    # Download video content
                    logger.info(f"Downloading video content...")
                    # Assuming download().content provides bytes. If it saves to file, this needs change.
                    # Based on some SDKs, download might return a response-like object with a content attribute.
                    video_bytes = sync_client.files.download(file=first_video.video)
                    
                    logger.info(f"Video downloaded successfully: {len(video_bytes)} bytes, MIME type: {mime_type}")
                    return video_bytes, mime_type
                else:
                    logger.error("Generated video response does not have the expected 'video.name' structure.")
                    return None, None
            else:
                logger.warning("Video generation operation completed but no videos found in the response.")
                return None, None

        except Exception as e:
            logger.error(f"Exception during VEO video generation or polling: {e}", exc_info=True)
            return None, None

    try:
        return await asyncio.to_thread(_sync_generate_video)
    except Exception as e:
        logger.error(f"Error running _sync_generate_video in thread: {e}", exc_info=True)
        return None, None


async def generate_image_with_vertex(
    prompt: str,
    number_of_images: int = 4,
) -> List[bytes]:
    """Generate images using Vertex AI.

    Args:
        prompt: The user's description of the desired image.
        number_of_images: The number of images to generate.

    Returns:
        A list of image bytes (JPEG format), or an empty list if generation failed.
    """
    logger.info(
        f"Generating {number_of_images} images with Vertex AI (Model: {VERTEX_IMAGE_MODEL}) "
        f"using genai.Client for prompt: {prompt[:100]}..."
    )
    
    vertex_client = get_vertex_client()
    if not vertex_client:
        logger.error("Vertex AI client (genai.Client) not available or not configured for image generation.")
        return []
    
    if not VERTEX_IMAGE_MODEL: # Ensure model name is configured
        logger.error("VERTEX_IMAGE_MODEL not configured in .env file.")
        return []

    generated_images_bytes: List[bytes] = []

    try:

        # Configuration for image generation.        
        logger.info(f"Calling Vertex AI model {VERTEX_IMAGE_MODEL} via genai.Client.generate_content with candidate_count={number_of_images}")

        response = vertex_client.models.generate_images(
                model=VERTEX_IMAGE_MODEL,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=number_of_images,
                    enhance_prompt=True,
                    safety_filter_level="BLOCK_ONLY_HIGH",
                    person_generation="ALLOW_ADULT",
                    include_rai_reason=True
                )
            )

        if response and response.generated_images:
            logger.info(f"Vertex AI (genai.Client) response received. Images count: {len(response.generated_images)}")
            enhanced_prompt = None
            for generated_image in response.generated_images:
                if hasattr(generated_image, 'image'):
                    image = generated_image.image
                    if hasattr(image, 'image_bytes') and image.image_bytes is not None and len(image.image_bytes) > 0:
                        logger.info(f"Vertex AI (genai.Client): Found image data with mime type: {image.mime_type}")
                        image_bytes = image.image_bytes
                        try:
                            img = await asyncio.to_thread(Image.open, BytesIO(image_bytes))
                            if img.mode != 'RGB':
                                img = await asyncio.to_thread(img.convert, 'RGB')
                            
                            output_buffer = BytesIO()
                            await asyncio.to_thread(img.save, output_buffer, format='JPEG', quality=95)
                            generated_images_bytes.append(output_buffer.getvalue())
                            logger.info("Successfully processed one image from Vertex AI (genai.Client) response.")
                            if len(generated_images_bytes) >= number_of_images: # Stop if we have enough
                                break 
                        except Exception as img_proc_err:
                            logger.error(f"Vertex AI (genai.Client): Error processing image data: {img_proc_err}", exc_info=True)
                    if enhanced_prompt is None and generated_image.enhanced_prompt is not None:
                        enhanced_prompt = generated_image.enhanced_prompt
                if hasattr(generated_image, 'rai_filtered_reason'):
                    logger.error(f"Vertex AI (genai.Client): Error generating image with RAI reason: {generated_image.rai_filtered_reason}")
                if len(generated_images_bytes) >= number_of_images: # Stop if we have enough
                    break
            
            logger.info(f"Successfully processed {len(generated_images_bytes)} images from Vertex AI (genai.Client). With enhanced prompt {enhanced_prompt}")
        else:
            logger.warning("Vertex AI (genai.Client) image generation response did not contain any images or candidates.")

    except Exception as e:
        logger.error(f"Error generating image with Vertex AI (genai.Client): {e}", exc_info=True)
        if "quota" in str(e).lower():
            logger.error("Vertex AI image generation quota possibly exceeded.")
        # Add specific error checks for genai.Client if known
        return []

    if not generated_images_bytes:
        logger.warning(
            f"Vertex AI (genai.Client) image generation for prompt '{prompt[:100]}...' resulted in no usable images."
        )

    return generated_images_bytes[:number_of_images] # Ensure we don't return more than requested