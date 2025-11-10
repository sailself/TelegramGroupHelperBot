# Module & Function Summary

Comprehensive reference for the Python modules in this repository. Each entry lists the module-level purpose together with its primary functions, classes, and scripts so contributors can quickly orient themselves before making changes.

## Top-Level Utilities

### `tmp_query_chat_ids.py`
- Ad-hoc SQLite helper that scans every table for a `chat_id` column and prints aggregated counts per chat as JSON. Useful for manual audits of stored Telegram data.

### `get_chat_id.py`
- Minimal Telegram bot that echoes the user/chat identifiers for whitelist management.
- `handle_message(update, context)`: Logs sender metadata to stdout and replies with a Markdown-formatted cheat sheet.
- `main()`: Starts the polling bot after verifying that `BOT_TOKEN` is set.

## `bot` Package

### `bot/__init__.py`
- Re-exports configuration constants so callers can `from bot import BOT_TOKEN, ...` instead of importing from `bot.config` directly.

### `bot/config.py`
- Centralizes environment-variable driven configuration, logging bootstrap, and prompt templates.
- `OpenRouterModelConfig`: Dataclass describing each OpenRouter model’s capabilities.
- `_resolve_openrouter_models_path()`: Locates the JSON file that enumerates OpenRouter models.
- `_load_openrouter_models_from_path(path)`: Parses that JSON into `OpenRouterModelConfig` objects with validation and logging.
- `_load_legacy_openrouter_models()`: Builds configs from legacy environment variables when no JSON exists.
- `_build_openrouter_models(path)`: Orchestrates JSON vs. legacy fallback and produces the immutable tuple exposed to the rest of the codebase.
- `iter_openrouter_models()`: Returns the configured tuple (used for keyboards and validation).
- `get_openrouter_model_config(model_name)`: Looks up a single configuration by identifier.
- `_resolve_model_by_keyword(value, *keywords)`: Chooses a model ID by searching configured names when a shorthand flag (e.g., “llama”) is supplied.
- Module-level constants define bot tokens, Gemini/OpenRouter tuning, Vertex feature switches, rate limits, prompt templates (`TLDR_SYSTEM_PROMPT`, `FACTCHECK_SYSTEM_PROMPT`, `Q_SYSTEM_PROMPT`, etc.), access-control settings, and support messaging strings that the rest of the project imports.

### `bot/main.py`
- Entry point that wires together handlers, webhook/polling modes, whitelist warm-up, and graceful shutdown.
- `init_db_wrapper()`: Runs Alembic-backed table creation and preloads the whitelist cache.
- `post_init(application)`: Placeholder for any asynchronous setup after handlers are registered.
- `main()`: Creates a `telegram.ext.Application`, registers every command/handler, spins up the DB loop, optionally configures webhook hosting, and starts polling/webhook processing.

### `bot/utils/__init__.py`
- Namespace placeholder for future utilities; currently only exposes an empty `__all__`.

### `bot/utils/http.py`
- Provides a shared `aiohttp.ClientSession` for all outbound HTTP calls to reduce socket churn.
- `get_http_session()`: Lazily creates or returns a cached client session guarded by an asyncio lock.
- `close_http_session()`: Gracefully closes the shared session during bot shutdown.

### `bot/db/models.py`
- SQLAlchemy ORM definitions.
- `Base`: Declarative base shared across migrations.
- `Message`: Represents each Telegram message persisted for summarization, with a uniqueness constraint on `(chat_id, message_id)` to avoid duplicates plus indices for query performance.

### `bot/db/database.py`
- Async SQLAlchemy helpers plus a background writer queue so handler code can enqueue messages without blocking the event loop.
- `init_db()`: Creates tables (if needed) and starts the `db_writer` task.
- `drop_all_tables()`: Dangerous helper to purge the schema (disabled by default).
- `get_session()`: Async context manager that yields an `AsyncSession` and ensures commit/rollback semantics.
- `get_last_n_text_messages(chat_id, n, exclude_commands)`: Fetches the last `n` textual messages for `/tldr` and similar features.
- `get_messages_from_id(chat_id, from_message_id, exclude_commands)`: Streams messages starting at a lower message ID boundary (used when summarizing contiguous spans).
- `db_writer()`: Consumes the global `message_queue` and upserts rows, handling duplicate key races gracefully.
- `queue_message_insert(...)`: Public API for handlers to enqueue messages for storage without awaiting DB work.
- `select_messages(chat_id, limit)`: Convenience wrapper that pulls the latest `limit` entries (alias to `get_last_n_text_messages`).
- `select_messages_by_user(chat_id, user_id, limit, exclude_commands)`: Returns chronological history for a single user, used by `/profileme`, `/paintme`, etc.
- `select_messages_from_id(chat_id, message_id)`: Wrapper around `get_messages_from_id`.

### `bot/db/__init__.py`
- Re-exports the most commonly used DB helpers (`init_db`, `message_queue`, etc.) for cleaner imports.

## `bot.handlers` Package

### `bot/handlers/__init__.py`
- Collects and re-exports handler functions, LLM helpers, and shared state (e.g., `pending_q_requests`) so `bot.main` can import from a single module.

### `bot/handlers/access.py`
- Access control and rate limiting.
- `is_rate_limited(user_id)`: Returns `True` if the user has issued a command within `RATE_LIMIT_SECONDS`.
- `load_whitelist()`: Reads `WHITELIST_FILE_PATH` and caches the allowed IDs in memory.
- `is_user_whitelisted(user_id)`: Checks the cached list (auto-loading on first use).
- `is_chat_whitelisted(chat_id)`: Same as above for chats.
- `is_access_allowed(user_id, chat_id)`: Grants access when either the user or the chat appears in the whitelist.
- `requires_access_control(command)`: Allows fine-grained protection of selected commands via the `ACCESS_CONTROLLED_COMMANDS` env var.
- `check_access_control(update, command)`: Async guard that replies with an error when a user is not authorized.

### `bot/handlers/content.py`
- URL extraction, Telegraph authoring, and media downloading for fact-checking and QA commands.
- `markdown_to_telegraph_nodes(md_content)`: Converts Markdown into the node format expected by the Telegraph API.
- `html_to_telegraph_nodes(element)`: Recursively walks BeautifulSoup trees to produce Telegraph nodes, with custom handling for tables, lists, and code blocks.
- `create_telegraph_page(title, content)`: Calls the Telegraph API using the configured author metadata and returns a URL when long responses need to be offloaded.
- `extract_youtube_urls(text, max_urls)`: Returns up to `max_urls` valid YouTube URLs from command text.
- `extract_telegraph_urls_and_content(text, entities)`: Replaces Telegraph URLs with extracted content summary and returns both the rewritten text and metadata needed for media downloads.
- `download_telegraph_media(contents)`: Given extractor results, pulls associated images/videos so they can be fed into Gemini.
- `_is_twitter_url(url)`: Helper to validate candidate domains.
- `extract_twitter_urls_and_content(text, entities, source_text)`: Mirrors the Telegraph extractor but for Twitter/X links, including host allow-listing and deduplication.
- `download_twitter_media(twitter_contents, max_images, max_videos)`: Downloads up to `max_images` pictures and one video per fact-check to enrich Gemini context.

### `bot/handlers/responses.py`
- Output helpers and logging of inbound chat messages.
- `send_response(message, response, title, parse_mode)`: Writes responses back to Telegram, auto-creating a Telegraph page when the reply is longer than 22 lines or `TELEGRAM_MAX_LENGTH`.
- `log_message(update, context)`: Records every non-command text/caption via `queue_message_insert`, capturing language metadata with `langid`.

### `bot/handlers/commands.py`
- Implements high-level command handlers outside of `/q` routing.
- `tldr_handler(update, context)`: Summarizes recent chat traffic (default 100 messages) with Gemini using the `TLDR_SYSTEM_PROMPT`, respecting whitelist and rate limits.
- `factcheck_handler(update, context)`: Requires a reply, extracts linked Telegraph/Twitter content, downloads media, streams output via Gemini, and pipes final results through `send_response`.
- `handle_media_group(update, context)`: Keeps a transient cache of media-group messages inside `context.bot_data` so `/img` can operate on full albums.
- `img_handler(update, context)`: Uses Gemini or Vertex (if enabled) for image generation/editing, handles both inline prompts and replied media, uploads results via `send_response`, and persists the original request text for analytics.
- `vid_handler(update, context)`: Bridges `/vid` requests to `generate_video_with_veo`, optionally attaching image bytes from the replied message, and sends the resulting MP4/WebM back to the user.
- `start_handler(update, context)`: Replies with a command overview when someone starts the bot in private or in a chat.
- `paintme_handler(update, context)`: Builds an imaginative persona-based image prompt from the user’s recent chat history and calls `call_gemini` + image generation to return either a representation or portrait (for `/portraitme` alias).
- `profileme_handler(update, context)`: Similar history retrieval but produces a textual profile, optionally influenced by user-provided style hints.
- `help_handler(update, context)`: Sends the Markdown command reference.
- `support_handler(update, context)`: Displays the configured Ko-fi/support button.

### `bot/handlers/qa.py`
- Everything related to `/q`, model selection, and OpenRouter integrations.
- `resolve_alias_to_model_id(alias)`: Maps human-readable aliases to configured model IDs, including fallbacks for JSON-defined entries.
- `normalize_model_identifier(identifier)`: Standardizes user input and defaults to Gemini if blanks are provided.
- `get_model_capabilities(model_key)`: Advertises whether the model can process images/video/audio so the UI can hide incompatible options.
- `is_model_configured(model_key)`: Sanity-check for `/gpt`, `/llama`, etc.
- `is_openrouter_available()`: Returns `True` only when OpenRouter is enabled and keyed.
- `create_model_selection_keyboard(request_key, include_openrouter_models, include_gemini)`: Builds the inline keyboard shown after `/q` when multiple models are on tap.
- `process_q_request_with_gemini(request_data)`: Handles the full `/q` flow for Gemini-only cases, including history threading, media extraction, language detection, and invoking `call_gemini`.
- `process_q_request_with_specific_model(request_data, call_model, model_name)`: Shared worker for any OpenRouter model selection result.
- `q_handler(update, context, call_model, model_name)`: Entry point for `/q` (and alias commands) that enforces access control, collects context (e.g., replied-to messages, attachments), schedules timeouts, and sends the model-selection keyboard.
- `model_selection_callback(update, context)`: Processes button presses, dispatching to the appropriate Gemini/OpenRouter handler and ensuring duplicate presses are ignored.
- `get_model_display_name(model_key)`: Pretty name for inline buttons.
- `get_model_function_and_name(model_key)`: Maps a button press back to the callable (Gemini vs. OpenRouter).
- `_build_openrouter_command(alias, fallback)`: Utility for `/deepseek`, `/qwen`, `/llama`, `/gpt` commands to ensure the configured model exists.
- `cleanup_expired_requests(bot)`: Removes pending requests that exceeded `MODEL_SELECTION_TIMEOUT`.
- `handle_model_timeout(request_key, bot)`: Notifies the chat when a requester never chose a model.
- `periodic_cleanup_task(bot)`, `start_periodic_cleanup(bot)`, `stop_periodic_cleanup()`: Maintain the background cleanup loop.
- Shortcut handlers `deepseek_handler`, `qwen_handler`, `llama_handler`, `gpt_handler` simply call `_build_openrouter_command` and then invoke `q_handler`.

## `bot.llm` Package

### `bot/llm/__init__.py`
- Aggregates public LLM helpers for easier imports (Gemini, OpenRouter, Exa search, media tools, etc.).

### `bot/llm/clients.py`
- Lazy client factories for external APIs.
- `get_gemini_client()`: Initializes the singleton `genai.Client` with `GEMINI_API_KEY`.
- `get_openrouter_client()`: Creates an `AsyncOpenAI` client targeting the standard OpenRouter base URL.
- `get_openrouter_responses_client()`: Same as above but configured for the alpha Responses API when tool-calling is required.
- `get_vertex_client()`: Creates a Vertex AI client only when image/video features are enabled.

### `bot/llm/gemini.py`
- Full-featured Gemini + Vertex helpers.
- `call_gemini(...)`: Core text/vision entry point that supports search grounding, URL context, inline media, and output-language hints.
- `call_gemini_with_media(...)`: Lower-level helper used when video, audio, or image bytes must accompany the prompt.
- `extract_and_process_image_from_text(text_response)`: Pulls any base64 image payloads from Gemini responses and returns usable bytes.
- `stream_gemini(...)`: Starts a streaming response and exposes an asyncio queue so handlers can progressively edit messages (used by `/factcheck`).
- `test_gemini_vision(image_url)`: Diagnostic helper to sanity-check image analysis end-to-end.
- `generate_image_with_gemini(prompt, image_urls, prompt_hint, upload_to_cwd)`: Orchestrates image generation/editing plus optional upload to CWD.PW.
- `generate_video_with_veo(system_prompt, user_prompt, image_data)`: Wraps the Veo/Gemini video generation flow, including polling for completion and downloading the final file bytes.
- `generate_image_with_vertex(prompt, image_urls, model_hint)`: Uses Vertex AI as an alternative to Gemini for `/img`.

### `bot/llm/openrouter.py`
- Glue for OpenRouter chat completions plus result parsing.
- `parse_gpt_content(content)`: Splits GPT-style reasoning traces into “analysis” vs. “final answer”.
- `parse_qwen_content(content)`: Similar logic for Qwen’s `<think>` blocks.
- `_parse_openrouter_response(model_name, content)`: Chooses the right parser for a given model ID.
- `_build_function_tools()`: Declares the Exa search tool for models that support function calling.
- `_coerce_text(value)`: Normalizes OpenRouter’s nested response structure into plain strings.
- `_extract_reasoning_text(payload)`: Collects any `reasoning_content` chunks for display.
- `_message_content_to_text(message)` / `_message_to_dict(message)`: Helpers for formatting OpenRouter API payloads.
- `_execute_function_tool(call)`: Calls the Exa search helper and returns Markdown responses when models request web data.
- `_chat_completion_with_tools(messages, model_name, temperature, top_p, top_k)`: Loops through tool calls up to `MAX_TOOL_CALL_ITERATIONS` and injects the tool limit warning when needed.
- `call_openrouter(request_data, model_identifier, response_title)`: High-level async interface used by handlers; handles tool availability, streaming vs. completion API selection, and error reporting.

### `bot/llm/media.py`
- `detect_mime_type(image_data)`: Quick MIME sniffing based on magic numbers with HEIC/HEIF support.
- `download_media(media_url)`: Shared async downloader with timeout and error logging, built atop the shared HTTP session.

### `bot/llm/jina_search.py`
- Legacy Jina search integration.
- `JinaSearchResult` / `JinaSearchResponse`: Dataclasses describing search hits.
- `_authorized_client(timeout)`: Configures an `httpx.Client` with bearer auth if a key is set.
- `_parse_search_text(payload, max_results)`: Converts the plain-text response from Jina into structured results.
- `search_jina_web(query, max_results)`: Makes the HTTP call and returns a `JinaSearchResponse`.
- `fetch_jina_reader(url)`: Pulls markdown content for a URL via Jina’s reader endpoint.
- `format_search_results_markdown(results)`: Produces markdown for LLM consumption.
- `jina_search_tool(query, max_results)`: Small wrapper that returns a dict payload for tool-calling contexts.

### `bot/llm/exa_search.py`
- Modern Exa search facility wired into OpenRouter tool calling.
- `ExaSearchError`: Custom exception for telemetry.
- `_authorized_client(timeout)`, `_normalise_snippet(value)`, `_extract_results(payload)`: Internal helpers for interacting with Exa’s API.
- `exa_search(query, max_results)`: Executes the HTTP POST and normalizes the results list.
- `format_results_markdown(query, results)`: Produces Markdown to feed back into LLM prompts.
- `exa_search_tool(query, max_results)`: Structured dict response for function-calling flows.

## `bot.tools` Package

### `bot/tools/cwd_uploader.py`
- Upload pipeline for pushing generated images to CWD.PW.
- `upload_base64_image_to_cwd(base64_data, api_key, model, prompt)`: Validates a data URI, builds a multipart request manually, and returns the hosted URL.
- `upload_image_bytes_to_cwd(image_bytes, api_key, mime_type, model, prompt)`: Convenience wrapper that converts raw bytes into the data-URI format before delegating to the base64 uploader.

### `bot/tools/telegraph_extractor.py`
- `extract_telegraph_content(url)`: Calls the Telegraph `getPage` API via the shared HTTP session, walks the node tree, and returns cleaned text plus lists of image/video URLs.

### `bot/tools/twitter_extractor.py`
- Collection of helpers to normalize, fetch, and clean Twitter/X posts via the Jina proxy.
- Private helpers handle URL normalization (`_normalize_status_url`), host validation, query stripping, stop-word filtering, and deduplication.
- `extract_twitter_content(url)`: Main coroutine that fetches Markdown from r.jina.ai, extracts metadata (author, timestamp), collects referenced media URLs, and returns a formatted snippet ready for Gemini.

## Tests

### `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
- Placeholder modules to mark the respective packages; no executable code.

### `tests/unit/test_llm.py`
- Extensive async tests for `generate_video_with_veo`, `generate_image_with_gemini`, and `generate_image_with_vertex`, including success paths, timeout/error handling, MIME detection, and fallback logic using patched Google GenAI clients.

### `tests/unit/test_handlers.py`
- Covers `vid_handler`, `img_handler`, `paintme_handler`, and `profileme_handler`, ensuring rate limits are respected, media is downloaded when expected, and responses/editing behave correctly when LLM calls succeed or fail.

### `tests/unit/test_factcheck_lang.py`
- Verifies that `call_gemini` used by `/factcheck` preserves the input language without forcing overrides.

### `tests/unit/test_cwd_uploader.py`
- Exercises base64 parsing, MIME validation, HTTP error handling, and success flows for the CWD uploader helpers.

### `tests/unit/test_whitelist.py`
- Ensures whitelist loading tolerates missing files, ignores comments, and enforces restrictive defaults when errors occur; also verifies combined access control logic.

### `tests/unit/test_tldr_language.py`
- Confirms the TLDR system prompt is used as-is (Chinese) and that no explicit response language injection occurs.

### `tests/unit/test_telegraph.py`
- Validates Markdown/HTML conversion to Telegraph nodes plus the `create_telegraph_page` API integration (both success and failure scenarios).

### `tests/integration/test_telegraph.py`
- Integration-style tests that patch HTTP calls but exercise `send_response`, `q_handler`, and `factcheck_handler` when long responses require Telegraph offloading.

### `tests/integration/test_q_reply.py`
- Ensures replying with `/q` stitches the replied-to content into the Gemini prompt and sends the final answer back to the user.

### `tests/integration/test_tldr.py`
- Walks the `/tldr` command end-to-end with mocked database results and verifies the interaction with Gemini plus response length enforcement.

## Migrations

### `migrations/env.py`
- Standard Alembic env file adapted for async SQLAlchemy engines; wires migrations to the runtime `DATABASE_URL`.

### `migrations/versions/53244dba3a71_initial_schema.py`
- Initial Alembic revision creating the `messages` table with indices and the `(chat_id, message_id)` uniqueness constraint.

### `migrations/versions/add_unique_constraint.py`
- Follow-up revision that recreates the `messages` table on SQLite to enforce the unique constraint retroactively while deduplicating rows.

