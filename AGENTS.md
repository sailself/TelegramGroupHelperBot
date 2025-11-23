# AGENTS Handbook

Guide for autonomous contributors working inside `TelegramGroupHelperBot`. Keep this close when modifying code or running automation.

## Architecture Overview
- Telegram entrypoint is `bot/main.py`, which builds the async `Application`, registers handlers, warms the DB/whitelist, and runs in webhook or polling mode. Shutdown closes the shared HTTP session.
- Configuration lives in `bot/config.py`: environment loading, logging bootstrap (timed rotating file + console), prompt constants, model toggles, rate limits, and OpenRouter model discovery (JSON or legacy env).
- Handlers in `bot/handlers/` orchestrate Telegram flows, call LLM helpers from `bot/llm/`, and persist messages via the async queue in `bot/db/database.py`.
- Data persistence uses SQLAlchemy async engine with a background writer task consuming `message_queue`; migrations are under `migrations/`.
- HTTP calls share a single `aiohttp` session (`bot/utils/http.py`). External content helpers live in `bot/tools/`.
- Tests mirror runtime modules under `tests/unit` and `tests/integration` (pytest + pytest-asyncio).

## Module Guide
- **Top-level scripts**
  - `get_chat_id.py`: Tiny polling bot that echoes user/chat identifiers for whitelist management.
  - `tmp_query_chat_ids.py`: SQLite inspector that aggregates `chat_id` usage across tables.
- **bot/config.py**
  - Loads `.env`, creates `logs/bot.log` via `TimedRotatingFileHandler`, and sets console logging.
- Defines all env-driven toggles (Gemini, Vertex, OpenRouter, Exa, Jina, CWD.PW, whitelist, webhook/polling, support copy) and prompt templates (`TLDR_SYSTEM_PROMPT` etc. - leave as-is even if they look garbled).
  - `OpenRouterModelConfig` plus helpers `iter_openrouter_models`, `get_openrouter_model_config`, `_resolve_model_by_keyword` to wire JSON/env model configs.
- **bot/main.py**
  - Builds the Telegram `Application`, registers every command and callback, attaches message logging, runs `init_db_wrapper()` (init DB + load whitelist), and starts webhook/polling. Uses `close_http_session` on shutdown.
- **bot/utils/http.py**
  - Provides a process-wide `aiohttp.ClientSession` guarded by an asyncio lock (`get_http_session` / `close_http_session`); default timeout 30s.
- **bot/db/models.py**
  - SQLAlchemy `Message` model with unique `(chat_id, message_id)`, indexed fields for chat/user/date/text metadata.
- **bot/db/database.py**
  - Async engine/session factory; `init_db` creates tables and spawns `db_writer`.
  - `db_writer` upserts messages from `message_queue`, handling race `IntegrityError` with an update retry.
  - Query helpers: `select_messages`, `select_messages_by_user`, `select_messages_from_id`, `get_last_n_text_messages`, `get_messages_from_id`.
  - `queue_message_insert` enqueues message data (defaults `chat_id` to user_id when missing).
- **bot/handlers/access.py**
  - Rate limiting via in-memory timestamps. Whitelist loader/cacher (`load_whitelist`), checks (`is_user_whitelisted`, `is_chat_whitelisted`, `is_access_allowed`), command gating (`requires_access_control`, `check_access_control`).
- **bot/handlers/responses.py**
  - `send_response` edits messages and auto-offloads to Telegraph when >22 lines or `TELEGRAM_MAX_LENGTH`; truncates as last resort.
  - `log_message` detects language via `langid`/`pycountry`, normalizes username, and enqueues DB inserts.
- **bot/handlers/content.py**
- Markdown/HTML -> Telegraph nodes, `create_telegraph_page`, YouTube URL extraction, Telegraph/Twitter content expansion with media download helpers (`download_telegraph_media`, `download_twitter_media`). Strict Twitter/X host allowlist.
- **bot/handlers/commands.py**
  - Implements `/tldr`, `/factcheck`, `/img`, `/vid`, `/paintme` `/portraitme`, `/profileme`, `/start`, `/help`, `/support`, plus media-group caching.
- Pattern: access check -> rate limit -> "processing..." reply -> gather context/media (including Telegraph/Twitter extraction) -> call LLM/media helpers -> respond via `send_response` or direct media send. Logs errors with `exc_info`.
- **bot/handlers/qa.py**
  - `/q` flow and model-selection callbacks; alias resolution for OpenRouter models vs Gemini; capability filtering for media; pending request registry with timeouts (`handle_model_timeout`, `cleanup_expired_requests`, `periodic_cleanup_task`).
  - `process_q_request_with_gemini` / `process_q_request_with_specific_model` route to Gemini/OpenRouter; enforces original requester on callbacks and hides unsupported models when media is present.
- **bot/handlers/__init__.py**
  - Re-exports handlers, access helpers, and shared constants for easy imports in `bot.main`.
- **bot/llm/clients.py**
  - Lazy singletons for Gemini, OpenRouter (standard + Responses API), and Vertex (when enabled).
- **bot/llm/gemini.py**
  - `call_gemini` (text or media-aware with search grounding/URL context), `call_gemini_with_media`, streaming/test helpers, image generation (`generate_image_with_gemini`, `generate_image_with_vertex`), video generation via VEO (`generate_video_with_veo`), base64 image extraction, MIME detection integration.
- **bot/llm/openrouter.py**
  - OpenRouter chat completions with optional Exa function-calling (`exa_web_search`), GPT/Qwen reasoning parsers, media embedding via base64, retries without media on model errors, tool-call iteration cap.
- **bot/llm/exa_search.py**
  - Exa API client/formatter and function-call payload builder; raises `ExaSearchError` when misconfigured.
- **bot/llm/jina_search.py**
  - Legacy Jina search/reader with regex parsing into dataclasses and Markdown formatter.
- **bot/llm/media.py**
  - `detect_mime_type` via magic numbers; `download_media` using the shared HTTP session with robust error logging.
- **bot/tools/cwd_uploader.py**
  - Manual multipart upload to cwd.pw from base64 or bytes; validates MIME/extension and logs failures.
- **bot/tools/telegraph_extractor.py**
  - Fetches Telegraph pages via API, extracts text plus image/video URLs, and normalizes relative links.
- **bot/tools/twitter_extractor.py**
  - Validates/normalizes Twitter/X URLs, fetches via r.jina.ai proxy, cleans Markdown, deduplicates media, and returns formatted content + media lists.
- **tests/**
  - Unit tests cover LLM helpers, handlers, language handling, CWD uploader, whitelist, and content conversion; integration tests exercise `/q`, `/tldr`, and Telegraph flows with mocked HTTP.
- **migrations/**
  - Alembic async `env.py` plus revisions for initial schema and unique constraint enforcement.

## Coding Style & Conventions
- Target Python 3.11+ (per `pyproject.toml`); async-first codebase. 4-space indent, Ruff line length 100, imports sorted (`ruff I`). Mypy is strict (`disallow_untyped_defs`, `no_implicit_optional`, etc.) - type new code accordingly.
- Logging: use `logging.getLogger(__name__)`; avoid `print` and avoid reconfiguring handlers (config.py sets rotating file + console).
- HTTP: reuse `get_http_session()`; do not create ad-hoc sessions; handle `ClientError`/timeouts explicitly.
- DB: never block the event loop with sync calls; enqueue via `queue_message_insert` and let `db_writer` upsert. Respect `(chat_id, message_id)` uniqueness and keep `get_session` context managers around DB work.
- Telegram handlers: always guard `effective_message`/`effective_chat`, run `check_access_control`, enforce `is_rate_limited`, send a processing placeholder, and route long replies through `send_response`. Register handlers in `bot/main.py` and export via `bot/handlers/__init__.py`; update help text when user-facing commands change.
- Access control: whitelist IDs live in `allowed_chat.txt` (path via `WHITELIST_FILE_PATH`). Missing file means "allow all." Keep cache coherent by calling `load_whitelist()` on startup.
- LLM usage:
  - Prefer `call_gemini`; set `use_pro_model` when media is present. Pass `youtube_urls`, `image_data_list`, `video_data`, or `audio_data` instead of embedding URLs to keep capability filtering accurate.
  - Use `call_openrouter` only for configured models; provide `supports_tools=False` for models lacking tool support. Keep Exa search enabled via config when expecting tool calls.
- Media/content handling: use helpers in `content.py` and `media.py`; respect host allowlists; observe `max_images`/`max_videos` defaults in download helpers; prefer `download_media` instead of new fetchers.
- Error handling: wrap risky sections with broad `except Exception as exc` and log `exc_info=True`; present user-friendly messages without leaking secrets. `send_response` already handles Telegram length errors via Telegraph or truncation.
- Prompts/config: prompt constants (e.g., `TLDR_SYSTEM_PROMPT`) include non-ASCII artifacts - do not "clean" them. Keep env defaults centralized in `config.py`.
- Testing: use pytest/pytest-asyncio. Ruff/mypy configs live in `pyproject.toml`. Run unit/integration tests matching touched areas.

## Modularity & Layering
- Handlers manage Telegram I/O and assemble context; LLM modules perform API calls; tools/content helpers handle remote data extraction; DB modules isolate persistence.
- Reuse shared HTTP session and download helpers; do not let handlers talk directly to low-level HTTP without going through utilities.
- Keep DB writes asynchronous via the queue; avoid business logic inside `db_writer`.
- Preserve separation: LLM helpers are Telegram-agnostic; only handlers should import Telegram types.

## Feature & Refactor Recipes
- **New command/handler**: create async handler in `bot/handlers/commands.py` (or dedicated module), apply access/rate guards, send processing message, call LLM/content helpers, reply via `send_response` or Telegram media APIs. Register in `bot/main.py`, export via `bot/handlers/__init__.py`, and update `/help` + `docs/module_summary.md`. Add pytest coverage (unit for logic, integration for end-to-end flows).
- **Add OpenRouter model**: update `openrouter_models.json` (or env variables) with capability flags (`image`/`video`/`audio`/`tools`). Ensure aliases in `config.py` if you expect `/deepseek`/`/llama`/etc. to resolve. Mention in README/help if user-facing.
- **DB schema change**: create Alembic revision under `migrations/versions/`, update `bot/db/models.py`, adjust selectors, and avoid mixing schema changes with feature code.
- **New HTTP helper or extractor**: place in `bot/utils` or `bot/tools`, reuse `get_http_session`, and add targeted unit tests.
- **Logging additions**: use module loggers and include context (chat_id/user_id/model); avoid printing secrets or raw prompts.

## Pitfalls & Invariants
- Logging is configured in `config.py`; re-calling `basicConfig` may interfere. Prefer module loggers.
- `pending_q_requests` keys combine chat/user/selection message; ensure cleanup paths are preserved when altering `/q` flow (timeouts and `periodic_cleanup_task`).
- `queue_message_insert` defaults `chat_id` to `user_id` when absent - supply chat_id explicitly for group context.
- `send_response` will create Telegraph pages when long; give meaningful titles and expect edits to the processing message rather than new sends.
- Twitter extraction enforces host allowlist; unsupported hosts will be skipped with logged warnings.
- Rate limits are in-memory; restarts reset them.
- Some help/response strings include playful Unicode; keep them stable to avoid test breaks.

## Agent Activity Logging
- Every autonomous session must append a diary under `agent_logs/agent_log_<timestamp>Z.md` (UTC ISO 8601 with colons replaced by hyphens).
- Each entry should record, in order: **Step** (command/action), **Thought** (why/considerations), **Result** (outcome, files touched/modified, follow-ups). Include code/diff summaries for edits.
- Append as you work; do not rewrite history except to fix typos. Plaintext only; never include secrets.
