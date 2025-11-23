# AGENTS Guide

Authoritative checklist for contributors working inside `TelegramGroupHelperBot`. Read this document before opening a PR or running automation.

## Repository Map
- `bot/` — runtime code (`main.py`, `config.py`, handlers, db helpers, llm adapters, utils).
- `bot/tools/` — standalone async helpers (Telegraph/Twitter extractors, CWD uploader).
- `bot/db/` + `migrations/` — SQLAlchemy models, async engine bootstrap, Alembic revisions.
- `tests/unit/` & `tests/integration/` — pytest suites mirroring runtime modules.
- `docs/module_summary.md` — up-to-date description of every module/function (read before editing unfamiliar code).
- Supporting assets live under `.github/`, `fly.toml`, `logs/`, and deployment scripts.

## Setup & Tooling
1. **Environment**: Python 3.13+, Poetry installed. Copy `.env.example` to `.env` and populate required secrets (Telegram BOT_TOKEN, Gemini/OpenRouter keys, database URL). Never commit filled `.env`.
2. **Bootstrap**:
   ```bash
   poetry install          # or: make setup
   poetry run alembic upgrade head
   ```
3. **Common commands**:
   - `make run` / `poetry run python -m bot.main` — start the bot (loads whitelist & DB writer).
   - `make test` / `pytest` — run full suite. Use `pytest -m "not integration"` for quick checks.
   - `make lint` (`poetry run ruff check .`) and `make typecheck` (`poetry run mypy bot tests`) must be clean before pushing.
   - `make migrate` wraps `alembic upgrade head`. Never mix schema changes with feature code.
   - `make deploy` assumes Fly.io CLI is authenticated.

## Coding Practices
### Language & Style
- Target Python 3.13 async/await. 4-space indents, ≤100 char lines, Black-compatible formatting (enforced by Ruff). Keep imports sorted (`ruff I`) and prefer explicit relative imports inside packages.
- Public constants go in `bot/config.py`. Keep prompt strings and env defaults there.
- Use dataclasses or TypedDicts when exchanging structured data with LLM helpers to satisfy mypy’s strict mode.

### Async & IO
- All Telegram handlers must be `async` and guard against `None` `effective_message`/`effective_chat`.
- Use `bot.utils.http.get_http_session()` instead of creating ad-hoc `aiohttp` clients.
- Never block the event loop with synchronous DB calls; enqueue writes via `queue_message_insert` and use the async selectors in `bot.db.database`.
- When downloading or uploading media, respect the max limits used in `content.py` (`max_images`, `max_videos`) and reuse the helpers already there.

### Handlers & Commands
- Every command must:
  1. `await check_access_control(update, "<command>")`
  2. Enforce `is_rate_limited(...)`
  3. Provide a “processing…” placeholder via `reply_text`
  4. Call `send_response(...)` so long replies auto-migrate to Telegraph.
- Register new handlers in `bot/main.py` **and** export them through `bot/handlers/__init__.py`. Update `/help` text when adding user-facing commands.
- When touching `/q` flows, keep `pending_q_requests` coherence: write through `q_handler`, make sure keyboards use `MODEL_CALLBACK_PREFIX`, and update `docs/module_summary.md` plus inline comments to explain any new callback data.

### LLM Integrations
- Prefer `call_gemini` for Gemini tasks and `call_openrouter` for OpenRouter. Only bypass them if you are adding low-level capabilities (e.g., new tool-calling support) and mirror their error-handling patterns.
- When enabling media-aware prompts, set `youtube_urls`, `image_data_list`, `video_data`, etc., instead of embedding URLs manually. This keeps `get_model_capabilities` filtering accurate.
- Respect `ENABLE_EXA_SEARCH`, `ENABLE_JINA_MCP`, and other feature toggles; don’t assume APIs are available.
- When adding prompts, place the template in `bot/config.py` alongside existing system prompts and document the placeholder variables.

### Database & Persistence
- Schema changes go through Alembic revisions under `migrations/versions/`. Use `alembic revision --autogenerate -m "..."` and review the diff for unintended drops.
- Do not bypass the message queue. If you need new tables, expose typed selectors similar to `select_messages_by_user`.
- For cron-like logic (cleanup jobs, cache refresh), follow the pattern in `bot/handlers/qa.py` (`start_periodic_cleanup` / `stop_periodic_cleanup`) to avoid orphaned tasks.

### Logging, Errors, & Telemetry
- Obtain module-level loggers (`logging.getLogger(__name__)`). Avoid `print`.
- Wrap risky sections in `try/except Exception as exc` and log with `exc_info=True`. Surface user-friendly error text while keeping sensitive details in logs only.
- Use the rotating file handler configured in `bot/config.py`; do not reconfigure logging elsewhere.

### Security & Secrets
- Keep Telegram allowlisted IDs in `allowed_chat.txt`. Update `_whitelist_cache` via `load_whitelist()` after editing or expose an admin command if runtime reload is needed.
- Never log API keys, user tokens, or raw prompts that may contain secrets. Scrub them before logging.
- When adding downloads/uploads, validate hosts (see `_is_twitter_url`) and sanitize URLs to avoid SSRF.

## Testing & QA
- Unit tests belong beside the module they exercise (`tests/unit/test_<module>.py`). Integration tests should cover async Telegram flows end-to-end using mocks (see existing integration suites).
- For handler changes, add regression tests that simulate Telegram `Update` objects. Favor `IsolatedAsyncioTestCase` or `pytest.mark.asyncio`.
- When fixing bugs, add assertions reproducing the failure (e.g., whitelist edge cases, language detection). Mention new tests in the PR description.
- Run `pytest --asyncio-mode=auto --cov=bot` locally when touching handlers or LLM integrations; keep coverage deltas neutral or positive.

## Deployment & Operations
- The bot can run in webhook or polling mode. When toggling `USE_WEBHOOK`, ensure Fly.io/Gateway configs match and environment variables (`WEBHOOK_URL`, `WEBHOOK_PORT`) are updated.
- Before deploying:
  1. `make lint typecheck test`
  2. `poetry run pytest -m "integration"` if you touched long response flows, Telegraph, or OpenRouter.
  3. Verify `docs/module_summary.md` remains accurate.
  4. Confirm `logs/bot.log` stays under rotation control—do not commit it.
- Telemetry/resiliency:
  - Long replies must call `create_telegraph_page` via `send_response`.
  - For streaming LLM responses, reuse `stream_gemini` and ensure queues are drained to avoid memory leaks.

## Contribution Workflow
1. Create a descriptive branch: `feature/<short-description>` or `fix/<issue-id>`.
2. Keep commits focused. Subject lines should be imperative and ≤72 chars (`Add portrait mode handler`).
3. Summaries in PRs must cover:
   - Behaviour changes
   - Manual validation (commands run, sample chat flows)
   - Tests executed (`make test`, selective pytest markers)
   - Any new env vars/secrets or Fly.io steps.
4. Before requesting review, ensure CI is green and lint/typecheck/test have been run locally.

## Agent Activity Logging
- Every autonomous agent session must emit an execution diary under `agent_logs/agent_log_<timestamp>.md` (UTC ISO 8601, e.g., `agent_log_2025-11-08T21-45-00Z.md`).
- Each log entry documents, in order:
  1. **Step name / command** (what is about to run).
  2. **Thought brief** (why this step is necessary, key considerations).
  3. **Result** (outputs, files touched, success/failure, follow-ups).
  4. **Detail** (list of code changes with file name and line number).
- Append entries as you work so reviewers can audit decision flow. Do not retroactively edit old entries except to fix typos; add corrections as new steps.
- Store only plaintext Markdown; never include secrets. If a step is skipped because of policy or missing data, note that explicitly.

Following this guide keeps the bot stable, secure, and ready for rapid iteration. When in doubt, inspect `docs/module_summary.md` and existing tests to mirror established patterns.
