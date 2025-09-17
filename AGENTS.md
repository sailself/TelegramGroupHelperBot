# Repository Guidelines

## Project Structure & Module Organization
Core bot logic sits in `bot/` (entrypoint `bot/main.py`, command handlers `bot/handlers.py`, config `bot/config.py`). Database helpers and migrations support live in `bot/db/` and `migrations/`. Automation scripts appear in `bot/tools/`; test suites mirror runtime code under `tests/unit/` and `tests/integration/`. Runtime data such as `bot.db` and exported logs belong under `logs/`, while deployment manifests stay in `.github/` and `fly.toml`.

## Build, Test, and Development Commands
- `poetry install` or `make setup`: install dependencies and apply the latest Alembic migration.
- `poetry run python -m bot.main` / `make run`: start the bot with env vars from `.env`.
- `poetry run pytest -m "not integration"` for quick cycles, or `make test` for everything.
- `poetry run ruff check .` with optional `--fix`, and `poetry run mypy bot tests` / `make typecheck` before pushing.

## Coding Style & Naming Conventions
Ruff enforces formatting with a 100-character line limit and Black-compatible spacing; stick to 4-space indents. Modules, functions, and async coroutines use `snake_case`, classes stay `PascalCase`, and constants live in `UPPER_CASE` near module tops. Keep imports sorted (ruff `I` rule) and lean on dataclasses or TypedDicts to preserve the strict mypy configuration in `pyproject.toml`.

## Testing Guidelines
Pytest powers the suite with async fixtures in `tests/conftest.py`. Name files after the module under test (`bot/handlers.py` -> `tests/unit/test_handlers.py`). Exercise database surface area through integration specs that run against the SQLite test DB. Run `poetry run pytest --asyncio-mode=auto --cov=bot` when touching handlers or LLM adapters and include regression tests for every bugfix.

## Commit & Pull Request Guidelines
Commit subjects follow the imperative voice and stay under roughly 72 characters (`Improve DeepSeek switcher`). Group related file changes; do not mix migrations with feature work. Pull requests must outline behaviour changes, test evidence (`make test`, manual chat flows), and any new environment variables or Fly.io deployment steps. Verify CI status before requesting review.

## Environment & Secrets
Copy `.env.example` to `.env`, supply Telegram, Gemini/OpenRouter, and database credentials, and keep local overrides out of version control. Store allow-listed chat IDs in `allowed_chat.txt`; the `.example` file documents expected format. If you add new sensitive files, extend `.gitignore` and confirm they are excluded from commits.
