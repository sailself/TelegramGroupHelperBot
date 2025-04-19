# TelegramGroupHelperBot

A Telegram bot powered by Google Gemini 2.5 Flash AI that provides chat summarization, fact-checking, and Q&A capabilities.

## Features

- **Message Logging**: Records all text messages visible to the bot.
- **`/tldr <n>`**: Generates a Chinese-language summary of the last n messages.
- **`/factcheck`**: Fact-checks the replied message content against reliable sources.
- **`/q <question>`**: Answers questions in the detected language of the query.

## Requirements

- Python 3.11+
- python-telegram-bot 21.0+
- Google Gemini API key
- SQLite (dev) or PostgreSQL (production)

## Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Or with Poetry:
   ```
   poetry install
   ```
4. Copy `.env.example` to `.env` and fill in your credentials:
   ```
   cp .env.example .env
   ```
5. Run the database migrations:
   ```
   alembic upgrade head
   ```

## Running the Bot

### Development
```
python -m bot.main
```

### Production
The bot is designed to be deployed using Docker on platforms like Railway or Fly.io.

## Testing
Run tests with:
```
pytest
```

## License
MIT 