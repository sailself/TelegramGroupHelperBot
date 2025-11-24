# Telegram Group Helper Bot

A Telegram bot for group chats that provides summarization, fact-checking, and question-answering capabilities using Google's Gemini AI with Google Search grounding.

## Features

- **TLDR Summary**: Summarize the last N messages in a group chat with `/tldr [number]`
- **Fact Checking**: Fact-check messages by replying with `/factcheck`
- **Multi-Model Question Answering**: Ask questions with `/q <your question>` and choose from multiple AI models (Gemini 3 ✨, Llama 4, Qwen 3, DeepSeek 3.1) via interactive buttons
- **Smart Model Selection**: Automatic filtering to show only media-capable models (Gemini, Llama) when images, videos, or audio are present
- **Image Generation**: Generate images with `/img <description>`. Uses Gemini by default, or Vertex AI if configured (can return multiple images). Generated images are automatically uploaded to CWD.PW for external hosting with AI generation metadata (model and prompt information).
- **Video Generation**: Generates a video based on a text prompt and/or a replied-to image with `/vid <prompt>`.
- **Image Understanding**: Analyze and understand images when replying to a photo with `/factcheck` or `/q`
- **Google Search Grounding**: All responses are grounded in current information from Google Search
- **Telegraph Integration**: Automatically creates Telegraph pages for lengthy responses
- **Database Logging**: Messages are stored in a database for summarization and analysis
- **Multi-language Support**: Automatically detects and responds in the same language as the query
- **Support Integration**: Customizable support message with Ko-fi link for tip collection
- **Proactive Cleanup**: Automatic cleanup of expired model selection messages every 5 seconds
- **OpenRouter Integration**: Configurable support for OpenRouter models with fallback to Gemini-only mode

## Requirements

- Python 3.13 or higher
- A Telegram Bot Token from [BotFather](https://t.me/botfather)
- Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
- Optional: Telegraph API access token (for optimal Telegraph integration)
- Optional: CWD.PW API key (for image hosting when using image generation features)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TelegramGroupHelperBot.git
   cd TelegramGroupHelperBot
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source ./venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration (see Configuration section below for all options):
   ```bash
   # Copy and customize the configuration template
   touch .env
   # Edit .env with your Telegram Bot Token, Gemini API key, and optional settings
   ```

5. Run the database migrations:
   ```bash
   alembic upgrade head
   ```

6. Run the bot:
   ```bash
   python -m bot.main
   ```

## Configuration

Configure the bot by editing the `.env` file:

### Required settings:
- `BOT_TOKEN`: Your Telegram Bot token from BotFather
- `GEMINI_API_KEY`: Google Gemini API key

### OpenRouter settings (Optional):
- `ENABLE_OPENROUTER`: Enable/disable OpenRouter model selection (default: "true")
- `OPENROUTER_API_KEY`: Your OpenRouter API key for accessing Llama, Qwen, DeepSeek, and GPT models
- `OPENROUTER_MODELS_CONFIG_PATH`: Optional path to a JSON file that lists the OpenRouter models exposed to the bot (defaults to `openrouter_models.json`)
- `OPENROUTER_ALPHA_BASE_URL`: Base URL for the alpha Responses API (default: "https://openrouter.ai/api/alpha")
- `ENABLE_JINA_MCP`: Legacy flag for the deprecated Jina MCP integration (default: "false")
- `JINA_AI_API_KEY`: Optional Jina API key (only used if you re-enable the legacy MCP integration)
- `JINA_SEARCH_ENDPOINT`: Jina search API endpoint (default: "https://s.jina.ai/search")
- `JINA_READER_ENDPOINT`: Jina reader API endpoint (default: "https://r.jina.ai/")
- `ENABLE_EXA_SEARCH`: Enable/disable the Exa-powered search function available to OpenRouter models (default: "true")
- `EXA_API_KEY`: Exa API key used for the search function
- `EXA_SEARCH_ENDPOINT`: Exa search API endpoint (default: "https://api.exa.ai/search")
- `DEEPSEEK_MODEL`, `QWEN_MODEL`, `LLAMA_MODEL`, `GROK_MODEL`, `GPT_MODEL`: Legacy fallback identifiers for direct command shortcuts. These are auto-populated from the JSON config when possible.

Create `openrouter_models.json` (or point `OPENROUTER_MODELS_CONFIG_PATH` to a custom location) to control which OpenRouter models are offered in the `/q` picker. A sample is provided in `openrouter_models.json.example`:

```json
{
  "models": [
    {
      "name": "Llama 4",
      "model": "meta-llama/llama-4-maverick:free",
      "image": true,
      "video": true,
      "audio": true,
      "tools": true
    },
    {
      "name": "DeepSeek R1",
      "model": "deepseek/deepseek-r1-0528:free",
      "image": false,
      "video": false,
      "audio": false,
      "tools": false
    }
  ]
}
```

Field meanings:

- `name`: Friendly label shown in the inline keyboard.
- `model`: Exact model identifier passed to OpenRouter.
- `image`/`video`/`audio`: Whether the model can consume the corresponding media types. Models that lack support are hidden when a request includes that media.
- `tools`: Set to `false` for models that cannot execute tool calls. The bot will fall back to the standard Chat Completions API instead of the OpenRouter Responses API in that case.

### Telegraph settings:
- `TELEGRAM_MAX_LENGTH`: Maximum character length before using Telegraph (default: 4000)
- `TELEGRAPH_ACCESS_TOKEN`: Your Telegraph API access token
- `TELEGRAPH_AUTHOR_NAME`: Name to display as author on Telegraph pages
- `TELEGRAPH_AUTHOR_URL`: URL for the author name

### Image hosting settings:
- `CWD_PW_API_KEY`: Your CWD.PW API key for image hosting (optional, required for external image hosting)

### Support/Ko-fi settings:
- `SUPPORT_MESSAGE`: Custom message displayed with the `/support` command (default: "☕ If you find this bot helpful, consider supporting its development!")
- `KOFI_LINK`: Your Ko-fi profile URL for accepting tips (default: "https://ko-fi.com/yourusername")

### Whitelist settings:
- `WHITELIST_FILE_PATH`: Path to the whitelist file containing allowed user IDs (default: "allowed_chat.txt")

### Optional settings:
- `DATABASE_URL`: Database connection URL (default: SQLite)
- `USE_WEBHOOK`: Whether to use webhook mode (default: false)
- `WEBHOOK_URL`: URL for the webhook in production
- `RATE_LIMIT_SECONDS`: Rate limiting between user requests (default: 15)
- `MODEL_SELECTION_TIMEOUT`: Timeout for model selection in seconds (default: 30)
- `GEMINI_MODEL`: Gemini model to use for general text tasks (default: gemini-pro)
- `GEMINI_PRO_MODEL`: Gemini Pro model for more complex tasks, including media analysis (default: gemini-2.5-pro-exp-03-25)
- `GEMINI_IMAGE_MODEL`: Gemini model for image generation (default: gemini-3-pro-image-preview)
- `GEMINI_VIDEO_MODEL`: The Gemini model to use for video generation. Defaults to "veo-3.0-generate-preview".

### Vertex AI Settings (Optional):
- `VERTEX_PROJECT_ID`: Your Google Cloud Project ID (required if using Vertex AI for any feature).
- `VERTEX_LOCATION`: The Google Cloud region for your Vertex AI resources (e.g., `us-central1`, required if using Vertex AI).
- `USE_VERTEX_IMAGE`: Set to "true" to use Vertex AI for image generation via the `/img` command (default: "false").
- `VERTEX_IMAGE_MODEL`: The specific Vertex AI image generation model to use (e.g., `imagegeneration@006`).
- `USE_VERTEX_VIDEO`: Set to "true" to use Vertex AI for video generation via the `/vid` command (default: "false", currently uses Gemini VEO).
- `VERTEX_VIDEO_MODEL`: The specific Vertex AI video generation model to use if `USE_VERTEX_VIDEO` is true.

## Example Configuration

Here's a sample `.env` file with common configurations:

### Minimal Configuration (Gemini-only)
```env
# Required settings
BOT_TOKEN=your_telegram_bot_token_here
GEMINI_API_KEY=your_gemini_api_key_here

# Disable OpenRouter for cost control
ENABLE_OPENROUTER=false
```

### Full Multi-Model Configuration
```env
# Required settings
BOT_TOKEN=your_telegram_bot_token_here
GEMINI_API_KEY=your_gemini_api_key_here

# Multi-model support
ENABLE_OPENROUTER=true
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_ALPHA_BASE_URL=https://openrouter.ai/api/alpha
ENABLE_JINA_MCP=false
JINA_AI_API_KEY=
JINA_SEARCH_ENDPOINT=https://s.jina.ai/search
JINA_READER_ENDPOINT=https://r.jina.ai/
ENABLE_EXA_SEARCH=true
EXA_API_KEY=your_exa_api_key
EXA_SEARCH_ENDPOINT=https://api.exa.ai/search
DEEPSEEK_MODEL=deepseek/deepseek-v3
QWEN_MODEL=qwen/qwen-2.5-72b-instruct
LLAMA_MODEL=meta-llama/llama-3.1-405b-instruct
GPT_MODEL=openai/gpt-4o

# Customization
MODEL_SELECTION_TIMEOUT=60
TELEGRAM_MAX_LENGTH=4000
RATE_LIMIT_SECONDS=10

# Optional features
TELEGRAPH_ACCESS_TOKEN=your_telegraph_token
CWD_PW_API_KEY=your_cwd_pw_api_key
SUPPORT_LINK=https://ko-fi.com/yourusername
```

## Deployment

The bot can be deployed in various ways:

1. **Local Development**: Run the bot locally with polling mode
2. **Production Webhook**: Deploy to a server and use webhook mode
3. **Docker**: Use the provided Dockerfile for containerized deployment
4. **Fly.io**: Deploy using the included `fly.toml` configuration

## Using Docker

1. Build the Docker image:
   ```bash
   docker build -t telegram-group-helper-bot .
   ```

2. Run the container:
   ```bash
   docker run --env-file .env telegram-group-helper-bot
   ```

## Using Fly.io

1. Install the Fly.io CLI
2. Log in to Fly.io:
   ```bash
   fly auth login
   ```

3. Launch the app:
   ```bash
   fly launch --generate-name
   ```

4. Set secrets:
   ```bash
   fly secrets set BOT_TOKEN=your_bot_token GEMINI_API_KEY=your_gemini_key CWD_PW_API_KEY=your_cwd_pw_key
   ```

5. Deploy:
   ```bash
   fly deploy
   ```

## Telegraph Integration

For responses that exceed the Telegram message size limit, the bot automatically:

1. Creates a Telegraph page with proper formatting
2. Sends a link to the page instead of the full content
3. Preserves markdown formatting in the Telegraph page

This feature triggers automatically when:
- The message exceeds the character limit set in `TELEGRAM_MAX_LENGTH`
- The message contains too many lines (22+)

## Google Search Grounding

This bot uses Google's Gemini AI with Google Search grounding to ensure responses are based on current, accurate information. When you ask a question or fact-check a statement, the bot:

1. Searches the web for relevant information
2. Grounds its response in the search results
3. Provides a factual, up-to-date answer
4. Includes citations where appropriate

## Multi-Model AI Integration

The bot provides flexible AI model selection through an intuitive interface:

### Interactive Model Selection
- **Smart Interface**: Use `/q <question>` to get interactive buttons for model selection
- **4 AI Models**: Choose from Gemini 3 ✨, Llama 4, Qwen 3, and DeepSeek 3.1
- **Media-Aware**: When images, videos, or audio are present, only media-capable models (Gemini, Llama) are shown
- **User-Specific**: Only the original requester can select the model (prevents button hijacking)
- **Auto-Timeout**: Selection expires after 30 seconds (configurable) with automatic message cleanup

### Direct Model Access

### OpenRouter Fallback
- **Flexible Configuration**: Can be disabled via `ENABLE_OPENROUTER=false`
- **Graceful Fallback**: Automatically uses Gemini-only mode when OpenRouter is unavailable
- **Cost Control**: Disable OpenRouter to avoid API costs while keeping full functionality

## Image Understanding

The bot leverages Gemini's advanced image understanding capabilities to:

- Analyze photos shared in the group
- Answer questions about image content
- Fact-check claims in images or captions
- Detect objects and scenes in photos
- Process multiple image formats (JPEG, PNG, WEBP, HEIC, HEIF)

Simply reply to an image with `/q What's in this picture?` or `/factcheck` to analyze visual content.

## User and Chat Whitelist

The bot includes a comprehensive whitelist feature to restrict access to authorized users and groups:

### Features:
- **Access Control**: Only users or groups listed in the whitelist file can use bot commands
- **Flexible Access**: Users can access the bot if either their user ID OR their group chat ID is whitelisted
- **Backward Compatibility**: If no whitelist file exists, all users and groups are allowed (default behavior)
- **Easy Management**: Simple text file with one ID per line (user IDs or group chat IDs)
- **Security**: Failed whitelist checks are logged for monitoring

### Configuration:
1. Create a file named `allowed_chat.txt` (or set `WHITELIST_FILE_PATH` in your `.env`)
2. Add one Telegram user ID or group chat ID per line
3. Restart the bot

### Getting User IDs:
- Ask users to send a message to [@userinfobot](https://t.me/userinfobot)
- Check bot logs when users send commands (temporarily disable whitelist first)
- Use Telegram's API to get user information

### Getting Group Chat IDs:
- Add the bot to the group
- Send a command in the group
- Check the bot logs - you'll see the chat_id (negative number)
- Or use [@userinfobot](https://t.me/userinfobot) in the group to get the group ID
- Or use the included `get_chat_id.py` script to easily get IDs

### Example whitelist file:
```
# Telegram User and Chat Whitelist
# Add one ID per line (user IDs or group chat IDs)

# User IDs (positive numbers):
123456789
987654321

# Group Chat IDs (negative numbers):
-1001234567890
-1009876543210
```

### Disabling Whitelist:
To allow all users (disable whitelist), either:
- Delete the `allowed_chat.txt` file
- Set `WHITELIST_FILE_PATH` to a non-existent file path

### Helper Script:
Use the included `get_chat_id.py` script to easily get user and chat IDs:
1. Edit the script and set your bot token
2. Run: `python get_chat_id.py`
3. Send a message to your bot
4. The script will display all the ID information you need for the whitelist

## Image Hosting Integration

The bot includes automatic image hosting through CWD.PW for generated images:

### Features:
- **Automatic Upload**: All generated images are automatically uploaded to CWD.PW for external hosting
- **Multiple Formats**: Supports PNG, JPEG, and WebP image formats
- **Error Handling**: Upload failures don't affect image generation functionality
- **Logging**: Upload status and URLs are logged for debugging

### Configuration:
To enable image hosting, set your CWD.PW API key in the `.env` file:
```
CWD_PW_API_KEY=your_cwd_pw_api_key_here
```

### Behavior:
- When `CWD_PW_API_KEY` is configured: Generated images are uploaded to CWD.PW and URLs are logged
- When `CWD_PW_API_KEY` is not configured: Images are generated normally but not uploaded
- Upload failures are logged but don't prevent image generation or delivery to users

## Database Management

The bot uses SQLAlchemy for database operations. By default, it uses SQLite, but you can configure PostgreSQL for production.

See [README-MIGRATIONS.md](README-MIGRATIONS.md) for instructions on managing database migrations.

## Testing

Run tests with pytest:

```bash
pytest
```

To run specific test categories:

```bash
# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/
```

## Authors

- **Kevin Z** - Original project creator and main developer
- **Frank L** - CWD.PW image hosting integration and AI metadata enhancement

See [AUTHORS](AUTHORS) for detailed contribution information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
