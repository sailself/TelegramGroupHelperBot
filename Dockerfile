FROM python:3.11-slim

# Set non-root user
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Set up workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Create volume for database
VOLUME /app/data

# Set DATABASE_URL environment variable for container
ENV DATABASE_URL=sqlite+aiosqlite:///data/bot.db

# Set permissions
RUN chown -R botuser:botuser /app
USER botuser

# Run migrations on startup
CMD alembic upgrade head && python -m bot.main 