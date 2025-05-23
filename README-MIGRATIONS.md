# Run migration to update the database schema

# Database Migration Instructions

This document provides instructions for managing database migrations using Alembic with the TelegramGroupHelperBot.

## Prerequisites

Ensure you have Alembic installed:

```bash
pip install alembic
```

## Creating a New Migration

When you make changes to the database models in `bot/db/models.py`, you need to create a migration to update the database schema:

1. Make sure your virtual environment is activated (if you're using one)

2. Run the following command to generate a new migration:

```bash
alembic revision --autogenerate -m "Add new fields to Message model"
```

This will create a new migration script in the `migrations/versions/` directory.

3. Review the generated migration script to ensure it correctly captures the changes you made to the models.

## Applying Migrations

To apply migrations and update your database schema:

```bash
alembic upgrade head
```

This will apply all pending migrations to bring your database schema up to date.

## Rolling Back Migrations

If you need to revert a migration:

1. To revert to the previous migration:

```bash
alembic downgrade -1
```

2. To revert to a specific migration, use its revision identifier:

```bash
alembic downgrade <revision_id>
```

3. To revert all migrations:

```bash
alembic downgrade base
```

## Checking Migration Status

To check the current migration status:

```bash
alembic current
```

To see the migration history:

```bash
alembic history
```

## Handling the Latest Schema Changes

The latest changes to the database schema include:

1. Adding a unique constraint on `(chat_id, message_id)` columns in the `messages` table
2. Ensuring that records with the same `chat_id` and `message_id` are updated rather than trying to insert a duplicate

To apply these changes to your database:

1. Run the migration:

```bash
alembic upgrade head
```

## Upsert Functionality

The database has been enhanced with upsert functionality:

1. When a message with the same `chat_id` and `message_id` already exists in the database, it will be updated rather than rejected.
2. This ensures data consistency while avoiding duplicate entries.
3. The implementation handles potential race conditions with proper error management.

## Troubleshooting

If you encounter issues with the migrations:

1. Make sure your database URL in `.env` is correct
2. Check the Alembic logs for detailed error messages
3. For SQLite databases, note that it has limited support for column alterations. You might need to manually modify the migration script.
4. If you encounter constraint violations during migration, you might need to first clean up any duplicate records in your database before applying the unique constraint.
