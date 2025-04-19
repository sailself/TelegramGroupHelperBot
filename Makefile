.PHONY: help setup run test lint typecheck export-req clean migrate deploy

help:
	@echo "Available commands:"
	@echo "  make setup        Install dependencies and setup the project"
	@echo "  make run          Run the bot locally"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make typecheck    Run type checker (mypy)"
	@echo "  make export-req   Export requirements.txt from Poetry"
	@echo "  make clean        Clean temporary files"
	@echo "  make migrate      Run database migrations"
	@echo "  make deploy       Deploy to Fly.io"

setup:
	@echo "Setting up the project..."
	pip install -e .
	pip install -r requirements.txt
	alembic upgrade head

run:
	@echo "Running the bot..."
	python -m bot.main

test:
	@echo "Running tests..."
	pytest

lint:
	@echo "Running linter..."
	ruff check .

typecheck:
	@echo "Running type checker..."
	mypy bot tests

export-req:
	@echo "Exporting requirements.txt..."
	pip install poetry
	poetry export -f requirements.txt --without-hashes > requirements.txt

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} +

migrate:
	@echo "Running database migrations..."
	alembic upgrade head

deploy:
	@echo "Deploying to Fly.io..."
	flyctl deploy 