"""Utility for managing Gemini File Search stores for TelegramGroupHelperBot."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai

from bot.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_PRO_MODEL


def _require_api_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    return GEMINI_API_KEY


def _wait_for_operation(client: genai.Client, operation) -> None:
    while not operation.done:
        time.sleep(5)
        operation = client.operations.get(operation)


def _build_metadata(metadata_items: List[str]) -> Optional[List[Dict[str, Any]]]:
    if not metadata_items:
        return None
    parsed: List[Dict[str, Any]] = []
    for item in metadata_items:
        if "=" not in item:
            raise ValueError(f"Invalid metadata '{item}'. Expected key=value format.")
        key, value = item.split("=", 1)
        entry: Dict[str, Any] = {"key": key.strip()}
        value = value.strip()
        if value.isdigit():
            entry["numeric_value"] = int(value)
        else:
            entry["string_value"] = value
        parsed.append(entry)
    return parsed


def _get_store_by_display_name(
    client: genai.Client, display_name: str
) -> Optional[Any]:
    stores = client.file_search_stores.list()
    for store in stores:
        if getattr(store, "display_name", None) == display_name:
            return store
    return None


def _ensure_store(client: genai.Client, display_name: str) -> str:
    store = _get_store_by_display_name(client, display_name)
    if store is not None:
        return store.name

    store = client.file_search_stores.create(config={"display_name": display_name})
    return store.name


def _require_existing_store(client: genai.Client, display_name: str) -> str:
    store = _get_store_by_display_name(client, display_name)
    if store is None:
        raise ValueError(
            f"No File Search store found with display name '{display_name}'."
        )
    return store.name


def _upload_file(
    client: genai.Client,
    store_name: str,
    file_path: Path,
    display_name: Optional[str],
    metadata: Optional[List[Dict[str, Any]]],
) -> None:
    config: Dict[str, Any] = {}
    if display_name:
        config["display_name"] = display_name
    if metadata:
        config["custom_metadata"] = metadata

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            operation = client.file_search_stores.upload_to_file_search_store(
                file=str(file_path),
                file_search_store_name=store_name,
                config=config or None,
            )
            _wait_for_operation(client, operation)
            return
        except genai.errors.ClientError as exc:
            message = ""
            if hasattr(exc, "response_json") and isinstance(exc.response_json, dict):
                message = exc.response_json.get("message", "")
            message = message or str(exc)
            status_code = getattr(exc, "status_code", None)
            retriable = (
                "Upload has already been terminated" in message
                or (status_code is not None and status_code in {500, 502, 503, 504})
            )
            if attempt >= max_attempts or not retriable:
                raise
            backoff = attempt * 2
            print(
                f"Upload failed for {file_path.name} ({message}). "
                f"Retrying in {backoff}s ({attempt}/{max_attempts})..."
            )
            time.sleep(backoff)


def _build_file_search_tool(store_name: str) -> Dict[str, Any]:
    return {"file_search": {"file_search_store_names": [store_name]}}


def _query_store(
    client: genai.Client, model: str, store_name: str, question: str
) -> str:
    config = {"tools": [_build_file_search_tool(store_name)]}
    response = client.models.generate_content(
        model=model,
        contents=question,
        config=config,
    )
    return getattr(response, "text", str(response))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or reuse a Gemini File Search store for a Telegram chat, "
            "optionally upload every file in a chat-history folder, and run a query."
        )
    )
    parser.add_argument(
        "--folder",
        type=Path,
        help="Path to the folder containing chat history files to upload (required unless --query-only).",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to run against the File Search store.",
    )
    parser.add_argument(
        "--store-display-name",
        required=True,
        help="Display name for the File Search store to create or reuse.",
    )
    parser.add_argument(
        "--file-display-name",
        help="Optional display name to associate with each uploaded file (defaults to the file name).",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Attach metadata to the upload (repeatable, e.g., author=alice).",
    )
    parser.add_argument(
        "--model",
        default=GEMINI_PRO_MODEL or GEMINI_MODEL,
        help="Gemini model to query (defaults to the configured Pro model).",
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Only query an existing File Search store without uploading new files.",
    )

    args = parser.parse_args(argv)
    if args.query_only:
        if args.folder is not None:
            parser.error("--folder cannot be used when --query-only is set.")
    else:
        if args.folder is None:
            parser.error("--folder is required unless --query-only is set.")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    api_key = _require_api_key()

    client = genai.Client(api_key=api_key)

    if args.query_only:
        store_name = _require_existing_store(client, args.store_display_name)
    else:
        folder = args.folder
        assert folder is not None
        if not folder.exists():
            raise FileNotFoundError(folder)
        if not folder.is_dir():
            raise NotADirectoryError(folder)

        store_name = _ensure_store(client, args.store_display_name)
        metadata = _build_metadata(args.metadata)
        files = sorted(p for p in folder.iterdir() if p.is_file())
        if not files:
            raise ValueError(f"No files found in folder {folder}")

        for file_path in files:
            display_name = args.file_display_name or file_path.name
            print(f"Uploading {file_path} as {display_name} ...")
            _upload_file(client, store_name, file_path, display_name, metadata)

    answer = _query_store(client, args.model, store_name, args.question)
    print("=== Gemini response ===")
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
