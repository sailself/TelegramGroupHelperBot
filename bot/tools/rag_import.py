"""Import chat history JSON into Pinecone for RAG."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List

from bot.config import ENABLE_PINECONE_RAG
from bot.llm import upsert_raw_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import a JSON chat history file into the Pinecone RAG index "
            "(per-chat namespace)."
        )
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to the JSON file with chat messages.",
    )
    parser.add_argument(
        "--chat-id",
        required=True,
        type=int,
        help="Chat ID / namespace to which the records will be written.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optionally limit how many records are imported (0 = all).",
    )
    return parser.parse_args()


def _load_records(path: Path, limit: int | None) -> List[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        raw = raw[:limit]
    records = []
    for item in raw:
        dt_val = item.get("datetime")
        parsed_date = (
            datetime.fromisoformat(dt_val) if dt_val else datetime.utcnow()
        )
        records.append(
            {
                "message_id": item.get("id"),
                "user_id": item.get("user_id"),
                "username": item.get("username"),
                "text": item.get("text"),
                "reply_to_message_id": item.get("reply_to_message_id"),
                "date": parsed_date,
            }
        )
    return records


async def _main_async() -> int:
    args = _parse_args()
    if not ENABLE_PINECONE_RAG:
        raise RuntimeError("ENABLE_PINECONE_RAG is false; enable it before importing.")

    if not args.file.exists():
        raise FileNotFoundError(args.file)

    records = _load_records(args.file, args.max_records or None)
    total = len(records)
    print(
        f"[import] Starting import for chat_id={args.chat_id} "
        f"from {args.file} (records={total}, limit={args.max_records or 'all'})"
    )

    # Basic progress logging
    if total > 0:
        print(f"[import] Sample record keys: {list(records[0].keys())}")

    print("[import] Upserting records into Pinecone...")
    count = await upsert_raw_records(args.chat_id, records)
    print(f"[import] Completed. Imported {count}/{total} messages into namespace {args.chat_id}")
    return 0


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
