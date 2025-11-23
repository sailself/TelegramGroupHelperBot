"""Pinecone-backed RAG utilities with optional local embeddings."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions.exceptions import PineconeApiException
from sqlalchemy import text as sa_text

from bot.config import (
    ENABLE_PINECONE_RAG,
    GEMINI_EMBED_MODEL,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_HF_TOKEN,
    LOCAL_EMBEDDING_DEVICE,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    RAG_CHAT_TOP_K,
    RAG_EMBEDDING_PROVIDER,
    RAG_POLL_INTERVAL_SECONDS,
)
from bot.db.database import get_last_n_text_messages, get_messages_from_id, get_session
from bot.db.models import Message
from bot.llm.clients import get_gemini_client

logger = logging.getLogger(__name__)

_pc_client: Optional[Pinecone] = None
_pinecone_index = None
_local_embedder = None
_last_vectorized: Dict[int, int] = {}
_vectorizer_task: Optional[asyncio.Task] = None
_MAX_CHARS = 4000  # keep payloads well under Pinecone 4MB limit
_UPSERT_BATCH = 64  # keep request bodies small
_EMBED_BATCH = 64  # batch size for local embedding


def _get_pinecone_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is required for Pinecone RAG.")
    global _pc_client  # noqa: PLW0603
    if _pc_client is None:
        _pc_client = Pinecone(api_key=PINECONE_API_KEY)
    return _pc_client


def _ensure_index(dimension: int):
    """Create the Pinecone index if it does not exist and return a handle."""
    global _pinecone_index  # noqa: PLW0603
    if _pinecone_index is not None:
        return _pinecone_index

    client = _get_pinecone_client()
    try:
        info = client.describe_index(PINECONE_INDEX_NAME)
        existing_dim = getattr(info, "dimension", None) or info.get("dimension")
        if existing_dim and existing_dim != dimension:
            raise RuntimeError(
                f"Pinecone index {PINECONE_INDEX_NAME} dimension {existing_dim} "
                f"does not match embedding dimension {dimension}. "
                "Please recreate the index or switch embedding models."
            )
        _pinecone_index = client.Index(PINECONE_INDEX_NAME)
        return _pinecone_index
    except Exception as exc:  # noqa: BLE001
        logger.info("Creating Pinecone index %s (%s)", PINECONE_INDEX_NAME, exc)
        client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        _pinecone_index = client.Index(PINECONE_INDEX_NAME)
        return _pinecone_index


def _get_local_embedder():
    global _local_embedder  # noqa: PLW0603
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading local embedding model: %s (using auth token: %s)",
            LOCAL_EMBEDDING_MODEL,
            "yes" if LOCAL_EMBEDDING_HF_TOKEN else "no",
        )
        device = LOCAL_EMBEDDING_DEVICE or "cpu"
        model_kwargs: Dict[str, Any] = {"dtype": "auto"}
        try:
            _local_embedder = SentenceTransformer(
                LOCAL_EMBEDDING_MODEL,
                use_auth_token=LOCAL_EMBEDDING_HF_TOKEN,
                device=device,
                model_kwargs=model_kwargs,
            )
        except OSError as exc:
            raise RuntimeError(
                "Failed to load local embedding model. "
                "If the model is gated (e.g., embedding-gemma), set LOCAL_EMBEDDING_HF_TOKEN "
                "or HUGGINGFACE_HUB_TOKEN with a valid access token."
            ) from exc
        except AssertionError:
            # Likely CUDA not available; fall back to CPU and retry once
            logger.warning("CUDA not available; retrying local embedder on CPU.")
            _local_embedder = SentenceTransformer(
                LOCAL_EMBEDDING_MODEL,
                use_auth_token=LOCAL_EMBEDDING_HF_TOKEN,
                device="cpu",
                model_kwargs=model_kwargs,
            )
    return _local_embedder


async def _embed_with_gemini(texts: Sequence[str], *, is_query: bool = False) -> List[List[float]]:
    client = get_gemini_client()
    embeddings: List[List[float]] = []
    for text in texts:
        response = await client.aio.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            content=text,
        )
        if hasattr(response, "embedding"):
            embeddings.append(list(response.embedding.values))
        elif isinstance(response, dict):
            embeddings.append(list(response["embedding"]["values"]))
        else:
            raise RuntimeError("Unexpected Gemini embedding response format.")
    return embeddings


async def _embed_with_local(
    texts: Sequence[str], *, is_query: bool = False
) -> List[List[float]]:
    embedder = _get_local_embedder()
    loop = asyncio.get_running_loop()
    prompt_name = "query" if is_query else "document"

    def _run() -> Any:
        return embedder.encode(
            list(texts),
            prompt_name=prompt_name,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=_EMBED_BATCH,
        )

    vectors = await loop.run_in_executor(None, _run)
    return [vec.tolist() for vec in vectors]


async def _embed_texts(texts: Sequence[str], *, is_query: bool = False) -> List[List[float]]:
    if RAG_EMBEDDING_PROVIDER.lower() == "local":
        return await _embed_with_local(texts, is_query=is_query)
    return await _embed_with_gemini(texts, is_query=is_query)


def _chunk_text(text: str, *, max_chars: int = _MAX_CHARS) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _build_metadata(message: Message) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "chat_id": int(message.chat_id),
        "message_id": int(message.message_id),
        "text": message.text,
    }
    if message.user_id is not None:
        meta["user_id"] = int(message.user_id)
    if message.username:
        meta["username"] = message.username
    if isinstance(message.date, datetime):
        meta["date"] = message.date.isoformat()
    if message.reply_to_message_id is not None:
        meta["reply_to_message_id"] = int(message.reply_to_message_id)
    return meta


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Remove Pinecone-invalid values (None) from metadata."""
    return {k: v for k, v in meta.items() if v is not None}


async def upsert_messages_to_pinecone(
    messages: Sequence[Message], chat_id: int
) -> int:
    """Embed and upsert messages into Pinecone for a specific chat namespace."""
    text_messages = [m for m in messages if m.text]
    if not text_messages:
        return 0

    expanded_texts: List[str] = []
    expanded_metas: List[Dict[str, Any]] = []

    for msg in text_messages:
        chunks = _chunk_text(msg.text or "")
        for idx, chunk in enumerate(chunks):
            meta = _build_metadata(msg)
            meta["text"] = chunk
            meta["chunk"] = idx
            expanded_texts.append(chunk)
            expanded_metas.append(_sanitize_metadata(meta))

    if not expanded_texts:
        return 0

    embeddings = await _embed_texts(expanded_texts, is_query=False)
    if not embeddings:
        return 0

    index = _ensure_index(len(embeddings[0]))
    vectors = []
    for meta, embedding in zip(expanded_metas, embeddings):
        base_id = meta.get("message_id")
        chunk_id = meta.get("chunk", 0)
        vectors.append(
            {
                "id": f"{base_id}:{chunk_id}",
                "values": embedding,
                "metadata": meta,
            }
        )

    # Batch upserts to keep request bodies under Pinecone limits
    total = 0
    for idx in range(0, len(vectors), _UPSERT_BATCH):
        batch = vectors[idx : idx + _UPSERT_BATCH]
        try:
            await asyncio.to_thread(index.upsert, vectors=batch, namespace=str(chat_id))
        except PineconeApiException as exc:
            if "dimension" in str(exc).lower():
                raise RuntimeError(
                    f"Pinecone index dimension mismatch. Recreate index {PINECONE_INDEX_NAME} "
                    "or align the embedding model to the index dimension."
                ) from exc
            raise
        total += len(batch)
    return total


async def retrieve_context(
    chat_id: int, question: str, *, top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Retrieve top-k similar messages for a chat using the provided question."""
    top_k = top_k or RAG_CHAT_TOP_K
    embeddings = await _embed_texts([question], is_query=True)
    if not embeddings:
        return []

    vector = embeddings[0]
    index = _ensure_index(len(vector))
    result = await asyncio.to_thread(
        index.query,
        vector=vector,
        top_k=top_k,
        namespace=str(chat_id),
        include_metadata=True,
    )

    matches = getattr(result, "matches", None) or result.get("matches", [])
    return matches


def format_matches(matches: Iterable[Dict[str, Any]]) -> str:
    """Turn Pinecone matches into a concise context block."""
    lines: List[str] = []
    for match in matches:
        meta = match.metadata if hasattr(match, "metadata") else match.get("metadata", {})  # type: ignore[union-attr]
        text = meta.get("text") or meta.get("message_text")
        # Older records won't have text metadata; skip those.
        if not text:
            continue
        msg_id = meta.get("message_id")
        username = meta.get("username") or f"user {meta.get('user_id')}"
        lines.append(f"[{msg_id}] {username}: {text}")
    return "\n".join(lines)


async def retrieve_formatted_context(chat_id: int, question: str) -> str:
    matches = await retrieve_context(chat_id, question)
    return format_matches(matches)


async def get_distinct_chat_ids() -> List[int]:
    """Return all chat IDs seen in the messages table."""
    async with get_session() as session:
        result = await session.execute(sa_text("SELECT DISTINCT chat_id FROM messages"))
        return [row[0] for row in result.fetchall()]


async def _vectorize_chat(chat_id: int, batch_size: int = 200) -> int:
    last_seen = _last_vectorized.get(chat_id, 0)
    if last_seen == 0:
        messages = await get_last_n_text_messages(chat_id, batch_size)
    else:
        messages = await get_messages_from_id(chat_id, last_seen + 1)
        if len(messages) > batch_size:
            messages = messages[-batch_size:]

    if not messages:
        return 0
    processed = await upsert_messages_to_pinecone(messages, chat_id)
    _last_vectorized[chat_id] = max(msg.message_id for msg in messages)
    return processed


async def run_vectorizer_once(batch_size: int = 200) -> int:
    """Run a single sync cycle across all chats."""
    if not ENABLE_PINECONE_RAG:
        return 0
    chat_ids = await get_distinct_chat_ids()
    total = 0
    for chat_id in chat_ids:
        try:
            total += await _vectorize_chat(chat_id, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error vectorizing chat %s: %s", chat_id, exc, exc_info=True)
    return total


async def _vectorizer_loop(interval_seconds: int) -> None:
    while True:
        try:
            ingested = await run_vectorizer_once()
            if ingested:
                logger.info("Vectorizer ingested %s messages into Pinecone.", ingested)
        except Exception as exc:  # noqa: BLE001
            logger.error("Vectorizer loop error: %s", exc, exc_info=True)
        await asyncio.sleep(interval_seconds)


async def start_background_vectorizer() -> None:
    """Start the periodic vectorizer if enabled and not already running."""
    global _vectorizer_task  # noqa: PLW0603
    if not ENABLE_PINECONE_RAG:
        logger.info("Pinecone RAG disabled; not starting vectorizer.")
        return
    if not PINECONE_API_KEY:
        logger.warning(
            "Pinecone RAG enabled but PINECONE_API_KEY is missing; "
            "vectorizer will not start."
        )
        return
    if _vectorizer_task and not _vectorizer_task.done():
        return
    _vectorizer_task = asyncio.create_task(
        _vectorizer_loop(RAG_POLL_INTERVAL_SECONDS)
    )
    logger.info(
        "Pinecone vectorizer started (interval=%ss).", RAG_POLL_INTERVAL_SECONDS
    )


async def upsert_raw_records(
    chat_id: int, records: Sequence[Dict[str, Any]]
) -> int:
    """Embed and upsert raw dict records (for CLI import)."""
    messages: List[Message] = []
    for rec in records:
        text = rec.get("text")
        if not text:
            continue
        msg = Message(
            chat_id=chat_id,
            message_id=int(rec.get("message_id") or rec.get("id") or 0),
            user_id=(
                int(rec.get("user_id")) if rec.get("user_id") not in (None, "") else None
            ),
            username=rec.get("username"),
            text=text,
            language=None,
            date=rec.get("date") or datetime.utcnow(),
            reply_to_message_id=rec.get("reply_to_message_id"),
        )
        messages.append(msg)

    return await upsert_messages_to_pinecone(messages, chat_id)
