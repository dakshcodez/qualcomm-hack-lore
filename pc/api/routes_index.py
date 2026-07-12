"""Phase 2 — POST /index: maps a browser-extension page capture
{text, url, title} onto the LanceDB text schema (chunked + embedded via
the same shared pipeline used for filesystem indexing) and stores it.

Input: IndexRequest {text, url, title}.
Output: IndexResponse {status, chunks_indexed}.
Side effects: runs embedder inference; deletes any previously-indexed
chunks for the same URL and writes the new ones to LanceDB (so re-visiting
a page doesn't duplicate rows); logs request lifecycle.
"""

import time

from fastapi import APIRouter, Depends

from pc.api.dependencies import get_embedder, get_vector_store
from pc.api.logging_config import get_request_logger
from pc.api.schemas import IndexRequest, IndexResponse
from pc.indexer.chunker import chunk_text

router = APIRouter()


@router.post("/index", response_model=IndexResponse)
def index(request: IndexRequest, embedder=Depends(get_embedder), vector_store=Depends(get_vector_store)):
    logger = get_request_logger()
    logger.info("POST /index text_len=%d url=%s", len(request.text), request.url)

    try:
        chunks = chunk_text(request.text)
        if not chunks:
            logger.info("No chunks extracted from /index payload for %s", request.url)
            return IndexResponse(status="ok", chunks_indexed=0)

        embed_start = time.perf_counter()
        embeddings = embedder.embed([chunk["text"] for chunk in chunks])
        embed_ms = (time.perf_counter() - embed_start) * 1000
        logger.debug("Embedded %d chunk(s) in %.2fms", len(chunks), embed_ms)

        records = [
            {
                "location": request.url,
                "title": request.title,
                "chunk": chunk["text"],
                "embedding": embedding,
                "file_type": "web",
                "chunk_index": chunk["chunk_index"],
                "metadata": {"source": "browser", "url": request.url},
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        # Replace this URL's previous chunks (if any) so re-visiting the
        # same page doesn't duplicate rows on every dwell-time trigger.
        vector_store.delete_by_location(request.url)
        vector_store.upsert_chunks(records)
    except Exception:
        logger.exception("Failed to index /index payload for %s", request.url)
        return IndexResponse(status="error", chunks_indexed=0)

    logger.info("Indexed %d chunk(s) from %s", len(records), request.url)
    return IndexResponse(status="ok", chunks_indexed=len(records))
