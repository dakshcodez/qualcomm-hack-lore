"""Phase 2/3 — POST /query: embed the query, search LanceDB for the top-5
nearest chunks, hand them to the Cloud AI 100 client (cloud_client.py) for
reranking + answer generation, and return the final answer + sources to
the caller (the mobile app).

Input: QueryRequest {text, modality}.
Output: QueryResponse {answer, sources[]}.
Side effects: runs embedder inference; queries LanceDB; calls
cloud_client.rerank_and_generate(); logs request lifecycle timings.
"""

import time

from fastapi import APIRouter, Depends

from pc.api import cloud_client
from pc.api.dependencies import get_embedder, get_vector_store
from pc.api.logging_config import get_request_logger
from pc.api.schemas import QueryRequest, QueryResponse, SourceItem

router = APIRouter()

TOP_K = 5
EXCERPT_MAX_CHARS = 300


def _excerpt(text, max_chars=EXCERPT_MAX_CHARS):
    text = (text or "").strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, embedder=Depends(get_embedder), vector_store=Depends(get_vector_store)):
    logger = get_request_logger()
    logger.info("POST /query text_len=%d modality=%s", len(request.text), request.modality)

    embed_start = time.perf_counter()
    [embedding] = embedder.embed([request.text], prefix="query: ")
    embed_ms = (time.perf_counter() - embed_start) * 1000
    logger.debug("Embedding took %.2fms", embed_ms)

    search_start = time.perf_counter()
    candidates = vector_store.search(embedding, top_k=TOP_K)
    search_ms = (time.perf_counter() - search_start) * 1000
    logger.debug("LanceDB search took %.2fms, %d candidates", search_ms, len(candidates))

    cloud_start = time.perf_counter()
    result = cloud_client.rerank_and_generate(request.text, embedding, candidates, request_logger=logger)
    cloud_ms = (time.perf_counter() - cloud_start) * 1000
    logger.debug("Cloud AI 100 round-trip took %.2fms", cloud_ms)

    sources = [
        SourceItem(
            title=source.get("title", ""),
            location=source.get("location", ""),
            excerpt=_excerpt(source.get("chunk", "")),
            file_type=source.get("file_type", ""),
        )
        for source in result["ranked_sources"]
    ]

    logger.info(
        "Served /query in %.2fms (embed=%.2fms search=%.2fms cloud=%.2fms)",
        embed_ms + search_ms + cloud_ms,
        embed_ms,
        search_ms,
        cloud_ms,
    )

    return QueryResponse(answer=result["answer"], sources=sources)
