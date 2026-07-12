"""Phase 3 — Cloud AI 100 client wiring for the /query route: reranks
candidates via cloud.reranker, attempts real generation via
cloud.inference.CloudAI100Client (a Cloud AI 100-hosted model, reached
via Cirrascale's Imagine SDK), and falls back to a deterministic
templated answer (Phase 2's original behavior) if that's unavailable —
which it is until IMAGINE_API_KEY is set (see cloud/inference.py's
docstring for the full env-var contract).

When nothing was found to ground an answer in at all (nothing's been
indexed yet, or nothing matched the query), this no longer just says so
— it asks the Cloud AI 100 LLM to answer from general knowledge instead
(clearly flagged as not coming from the user's own files), so a search
with no matches doesn't just dead-end. That path uses the same
NotImplementedError-vs-real-error fallback as the grounded path, so it
degrades to the old "no relevant results" message if Cloud AI 100 isn't
configured either.

Input: a query string + its embedding + the top-k candidate chunk rows
from VectorStore.search() (candidates may be empty).
Output: {"answer": str, "ranked_sources": list[dict]}.
Side effects: attempts to construct a CloudAI100Client and call it (a
real HTTP call to the Cloud AI 100-hosted model, once IMAGINE_API_KEY is
set); logs a warning and falls back to a local templated answer if that
fails, so /query keeps working before credentials are provided.
"""

import logging

from cloud import reranker
from cloud.inference import CloudAI100Client, generate_answer, generate_general_answer

logger = logging.getLogger("lore")

_EXCERPT_MAX_CHARS = 200


def _template_fallback_answer(query, candidates):
    """Deterministic templated answer over the top candidate (Phase 2's
    original behavior) — used whenever Cloud AI 100 generation is
    unavailable, so /query keeps working before real hardware is wired up."""
    if not candidates:
        return f'No relevant results found for "{query}".'

    top = candidates[0]
    excerpt = (top.get("chunk") or "").strip()
    if len(excerpt) > _EXCERPT_MAX_CHARS:
        excerpt = excerpt[:_EXCERPT_MAX_CHARS].rstrip() + "..."

    title = top.get("title") or "an indexed document"
    return f'Based on "{title}": {excerpt}' if excerpt else f'Found a match in "{title}".'


def _run_cloud_ai100(generate_fn, fallback_fn, log):
    """Try the real Cloud AI 100 client via generate_fn(client); fall back
    to fallback_fn() if it's not configured yet or fails unexpectedly.
    Shared by both the grounded and no-candidates-found generation paths
    so they degrade identically and log identically.

    Args:
        generate_fn: callable(client) -> str, the actual generation call.
        fallback_fn: callable() -> str, used if the client is unavailable
            or generation fails.
        log: logger (or LoggerAdapter) to log the WARNING/ERROR through.

    Returns:
        The generated answer, or fallback_fn()'s result.
    """
    try:
        client = CloudAI100Client()
        return generate_fn(client)
    except NotImplementedError as exc:
        # Expected until CloudAI100Client is wired to real hardware — see
        # cloud/inference.py. Falling back keeps /query working in the
        # meantime, so this is a WARNING, not an error.
        log.warning("Cloud AI 100 unavailable, falling back to local templated answer: %s", exc)
        return fallback_fn()
    except Exception:
        # Anything else is a real bug (in generate_answer, the client once
        # it's real, etc.) — log it distinctly, with a traceback, so it
        # doesn't get silently mislabeled as "hardware not wired up yet".
        # Still falls back rather than raising, so a live demo survives it.
        log.exception("Unexpected error generating a Cloud AI 100 answer; falling back to local templated answer")
        return fallback_fn()


def rerank_and_generate(query, query_embedding, candidates, request_logger=None):
    """Rerank candidates and generate an answer — via the real Cloud AI
    100 client if available, else a local templated fallback. If nothing
    matched the query at all, asks the LLM to answer generally instead of
    just reporting no results (still via the same client/fallback logic).

    Args:
        query: the original user query text.
        query_embedding: the query's embedding vector (used for reranking).
        candidates: list of chunk row dicts (from VectorStore.search()).
            May be empty.
        request_logger: optional logger (e.g. logging_config.get_request_logger()'s
            LoggerAdapter) to log through, so the fallback warning carries
            the caller's request ID. Defaults to the plain module logger.

    Returns:
        {"answer": str, "ranked_sources": list[dict]}. `ranked_sources` is
        `candidates` reordered by cloud.reranker.rerank() (empty if
        `candidates` was empty). `answer` comes from the real Cloud AI 100
        client when available, else a deterministic templated fallback.
    """
    log = request_logger or logger

    if not candidates:
        answer = _run_cloud_ai100(
            lambda client: generate_general_answer(query, client),
            lambda: f'No relevant results found for "{query}".',
            log,
        )
        return {"answer": answer, "ranked_sources": []}

    ranked_sources = reranker.rerank(query_embedding, candidates)

    answer = _run_cloud_ai100(
        lambda client: generate_answer(query, ranked_sources, client),
        lambda: _template_fallback_answer(query, ranked_sources),
        log,
    )

    return {"answer": answer, "ranked_sources": ranked_sources}
