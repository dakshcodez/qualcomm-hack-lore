import logging

from pc.api import cloud_client
from pc.api.cloud_client import rerank_and_generate


def test_empty_candidates_returns_no_results_message():
    result = rerank_and_generate("neural embeddings research", [1.0, 0.0], [])
    assert result["ranked_sources"] == []
    assert "neural embeddings research" in result["answer"]
    assert "No relevant results" in result["answer"]


def test_candidates_are_reranked_by_similarity_to_query_embedding():
    query_embedding = [1.0, 0.0]
    candidates = [
        {"title": "orthogonal", "chunk": "unrelated", "embedding": [0.0, 1.0]},
        {"title": "match", "chunk": "relevant content", "embedding": [1.0, 0.0]},
    ]
    result = rerank_and_generate("query", query_embedding, candidates)
    assert [c["title"] for c in result["ranked_sources"]] == ["match", "orthogonal"]


def test_falls_back_to_templated_answer_when_cloud_ai100_unavailable():
    # CloudAI100Client isn't wired to real hardware in this sandbox (see
    # cloud/inference.py), so rerank_and_generate() must fall back to the
    # local templated answer rather than raising.
    candidates = [{"title": "notes.txt", "chunk": "the hackathon starts July 11", "embedding": [1.0, 0.0]}]
    result = rerank_and_generate("when does it start", [1.0, 0.0], candidates)
    assert "notes.txt" in result["answer"]
    assert "the hackathon starts July 11" in result["answer"]


def test_fallback_logs_a_warning(caplog):
    candidates = [{"title": "a", "chunk": "b", "embedding": [1.0, 0.0]}]
    with caplog.at_level(logging.WARNING, logger="lore"):
        rerank_and_generate("query", [1.0, 0.0], candidates)
    assert "Cloud AI 100 unavailable" in caplog.text


def test_fallback_warning_carries_the_caller_supplied_request_logger(caplog):
    from pc.api.logging_config import get_request_logger

    request_logger = get_request_logger(request_id="req-smoke-test")
    candidates = [{"title": "a", "chunk": "b", "embedding": [1.0, 0.0]}]

    with caplog.at_level(logging.WARNING, logger="lore"):
        rerank_and_generate("query", [1.0, 0.0], candidates, request_logger=request_logger)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(getattr(r, "request_id", None) == "req-smoke-test" for r in warning_records)


def test_unexpected_client_error_still_falls_back_but_logs_as_error(monkeypatch, caplog):
    # A bug in generate_answer/a real (broken) client shouldn't be
    # mislabeled as "Cloud AI 100 unavailable" (that's specifically for
    # the expected NotImplementedError case) — it should still fall back
    # so /query survives, but be logged distinctly, with a traceback.
    class BrokenClient:
        def generate(self, prompt):
            raise RuntimeError("boom")

    monkeypatch.setattr(cloud_client, "CloudAI100Client", BrokenClient)

    candidates = [{"title": "notes.txt", "chunk": "the hackathon starts July 11", "embedding": [1.0, 0.0]}]
    with caplog.at_level(logging.WARNING, logger="lore"):
        result = rerank_and_generate("query", [1.0, 0.0], candidates)

    assert "notes.txt" in result["answer"]  # still falls back to a real answer
    assert "Cloud AI 100 unavailable" not in caplog.text
    assert "Unexpected error" in caplog.text
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert error_records[0].exc_info is not None  # traceback was captured


def test_uses_real_client_answer_when_cloud_ai100_available(monkeypatch):
    class FakeWorkingClient:
        def generate(self, prompt):
            return "a real Cloud AI 100 generated answer"

    monkeypatch.setattr(cloud_client, "CloudAI100Client", FakeWorkingClient)

    candidates = [{"title": "a", "chunk": "b", "embedding": [1.0, 0.0]}]
    result = rerank_and_generate("query", [1.0, 0.0], candidates)

    assert result["answer"] == "a real Cloud AI 100 generated answer"


def test_long_chunk_text_is_truncated_in_the_fallback_answer():
    long_chunk = "word " * 100  # far more than 200 chars
    candidates = [{"title": "long.txt", "chunk": long_chunk, "embedding": [1.0, 0.0]}]
    result = rerank_and_generate("query", [1.0, 0.0], candidates)
    assert result["answer"].endswith("...")
    assert len(result["answer"]) < len(long_chunk)


def test_missing_chunk_text_falls_back_to_generic_answer():
    candidates = [{"title": "empty.txt", "chunk": "", "embedding": [1.0, 0.0]}]
    result = rerank_and_generate("query", [1.0, 0.0], candidates)
    assert "empty.txt" in result["answer"]


def test_missing_title_falls_back_to_generic_label():
    candidates = [{"chunk": "some content", "embedding": [1.0, 0.0]}]
    result = rerank_and_generate("query", [1.0, 0.0], candidates)
    assert "an indexed document" in result["answer"]


def test_does_not_mutate_input_candidates():
    candidates = [{"title": "a", "chunk": "b", "embedding": [1.0, 0.0]}]
    original = [dict(c) for c in candidates]
    rerank_and_generate("query", [1.0, 0.0], candidates)
    assert candidates == original
