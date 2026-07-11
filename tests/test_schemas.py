import pytest
from pydantic import ValidationError

from pc.api.schemas import IndexRequest, IndexResponse, QueryRequest, QueryResponse, SourceItem


def test_query_request_defaults_modality_to_text():
    request = QueryRequest(text="neural embeddings research")
    assert request.modality == "text"


def test_query_request_accepts_image_modality():
    request = QueryRequest(text="a photo", modality="image")
    assert request.modality == "image"


def test_query_request_rejects_invalid_modality():
    with pytest.raises(ValidationError):
        QueryRequest(text="hello", modality="audio")


def test_query_request_requires_text():
    with pytest.raises(ValidationError):
        QueryRequest()


def test_query_response_defaults_sources_to_empty_list():
    response = QueryResponse(answer="no results found")
    assert response.sources == []


def test_query_response_holds_source_items():
    response = QueryResponse(
        answer="here is what I found",
        sources=[
            SourceItem(title="notes.txt", location="/home/user/notes.txt", excerpt="...", file_type="txt"),
        ],
    )
    assert len(response.sources) == 1
    assert response.sources[0].file_type == "txt"


def test_query_response_serializes_to_expected_shape():
    response = QueryResponse(
        answer="answer text",
        sources=[SourceItem(title="a", location="/a", excerpt="e", file_type="pdf")],
    )
    dumped = response.model_dump()
    assert dumped == {
        "answer": "answer text",
        "sources": [{"title": "a", "location": "/a", "excerpt": "e", "file_type": "pdf"}],
    }


def test_index_request_requires_all_fields():
    with pytest.raises(ValidationError):
        IndexRequest(text="page content", url="https://example.com/article")


def test_index_request_valid_payload():
    request = IndexRequest(text="page content...", url="https://example.com/article", title="Article Title")
    assert request.url == "https://example.com/article"
    assert request.title == "Article Title"


def test_index_response_defaults():
    response = IndexResponse()
    assert response.status == "ok"
    assert response.chunks_indexed == 0


def test_index_response_rejects_invalid_status():
    with pytest.raises(ValidationError):
        IndexResponse(status="maybe")
