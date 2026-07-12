import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from starlette.testclient import TestClient

INPUT_DIM = 16
EMBEDDING_DIM = 8


def _build_toy_onnx_model(path, input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM):
    weight = np.random.RandomState(0).randn(input_dim, embedding_dim).astype(np.float32)
    weight_initializer = helper.make_tensor(
        "weight", TensorProto.FLOAT, weight.shape, weight.flatten().tolist()
    )
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, input_dim])
    output_tensor = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, embedding_dim])
    node = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["embedding"])
    graph = helper.make_graph(
        [node], "toy_embedder", [input_tensor], [output_tensor], initializer=[weight_initializer]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


@pytest.fixture
def client(tmp_path, monkeypatch):
    model_path = _build_toy_onnx_model(tmp_path / "toy.onnx")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(model_path))
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "lancedb"))
    monkeypatch.setenv("EMBEDDING_DIM", str(EMBEDDING_DIM))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "lore.log"))

    from pc.api.main import app

    with TestClient(app) as test_client:
        yield test_client, app


def test_index_valid_payload_returns_ok_and_chunk_count(client):
    test_client, app = client
    response = test_client.post("/index", json={
        "text": "This article explains private on-device semantic search.",
        "url": "https://example.com/article",
        "title": "Article Title",
    })

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] >= 1
    assert app.state.vector_store.text_table.count_rows() == body["chunks_indexed"]


def test_index_writes_rows_with_web_file_type_and_browser_metadata(client):
    test_client, app = client
    test_client.post("/index", json={
        "text": "Some page content worth remembering.",
        "url": "https://example.com/page",
        "title": "Page Title",
    })

    rows = app.state.vector_store.text_table.to_arrow().to_pylist()
    assert len(rows) >= 1
    row = rows[0]
    assert row["file_type"] == "web"
    assert row["location"] == "https://example.com/page"
    assert row["title"] == "Page Title"
    assert row["page"] is None and row["sheet"] is None and row["slide"] is None and row["section"] is None
    assert json.loads(row["metadata"]) == {"source": "browser", "url": "https://example.com/page"}


def test_index_empty_text_writes_nothing(client):
    test_client, app = client
    response = test_client.post("/index", json={"text": "   ", "url": "https://example.com/empty", "title": "Empty"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] == 0
    assert app.state.vector_store.text_table.count_rows() == 0


def test_index_rejects_missing_fields(client):
    test_client, _ = client
    response = test_client.post("/index", json={"text": "content", "url": "https://example.com"})
    assert response.status_code == 422


def test_reindexing_same_url_replaces_rather_than_duplicates(client):
    test_client, app = client
    payload = {"url": "https://example.com/page", "title": "Page Title"}

    first = test_client.post("/index", json={**payload, "text": "original page content"})
    second = test_client.post("/index", json={**payload, "text": "the page was edited since"})

    assert first.status_code == 200 and second.status_code == 200
    rows = app.state.vector_store.text_table.to_arrow().to_pylist()
    assert app.state.vector_store.text_table.count_rows() == second.json()["chunks_indexed"]
    all_chunk_text = " ".join(r["chunk"] for r in rows)
    assert "original page content" not in all_chunk_text
    assert "the page was edited since" in all_chunk_text
