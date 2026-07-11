import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from starlette.testclient import TestClient

from pc.api.dependencies import get_embedder, get_vector_store
from pc.indexer.embedder import Embedder
from pc.indexer.vector_store import VectorStore

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
def configured_app(tmp_path, monkeypatch):
    """Points main.app's lifespan env vars at a synthetic ONNX model and a
    temp LanceDB dir, so app startup exercises the real wiring end-to-end."""
    model_path = _build_toy_onnx_model(tmp_path / "toy.onnx")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(model_path))
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "lancedb"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "lore.log"))

    from pc.api.main import app

    return app


def test_health_endpoint(configured_app):
    with TestClient(configured_app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lifespan_builds_real_embedder_and_vector_store_on_state(configured_app):
    with TestClient(configured_app) as client:
        client.get("/health")  # triggers startup within the `with` context
        assert isinstance(configured_app.state.embedder, Embedder)
        assert isinstance(configured_app.state.vector_store, VectorStore)


def test_get_embedder_dependency_reads_from_app_state():
    class DummyState:
        embedder = object()

    class DummyApp:
        state = DummyState()

    class DummyRequest:
        app = DummyApp()

    assert get_embedder(DummyRequest()) is DummyApp.state.embedder


def test_get_vector_store_dependency_reads_from_app_state():
    class DummyState:
        vector_store = object()

    class DummyApp:
        state = DummyState()

    class DummyRequest:
        app = DummyApp()

    assert get_vector_store(DummyRequest()) is DummyApp.state.vector_store
