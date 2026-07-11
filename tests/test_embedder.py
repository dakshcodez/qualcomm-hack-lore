import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from pc.indexer.embedder import Embedder, ExecutionProviderUnavailableError
from pc.indexer.profiler import InferenceProfiler

INPUT_DIM = 16
EMBEDDING_DIM = 8


def _build_toy_onnx_model(path, input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM):
    """A single-MatMul ONNX graph standing in for a real embedding model:
    input "input" [None, input_dim] -> output "embedding" [None, embedding_dim]."""
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
def toy_model_path(tmp_path):
    return _build_toy_onnx_model(tmp_path / "toy.onnx")


def test_embed_returns_vectors_of_expected_dimension(toy_model_path):
    embedder = Embedder(toy_model_path, preferred_providers=["CPUExecutionProvider"])

    vectors = embedder.embed(["hello world", "goodbye world"])

    assert len(vectors) == 2
    assert all(len(v) == EMBEDDING_DIM for v in vectors)


def test_embed_empty_list_returns_empty_list(toy_model_path):
    embedder = Embedder(toy_model_path, preferred_providers=["CPUExecutionProvider"])
    assert embedder.embed([]) == []


def test_embed_is_deterministic_for_the_same_text(toy_model_path):
    embedder = Embedder(toy_model_path, preferred_providers=["CPUExecutionProvider"])
    v1 = embedder.embed(["consistent text"])
    v2 = embedder.embed(["consistent text"])
    assert v1 == v2


def test_falls_back_to_cpu_when_qnn_and_dml_unavailable(toy_model_path):
    # This sandbox's onnxruntime build has no QNN/DirectML providers, so
    # requesting them exercises the real NPU->GPU->CPU fallback path.
    embedder = Embedder(
        toy_model_path,
        preferred_providers=["QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
    )
    assert embedder.active_provider == "CPUExecutionProvider"


def test_fallback_is_logged_as_a_warning_via_profiler(toy_model_path, caplog):
    import logging

    embedder = Embedder(
        toy_model_path,
        preferred_providers=["QNNExecutionProvider", "CPUExecutionProvider"],
    )
    with caplog.at_level(logging.WARNING, logger="pc.indexer.profiler"):
        embedder.embed(["trigger a profiled inference"])

    assert "fell back" in caplog.text
    assert "QNNExecutionProvider" in caplog.text
    summary = embedder.profiler.summary()
    assert summary["any_fallback"] is True
    assert summary["backends"]["CPU"]["count"] == 1


def test_no_fallback_warning_when_cpu_is_the_only_preference(toy_model_path, caplog):
    import logging

    embedder = Embedder(toy_model_path, preferred_providers=["CPUExecutionProvider"])
    with caplog.at_level(logging.WARNING, logger="pc.indexer.profiler"):
        embedder.embed(["no fallback expected here"])

    assert caplog.text == ""
    assert embedder.profiler.summary()["any_fallback"] is False


def test_raises_when_no_preferred_provider_is_available(toy_model_path):
    with pytest.raises(ExecutionProviderUnavailableError):
        Embedder(toy_model_path, preferred_providers=["TotallyMadeUpExecutionProvider"])


def test_custom_preprocess_fn_is_used(toy_model_path):
    calls = []

    def spy_preprocess(texts, input_dim):
        calls.append((tuple(texts), input_dim))
        return np.ones((len(texts), input_dim), dtype=np.float32)

    embedder = Embedder(
        toy_model_path,
        preferred_providers=["CPUExecutionProvider"],
        preprocess_fn=spy_preprocess,
    )
    embedder.embed(["a", "b"])

    assert calls == [(("a", "b"), INPUT_DIM)]


def test_shared_profiler_accumulates_across_embedders(toy_model_path):
    profiler = InferenceProfiler(preferred_providers=["CPUExecutionProvider"])
    embedder1 = Embedder(toy_model_path, preferred_providers=["CPUExecutionProvider"], profiler=profiler)
    embedder2 = Embedder(toy_model_path, preferred_providers=["CPUExecutionProvider"], profiler=profiler)

    embedder1.embed(["one"])
    embedder2.embed(["two"])

    assert profiler.summary()["total_inferences"] == 2
