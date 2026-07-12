import time

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from pc.indexer.embedder import Embedder
from pc.indexer.vector_store import VectorStore
from pc.indexer.watcher import DEFAULT_EXCLUDED_DIR_NAMES, Watcher, index_file, is_excluded

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
def embedder(tmp_path):
    model_path = _build_toy_onnx_model(tmp_path / "toy.onnx")
    return Embedder(model_path, preferred_providers=["CPUExecutionProvider"])


@pytest.fixture
def vector_store(tmp_path):
    return VectorStore(tmp_path / "lancedb", embedding_dim=EMBEDDING_DIM)


# --- is_excluded ---------------------------------------------------------

@pytest.mark.parametrize("relative", [
    "AppData/Local/file.txt",
    "project/node_modules/pkg/file.txt",
    "repo/.git/file.txt",
    "repo/.hidden/file.txt",
])
def test_is_excluded_true_for_configured_and_hidden_dirs(tmp_path, relative):
    path = tmp_path / relative
    assert is_excluded(path, tmp_path, DEFAULT_EXCLUDED_DIR_NAMES) is True


def test_is_excluded_false_for_plain_path(tmp_path):
    path = tmp_path / "Documents" / "notes.txt"
    assert is_excluded(path, tmp_path, DEFAULT_EXCLUDED_DIR_NAMES) is False


def test_is_excluded_ignores_hidden_filename_itself(tmp_path):
    # a dotfile directly under root is not excluded by a *directory* check
    path = tmp_path / ".hidden_file.txt"
    assert is_excluded(path, tmp_path, DEFAULT_EXCLUDED_DIR_NAMES) is False


# --- index_file ------------------------------------------------------------

def test_index_file_unsupported_extension_returns_zero(tmp_path, embedder, vector_store):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"\x00\x01")
    count = index_file(path, embedder, vector_store)
    assert count == 0
    assert vector_store.text_table.count_rows() == 0


def test_index_file_txt_writes_chunks_to_vector_store(tmp_path, embedder, vector_store):
    path = tmp_path / "notes.txt"
    path.write_text("some notes about the hackathon project", encoding="utf-8")

    count = index_file(path, embedder, vector_store)

    assert count >= 1
    assert vector_store.text_table.count_rows() == count
    rows = vector_store.text_table.to_arrow().to_pylist()
    assert rows[0]["location"] == str(path)
    assert rows[0]["file_type"] == "txt"
    assert rows[0]["metadata"] is not None


def test_index_file_empty_file_writes_nothing(tmp_path, embedder, vector_store):
    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")

    count = index_file(path, embedder, vector_store)

    assert count == 0
    assert vector_store.text_table.count_rows() == 0


def test_index_file_md_stores_raw_chunk_text(tmp_path, embedder, vector_store):
    path = tmp_path / "readme.md"
    path.write_text("# Title\n\nSome **bold** content here.", encoding="utf-8")

    count = index_file(path, embedder, vector_store)

    assert count >= 1
    rows = vector_store.text_table.to_arrow().to_pylist()
    stored_chunk = rows[0]["chunk"]
    assert stored_chunk.startswith("#")  # raw markdown syntax preserved in storage


def test_index_file_reindexing_replaces_rather_than_duplicates(tmp_path, embedder, vector_store):
    path = tmp_path / "notes.txt"
    path.write_text("original content", encoding="utf-8")
    index_file(path, embedder, vector_store)

    path.write_text("completely different content now", encoding="utf-8")
    index_file(path, embedder, vector_store)

    rows = vector_store.text_table.to_arrow().to_pylist()
    assert vector_store.text_table.count_rows() == len(rows)  # sanity
    all_chunk_text = " ".join(r["chunk"] for r in rows)
    assert "original content" not in all_chunk_text
    assert "completely different content now" in all_chunk_text


def test_index_file_reindexing_same_content_does_not_duplicate(tmp_path, embedder, vector_store):
    path = tmp_path / "notes.txt"
    path.write_text("stable content", encoding="utf-8")

    first_count = index_file(path, embedder, vector_store)
    index_file(path, embedder, vector_store)  # no content_hashes passed: still re-indexes...

    # ...but delete-before-insert means row count matches one indexing pass, not two.
    assert vector_store.text_table.count_rows() == first_count


def test_index_file_skips_unchanged_content_when_content_hashes_provided(tmp_path, embedder, vector_store):
    path = tmp_path / "notes.txt"
    path.write_text("stable content", encoding="utf-8")
    content_hashes = {}

    first_count = index_file(path, embedder, vector_store, content_hashes=content_hashes)
    second_count = index_file(path, embedder, vector_store, content_hashes=content_hashes)

    assert first_count > 0
    assert second_count == 0  # skipped entirely, no re-embedding
    assert vector_store.text_table.count_rows() == first_count


def test_index_file_content_hashes_still_reindexes_on_real_change(tmp_path, embedder, vector_store):
    path = tmp_path / "notes.txt"
    path.write_text("version one", encoding="utf-8")
    content_hashes = {}

    index_file(path, embedder, vector_store, content_hashes=content_hashes)
    path.write_text("version two, quite different", encoding="utf-8")
    second_count = index_file(path, embedder, vector_store, content_hashes=content_hashes)

    assert second_count > 0
    rows = vector_store.text_table.to_arrow().to_pylist()
    all_chunk_text = " ".join(r["chunk"] for r in rows)
    assert "version one" not in all_chunk_text
    assert "version two" in all_chunk_text


# --- Watcher (live watchdog Observer) --------------------------------------

def _wait_until(predicate, timeout=5.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_watcher_indexes_new_supported_file(tmp_path, embedder, vector_store):
    watch_root = tmp_path / "watched"
    watch_root.mkdir()
    watcher = Watcher(watch_root, embedder, vector_store)
    watcher.start()
    try:
        (watch_root / "new_file.txt").write_text("content to be indexed", encoding="utf-8")
        indexed = _wait_until(lambda: vector_store.text_table.count_rows() > 0)
        assert indexed
    finally:
        watcher.stop()


def test_watcher_does_not_duplicate_rows_when_content_is_unchanged(tmp_path, embedder, vector_store):
    # Regression test: watchdog fires both on_created and on_modified for a
    # single save, and re-saving identical content shouldn't grow the index.
    watch_root = tmp_path / "watched"
    watch_root.mkdir()
    watcher = Watcher(watch_root, embedder, vector_store)
    watcher.start()
    try:
        target = watch_root / "notes.txt"
        target.write_text("same content every time", encoding="utf-8")
        assert _wait_until(lambda: vector_store.text_table.count_rows() > 0)

        # Give any duplicate on_created/on_modified events time to land,
        # then re-save the exact same content and confirm no growth.
        time.sleep(0.5)
        count_after_first_save = vector_store.text_table.count_rows()
        target.write_text("same content every time", encoding="utf-8")
        time.sleep(1.0)

        assert vector_store.text_table.count_rows() == count_after_first_save
    finally:
        watcher.stop()


def test_watcher_skips_excluded_directory(tmp_path, embedder, vector_store):
    watch_root = tmp_path / "watched"
    excluded_dir = watch_root / "node_modules"
    excluded_dir.mkdir(parents=True)
    watcher = Watcher(watch_root, embedder, vector_store)
    watcher.start()
    try:
        (excluded_dir / "ignored.txt").write_text("should not be indexed", encoding="utf-8")
        # give the observer a moment to (not) process the event, then confirm nothing landed
        time.sleep(1.0)
        assert vector_store.text_table.count_rows() == 0
    finally:
        watcher.stop()
