import json

import pytest

from pc.indexer.vector_store import IMAGES_TABLE_NAME, TEXT_TABLE_NAME, VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "lancedb", embedding_dim=4)


def _record(**overrides):
    record = {
        "location": "/tmp/a.txt",
        "title": "a",
        "chunk": "hello world",
        "embedding": [1.0, 0.0, 0.0, 0.0],
        "file_type": "txt",
        "chunk_index": 0,
        "metadata": {"source": "filesystem"},
    }
    record.update(overrides)
    return record


def test_creates_text_and_images_tables(store):
    table_names = set(store.db.table_names())
    assert TEXT_TABLE_NAME in table_names
    assert IMAGES_TABLE_NAME in table_names


def test_images_table_has_no_pipeline_and_starts_empty(store):
    assert store.images_table.count_rows() == 0
    assert store.images_table.schema.names == [
        "id", "location", "title", "embedding", "file_type", "created_at", "updated_at", "metadata",
    ]


def test_upsert_chunks_fills_defaults(store):
    rows = store.upsert_chunks([_record()])
    assert len(rows) == 1
    row = rows[0]
    assert row["page"] is None and row["sheet"] is None
    assert row["slide"] is None and row["section"] is None
    assert "id" in row and "chunk_id" in row
    assert "created_at" in row and "updated_at" in row
    # metadata is JSON-encoded on write
    assert json.loads(row["metadata"]) == {"source": "filesystem"}
    assert store.text_table.count_rows() == 1


def test_upsert_chunks_respects_explicit_ids_and_structural_fields(store):
    rows = store.upsert_chunks([_record(id="fixed-id", chunk_id="fixed-chunk-id", page="3")])
    assert rows[0]["id"] == "fixed-id"
    assert rows[0]["chunk_id"] == "fixed-chunk-id"
    assert rows[0]["page"] == "3"


def test_search_returns_nearest_first_with_decoded_metadata(store):
    store.upsert_chunks([
        _record(title="close", embedding=[1.0, 0.0, 0.0, 0.0]),
        _record(title="far", embedding=[0.0, 1.0, 0.0, 0.0]),
    ])

    results = store.search([0.9, 0.1, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0]["title"] == "close"
    assert results[1]["title"] == "far"
    assert results[0]["metadata"] == {"source": "filesystem"}


def test_search_respects_top_k(store):
    store.upsert_chunks([_record(title=f"chunk-{i}", embedding=[float(i), 0.0, 0.0, 0.0]) for i in range(5)])

    results = store.search([0.0, 0.0, 0.0, 0.0], top_k=3)

    assert len(results) == 3


def test_upsert_empty_list_is_a_noop(store):
    assert store.upsert_chunks([]) == []
    assert store.text_table.count_rows() == 0


def test_reopening_existing_db_reuses_tables(tmp_path):
    db_path = tmp_path / "lancedb"
    store1 = VectorStore(db_path, embedding_dim=4)
    store1.upsert_chunks([_record()])

    store2 = VectorStore(db_path, embedding_dim=4)
    assert store2.text_table.count_rows() == 1


def test_delete_by_location_removes_only_matching_rows(store):
    store.upsert_chunks([
        _record(location="/tmp/a.txt", title="a1"),
        _record(location="/tmp/a.txt", title="a2"),
        _record(location="/tmp/b.txt", title="b1"),
    ])

    store.delete_by_location("/tmp/a.txt")

    rows = store.text_table.to_arrow().to_pylist()
    assert store.text_table.count_rows() == 1
    assert [r["title"] for r in rows] == ["b1"]


def test_delete_by_location_is_a_noop_when_nothing_matches(store):
    store.upsert_chunks([_record(location="/tmp/a.txt")])
    store.delete_by_location("/tmp/does-not-exist.txt")
    assert store.text_table.count_rows() == 1


def test_delete_by_location_handles_single_quotes_safely(store):
    store.upsert_chunks([_record(location="/tmp/it's a file.txt")])
    store.delete_by_location("/tmp/it's a file.txt")
    assert store.text_table.count_rows() == 0
