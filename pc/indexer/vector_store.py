"""Phase 1 — LanceDB wrapper: the active `text` table and a `images` table
schema placeholder (no embedding pipeline wired up until Phase 4).

Input: chunk records (from chunker.py output + extractor.py structural
context + embedder.py vectors) or a query embedding.
Output: inserted rows (upsert_chunks) or ranked nearest-neighbor rows (search).
Side effects: reads/writes the LanceDB database at the configured path.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import lancedb
import pyarrow as pa

TEXT_TABLE_NAME = "text"
IMAGES_TABLE_NAME = "images"

_OPTIONAL_STRUCTURAL_FIELDS = ("page", "sheet", "slide", "section")


def _text_schema(embedding_dim):
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("location", pa.string()),
        pa.field("title", pa.string()),
        pa.field("chunk", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
        pa.field("file_type", pa.string()),
        pa.field("page", pa.string()),
        pa.field("sheet", pa.string()),
        pa.field("slide", pa.string()),
        pa.field("section", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("chunk_index", pa.int64()),
        pa.field("created_at", pa.string()),
        pa.field("updated_at", pa.string()),
        pa.field("metadata", pa.string()),  # JSON-encoded free-form dict
    ])


def _images_schema(embedding_dim):
    # TODO: Phase 4 — image indexing with jina-clip. Schema placeholder only;
    # no extraction/embedding pipeline writes to this table yet.
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("location", pa.string()),
        pa.field("title", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
        pa.field("file_type", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("updated_at", pa.string()),
        pa.field("metadata", pa.string()),
    ])


class VectorStore:
    """Owns the `text` table (active, per CLAUDE.md's schema) and the
    `images` table (Phase 4 placeholder, schema only)."""

    def __init__(self, db_path, embedding_dim=768, image_embedding_dim=768):
        # read_consistency_interval=0 makes every read check out the latest
        # version before querying. Without it, a connection defaults to
        # never auto-refreshing, so a long-lived reader (the API server)
        # would stay pinned to whatever version existed at connect time and
        # never see rows written later by another process (the watcher) —
        # exactly the two-process split this project's architecture requires.
        self.db = lancedb.connect(str(db_path), read_consistency_interval=timedelta(seconds=0))
        self.embedding_dim = embedding_dim
        self.text_table = self._open_or_create(TEXT_TABLE_NAME, _text_schema(embedding_dim))
        # TODO: Phase 4 — image indexing with jina-clip. Created up front so
        # the schema exists, but nothing indexes into it yet.
        self.images_table = self._open_or_create(IMAGES_TABLE_NAME, _images_schema(image_embedding_dim))

    def _open_or_create(self, name, schema):
        if name in self.db.table_names():
            return self.db.open_table(name)
        return self.db.create_table(name, schema=schema)

    def upsert_chunks(self, records):
        """Insert chunk records into the text table.

        Args:
            records: dicts matching the text schema's logical fields. `id`
                and `chunk_id` default to a fresh uuid4 if omitted,
                `created_at`/`updated_at` default to the current UTC ISO 8601
                timestamp if omitted, `page`/`sheet`/`slide`/`section`
                default to None, and `metadata` (a plain dict) is JSON-encoded.

        Returns:
            The rows actually written (post-defaulting), in insertion order.
            Side effects: writes to LanceDB.
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for record in records:
            row = dict(record)
            row.setdefault("id", str(uuid.uuid4()))
            row.setdefault("chunk_id", str(uuid.uuid4()))
            row.setdefault("created_at", now)
            row.setdefault("updated_at", now)
            row["metadata"] = json.dumps(row.get("metadata") or {})
            for field in _OPTIONAL_STRUCTURAL_FIELDS:
                row.setdefault(field, None)
            rows.append(row)
        if rows:
            self.text_table.add(rows)
        return rows

    def delete_by_location(self, location):
        """Delete all text-table rows for a given location (file path or URL).

        Used to clear a file's previous chunks before re-indexing it, so
        edits don't leave stale rows behind and re-saving unchanged content
        doesn't duplicate rows. A no-op if no rows match.

        Side effects: writes to LanceDB (row deletion).
        """
        escaped = location.replace("'", "''")
        self.text_table.delete(f"location = '{escaped}'")

    def search(self, query_embedding, top_k=5):
        """Return the top_k nearest text-table rows to query_embedding, nearest first.

        `metadata` is JSON-decoded back into a dict on each returned row.
        Side effects: none (read-only LanceDB query).
        """
        results = self.text_table.search(query_embedding).limit(top_k).to_list()
        for result in results:
            raw_metadata = result.get("metadata")
            result["metadata"] = json.loads(raw_metadata) if raw_metadata else {}
        return results
