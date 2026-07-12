"""Phase 1 — LanceDB wrapper: the active `text` table and a `images` table
schema placeholder (no embedding pipeline wired up until Phase 4).

Input: chunk records (from chunker.py output + extractor.py structural
context + embedder.py vectors) or a query embedding.
Output: inserted rows (upsert_chunks) or ranked nearest-neighbor rows (search).
Side effects: reads/writes the LanceDB database at the configured path.
"""

import json
import os
import uuid
from datetime import datetime, timezone

# Global flag to indicate if we are using the SQLite fallback
USING_SQLITE_FALLBACK = False

try:
    import lancedb
    import pyarrow as pa
except ImportError:
    USING_SQLITE_FALLBACK = True

if USING_SQLITE_FALLBACK:
    # Define Mock PyArrow
    class MockPyArrowField:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_

    class MockPyArrowSchema:
        def __init__(self, fields):
            self.fields = fields
            self.names = [f.name for f in fields]
        def __iter__(self):
            return iter(self.fields)

    class MockPyArrow:
        @staticmethod
        def field(name, type_, *args, **kwargs):
            return MockPyArrowField(name, type_)

        @staticmethod
        def schema(fields):
            return MockPyArrowSchema(fields)

        @staticmethod
        def string():
            return "string"

        @staticmethod
        def int64():
            return "int64"

        @staticmethod
        def float32():
            return "float32"

        @staticmethod
        def list_(type_, dim=None):
            return f"list<{type_}>"

    pa = MockPyArrow

    # Define SQLite-backed LanceDB Connection/Table Fallbacks
    import sqlite3

    class SQLiteArrow:
        def __init__(self, rows):
            self.rows = rows
        def to_pylist(self):
            return self.rows

    class SQLiteSearchQuery:
        def __init__(self, rows, query_embedding):
            self.rows = rows
            self.query_embedding = query_embedding
            self._limit = None

        def limit(self, k):
            self._limit = k
            return self

        def to_list(self):
            results_with_dist = []
            for r in self.rows:
                if r.get("embedding") is not None:
                    dist = sum((x - y) ** 2 for x, y in zip(r["embedding"], self.query_embedding))
                    results_with_dist.append((dist, r))
            results_with_dist.sort(key=lambda item: item[0])
            sorted_rows = [item[1] for item in results_with_dist]
            if self._limit is not None:
                return sorted_rows[:self._limit]
            return sorted_rows

    class SQLiteLanceDBTable:
        def __init__(self, conn, name, schema=None):
            self.conn = conn
            self.name = name
            if schema is not None:
                self.names = schema.names
            else:
                self.names = []
                cursor = self.conn.cursor()
                cursor.execute(f"PRAGMA table_info({self.name})")
                columns = cursor.fetchall()
                for col in columns:
                    self.names.append(col[1])

            class Schema:
                def __init__(self, names):
                    self.schema_names = names
                @property
                def names(self):
                    return self.schema_names

            self.schema = Schema(self.names)

        def count_rows(self):
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.name}")
            return cursor.fetchone()[0]

        def add(self, rows):
            cursor = self.conn.cursor()
            for row in rows:
                row_dict = dict(row)
                columns = []
                values = []
                for col in self.names:
                    if col in row_dict:
                        val = row_dict[col]
                        if isinstance(val, (list, dict)):
                            val = json.dumps(val)
                        columns.append(col)
                        values.append(val)
                placeholders = ",".join(["?"] * len(values))
                sql = f"INSERT INTO {self.name} ({','.join(columns)}) VALUES ({placeholders})"
                cursor.execute(sql, values)
            self.conn.commit()

        def _get_all_rows(self):
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT * FROM {self.name}")
            columns = [col[0] for col in cursor.description]
            db_rows = cursor.fetchall()
            rows = []
            for db_row in db_rows:
                r = dict(zip(columns, db_row))
                if "embedding" in r and r["embedding"] is not None:
                    try:
                        r["embedding"] = json.loads(r["embedding"])
                    except Exception:
                        pass
                rows.append(r)
            return rows

        def search(self, query_embedding):
            return SQLiteSearchQuery(self._get_all_rows(), query_embedding)

        def to_arrow(self):
            return SQLiteArrow(self._get_all_rows())

        def delete(self, where):
            cursor = self.conn.cursor()
            cursor.execute(f"DELETE FROM {self.name} WHERE {where}")
            self.conn.commit()


    class SQLiteLanceDBConnection:
        def __init__(self, db_path):
            self.db_path = db_path
            os.makedirs(self.db_path, exist_ok=True)
            sqlite_file = os.path.join(self.db_path, "lancedb.sqlite")
            self.conn = sqlite3.connect(sqlite_file, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=5000")

        def table_names(self):
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            return [row[0] for row in cursor.fetchall()]

        def open_table(self, name):
            if name not in self.table_names():
                raise ValueError(f"Table {name} not found")
            return SQLiteLanceDBTable(self.conn, name)

        def create_table(self, name, schema=None, **kwargs):
            cursor = self.conn.cursor()
            columns_sql = []
            for field in schema:
                field_name = field.name
                field_type = "TEXT"
                if field_name == "chunk_index":
                    field_type = "INTEGER"
                columns_sql.append(f"{field_name} {field_type}")
            sql = f"CREATE TABLE IF NOT EXISTS {name} ({','.join(columns_sql)})"
            cursor.execute(sql)
            self.conn.commit()
            return SQLiteLanceDBTable(self.conn, name, schema)


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
        if USING_SQLITE_FALLBACK:
            self.db = SQLiteLanceDBConnection(str(db_path))
        else:
            self.db = lancedb.connect(str(db_path))
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
