import os
import time

import pytest

from pc.indexer.watcher import Watcher
from pc.scripts.build_smoke_model import build_smoke_model
from pc.scripts.run_indexer import build_watcher

EMBEDDING_DIM = 8


@pytest.fixture
def env(tmp_path, monkeypatch):
    model_path = build_smoke_model(tmp_path / "smoke.onnx", input_dim=16, embedding_dim=EMBEDDING_DIM)
    watch_root = tmp_path / "watched"
    watch_root.mkdir()

    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(model_path))
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "lancedb"))
    monkeypatch.setenv("EMBEDDING_DIM", str(EMBEDDING_DIM))
    monkeypatch.setenv("WATCH_ROOT", str(watch_root))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "lore.log"))

    return {"watch_root": watch_root}


def test_build_watcher_wires_a_real_watcher_from_env_vars(env):
    watcher = build_watcher()
    assert isinstance(watcher, Watcher)
    assert str(watcher.root) == str(env["watch_root"])


def test_build_watcher_indexes_files_end_to_end(env):
    watcher = build_watcher()
    watcher.start()
    try:
        (env["watch_root"] / "notes.txt").write_text("hello from the indexer", encoding="utf-8")
        deadline = time.time() + 5
        while time.time() < deadline and watcher.handler.vector_store.text_table.count_rows() == 0:
            time.sleep(0.1)
        assert watcher.handler.vector_store.text_table.count_rows() > 0
    finally:
        watcher.stop()


def test_build_watcher_defaults_watch_root_to_home_dir(env, monkeypatch):
    monkeypatch.delenv("WATCH_ROOT", raising=False)
    watcher = build_watcher()
    assert str(watcher.root) == os.path.expanduser("~")
