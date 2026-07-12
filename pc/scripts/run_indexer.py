"""Phase 1 — CLI entry point: runs the background filesystem watcher as a
long-lived process, wiring Embedder + VectorStore + Watcher from env vars
(mirrors pc/api/main.py's lifespan configuration, so the indexer and the
API always agree on where the model/database live).

Environment variables:
    EMBEDDING_MODEL_PATH — path to the embedding ONNX model (on the real
        Snapdragon PC, the QNN-quantized EmbeddingGemma 300M artifact).
        Default: models/embedding_gemma_300m_qnn_int8.onnx
    LANCEDB_PATH — LanceDB directory. Default: lancedb
    EMBEDDING_DIM — embedding vector width, must match the model. Default: 768
    WATCH_ROOT — directory to watch recursively. Default: the current
        user's home directory.
    LOG_FILE — rotating log file path. Default: lore.log

Usage:
    python pc/scripts/run_indexer.py
Side effects: loads the embedding model, opens/creates the LanceDB
tables, and starts a watchdog Observer thread that runs until interrupted
(Ctrl+C).
"""

import os
import sys
import time
from pathlib import Path

from pc.api.logging_config import configure_logging
from pc.indexer.embedder import Embedder
from pc.indexer.vector_store import VectorStore
from pc.indexer.watcher import Watcher

DEFAULT_EMBEDDING_MODEL_PATH = "models/embedding_gemma_300m_qnn_int8.onnx"
DEFAULT_LANCEDB_PATH = "lancedb"
DEFAULT_LOG_FILE = "lore.log"
DEFAULT_EMBEDDING_DIM = 768


def build_watcher():
    """Wire Embedder + VectorStore + Watcher from env vars.

    Returns:
        An unstarted Watcher (call .start()/.stop() to run it).
    Side effects: loads the embedding model file (onnxruntime.InferenceSession);
        opens/creates the LanceDB tables.
    """
    model_path = os.environ.get("EMBEDDING_MODEL_PATH", DEFAULT_EMBEDDING_MODEL_PATH)
    lancedb_path = os.environ.get("LANCEDB_PATH", DEFAULT_LANCEDB_PATH)
    embedding_dim = int(os.environ.get("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM))
    watch_root = os.environ.get("WATCH_ROOT", os.path.expanduser("~"))
    log_file = os.environ.get("LOG_FILE", DEFAULT_LOG_FILE)

    embedder = Embedder(model_path)
    vector_store = VectorStore(lancedb_path, embedding_dim=embedding_dim)

    # If this project's own log file/LanceDB directory happen to live inside
    # watch_root (e.g. the repo sits under the user's home folder), writing
    # to them would otherwise re-trigger the watcher on every write — an
    # infinite self-referential loop. Exclude them explicitly.
    excluded_paths = [Path(lancedb_path).resolve(), Path(log_file).resolve().parent]
    return Watcher(watch_root, embedder, vector_store, excluded_paths=excluded_paths)


def main():
    configure_logging(log_file=os.environ.get("LOG_FILE", DEFAULT_LOG_FILE))
    watcher = build_watcher()
    watcher.start()
    print(f"Watching {watcher.root} for changes (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
