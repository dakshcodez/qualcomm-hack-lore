"""Phase 2 — FastAPI dependency providers for the shared Embedder and
VectorStore instances, built once at app startup (main.py's lifespan) and
stashed on app.state.

Route handlers take these via `Depends(get_embedder)` / `Depends(get_vector_store)`
instead of importing global instances directly, so tests can swap in
synthetic-model-backed instances via app.dependency_overrides without
touching app startup.
"""

from fastapi import Request

from pc.indexer.embedder import Embedder
from pc.indexer.vector_store import VectorStore


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store
