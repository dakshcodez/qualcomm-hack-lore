"""Phase 2 — Pydantic request/response models for the FastAPI backend.

Field names and shapes match CLAUDE.md's documented POST /query and
POST /index request/response bodies exactly.
"""

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    text: str
    modality: Literal["text", "image"] = "text"


class SourceItem(BaseModel):
    title: str
    location: str
    excerpt: str
    file_type: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)


class IndexRequest(BaseModel):
    text: str
    url: str
    title: str


class IndexResponse(BaseModel):
    status: Literal["ok", "error"] = "ok"
    chunks_indexed: int = 0
