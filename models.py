from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    k: int = Field(5, ge=1, le=10, description="How many chunks to use after reranking")


class SourceChunk(BaseModel):
    id: int
    doc_id: str
    source: str
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class IngestResponse(BaseModel):
    ok: bool
    chunks_indexed: int
    docs: int
    note: Optional[str] = None


class DebugIndex(BaseModel):
    ok: bool
    index_dir: str
    faiss_loaded: bool
    chunks: int
    index_exists: bool
    meta_exists: bool
    index_size: int
    meta_size: int
