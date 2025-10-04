from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    query: str
    k: int = 5

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
