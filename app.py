import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse, SourceChunk, IngestResponse
from rag import RAGPipeline
from settings import settings

app = FastAPI(title="LongevAI RAG Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    chunks, docs = pipeline.ingest_paths([data_dir])
    if chunks == 0:
        raise HTTPException(status_code=400, detail="No ingestible documents found under backend/data")
    return IngestResponse(ok=True, chunks_indexed=chunks, docs=docs)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if pipeline.vs.index is None or len(pipeline.vs.id2meta) == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest documents first.")
    answer, citations, scored = pipeline.chat(req.query, k=req.k)
    sources = []
    score_map = {i: s for i, s in scored}
    for c in citations:
        sources.append(SourceChunk(
            id=c["id"],
            doc_id=c["doc_id"],
            source=c["source"],
            text=c["text"],
            score=float(score_map.get(c["id"], 0.0))
        ))
    return ChatResponse(answer=answer, sources=sources)
