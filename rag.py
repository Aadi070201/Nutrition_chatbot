import os
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse, SourceChunk, IngestResponse
from rag import RAGPipeline
from settings import settings

app = FastAPI(title="LongevAI RAG Backend", version="0.1.0")

# CORS (tighten to your frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend URL when ready
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple API-key guard (optional but recommended)
def require_api_key(x_api_key: str = Header(default="")):
    expected = settings.BACKEND_API_KEY
    if not expected:
        # no key set -> allow (dev)
        return True
    if x_api_key == expected:
        return True
    raise HTTPException(status_code=401, detail="Invalid or missing API key.")

pipeline = RAGPipeline()

@app.on_event("startup")
def _startup():
    index_dir = pipeline.index_dir
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    index_path = os.path.join(index_dir, "index.faiss")
    meta_path  = os.path.join(index_dir, "meta.jsonl")

    has_index = os.path.exists(index_path) and os.path.exists(meta_path)
    if has_index:
        ok = pipeline.vs.load()
        print(f"[startup] loaded index: ok={ok} chunks={len(pipeline.vs.id2meta)} dir={index_dir}")
    else:
        print(f"[startup] no index found -> ingesting from {data_dir} ...")
        os.makedirs(index_dir, exist_ok=True)
        chunks, docs = pipeline.ingest_paths([data_dir])
        print(f"[startup] ingest complete: docs={docs}, chunks={chunks}")

@app.get("/health")
def health():
    return {"ok": True, **pipeline.index_stats()}

@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    chunks, docs = pipeline.ingest_paths([data_dir])
    if chunks == 0:
        raise HTTPException(status_code=400, detail="No ingestible documents found under backend/data")
    return IngestResponse(ok=True, chunks_indexed=chunks, docs=docs)

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    if pipeline.vs.index is None or len(pipeline.vs.id2meta) == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest documents first.")
    try:
        answer, citations, scored = pipeline.chat(req.query, k=req.k)
    except Exception as e:
        # surface the error instead of 502
        raise HTTPException(status_code=500, detail=f"chat_failed: {type(e).__name__}: {e}")
    score_map = {i: s for i, s in scored}
    sources = [
        SourceChunk(
            id=c["id"],
            doc_id=c["doc_id"],
            source=c["source"],
            text=c["text"],
            score=float(score_map.get(c["id"], 0.0)),
        )
        for c in citations
    ]
    return ChatResponse(answer=answer, sources=sources)
