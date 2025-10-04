import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse, SourceChunk, IngestResponse, DebugIndex
from rag import RAGPipeline
from settings import settings

app = FastAPI(title="LongevAI RAG Backend", version="0.1.0")

# Wide-open CORS for quick testing (tighten later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- create pipeline once ----
pipeline = RAGPipeline()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug/index", response_model=DebugIndex)
def debug_index():
    idx_dir = pipeline.index_dir
    index_path = os.path.join(idx_dir, "index.faiss")
    meta_path = os.path.join(idx_dir, "meta.jsonl")
    index_exists = os.path.exists(index_path)
    meta_exists = os.path.exists(meta_path)
    index_size = os.path.getsize(index_path) if index_exists else 0
    meta_size = os.path.getsize(meta_path) if meta_exists else 0
    chunks = len(pipeline.vs.id2meta) if pipeline.vs and pipeline.vs.id2meta else 0
    return DebugIndex(
        ok=True,
        index_dir=idx_dir,
        faiss_loaded=pipeline.vs.index is not None,
        chunks=chunks,
        index_exists=index_exists,
        meta_exists=meta_exists,
        index_size=index_size,
        meta_size=meta_size,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    (Optional) Ingest the repo's /data folder on demand.
    On Render we already ingest during the build step via `python ingest.py`.
    Calling this again will add new files, if any.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    chunks, docs = pipeline.ingest_paths([data_dir])
    if chunks == 0 and docs == 0:
        # no new chunks; but it's not an error — just report it clearly
        return IngestResponse(ok=True, chunks_indexed=0, docs=0, note="No new ingestible documents found.")
    return IngestResponse(ok=True, chunks_indexed=chunks, docs=docs)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if pipeline.vs.index is None or len(pipeline.vs.id2meta) == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Please ingest documents first.")
    answer, citations, scored = pipeline.chat(req.query, k=req.k)
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
