import os
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse, SourceChunk, IngestResponse
from rag import RAGPipeline
from settings import settings

app = FastAPI(title="LongevAI RAG Backend", version="0.1.0")

# ---- CORS (tighten to your frontend later) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # set to your frontend URL when live
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- auth helper (protects /chat and /ingest) ----
def require_api_key(x_api_key: str = Header(default="")):
    expected = getattr(settings, "BACKEND_API_KEY", "") or os.getenv("BACKEND_API_KEY", "")
    if not expected:
        # no key configured -> allow (dev), but warn in logs
        print("[warn] BACKEND_API_KEY not set; /chat and /ingest are open")
        return True
    if x_api_key == expected:
        return True
    raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# ---- pipeline (created once) ----
pipeline = RAGPipeline()

# ---- startup: load index or ingest from /data ----
@app.on_event("startup")
def _startup():
    # absolute paths so Render can find them
    index_dir = pipeline.index_dir
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    index_path = os.path.join(index_dir, "index.faiss")
    meta_path  = os.path.join(index_dir, "meta.jsonl")

    has_index = os.path.exists(index_path) and os.path.exists(meta_path)

    if has_index:
        # make sure in-memory structures are populated
        ok = pipeline.vs.load()
        print(f"[startup] existing index present -> load() = {ok}; "
              f"chunks={len(pipeline.vs.id2meta)} dir={index_dir}")
    else:
        # first boot (or non-persistent storage) → ingest now
        print(f"[startup] no index in {index_dir}; ingesting from {data_dir} ...")
        os.makedirs(index_dir, exist_ok=True)
        chunks, docs = pipeline.ingest_paths([data_dir])
        print(f"[startup] ingest complete: docs={docs}, chunks={chunks}")

# ---- health ----
@app.get("/health")
def health():
    index_dir = pipeline.index_dir
    ip = os.path.join(index_dir, "index.faiss")
    mp = os.path.join(index_dir, "meta.jsonl")
    return {
        "ok": True,
        "index_dir": index_dir,
        "faiss_loaded": bool(pipeline.vs.index),
        "chunks": len(pipeline.vs.id2meta) if pipeline.vs.id2meta else 0,
        "index_exists": os.path.exists(ip),
        "meta_exists": os.path.exists(mp),
        "index_size": os.path.getsize(ip) if os.path.exists(ip) else 0,
        "meta_size": os.path.getsize(mp) if os.path.exists(mp) else 0,
    }

# ---- ingest (manual) ----
@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    chunks, docs = pipeline.ingest_paths([data_dir])
    if chunks == 0:
        raise HTTPException(status_code=400, detail="No ingestible documents found under backend/data")
    return IngestResponse(ok=True, chunks_indexed=chunks, docs=docs)

# ---- chat ----
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    if pipeline.vs.index is None or len(pipeline.vs.id2meta) == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest documents first.")
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
