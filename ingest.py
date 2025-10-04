import os
from rag import RAGPipeline

if __name__ == "__main__":
    p = RAGPipeline()
    # data/ is alongside ingest.py at the repo root
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    data_dir = os.path.abspath(data_dir)
    print("[ingest] data_dir:", data_dir)
    if not os.path.isdir(data_dir):
        print("[ingest] ERROR: data directory not found")
    else:
        print("[ingest] files:", os.listdir(data_dir))
    chunks, docs = p.ingest_paths([data_dir])
    print(f"Ingested {docs} docs -> {chunks} chunks from {data_dir}")
