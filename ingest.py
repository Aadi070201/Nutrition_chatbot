import os
from rag import RAGPipeline

if __name__ == "__main__":
    p = RAGPipeline()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data_dir = os.path.abspath(data_dir)
    chunks, docs = p.ingest_paths([data_dir])
    print(f"Ingested {docs} docs -> {chunks} chunks from {data_dir}")
