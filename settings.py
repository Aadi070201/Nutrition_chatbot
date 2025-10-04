import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Embeddings
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
    LOCAL_EMBEDDING_MODEL = os.getenv(
        "LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",  # small & fast
    )

    # Generation (Groq)
    GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "groq")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Reranker (disable on free plan)
    RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "none")  # "none" or "local"
    LOCAL_RERANK_MODEL = os.getenv("LOCAL_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Index/Chunking
    INDEX_DIR = os.getenv("INDEX_DIR", "store")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Optional: simple header auth for /chat and /ingest
    BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")

settings = Settings()
