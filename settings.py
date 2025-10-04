import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Embeddings
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
    LOCAL_EMBEDDING_MODEL = os.getenv(
        "LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Generation (Groq)
    GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "groq")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Reranker
    RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "local")   # "local" or "none"
    LOCAL_RERANK_MODEL = os.getenv(
        "LOCAL_RERANK_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Index/Chunking (make index path absolute so Render can’t miss /store)
    BASE_DIR = os.path.dirname(__file__)
    INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(BASE_DIR, "store"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

settings = Settings()

