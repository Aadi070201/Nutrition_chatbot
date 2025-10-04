import os, json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import faiss
import tiktoken
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from groq import APIConnectionError, APIStatusError

from settings import settings

# ---------- config ----------
EXCLUDE_FILENAMES = {"index.md", "glossary.md", "sources.md"}
DEFAULT_TOPN = 20
DEFAULT_K = 6
MMR_LAMBDA = 0.65
TIKTOKEN_ENCODING = "cl100k_base"

SMALLTALK = {
    "hi": "Hey! I’m Aadi. Ask me anything about nutrition—foods, macros, vitamins, diets, hydration.",
    "hello": "Hello! I’m Aadi. What nutrition question can I help with today?",
    "hey": "Hey there! Fire away with your nutrition question.",
    "help": "You can ask about macronutrients, vitamins/minerals, diets like DASH or Mediterranean, hydration, or meal ideas."
}

# ---------- helpers ----------
def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    return vectors / norms

def read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        reader = PdfReader(path)
        texts = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(texts)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int, overlap: int, encoding_name: str = TIKTOKEN_ENCODING) -> List[str]:
    enc = tiktoken.get_encoding(encoding_name)
    toks = enc.encode(text)
    chunks, start = [], 0
    while start < len(toks):
        end = start + chunk_size
        chunks.append(enc.decode(toks[start:end]))
        if end >= len(toks):
            break
        start = max(0, end - overlap)
    return chunks

# ---------- data ----------
@dataclass
class Meta:
    doc_id: str     # filename only
    source: str     # full path
    text: str       # chunk text

# ---------- vector store ----------
class VectorStore:
    def __init__(self, dim: int, index_dir: str):
        self.dim = dim
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "meta.jsonl")
        self.id2meta: List[Meta] = []
        self.index = None

    def load(self) -> bool:
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            return False
        self.index = faiss.read_index(self.index_path)
        metas = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                metas.append(Meta(**json.loads(line)))
        self.id2meta = metas
        return (self.index is not None) and (len(self.id2meta) > 0)

    def save(self):
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.id2meta:
                f.write(json.dumps(m.__dict__) + "\n")

    def create(self, vectors: np.ndarray, metas: List[Meta]):
        self.index = faiss.IndexFlatIP(self.dim)   # cosine if vectors normalized
        self.index.add(vectors.astype("float32"))
        self.id2meta = metas

    def add(self, vectors: np.ndarray, metas: List[Meta]):
        if self.index is None:
            self.create(vectors, metas)
        else:
            self.index.add(vectors.astype("float32"))
            self.id2meta.extend(metas)

    def search(self, query_vec: np.ndarray, top_n: int = DEFAULT_TOPN) -> List[Tuple[int, float]]:
        if self.index is None or len(self.id2meta) == 0:
            return []
        sims, ids = self.index.search(query_vec.astype("float32"), top_n)
        out = []
        for i, score in zip(ids[0].tolist(), sims[0].tolist()):
            if i == -1:
                continue
            out.append((i, float(score)))
        return out

# ---------- pipeline ----------
class RAGPipeline:
    def __init__(self):
        # Embeddings
        self.embedder = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL, device="cpu")
        self.dim = self.embedder.get_sentence_embedding_dimension()

        # Reranker (lazy-load to save RAM on free tier)
        self.rerank_provider = settings.RERANK_PROVIDER.lower()
        self._cross_encoder_name = settings.LOCAL_RERANK_MODEL
        self.cross_encoder = None  # loaded when needed if provider == "local"

        # Groq generation
        if settings.GENERATION_PROVIDER.lower() != "groq":
            raise RuntimeError("GENERATION_PROVIDER must be 'groq' for this setup.")
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY missing in environment")
        self.groq = Groq(api_key=settings.GROQ_API_KEY)
        self.generative_model = settings.GROQ_MODEL

        # Index path (absolute so Render can’t miss /store)
        base_dir = os.path.dirname(__file__)
        idx_dir = settings.INDEX_DIR
        self.index_dir = idx_dir if os.path.isabs(idx_dir) else os.path.join(base_dir, idx_dir)

        # Chunking
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

        # Vector store
        self.vs = VectorStore(dim=self.dim, index_dir=self.index_dir)
        if not self.vs.load():
            os.makedirs(self.index_dir, exist_ok=True)

    # ----- embeddings -----
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embs = self.embedder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        return embs

    # ----- ingestion -----
    def ingest_paths(self, paths: List[str]) -> Tuple[int, int]:
        all_chunks: List[str] = []
        metas: List[Meta] = []
        docs = 0

        def allowed(fname: str) -> bool:
            name = os.path.basename(fname).lower()
            if name in EXCLUDE_FILENAMES:
                return False
            return name.endswith((".txt", ".md", ".pdf"))

        for p in paths:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for fn in files:
                        full = os.path.join(root, fn)
                        if not allowed(full):
                            continue
                        text = read_file(full)
                        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
                        if not chunks:
                            continue
                        all_chunks.extend(chunks)
                        metas.extend([Meta(doc_id=os.path.basename(full), source=full, text=c) for c in chunks])
                        docs += 1
            else:
                if not allowed(p):
                    continue
                text = read_file(p)
                chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
                if not chunks:
                    continue
                all_chunks.extend(chunks)
                metas.extend([Meta(doc_id=os.path.basename(p), source=p, text=c) for c in chunks])
                docs += 1

        if not all_chunks:
            return 0, 0

        vectors = self.embed_texts(all_chunks)
        if (self.vs.index is None) or (len(self.vs.id2meta) == 0) or (self.vs.index.d != self.dim):
            self.vs = VectorStore(dim=self.dim, index_dir=self.index_dir)
            self.vs.create(vectors, metas)
        else:
            self.vs.add(vectors, metas)
        self.vs.save()
        return len(all_chunks), docs

    # ----- retrieve -----
    def retrieve(self, query: str, top_n: int = DEFAULT_TOPN) -> List[Tuple[int, float]]:
        if self.vs.index is None or len(self.vs.id2meta) == 0:
            return []
        qvec = self.embed_texts([query])
        raw = self.vs.search(qvec, top_n=top_n)

        # Drop excluded filenames
        filtered = []
        for idx, score in raw:
            fname = os.path.basename(self.vs.id2meta[idx].source).lower()
            if fname in EXCLUDE_FILENAMES:
                continue
            filtered.append((idx, score))
        return filtered

    # ----- simple MMR (if no local cross-encoder) -----
    def _mmr(self, query: str, candidates: List[Tuple[int, float]], k: int) -> List[Tuple[int, float]]:
        if not candidates:
            return []
        docs = [self.vs.id2meta[i].text for i, _ in candidates]
        doc_embs = self.embed_texts(docs)  # normalized
        q_emb = self.embed_texts([query])[0:1]

        rel = (doc_embs @ q_emb.T).ravel()
        selected, selected_idx = [], set()
        while len(selected) < min(k, len(candidates)):
            if not selected:
                j = int(np.argmax(rel))
                selected.append((candidates[j][0], float(rel[j])))
                selected_idx.add(j)
                continue
            selected_embs = doc_embs[list(selected_idx)]
            sim_to_sel = selected_embs @ doc_embs.T
            max_sim = sim_to_sel.max(axis=0)
            mmr_score = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * max_sim
            mmr_score[list(selected_idx)] = -1e9
            j = int(np.argmax(mmr_score))
            selected.append((candidates[j][0], float(rel[j])))
            selected_idx.add(j)
        return selected

    # ----- rerank -----
    def _get_cross_encoder(self):
        if (self.cross_encoder is None) and (self.rerank_provider == "local"):
            self.cross_encoder = CrossEncoder(self._cross_encoder_name, device="cpu")
        return self.cross_encoder

    def rerank(self, query: str, candidates: List[Tuple[int, float]], k: int) -> List[Tuple[int, float]]:
        if not candidates:
            return []
        ce = self._get_cross_encoder()
        if ce is None:
            return self._mmr(query, candidates, k=k)
        docs = [self.vs.id2meta[i].text for i, _ in candidates]
        pairs = [(query, d) for d in docs]
        scores = ce.predict(pairs, show_progress_bar=False).tolist()
        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:k]
        final = []
        for pos, sc in ranked:
            real_idx = candidates[pos][0]
            final.append((real_idx, float(sc)))
        return final

    # ----- Groq LLM -----
    def _groq_generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self.groq.chat.completions.create(
                model=self.generative_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except (APIStatusError, APIConnectionError) as e:
            raise RuntimeError(f"Groq API error: {e}")

    def generate(self, query: str, selected_ids: List[int]) -> Tuple[str, List[Dict]]:
        context_blocks, citations = [], []
        for rank, idx in enumerate(selected_ids, start=1):
            meta = self.vs.id2meta[idx]
            context_blocks.append(f"[{rank}] {meta.text}")
            citations.append({
                "id": idx,
                "doc_id": meta.doc_id,
                "source": meta.source,
                "text": meta.text
            })

        system_prompt = (
            "You are Aadi from Aadi-verse, an expert nutrition assistant.\n"
            "- Answer clearly using short sections and bullet points when helpful.\n"
            "- Use ONLY the provided context.\n"
            "- Do NOT mention internal filenames or file paths.\n"
            "- If the context is insufficient, say so briefly and suggest a follow-up question."
        )
        user_prompt = f"Question: {query}\n\nContext blocks:\n" + "\n\n".join(context_blocks)

        answer = self._groq_generate(system_prompt, user_prompt)
        return answer, citations

    # ----- end-to-end chat -----
    def chat(self, query: str, k: int = DEFAULT_K) -> Tuple[str, List[Dict], List[Tuple[int, float]]]:
        t = query.strip().lower()
        if t in SMALLTALK:
            return SMALLTALK[t], [], []

        if (self.vs.index is None) or (len(self.vs.id2meta) == 0):
            fallback = (
                "I can’t see any knowledge base loaded yet. "
                "If you committed a prebuilt index, make sure INDEX_DIR points to it (e.g., 'store'). "
                "Otherwise run ingestion first."
            )
            return fallback, [], []

        candidates = self.retrieve(query, top_n=DEFAULT_TOPN)
        topk = self.rerank(query, candidates, k=k)
        if not topk:
            fallback = (
                "I couldn’t find relevant info in the knowledge base yet. "
                "Try asking about macronutrients, vitamins/minerals, dietary patterns (Mediterranean, DASH), "
                "hydration, or daily requirements."
            )
            return fallback, [], []

        selected_ids = [i for i, _ in topk]
        answer, citations = self.generate(query, selected_ids)
        return answer, citations, topk
