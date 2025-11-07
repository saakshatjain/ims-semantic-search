import os
import json
from typing import List, Dict, Any

import cohere
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------- Env ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
API_SECRET = os.getenv("API_SECRET")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("Missing COHERE_API_KEY")
if not API_SECRET:
    raise RuntimeError("Missing API_SECRET")

# ------------- Clients ---------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
co = cohere.Client(COHERE_API_KEY)

# ✅ LOAD MiniLM EMBEDDING MODEL (384-d)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------- FastAPI --------------
app = FastAPI(title="NSUT RAG Retrieval API")

# -------------- Schemas --------------
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    prefetch_k: int = 30

# -------------- Helpers --------------
def embed_minilm(text: str) -> List[float]:
    """Generate 384-dim MiniLM embeddings for compatibility with Supabase."""
    return embed_model.encode([text], convert_to_numpy=True)[0].tolist()

def supabase_vector_search(query_vec: List[float], k: int):
    """Call your RPC to fetch similar notice chunks."""
    res = supabase.rpc(
        "match_notice_chunks",
        {
            "query_embedding": query_vec,   # your RPC must accept vector[]
            "match_count": k
        }
    ).execute()
    return res.data or []

def rerank_with_cohere(query: str, docs: List[str], top_n: int):
    rr = co.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=min(top_n, len(docs))
    )
    return [r.index for r in rr.results]

# -------------- Routes --------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/retrieve")
def retrieve(req: RetrieveRequest, api_key: str = Header(None)):
    if api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 1) embed query with MiniLM (384-dim)
    q_vec = embed_minilm(req.query)

    # 2) fetch top-N candidates via pgvector
    candidates = supabase_vector_search(q_vec, req.prefetch_k)
    if not candidates:
        return {"query": req.query, "chunks": []}

    docs = [c["chunk_text"] for c in candidates]

    # 3) rerank using Cohere
    ordered = rerank_with_cohere(req.query, docs, req.top_k)

    # 4) return reranked top_k chunks
    final = []
    for idx in ordered:
        c = candidates[idx]
        final.append({
            "chunk_text": c["chunk_text"],
            "notice_id": c.get("notice_id"),
            "filename": c.get("filename"),
            "similarity": c.get("similarity"),
        })

    return {
        "query": req.query,
        "chunks": final
    }
