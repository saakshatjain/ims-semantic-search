# api_server.py

import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from supabase import create_client
from sentence_transformers import SentenceTransformer, CrossEncoder



load_dotenv()

app = FastAPI()

# ---- Supabase ----
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- Embedding + Reranker ----
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---- Request Schema ----
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


# ---- Step 1: Fetch approximate chunks using Supabase RPC ----
def fetch_top_chunks(query_embedding, top_k=10):
    response = supabase.rpc(
        "match_notice_chunks",
        {"query_embedding": query_embedding, "match_count": top_k}
    ).execute()

    return response.data or []


# ---- Step 2: Rerank using CrossEncoder ----
def rerank_chunks(query, chunks):
    if not chunks:
        return []

    pairs = [[query, c["chunk_text"]] for c in chunks]
    scores = reranker.predict(pairs)

    # attach scores
    for c, score in zip(chunks, scores):
        c["rerank_score"] = float(score)

    # sort best → worst
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)


# ---- API ROUTE: return only context ----
@app.post("/retrieve")
def retrieve(req: QueryRequest):
    # 1) Embed the query
    q_emb = embedder.encode(req.query, convert_to_numpy=True).tolist()

    # 2) Fetch approximate chunks (fast)
    candidates = fetch_top_chunks(q_emb, req.top_k)

    # 3) Rerank them (accuracy boost)
    best = rerank_chunks(req.query, candidates)

    # 4) Return ONLY the chunk text + metadata
    return {
        "context": [
            {
                "chunk_text": c["chunk_text"],
                "notice_id": c["notice_id"],
                "filename": c["filename"],
                "rerank_score": c["rerank_score"]
            }
            for c in best[:5]
        ]
    }
