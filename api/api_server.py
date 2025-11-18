import os
from typing import List, Dict, Any

import cohere
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv
from fastembed import TextEmbedding

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

# Embedding model
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------- FastAPI --------------
app = FastAPI(title="NSUT RAG Retrieval API")

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    prefetch_k: int = 50

# -------------- Helpers --------------

def embed_minilm(text: str) -> List[float]:
    return next(embedder.embed([text])).tolist()

def supabase_vector_search(query_vec: List[float], k: int) -> List[Dict[str, Any]]:
    """Call your RPC to fetch similar notice chunks."""
    res = supabase.rpc(
        "match_notice_chunks",
        {"query_embedding": query_vec, "match_count": k}
    ).execute()
    return res.data or []

def rerank_with_cohere(query: str, docs: List[str], top_n: int) -> List[int]:
    rr = co.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=min(top_n, len(docs))
    )
    return [r.index for r in rr.results]

def get_notice_link(notice_id: str, filename: str) -> str:
    """Constructs public link to notice stored in Supabase Storage (notices/notices/{filename})."""
    bucket = "notices"
    folder = "notices"
    base_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{folder}"
    if filename:
        return f"{base_url}/{filename}"
    return f"{base_url}/{notice_id}.pdf"

def fetch_notices_ocr(notice_ids: List[Any]) -> Dict[str, str]:
    """
    Fetch full ocr_text for given notice_ids from the 'notices' table.
    Returns a map: str(notice_id) -> ocr_text
    """
    if not notice_ids:
        return {}
    # Ensure we pass a list of primitive ids to the .in_ call
    res = supabase.table("notices").select("id, ocr_text").in_("id", notice_ids).execute()
    rows = res.data or []
    return {str(r["id"]): (r.get("ocr_text") or "") for r in rows}

# -------------- Routes --------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/retrieve")
def retrieve(req: RetrieveRequest, api_key: str = Header(None)):
    if api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 1) Embed the query
    q_vec = embed_minilm(req.query)

    # 2) Retrieve top-N candidate chunks (each chunk should include notice_id, chunk_text, ocr_text (snippet), ocr_tables, filename, similarity)
    candidates = supabase_vector_search(q_vec, req.prefetch_k)
    if not candidates:
        return {"query": req.query, "chunks": []}

    # 3) Create extended docs for reranking (chunk-level context still)
    docs = [
        "\n".join([
            c.get("chunk_text", ""),
            f"Full Notice Text (snippet): {c.get('ocr_text', '')}",
            f"Tables: {c.get('ocr_tables', '')}"
        ])
        for c in candidates
    ]

    # 4) Rerank using Cohere to get best indices
    ordered_indices = rerank_with_cohere(req.query, docs, req.top_k)

    # 5) Collect unique notice_ids (preserving order) from reranked results, to fetch full ocr_text only once per notice
    unique_notice_ids: List[Any] = []
    seen = set()
    for idx in ordered_indices:
        nid = candidates[idx].get("notice_id")
        if nid is None:
            continue
        if str(nid) not in seen:
            seen.add(str(nid))
            unique_notice_ids.append(nid)

    # 6) Fetch full ocr_text for each unique notice_id
    notice_ocr_map = fetch_notices_ocr(unique_notice_ids)  # keys are strings

    # 7) Build final results: attach full ocr_text only the first time a notice appears
    final = []
    notices_attached = set()
    for idx in ordered_indices:
        c = candidates[idx]
        notice_id = c.get("notice_id")
        filename = c.get("filename")
        notice_link = get_notice_link(notice_id, filename) if (notice_id or filename) else None

        # Attach full ocr_text only once per notice (first occurrence)
        notice_ocr = None
        if notice_id and str(notice_id) not in notices_attached:
            notice_ocr = notice_ocr_map.get(str(notice_id))
            notices_attached.add(str(notice_id))

        final.append({
            "chunk_text": c.get("chunk_text"),
            "notice_link": notice_link,
            "filename": filename,
            "similarity": c.get("similarity"),
            # include full OCR only when present (otherwise value will be None)
            "notice_ocr": notice_ocr,
        })

    return {"query": req.query, "chunks": final}
