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

from fastapi.middleware.cors import CORSMiddleware

# -------------- FastAPI --------------
app = FastAPI(title="NSUT RAG Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RetrieveRequest(BaseModel):
    query: str
    search_query: str | None = None
    top_k: int = 10          # how many chunks to return after reranking
    prefetch_k: int = 50     # how many candidates to fetch from Supabase before rerank

# -------------- Helpers --------------

def embed_minilm(text: str) -> List[float]:
    return next(embedder.embed([text])).tolist()

def supabase_vector_search(query_vec: List[float], k: int) -> List[Dict[str, Any]]:
    """
    Call your RPC to fetch similar notice chunks.
    RPC: match_notice_chunks(query_embedding vector, match_count int)
    Must return at least:
      id, notice_id, chunk_idx, chunk_text, filename, uploaded_at, similarity, notice_title
    """
    res = supabase.rpc(
        "match_notice_chunks",
        {"query_embedding": query_vec, "match_count": k}
    ).execute()
    return res.data or []

def supabase_keyword_search(query: str, k: int) -> List[Dict[str, Any]]:
    """
    Performs a full text search using Supabase's built-in textSearch.
    This acts as our sparse retrieval (BM25 equivalent).
    """
    clean_query = query.replace("'", "").replace('"', "")
    try:
        res = (
            supabase.table("notice_chunks_new_2")
            .select("id, notice_id, chunk_idx, chunk_text, filename, uploaded_at, notice_title")
            .textSearch("chunk_text", clean_query, options={"config": "english", "type": "websearch"})
            .limit(k)
            .execute()
        )
        return res.data or []
    except Exception as e:
        print(f"Keyword search failed: {e}")
        return []

def compute_rrf(vector_docs: List[Dict[str, Any]], keyword_docs: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) to combine ranks from dense and sparse retrieval.
    """
    rrf_scores = {}
    doc_map = {}
    
    def add_ranks(docs):
        for rank, doc in enumerate(docs):
            doc_id = doc.get("id")
            if not doc_id:
                continue
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
    add_ranks(vector_docs)
    add_ranks(keyword_docs)
    
    # Sort docs by RRF score descending
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    results = []
    for doc_id in sorted_doc_ids:
        doc = doc_map[doc_id]
        doc["similarity"] = rrf_scores[doc_id] # override similarity with standardized RRF score
        results.append(doc)
        
    return results

def rerank_with_cohere(query: str, docs: List[str], top_n: int) -> List[int]:
    """
    Rerank candidates using Cohere on SHORT docs.
    Returns a list of indices into `docs`, sorted by relevance.
    """
    if not docs:
        return []
    rr = co.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=min(top_n, len(docs)),
    )
    return [r.index for r in rr.results]

def get_notice_link(notice_id: str, filename: str) -> str:
    """
    Generates a signed URL with download prompt for a notice stored in Supabase Storage.
    Will append `download` parameter manually.
    """
    bucket = "notices_new_2"
    folder = ""
    file_path = f"{folder}/{filename}" if filename else f"{folder}/{notice_id}.pdf"

    try:
        signed_url_res = supabase.storage.from_(bucket).create_signed_url(
            path=file_path,
            expires_in=3600,
        )
        url = signed_url_res.get("signedURL")
        if not url:
            return None

        download_name = filename or f"{notice_id}.pdf"
        return f"{url}&download={download_name}"

    except Exception as e:
        print(f"Error creating signed URL for {file_path}: {e}")
        return None

def fetch_notices_ocr(notice_ids: List[Any]) -> Dict[str, str]:
    """
    Fetch full ocr_text for given notice_ids from the 'notices' table.
    Returns a map: str(notice_id) -> full ocr_text
    """
    if not notice_ids:
        return {}
    res = supabase.table("notices_new_2").select("id, ocr_text").in_("id", notice_ids).execute()
    rows = res.data or []
    return {str(r["id"]): (r.get("ocr_text") or "") for r in rows}

def short_doc(c: Dict[str, Any]) -> str:
    """
    Build a compact text for reranking:
    - use notice_title + a trimmed version of chunk_text
    - add a bit of lightweight metadata if needed
    """
    chunk_text = (c.get("chunk_text") or "")[:300]  # first 300 chars are enough
    notice_title = c.get("notice_title") or ""
    filename = c.get("filename", "unknown")

    # Prefer title + snippet so Cohere sees course code / exam type clearly
    if notice_title:
        return f"{notice_title} :: {chunk_text}"
    else:
        return f"[file={filename}] :: {chunk_text}"

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

    # 2) Retrieve candidates using Hybrid Search (Dense + Sparse)
    vector_candidates = supabase_vector_search(q_vec, req.prefetch_k)
    
    # Use search_query for better keyword matching if provided by orchestrator
    kw_query = req.search_query if req.search_query else req.query
    keyword_candidates = supabase_keyword_search(kw_query, req.prefetch_k)
    
    # Combine using Reciprocal Rank Fusion (RRF)
    candidates = compute_rrf(vector_candidates, keyword_candidates)
    
    if not candidates:
        return {"query": req.query, "chunks": []}

    # 3) Build compact docs for reranking (now includes notice_title when available)
    docs = [short_doc(c) for c in candidates]

    # 4) Rerank using Cohere to get best indices
    ordered_indices = rerank_with_cohere(req.query, docs, req.top_k)

    # 5) Collect unique notice_ids to fetch full ocr_text only once
    unique_notice_ids: List[Any] = []
    seen = set()
    for idx in ordered_indices:
        nid = candidates[idx].get("notice_id")
        if nid is not None and str(nid) not in seen:
            seen.add(str(nid))
            unique_notice_ids.append(nid)

    # 6) Fetch full ocr_text for each unique notice_id
    notice_ocr_map = fetch_notices_ocr(unique_notice_ids)

    # 7) Build final results: attach full ocr_text only the first time a notice appears
    final = []
    notices_attached = set()
    for idx in ordered_indices:
        c = candidates[idx]
        notice_id = c.get("notice_id")
        filename = c.get("filename")
        notice_title = c.get("notice_title")  # from chunks RPC
        notice_link = get_notice_link(notice_id, filename) if (notice_id or filename) else None

        notice_ocr = None
        if notice_id and str(notice_id) not in notices_attached:
            full_ocr = notice_ocr_map.get(str(notice_id)) or ""
            notice_ocr = full_ocr if full_ocr else None
            notices_attached.add(str(notice_id))

        final.append({
            "chunk_text": c.get("chunk_text"),
            "notice_link": notice_link,
            "filename": filename,
            "similarity": c.get("similarity"),   # similarity from RPC
            "notice_ocr": notice_ocr,            # full OCR, first chunk per notice
            "notice_id": notice_id,
            "notice_title": notice_title,        # expose for frontend / LLM
        })

    return {"query": req.query, "chunks": final}
