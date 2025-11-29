# retriever_service.py
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv
from fastembed import TextEmbedding
import cohere
import math

load_dotenv()

# ---------------- Env ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Hybrid tuning
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))  # weight given to semantic (0..1)
USE_COHERE_RERANK = os.getenv("USE_COHERE_RERANK", "true").lower() in ("1", "true", "yes")

# Cohere model for rerank
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")

# Basic checks
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")
if not COHERE_API_KEY and USE_COHERE_RERANK:
    raise RuntimeError("Missing COHERE_API_KEY while USE_COHERE_RERANK is enabled")
if not API_SECRET:
    raise RuntimeError("Missing API_SECRET")

# ------------- Clients ---------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

# Embedding model (same as you used)
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------- FastAPI --------------
app = FastAPI(title="NSUT RAG Retrieval API (Hybrid)")

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    prefetch_k: int = 50
    search_query: Optional[str] = None  # optional normalized query coming from upstream

# ----------------- Helpers -----------------

# simple deterministic normalizer (same idea as in search.py)
STOPWORDS = {
    "when", "what", "which", "who", "where", "how", "why",
    "is", "are", "was", "were", "will", "shall", "can", "could",
    "please", "tell", "me", "about", "the", "a", "an", "of", "for"
}

def normalize_query(q: str) -> str:
    q = (q or "").strip().lower()
    if not q:
        return ""
    if q.endswith("?"):
        q = q[:-1]
    tokens = [t for t in q.split() if t not in STOPWORDS]
    base = " ".join(tokens)
    return base.strip()

def embed_minilm(text: str) -> List[float]:
    # returns list[float]
    return next(embedder.embed([text])).tolist()

def get_notice_link(notice_id: str, filename: Optional[str]) -> Optional[str]:
    bucket = "notices"
    folder = "notices"
    file_path = f"{folder}/{filename}" if filename else f"{folder}/{notice_id}.pdf"
    try:
        signed_url_res = supabase.storage.from_(bucket).create_signed_url(
            path=file_path,
            expires_in=3600
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
    if not notice_ids:
        return {}
    res = supabase.table("notices").select("id, ocr_text").in_("id", notice_ids).execute()
    rows = res.data or []
    return {str(r["id"]): (r.get("ocr_text") or "") for r in rows}

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

# simple normalization utilities for combining scores
def normalize_list_minmax(vals: List[float], higher_better: bool = True) -> List[float]:
    if not vals:
        return []
    mn = min(vals)
    mx = max(vals)
    if math.isclose(mx, mn):
        # avoid zero division, return 1.0 for nonzero input or 0.0
        return [1.0 if v > 0 else 0.0 for v in vals]
    out = []
    for v in vals:
        if higher_better:
            out.append((v - mn) / (mx - mn))
        else:
            # lower is better -> invert
            out.append((mx - v) / (mx - mn))
    return out

def cohere_rerank_indices(query: str, docs: List[str], top_n: int) -> List[int]:
    if not USE_COHERE_RERANK or co is None:
        return list(range(min(top_n, len(docs))))
    rr = co.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=min(top_n, len(docs))
    )
    return [r.index for r in rr.results]

# ----------------- Endpoints -----------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/retrieve")
def retrieve(req: RetrieveRequest, api_key: str = Header(None)):
    # auth
    if api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Use provided normalized search_query if given, else compute locally
    search_q = req.search_query.strip() if req.search_query else normalize_query(req.query)
    if not search_q:
        return {"query": req.query, "chunks": []}

    # 1) Lexical FTS search (RPC: notice_chunks_fts_search)
    try:
        fts_resp = supabase.rpc("notice_chunks_fts_search", {"p_query": search_q, "p_limit": req.prefetch_k}).execute()
        fts_candidates = fts_resp.data or []
    except Exception as e:
        # fallback: if RPC missing or error, return nothing (or optionally perform full vector search)
        print("FTS RPC error:", e)
        raise HTTPException(status_code=500, detail=f"FTS RPC failed: {e}")

    if not fts_candidates:
        return {"query": req.query, "chunks": []}

    # Build a map for fts ranks and candidate ordering
    fts_rank_map = {c["id"]: safe_float(c.get("fts_rank", 0.0)) for c in fts_candidates}
    candidate_ids = [c["id"] for c in fts_candidates]

    # 2) Semantic vector search restricted to candidate IDs
    # compute embedding on the search_q (normalized)
    try:
        q_vec = embed_minilm(search_q)
    except Exception as e:
        print("Embedding error:", e)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        vec_resp = supabase.rpc("notice_chunks_vector_search_ids", {
            "p_query": q_vec,
            "p_ids": candidate_ids,
            "p_limit": req.prefetch_k
        }).execute()
        vec_hits = vec_resp.data or []
    except Exception as e:
        print("Vector RPC error:", e)
        raise HTTPException(status_code=500, detail=f"Vector RPC failed: {e}")

    if not vec_hits:
        return {"query": req.query, "chunks": []}

    # Collect vec_scores and fts ranks for normalization
    vec_scores = [safe_float(v.get("vec_score", 0.0)) for v in vec_hits]  # distances (lower is better)
    fts_vals_for_vec = [fts_rank_map.get(v.get("id")) or 0.0 for v in vec_hits]

    # Normalize: for vec_scores lower is better -> invert to higher_better
    norm_vec = normalize_list_minmax(vec_scores, higher_better=False)
    norm_fts = normalize_list_minmax(fts_vals_for_vec, higher_better=True)

    # Combine into a hybrid combined_score
    combined_list = []
    for i, v in enumerate(vec_hits):
        nv = norm_vec[i] if i < len(norm_vec) else 0.0
        nf = norm_fts[i] if i < len(norm_fts) else 0.0
        combined = HYBRID_ALPHA * nv + (1.0 - HYBRID_ALPHA) * nf
        combined_list.append(combined)
        # attach intermediate values to vec_hits entries
        v["_semantic_norm"] = nv
        v["_lexical_norm"] = nf
        v["_combined"] = combined

    # Attach combined score to each hit and sort by it
    for i, v in enumerate(vec_hits):
        v["_combined"] = combined_list[i]

    vec_hits_sorted = sorted(vec_hits, key=lambda x: x.get("_combined", 0.0), reverse=True)

    # 3) Optional: Cohere rerank on the top N semantic candidates (use raw user query for rerank)
    # Build docs for reranker from chunk_text + short ocr snippet (if present in vec_hits)
    # We'll rerank only the top M candidates (to limit cost). M = min(len(vec_hits_sorted), req.prefetch_k)
    M = min(len(vec_hits_sorted), max(req.top_k, 50))  # rerank up to 50 or at least top_k
    docs_for_rerank = []
    rerank_candidates = vec_hits_sorted[:M]
    # For fetching full notice OCR snippet we will query the notices table later;
    # here we can use any partial fields present in vec_hits (e.g., ocr_text, ocr_snippet) if available.
    for v in rerank_candidates:
        text = v.get("chunk_text", "") or ""
        # include any attached small ocr snippet field (if earlier ingestion attached it)
        ocr_snip = v.get("ocr_text") or v.get("ocr_snippet") or v.get("notice_ocr") or ""
        docs_for_rerank.append("\n".join([text, ocr_snip]))

    final_order_indices = list(range(min(req.top_k, len(rerank_candidates))))
    if USE_COHERE_RERANK and docs_for_rerank:
        try:
            reranked_indices = cohere_rerank_indices(req.query, docs_for_rerank, top_n=req.top_k)
            # reranked_indices are indices into rerank_candidates; map to selected positions
            final_order_indices = reranked_indices
        except Exception as e:
            print("Cohere rerank failed:", e)
            # fallback: use combined ordering (already sorted)
            final_order_indices = list(range(min(req.top_k, len(rerank_candidates))))

    # Build the selected list in final order (limited to top_k)
    selected_hits = []
    for idx in final_order_indices:
        # guard against out-of-range
        if idx < 0 or idx >= len(rerank_candidates):
            continue
        selected_hits.append(rerank_candidates[idx])

    # If we didn't use rerank or reranked fewer than top_k, fill from vec_hits_sorted
    if len(selected_hits) < req.top_k:
        needed = req.top_k - len(selected_hits)
        # append next best from vec_hits_sorted skipping ones already selected (by id)
        selected_ids = {s.get("id") for s in selected_hits}
        for v in vec_hits_sorted:
            if v.get("id") in selected_ids:
                continue
            selected_hits.append(v)
            if len(selected_hits) >= req.top_k:
                break

    # 4) Fetch full ocr_text for unique notice_ids (for attachments)
    unique_notice_ids: List[Any] = []
    seen_ids = set()
    for v in selected_hits:
        nid = v.get("notice_id")
        if nid is not None and str(nid) not in seen_ids:
            seen_ids.add(str(nid))
            unique_notice_ids.append(nid)
    notice_ocr_map = fetch_notices_ocr(unique_notice_ids)

    # 5) Build final result list (attach full ocr only first time per notice)
    final = []
    notices_attached = set()
    for v in selected_hits:
        notice_id = v.get("notice_id")
        filename = v.get("filename")
        notice_link = get_notice_link(notice_id, filename) if (notice_id or filename) else None

        notice_ocr = None
        if notice_id and str(notice_id) not in notices_attached:
            full_ocr = notice_ocr_map.get(str(notice_id)) or ""
            notice_ocr = full_ocr[:2000] if full_ocr else None
            notices_attached.add(str(notice_id))

        final.append({
            "chunk_text": v.get("chunk_text"),
            "notice_link": notice_link,
            "filename": filename,
            "similarity": float(v.get("_combined", 0.0)),  # combined hybrid score 0..1
            "semantic_score": float(v.get("_semantic_norm", 0.0)),
            "lexical_score": float(v.get("_lexical_norm", 0.0)),
            "notice_ocr": notice_ocr,
        })

    return {"query": req.query, "chunks": final}
