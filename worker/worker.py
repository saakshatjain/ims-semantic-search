#!/usr/bin/env python3
# worker/worker.py
"""
Robust worker:
- Extracts text (pdfplumber) + EasyOCR fallback (scanned PDFs)
- Extracts tables via pdfplumber (Camelot optional)
- Flattens tables into readable text
- Sentence-aware semantic chunking with overlap (220 words, 60 overlap)
- Short-doc special-case (single chunk)
- Batches embeddings & upserts to `notice_chunks`
- Adds a short PAGE header to each chunk to preserve provenance
- Safe for GH Actions (EasyOCR, no paddle)
"""

import os, io, json, uuid, tempfile, re
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np

import fitz              # PyMuPDF
import pdfplumber
from PIL import Image

from supabase import create_client
from sentence_transformers import SentenceTransformer

import easyocr

# Optional camelot (only if you installed it and ghostscript)
try:
    import camelot
    _HAS_CAMELOT = True
except:
    _HAS_CAMELOT = False

# ---------------- config ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment")

# chunking params (tunable)
TARGET_CHUNK_WORDS = int(os.environ.get("TARGET_CHUNK_WORDS", 220))
CHUNK_OVERLAP_WORDS = int(os.environ.get("CHUNK_OVERLAP_WORDS", 60))
SHORT_DOC_WORDS = int(os.environ.get("SHORT_DOC_WORDS", 400))  # use single chunk if doc shorter
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", 32))
INSERT_BATCH = int(os.environ.get("INSERT_BATCH", 32))
OCR_THRESHOLD_CHARS = int(os.environ.get("OCR_THRESHOLD_CHARS", 40))  # if text < -> OCR page

# ---------------- clients ----------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
ocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU

# ---------------- utilities ----------------
_sentence_split_re = re.compile(r'(?<=[.!?])\s+|\n+')

def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in _sentence_split_re.split(text) if p.strip()]
    if not parts:
        # fallback: split by newline
        parts = [p.strip() for p in text.splitlines() if p.strip()]
    return parts

def words_in(text: str) -> int:
    return len(text.split())

def image_from_pixmap(pix) -> Image.Image:
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

def ocr_easy(img: Image.Image) -> str:
    arr = np.array(img)
    res = ocr_reader.readtext(arr)
    lines = []
    for r in res:
        if len(r) >= 2 and r[1].strip():
            lines.append(r[1].strip())
    return "\n".join(lines).strip()

# ---------------- PDF extraction ----------------
def extract_text_and_tables(pdf_bytes: bytes) -> Tuple[str, Optional[List]]:
    """Extracts page-by-page text and tables. Returns combined_text and tables list (or None)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes); tmp.flush(); pdf_path = tmp.name

    page_texts = []
    table_blocks = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            doc = fitz.open(pdf_path)
            for p_idx, page in enumerate(pdf.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                # if page_text is very small → OCR the page image
                if not page_text or len(page_text) < OCR_THRESHOLD_CHARS:
                    try:
                        pix = doc[p_idx-1].get_pixmap(dpi=300)
                        img = image_from_pixmap(pix)
                        page_text = ocr_easy(img)
                    except Exception as e:
                        # fallback: keep whatever small text we had
                        page_text = page_text or ""
                if page_text:
                    # prefix page header to preserve provenance
                    page_text = f"[PAGE:{p_idx}]\n" + page_text
                    page_texts.append(page_text)

                # extract tables via pdfplumber
                try:
                    ptables = page.extract_tables()
                    for t in ptables:
                        # t is list of row-lists
                        if t and any(any(cell for cell in row if cell) for row in t):
                            # flatten table into text block
                            rows = [" | ".join([str(cell) if cell is not None else "" for cell in row]) for row in t]
                            table_text = f"[PAGE:{p_idx}] TABLE\n" + "\n".join(rows)
                            table_blocks.append(table_text)
                except Exception:
                    pass
    except Exception as e:
        # Very defensive: full OCR fallback with fitz
        try:
            doc = fitz.open(pdf_path)
            for p_idx, page in enumerate(doc, start=1):
                pix = page.get_pixmap(dpi=200)
                img = image_from_pixmap(pix)
                page_text = ocr_easy(img)
                if page_text:
                    page_texts.append(f"[PAGE:{p_idx}]\n" + page_text)
        except Exception as e2:
            print("Critical: full fallback failed:", e2)

    # Camelot optional fallback for tables (only usable if installed & ghostscript present)
    if not table_blocks and _HAS_CAMELOT:
        try:
            ctbls = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
            for t in ctbls:
                df = t.df.values.tolist()
                rows = [" | ".join([str(c) for c in r]) for r in df]
                table_blocks.append("TABLE\n" + "\n".join(rows))
        except Exception:
            pass

    try:
        os.remove(pdf_path)
    except:
        pass

    # Combine: We keep table blocks separate so they can form their own chunks
    combined_text = "\n\n".join(page_texts).strip()
    return combined_text, (table_blocks if table_blocks else None)

# ---------------- semantic chunking ----------------
def chunk_text_semantic(text: str,
                        target_words: int = TARGET_CHUNK_WORDS,
                        overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    """
    Sentence-aware sliding window chunker.
    - If total_words < SHORT_DOC_WORDS -> single chunk (preserve small docs)
    - Each chunk stores sentences from i..j-1; slide back by overlap_sentences
    """
    if not text or not text.strip():
        return []

    total_words = words_in(text)
    if total_words <= SHORT_DOC_WORDS:
        # single chunk for small docs (improves accuracy)
        return [text.strip()]

    sentences = split_into_sentences(text)
    sent_word_counts = [len(s.split()) for s in sentences]
    cum_words = [0]
    for c in sent_word_counts:
        cum_words.append(cum_words[-1] + c)

    chunks = []
    i = 0
    n = len(sentences)

    # helper to compute overlap in sentences
    def overlap_sentences_from_end(j, overlap_words_target):
        # accumulate from j-1 backwards to approximate overlap_words_target
        k = 0
        acc = 0
        while j - 1 - k >= 0 and acc < overlap_words_target:
            acc += sent_word_counts[j - 1 - k]
            k += 1
        return k

    while i < n:
        # grow j until target_words reached
        j = i
        acc = 0
        while j < n and acc < target_words:
            acc += sent_word_counts[j]
            j += 1
        # build chunk text
        chunk = " ".join(sentences[i:j]).strip()
        if chunk:
            chunks.append(chunk)
        # compute overlap in sentences and move i
        overlap_sentences = overlap_sentences_from_end(j, overlap_words)
        # ensure at least 1 sentence advance to avoid infinite loop
        i = max(j - overlap_sentences, j - 1, i + 1)

    # merge tiny trailing chunk
    if len(chunks) >= 2 and words_in(chunks[-1]) < target_words // 4:
        chunks[-2] = chunks[-2] + "\n\n" + chunks[-1]
        chunks.pop()

    return chunks

# ---------------- embed & store ----------------
def create_chunk_embeddings_and_store(notice_id: str, filename: str, text: str, uploaded_at: Optional[str]=None):
    # If tables exist inside text, ensure they are separated — (worker.extract_text_and_tables already provides)
    chunks = chunk_text_semantic(text)
    if not chunks:
        return 0

    # Build rows with small provenance header (we don't change DB schema)
    rows = []
    for idx, c in enumerate(chunks):
        rows.append({
            "id": str(uuid.uuid4()),
            "notice_id": notice_id,
            "chunk_idx": idx,
            "chunk_text": c,
            "filename": filename,
            "uploaded_at": uploaded_at,
            "embedding": None,
            "processed_at": None
        })

    total = len(rows)
    # batch embed
    for s in range(0, total, EMBED_BATCH):
        eend = min(total, s + EMBED_BATCH)
        texts = [r["chunk_text"] for r in rows[s:eend]]
        vecs = embed_model.encode(texts, convert_to_numpy=True).tolist()
        for i, v in enumerate(vecs, start=s):
            rows[i]["embedding"] = v
            rows[i]["processed_at"] = datetime.utcnow().isoformat()

    # upsert in batches
    for s in range(0, total, INSERT_BATCH):
        eend = min(total, s + INSERT_BATCH)
        batch = rows[s:eend]
        # Use upsert if available — if not, fallback to insert
        try:
            supabase.table("notice_chunks").upsert(batch, on_conflict="id").execute()
        except Exception:
            for r in batch:
                try:
                    supabase.table("notice_chunks").insert(r).execute()
                except Exception as e:
                    print("chunk insert failed", r["id"], str(e))
    return total

# ---------------- main worker ----------------
def process_pending(limit:int=100):
    print("Fetching pending notices...")
    res = supabase.table("notices").select("*").eq("status", "pending").limit(limit).execute()
    notices = res.data or []
    print(f"Found {len(notices)} pending notices")
    for n in notices:
        nid = n["id"]
        fname = n.get("filename") or nid
        print("Processing:", fname)
        # mark processing
        supabase.table("notices").update({"status": "processing"}).eq("id", nid).execute()
        try:
            dl = supabase.storage.from_("notices").download(n["file_path"])
            pdf_bytes = dl.read() if hasattr(dl, "read") else dl
            text, tables = extract_text_and_tables(pdf_bytes)
            # Append table blocks (if any) as separate text sections to preserve them
            if tables:
                text = text + "\n\n" + "\n\n".join(tables)
            # store OCR text for debugging & search
            supabase.table("notices").update({
                "ocr_text": text,
                "ocr_tables": json.dumps(tables) if tables else None,
                "processed_at": datetime.utcnow().isoformat()
            }).eq("id", nid).execute()
            # create chunks + embeddings
            cnt = create_chunk_embeddings_and_store(nid, fname, text, uploaded_at=n.get("uploaded_at"))
            # finalize
            supabase.table("notices").update({
                "status": "processed",
                "embedding_model": "all-MiniLM-L6-v2"
            }).eq("id", nid).execute()
            print(f"✅ processed {fname} → {cnt} chunks")
        except Exception as e:
            print("❌ failed:", fname, str(e))
            supabase.table("notices").update({"status":"failed", "error_msg": str(e)}).eq("id", nid).execute()

# ---------------- backfill existing processed ----------------
def rechunk_all_processed(batch_size:int=50):
    offset = 0
    while True:
        res = supabase.table("notices").select("id, filename, ocr_text, uploaded_at").eq("status","processed").range(offset, offset+batch_size-1).execute()
        rows = res.data or []
        if not rows:
            break
        for r in rows:
            nid = r["id"]
            # skip if chunks exist
            chk = supabase.table("notice_chunks").select("id").eq("notice_id", nid).limit(1).execute()
            if chk.data:
                print("Skipping:", nid)
                continue
            text = r.get("ocr_text","") or ""
            fname = r.get("filename") or nid
            cnt = create_chunk_embeddings_and_store(nid, fname, text, uploaded_at=r.get("uploaded_at"))
            print("Backfilled", cnt, "chunks for", nid)
        offset += batch_size

if __name__ == "__main__":
    process_pending(limit=8)
