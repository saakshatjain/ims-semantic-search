#!/usr/bin/env python3
# worker/worker.py
import os
import io
import json
import uuid
import tempfile
from datetime import datetime
from typing import List, Optional, Tuple
from math import ceil

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import numpy as np

from supabase import create_client
from sentence_transformers import SentenceTransformer

# PaddleOCR (good for scanned / camscanned pages)
from paddleocr import PaddleOCR

# Optional: camelot for advanced table extraction if ghostscript is installed
try:
    import camelot
    _HAS_CAMELOT = True
except Exception:
    _HAS_CAMELOT = False

# --------------------------
# Configuration / constants
# --------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY env vars")

# chunking parameters (tune if needed)
TARGET_CHUNK_WORDS = int(os.environ.get("TARGET_CHUNK_WORDS", 120))
CHUNK_OVERLAP_WORDS = int(os.environ.get("CHUNK_OVERLAP_WORDS", 20))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", 32))
INSERT_BATCH = int(os.environ.get("INSERT_BATCH", 32))
OCR_THRESHOLD_CHARS = int(os.environ.get("OCR_THRESHOLD_CHARS", 40))

# --------------------------
# Init clients
# --------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # add 'hi' or others if needed

# --------------------------
# Utilities
# --------------------------
_sentence_split_re = __import__('re').compile(r'(?<=[.!?])\s+|\n+')

def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in _sentence_split_re.split(text) if p.strip()]
    if len(parts) == 1 and '\n' in text:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts

def chunk_text_semantic(text: str,
                        target_words: int = TARGET_CHUNK_WORDS,
                        overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    if not text or not text.strip():
        return []
    sentences = split_into_sentences(text)
    if not sentences:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i+target_words]
            chunks.append(" ".join(chunk_words))
            i += target_words - overlap_words
        return chunks

    sent_word_counts = [len(s.split()) for s in sentences]
    chunks = []
    i = 0
    n = len(sentences)
    while i < n:
        current_chunk = []
        current_words = 0
        j = i
        while j < n and current_words < target_words:
            current_chunk.append(sentences[j])
            current_words += sent_word_counts[j]
            j += 1
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)
        # compute advance using approximate words
        if j == i:
            i += 1
        else:
            advance = 0
            adv_words = 0
            k = 0
            while k < (j - i) and adv_words < (target_words - overlap_words):
                adv_words += sent_word_counts[i + k]
                k += 1
            i = i + k if k > 0 else j
    # merge tiny final chunk
    if len(chunks) >= 2 and len(chunks[-1].split()) < target_words // 3:
        chunks[-2] = chunks[-2] + "\n\n" + chunks[-1]
        chunks.pop()
    return chunks

def image_from_pixmap(pix) -> Image.Image:
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img.convert("RGB")

def ocr_image_paddle(img: Image.Image) -> str:
    np_img = np.array(img)
    result = ocr.ocr(np_img, cls=True)
    lines = []
    # normalize various result shapes
    for item in result:
        if isinstance(item, list):
            for entry in item:
                try:
                    txt = entry[1][0] if isinstance(entry[1], (list, tuple)) else str(entry[1])
                except Exception:
                    txt = str(entry)
                if txt and txt.strip():
                    lines.append(txt.strip())
        else:
            try:
                lines.append(str(item).strip())
            except Exception:
                pass
    return "\n".join(lines).strip()

def extract_text_and_tables(pdf_bytes: bytes) -> Tuple[str, Optional[List]]:
    # write to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        pdf_path = tmp.name

    all_text_pages = []
    all_tables = []

    # primary pass using pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p_idx, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                if not page_text or len(page_text.strip()) < OCR_THRESHOLD_CHARS:
                    # fallback to Paddle OCR on page image via fitz
                    try:
                        doc = fitz.open(pdf_path)
                        p = doc[p_idx - 1]
                        pix = p.get_pixmap(dpi=300)
                        pil_img = image_from_pixmap(pix)
                        page_text = ocr_image_paddle(pil_img)
                    except Exception:
                        try:
                            pil_img = page.to_image(resolution=300).original
                            page_text = ocr_image_paddle(pil_img)
                        except Exception:
                            page_text = page_text or ""
                if page_text and page_text.strip():
                    all_text_pages.append(page_text.strip())
                # extract tables via pdfplumber
                try:
                    page_tables = page.extract_tables()
                    for t in page_tables:
                        if t and any(any(cell for cell in row if cell) for row in t):
                            all_tables.append(t)
                except Exception:
                    pass
    except Exception:
        # fallback: full OCR by fitz/Paddle
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                pil_img = image_from_pixmap(pix)
                page_text = ocr_image_paddle(pil_img)
                if page_text:
                    all_text_pages.append(page_text)
        except Exception as e:
            print("Critical PDF parse failure:", e)

    # optional camelot fallback for tables (if available)
    if not all_tables and _HAS_CAMELOT:
        try:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
            for t in tables:
                df = t.df.values.tolist()
                if df:
                    all_tables.append(df)
        except Exception:
            pass

    # combine pages
    combined_text = "\n\n".join(all_text_pages).strip()

    # cleanup
    try:
        os.remove(pdf_path)
    except Exception:
        pass

    return combined_text, (all_tables if all_tables else None)

def create_chunk_embeddings_and_store(notice_id: str, filename: str, text: str, uploaded_at: Optional[str] = None):
    chunks = chunk_text_semantic(text)
    if not chunks:
        return 0
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
    # batch embed
    total = len(rows)
    for start in range(0, total, EMBED_BATCH):
        end = min(total, start + EMBED_BATCH)
        texts = [r["chunk_text"] for r in rows[start:end]]
        emb_batch = embed_model.encode(texts, convert_to_numpy=True).tolist()
        for i, emb in enumerate(emb_batch, start=start):
            rows[i]["embedding"] = emb
            rows[i]["processed_at"] = datetime.utcnow().isoformat()
    # insert in batches using upsert to avoid duplicates
    for start in range(0, total, INSERT_BATCH):
        end = min(total, start + INSERT_BATCH)
        batch = rows[start:end]
        try:
            supabase.table("notice_chunks").upsert(batch, on_conflict="id").execute()
        except Exception as e:
            # fallback to insert single rows to surface problematic rows
            for r in batch:
                try:
                    supabase.table("notice_chunks").upsert(r, on_conflict="id").execute()
                except Exception as e2:
                    print("Chunk insert failed for:", r["id"], str(e2))
    return total

# --------------------------
# Main worker loop
# --------------------------
def process_pending(limit: int = 8):
    resp = supabase.table("notices").select("*").eq("status", "pending").limit(limit).execute()
    rows = resp.data or []
    for n in rows:
        nid = n["id"]
        fname = n.get("filename") or n.get("file_path") or nid
        print(f"Processing notice: {fname} ({nid})")
        # mark processing
        supabase.table("notices").update({"status": "processing"}).eq("id", nid).execute()
        try:
            dl = supabase.storage.from_("notices").download(n["file_path"])
            if hasattr(dl, "read"):
                pdf_bytes = dl.read()
            else:
                pdf_bytes = dl
            # extract text + tables
            text, tables = extract_text_and_tables(pdf_bytes)
            # write basic notice metadata
            supabase.table("notices").update({
                "ocr_text": text,
                "ocr_tables": json.dumps(tables) if tables else None,
                "processed_at": datetime.utcnow().isoformat()
            }).eq("id", nid).execute()
            # create chunks & embeddings
            num_chunks = create_chunk_embeddings_and_store(nid, fname, text, uploaded_at=n.get("uploaded_at"))
            # finalize notice
            supabase.table("notices").update({"status": "processed", "embedding_model": "all-MiniLM-L6-v2"}).eq("id", nid).execute()
            print(f"✅ {fname} processed --> {num_chunks} chunks")
        except Exception as e:
            print("❌ Failed processing:", fname, str(e))
            supabase.table("notices").update({"status": "failed", "error_msg": str(e)}).eq("id", nid).execute()

# Convenience: backfill already processed notices
def rechunk_all_processed(batch_size: int = 50):
    offset = 0
    while True:
        res = supabase.table("notices").select("id, filename, ocr_text, uploaded_at").eq("status", "processed").range(offset, offset+batch_size-1).execute()
        rows = res.data or []
        if not rows:
            break
        for r in rows:
            nid = r["id"]
            # skip if chunks exist
            chk = supabase.table("notice_chunks").select("id").eq("notice_id", nid).limit(1).execute()
            if chk.data:
                print("Skipping already chunked:", nid)
                continue
            text = r.get("ocr_text","") or ""
            fname = r.get("filename") or nid
            cnt = create_chunk_embeddings_and_store(nid, fname, text, uploaded_at=r.get("uploaded_at"))
            print("Backfilled", cnt, "chunks for", nid)
        offset += batch_size

if __name__ == "__main__":
    process_pending(limit=8)
