#!/usr/bin/env python3
# worker/worker.py

import os
import io
import json
import uuid
import tempfile
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from supabase import create_client
from sentence_transformers import SentenceTransformer

import easyocr  # ✅ REPLACED PaddleOCR

# Try camelot (optional)
try:
    import camelot
    _HAS_CAMELOT = True
except Exception:
    _HAS_CAMELOT = False

# --------------------- CONFIG ---------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL & SUPABASE_KEY must be set")

TARGET_CHUNK_WORDS = 120
CHUNK_OVERLAP_WORDS = 20
EMBED_BATCH = 32
INSERT_BATCH = 32
OCR_THRESHOLD_CHARS = 40

# --------------------- INIT ------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reader = easyocr.Reader(['en'], gpu=False)  # ✅ EasyOCR CPU

# --------------------- UTILITIES ---------------------
import re
_sentence_split_re = re.compile(r'(?<=[.!?])\s+|\n+')

def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in _sentence_split_re.split(text) if p.strip()]
    return parts

def chunk_text_semantic(text: str) -> List[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return [text]

    chunks = []
    i = 0

    while i < len(sentences):
        curr = []
        words = 0
        j = i

        while j < len(sentences) and words < TARGET_CHUNK_WORDS:
            curr.append(sentences[j])
            words += len(sentences[j].split())
            j += 1

        chunks.append(" ".join(curr))
        i += max(1, (TARGET_CHUNK_WORDS - CHUNK_OVERLAP_WORDS) // 10)

    return chunks

def ocr_easy(img: Image.Image) -> str:
    np_img = np.array(img)
    result = reader.readtext(np_img)
    lines = [res[1] for res in result if len(res) >= 2]
    return "\n".join(lines).strip()

def extract_text_and_tables(pdf_bytes: bytes) -> Tuple[str, Optional[List]]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        pdf_path = tmp.name

    all_text_pages = []
    all_tables = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                txt = page.extract_text() or ""

                if len(txt.strip()) < OCR_THRESHOLD_CHARS:
                    doc = fitz.open(pdf_path)
                    pix = doc[idx].get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    txt = ocr_easy(img)

                if txt:
                    all_text_pages.append(txt)

                try:
                    tables = page.extract_tables()
                    for t in tables:
                        if any(any(cell for cell in row) for row in t):
                            all_tables.append(t)
                except:
                    pass
    except:
        pass

    # Camelot fallback
    if not all_tables and _HAS_CAMELOT:
        try:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
            for t in tables:
                all_tables.append(t.df.values.tolist())
        except:
            pass

    os.remove(pdf_path)
    return "\n\n".join(all_text_pages), all_tables or None

def create_chunk_embeddings_and_store(notice_id, filename, text, uploaded_at):
    chunks = chunk_text_semantic(text)
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
    for start in range(0, total, EMBED_BATCH):
        end = min(total, start + EMBED_BATCH)
        emb = embed_model.encode([r["chunk_text"] for r in rows[start:end]],
                                 convert_to_numpy=True).tolist()

        for i, e in enumerate(emb, start=start):
            rows[i]["embedding"] = e
            rows[i]["processed_at"] = datetime.utcnow().isoformat()

    for start in range(0, total, INSERT_BATCH):
        supabase.table("notice_chunks").upsert(
            rows[start:start+INSERT_BATCH], on_conflict="id"
        ).execute()

    return total

# --------------------- MAIN ---------------------
def process_pending(limit=8):
    res = supabase.table("notices").select("*").eq("status", "pending").limit(limit).execute()
    notices = res.data or []

    for n in notices:
        nid = n["id"]
        fname = n.get("filename") or nid

        supabase.table("notices").update({"status": "processing"}).eq("id", nid).execute()

        try:
            file_bytes = supabase.storage.from_("notices").download(n["file_path"])
            file_bytes = file_bytes.read() if hasattr(file_bytes, "read") else file_bytes

            text, table = extract_text_and_tables(file_bytes)

            supabase.table("notices").update({
                "ocr_text": text,
                "ocr_tables": json.dumps(table),
                "processed_at": datetime.utcnow().isoformat()
            }).eq("id", nid).execute()

            create_chunk_embeddings_and_store(
                nid, fname, text, uploaded_at=n.get("uploaded_at")
            )

            supabase.table("notices").update({
                "status": "processed",
                "embedding_model": "all-MiniLM-L6-v2"
            }).eq("id", nid).execute()

            print(f"✅ Processed {fname}")
        except Exception as e:
            supabase.table("notices").update({
                "status": "failed",
                "error_msg": str(e)
            }).eq("id", nid).execute()
            print(f"❌ Failed {fname}:", e)


if __name__ == "__main__":
    process_pending(limit=8)
