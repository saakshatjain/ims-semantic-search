#!/usr/bin/env python3
# worker/worker.py
"""
Worker with table->row conversion + row-level embeddings + optional multi-row chunks.
"""

import os, io, json, uuid, tempfile, re
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
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

# row chunking params (tunable)
MAX_ROWS_PER_CHUNK = int(os.environ.get("MAX_ROWS_PER_CHUNK", 25))   # how many rows per multi-row chunk
ROW_CHUNK_OVERLAP = int(os.environ.get("ROW_CHUNK_OVERLAP", 2))      # overlap rows between chunks

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
def extract_text_and_tables(pdf_bytes: bytes) -> Tuple[str, Optional[List[str]]]:
    """Extracts page-by-page text and tables. Returns combined_text and tables list (or None)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        pdf_path = tmp.name

    page_texts = []
    table_blocks: List[str] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            doc = fitz.open(pdf_path)
            for p_idx, page in enumerate(pdf.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                # if page_text is very small → OCR the page image
                if not page_text or len(page_text) < OCR_THRESHOLD_CHARS:
                    try:
                        pix = doc[p_idx - 1].get_pixmap(dpi=300)
                        img = image_from_pixmap(pix)
                        page_text = ocr_easy(img)
                    except Exception:
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
                            rows = [
                                " | ".join([str(cell) if cell is not None else "" for cell in row])
                                for row in t
                            ]
                            table_text = f"[PAGE:{p_idx}] TABLE\n" + "\n".join(rows)
                            table_blocks.append(table_text)
                except Exception:
                    pass
    except Exception:
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

    combined_text = "\n\n".join(page_texts).strip()
    return combined_text, (table_blocks if table_blocks else None)

# ---------------- table -> row parsing ----------------
# heuristics to detect common identifiers; tweak regexes for your corpus
ROLL_RE = re.compile(r'\b20\d{2}[A-Z]{1,4}\d{2,4}\b')          # e.g. 2025UCA1849
APP_RE  = re.compile(r'\b[A-Z]{2}\d{12,}\b')                  # e.g. HR202526002466664 (len heuristic)
COURSE_CODE_RE = re.compile(r'\b[A-Z]{2,6}\d{2,3}\b')        # e.g. MPMEC14, BTBTC15

def rows_from_table_blocks(table_blocks: Optional[List[str]]) -> List[Dict[str, Any]]:
    """
    Convert table blocks of the form:
      "[PAGE:1] TABLE\nH1 | H2 | H3\nr1c1 | r1c2 | r1c3\n..."
    into a list of structured row dicts:
      {
        "page": int,
        "row_idx": int,
        "header": [...],
        "cells": [...],
        "sentence": "PAGE ROW -> H1: v1; H2: v2; ...",
        "row_meta": {"roll": ..., "appid": ..., "code": ...}
      }
    """
    out: List[Dict[str, Any]] = []
    if not table_blocks:
        return out

    for block in table_blocks:
        if not block:
            continue
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        # page
        page = None
        mpage = re.search(r'\[PAGE:(\d+)\]', lines[0])
        if mpage:
            page = int(mpage.group(1))

        # find the first line that contains pipe '|' and treat as header
        header = None
        header_idx = None
        for i, ln in enumerate(lines[1:], start=1):
            if '|' in ln:
                header = [c.strip() for c in re.sub(r'\s+', ' ', ln).split('|')]
                header_idx = i
                break

        # if no header found, fallback - treat every '|' line as a row with no header
        if header is None:
            r_idx = 0
            for ln in lines[1:]:
                if '|' not in ln:
                    continue
                cells = [c.strip() for c in re.sub(r'\s+', ' ', ln).split('|')]
                sentence = f"[PAGE:{page}] ROW -> " + " | ".join(cells)
                meta = _extract_row_meta(cells)
                out.append({
                    "page": page,
                    "row_idx": r_idx,
                    "header": None,
                    "cells": cells,
                    "sentence": sentence,
                    "row_meta": meta
                })
                r_idx += 1
            continue

        # parse rows after header
        r_idx = 0
        for ln in lines[header_idx+1:]:
            if '|' not in ln:
                continue
            cells = [c.strip() for c in re.sub(r'\s+', ' ', ln).split('|')]
            if len(cells) != len(header):
                # if mismatch, keep raw row text
                sentence = f"[PAGE:{page}] ROW -> " + " | ".join(cells)
            else:
                pairs = [f"{header[i]}: {cells[i]}" for i in range(len(header))]
                sentence = f"[PAGE:{page}] ROW -> " + "; ".join(pairs)
            meta = _extract_row_meta(cells)
            out.append({
                "page": page,
                "row_idx": r_idx,
                "header": header,
                "cells": cells,
                "sentence": sentence,
                "row_meta": meta
            })
            r_idx += 1

    return out

def _extract_row_meta(cells: List[str]) -> Dict[str, Optional[str]]:
    """Extract simple metadata tokens (roll, appid, course code) from row cell texts."""
    roll = None
    appid = None
    code = None
    # scan cells
    for c in cells:
        if not roll:
            m = ROLL_RE.search(c)
            if m:
                roll = m.group(0)
        if not appid:
            m = APP_RE.search(c)
            if m:
                appid = m.group(0)
        if not code:
            m = COURSE_CODE_RE.search(c)
            if m:
                code = m.group(0)
    return {"roll": roll, "appid": appid, "code": code}

# ---------------- chunk table rows into multi-row chunks ----------------
def chunk_table_rows(rows: List[Dict[str, Any]], max_rows: int = MAX_ROWS_PER_CHUNK, overlap: int = ROW_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Group consecutive row dicts into multi-row chunks.
    Returns list of chunk dicts: {start_row, end_row, text, rows_meta}
    """
    out = []
    if not rows:
        return out
    n = len(rows)
    i = 0
    chunk_seq = 0
    while i < n:
        j = min(n, i + max_rows)
        rows_slice = rows[i:j]
        text = "\n".join(r["sentence"] for r in rows_slice)
        rows_meta = [{"row_id": f"{r['page']}_{r['row_idx']}", "roll": r["row_meta"].get("roll"), "appid": r["row_meta"].get("appid"), "code": r["row_meta"].get("code"), "page": r["page"], "row_idx": r["row_idx"]} for r in rows_slice]
        out.append({
            "chunk_seq": chunk_seq,
            "start_row": i,
            "end_row": j-1,
            "text": text,
            "rows_meta": rows_meta
        })
        chunk_seq += 1
        i = j - overlap
        if i <= j - overlap - 1:
            i = j
    return out

# ---------------- embed & store helpers ----------------
def upsert_row_embeddings(notice_id: str, filename: str, rows: List[Dict[str, Any]], notice_title: Optional[str] = None) -> int:
    """
    Create embeddings for each table row (one embedding per row) and upsert to notice_chunks_new_2.
    Uses chunk_idx offset to avoid colliding with text chunk_idx values.
    """
    if not rows:
        return 0

    ROW_OFFSET = 100000  # to avoid colliding with regular text chunk_idx
    records = []
    for idx, r in enumerate(rows):
        rec = {
            "id": str(uuid.uuid4()),
            "notice_id": notice_id,
            "chunk_idx": ROW_OFFSET + idx,
            "chunk_text": r["sentence"],
            "filename": filename,
            "notice_title": None,   # keep title separate for row chunks
            "uploaded_at": None,
            "embedding": None,
            "processed_at": None,
            "is_table_row": True,
            "row_meta": json.dumps(r["row_meta"])
        }
        records.append(rec)

    # embed in batches
    total = len(records)
    for s in range(0, total, EMBED_BATCH):
        e = min(total, s + EMBED_BATCH)
        texts = [rec["chunk_text"] for rec in records[s:e]]
        vecs = embed_model.encode(texts, convert_to_numpy=True).tolist()
        for i, v in enumerate(vecs):
            records[s + i]["embedding"] = v
            records[s + i]["processed_at"] = datetime.utcnow().isoformat()

    # upsert in batches
    for s in range(0, total, INSERT_BATCH):
        e = min(total, s + INSERT_BATCH)
        batch = records[s:e]
        try:
            supabase.table("notice_chunks_new_2").upsert(batch, on_conflict="notice_id,chunk_idx").execute()
        except Exception:
            for r in batch:
                try:
                    supabase.table("notice_chunks_new_2").insert(r).execute()
                except Exception as e:
                    print("row chunk insert failed", r["id"], str(e))
    return total

def upsert_multirow_chunks(notice_id: str, filename: str, chunks: List[Dict[str, Any]], notice_title: Optional[str] = None) -> int:
    """
    Create embeddings for multi-row chunks and upsert them.
    Use a different chunk_idx offset to avoid collisions.
    """
    if not chunks:
        return 0

    MULTI_OFFSET = 200000
    records = []
    for idx, c in enumerate(chunks):
        rec = {
            "id": str(uuid.uuid4()),
            "notice_id": notice_id,
            "chunk_idx": MULTI_OFFSET + idx,
            "chunk_text": c["text"],
            "filename": filename,
            "notice_title": notice_title,
            "uploaded_at": None,
            "embedding": None,
            "processed_at": None,
            "is_table_chunk": True,
            "rows_meta": json.dumps(c["rows_meta"])
        }
        records.append(rec)

    total = len(records)
    for s in range(0, total, EMBED_BATCH):
        e = min(total, s + EMBED_BATCH)
        texts = [rec["chunk_text"] for rec in records[s:e]]
        vecs = embed_model.encode(texts, convert_to_numpy=True).tolist()
        for i, v in enumerate(vecs):
            records[s + i]["embedding"] = v
            records[s + i]["processed_at"] = datetime.utcnow().isoformat()

    for s in range(0, total, INSERT_BATCH):
        e = min(total, s + INSERT_BATCH)
        batch = records[s:e]
        try:
            supabase.table("notice_chunks_new_2").upsert(batch, on_conflict="notice_id,chunk_idx").execute()
        except Exception:
            for r in batch:
                try:
                    supabase.table("notice_chunks_new_2").insert(r).execute()
                except Exception as e:
                    print("multi-row chunk insert failed", r["id"], str(e))
    return total
def chunk_text_semantic(
    text: str,
    target_words: int = TARGET_CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS
) -> List[str]:
    """
    Sentence-aware sliding window chunker.
    If total_words < SHORT_DOC_WORDS -> single chunk.
    """
    if not text or not text.strip():
        return []

    total_words = len(text.split())
    if total_words <= SHORT_DOC_WORDS:
        return [text.strip()]

    sentences = split_into_sentences(text)
    sent_word_counts = [len(s.split()) for s in sentences]

    chunks = []
    i = 0
    n = len(sentences)

    def overlap_sentences_from_end(j, overlap_words_target):
        k = 0
        acc = 0
        while j - 1 - k >= 0 and acc < overlap_words_target:
            acc += sent_word_counts[j - 1 - k]
            k += 1
        return k

    while i < n:
        j = i
        acc = 0
        while j < n and acc < target_words:
            acc += sent_word_counts[j]
            j += 1

        chunk = " ".join(sentences[i:j]).strip()
        if chunk:
            chunks.append(chunk)

        overlap_sentences = overlap_sentences_from_end(j, overlap_words)
        i = max(j - overlap_sentences, j - 1, i + 1)

    if len(chunks) >= 2 and len(chunks[-1].split()) < target_words // 4:
        chunks[-2] += "\n\n" + chunks[-1]
        chunks.pop()

    return chunks

# Keep your existing text chunk + embed function (slightly renamed for clarity)
def create_text_chunk_embeddings_and_store(
    notice_id: str,
    filename: str,
    text: str,
    notice_title: Optional[str] = None,
    uploaded_at: Optional[str] = None,
) -> int:
    """
    Chunk prose text semantics and store in notice_chunks_new_2 using previous logic.
    """
    if not text or not text.strip():
        return 0

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
            "notice_title": notice_title,
            "uploaded_at": uploaded_at,
            "embedding": None,
            "processed_at": None,
            "is_table_row": False
        })

    total = len(rows)
    for s in range(0, total, EMBED_BATCH):
        eend = min(total, s + EMBED_BATCH)
        texts = []
        for r in rows[s:eend]:
            # note: avoid prepending notice_title to row-level table embeddings (we only apply here to prose chunks)
            if r["notice_title"]:
                texts.append(f"{r['notice_title']}. {r['chunk_text']}")
            else:
                texts.append(r["chunk_text"])
        vecs = embed_model.encode(texts, convert_to_numpy=True).tolist()
        for offset, v in enumerate(vecs):
            idx = s + offset
            rows[idx]["embedding"] = v
            rows[idx]["processed_at"] = datetime.utcnow().isoformat()

    for s in range(0, total, INSERT_BATCH):
        eend = min(total, s + INSERT_BATCH)
        batch = rows[s:eend]
        try:
            supabase.table("notice_chunks_new_2").upsert(batch, on_conflict="notice_id,chunk_idx").execute()
        except Exception:
            for r in batch:
                try:
                    supabase.table("notice_chunks_new_2").insert(r).execute()
                except Exception as e:
                    print("text chunk insert failed", r["id"], str(e))
    return total

# ---------------- main worker ----------------
def process_pending(limit: int = 15):
    print("Fetching pending notices...")
    res = (
    supabase.table("notices_new_2")
    .select("*")
    .in_("status", ["pending", "failed"])
    .execute()
)

    notices = res.data or []
    print(f"Found {len(notices)} pending notices")
    for n in notices:
        nid = n["id"]
        fname = n.get("filename") or nid
        title = n.get("notice_title")   # pull notice_title from notices table
        # storage path: some older rows may have 'file_path', newer may only have filename
        file_path = n.get("file_path") or n.get("filename") or fname
        # for storage download we expect the filename/path stored in storage bucket
        raw_name = n.get("filename") or n.get("file_path") or nid
        clean_filename = os.path.basename(raw_name)
        print("Processing:", fname, "| title:", title)
        # mark processing
        supabase.table("notices_new_2").update({"status": "processing"}).eq("id", nid).execute()
        try:
            dl = supabase.storage.from_("notices_new_2").download(clean_filename)
            pdf_bytes = dl.read() if hasattr(dl, "read") else dl
            text, tables = extract_text_and_tables(pdf_bytes)

            # parse tables into rows (structured)
            table_rows = rows_from_table_blocks(tables)

            # Optionally: for very large date-sheets you may choose to skip prose chunking,
            # but here we do both: prose + row embeddings.
            # 1) create prose text chunks and embeddings (existing behavior)
            text_cnt = create_text_chunk_embeddings_and_store(
                notice_id=nid,
                filename=fname,
                text=text,
                notice_title=title,
                uploaded_at=n.get("uploaded_at")
            )

            # 2) upsert one embedding per table row
            rows_cnt = upsert_row_embeddings(
                notice_id=nid,
                filename=fname,
                rows=table_rows,
                notice_title=title
            )

            # 3) optionally create multi-row chunks for context
            multi_chunks = chunk_table_rows(table_rows, max_rows=MAX_ROWS_PER_CHUNK, overlap=ROW_CHUNK_OVERLAP)
            multi_cnt = upsert_multirow_chunks(
                notice_id=nid,
                filename=fname,
                chunks=multi_chunks,
                notice_title=title
            )

            # store OCR text + raw tables JSON for debugging & possible future reprocessing
            supabase.table("notices_new_2").update({
                "ocr_text": text,
                "ocr_tables": json.dumps(tables) if tables else None,
                "processed_at": datetime.utcnow().isoformat()
            }).eq("id", nid).execute()

            # finalize
            supabase.table("notices_new_2").update({
                "status": "processed",
                "embedding_model": "all-MiniLM-L6-v2"
            }).eq("id", nid).execute()

            total_chunks = text_cnt + rows_cnt + multi_cnt
            print(f"✅ processed {fname} → text_chunks: {text_cnt}, row_chunks: {rows_cnt}, multi_chunks: {multi_cnt} (total {total_chunks})")
        except Exception as e:
            print("❌ failed:", fname, str(e))
            supabase.table("notices_new_2").update(
                {"status": "failed", "error_msg": str(e)}
            ).eq("id", nid).execute()

# ---------------- backfill existing processed ----------------
def rechunk_all_processed(batch_size: int = 50):
    offset = 0
    while True:
        res = supabase.table("notices_new_2").select(
            "id, filename, ocr_text, uploaded_at, notice_title"
        ).eq("status", "processed").range(offset, offset + batch_size - 1).execute()
        rows = res.data or []
        if not rows:
            break
        for r in rows:
            nid = r["id"]
            # skip if chunks exist
            chk = supabase.table("notice_chunks_new_2").select("id").eq("notice_id", nid).limit(1).execute()
            if chk.data:
                print("Skipping:", nid)
                continue
            text = r.get("ocr_text", "") or ""
            fname = r.get("filename") or nid
            title = r.get("notice_title")
            cnt = create_text_chunk_embeddings_and_store(
                notice_id=nid,
                filename=fname,
                text=text,
                notice_title=title,
                uploaded_at=r.get("uploaded_at")
            )
            print("Backfilled text chunks", cnt, "for", nid)
        offset += batch_size

if __name__ == "__main__":
    process_pending(limit=15)
