import re
import datetime
import numpy as np
import os
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# --- Supabase Client ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Embedding Model (local) ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clean_text(text: str) -> str:
    """Normalize text: remove extra spaces, line breaks, junk chars."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    text = re.sub(r"[^\x20-\x7E]", "", text)   # remove non-printable chars
    return text.strip()

# New helper: split into sentences (lightweight, dependency-free)
def split_sentences(text: str):
    # split on sentence end punctuation while keeping short abbreviations reasonably ok
    sents = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    processed = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        # if sentence is excessively long, split on commas preserving fragments
        if len(s) > 1200:
            parts = re.split(r',\s+', s)
            for p in parts:
                if p:
                    processed.append(p.strip())
        else:
            processed.append(s)
    return processed

# New helper: cosine similarity
def cosine_sim(a: np.ndarray, b: np.ndarray):
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))

# New helper: MMR selection to pick non-redundant important sentences
def mmr_select(sentence_embs: np.ndarray, doc_emb: np.ndarray, top_k: int = 10, diversity: float = 0.7):
    # sentence_embs: (n_sentences, dim)
    n = sentence_embs.shape[0]
    if n == 0:
        return []
    sim_to_doc = np.dot(sentence_embs, doc_emb / (np.linalg.norm(doc_emb) + 1e-10))
    selected = []
    remaining = set(range(n))

    # pick the highest scoring to start
    first = int(np.argmax(sim_to_doc))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < min(top_k, n) and remaining:
        best = None
        best_score = -1e9
        for idx in list(remaining):
            # relevance
            relevance = sim_to_doc[idx]
            # redundancy: max similarity to any selected
            red = 0.0
            for s in selected:
                red = max(
                    red,
                    np.dot(sentence_embs[idx], sentence_embs[s]) /
                    ((np.linalg.norm(sentence_embs[idx]) + 1e-10) *
                     (np.linalg.norm(sentence_embs[s]) + 1e-10))
                )
            score = (1 - diversity) * relevance - diversity * red
            if score > best_score:
                best_score = score
                best = idx
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)

    return selected

# Modified build_chunks -> returns (chunks, used_flags)
def build_chunks(sentences, important_indices, min_chars=200, max_chars=1200):
    n = len(sentences)
    # sort important indices to process in document order
    important_indices = sorted(set(important_indices))
    chunks = []
    used = [False] * n

    for imp_idx in important_indices:
        # if this sentence is already included in a previous chunk, skip
        if used[imp_idx]:
            continue
        left = imp_idx
        right = imp_idx
        current_text = sentences[imp_idx]
        # expand alternatingly left and right while under min_chars and within bounds
        while len(current_text) < min_chars:
            can_expand_left = left - 1 >= 0 and not used[left - 1]
            can_expand_right = right + 1 < n and not used[right + 1]
            # choose side by which side yields nearer context (prefer right for forward context)
            if can_expand_right:
                right += 1
                current_text = " ".join(sentences[left:right+1])
            elif can_expand_left:
                left -= 1
                current_text = " ".join(sentences[left:right+1])
            else:
                break
            if len(current_text) >= max_chars:
                # if exceeded max, try to trim by shifting right back or left back
                if right > imp_idx:
                    right -= 1
                    current_text = " ".join(sentences[left:right+1])
                    break
                if left < imp_idx:
                    left += 1
                    current_text = " ".join(sentences[left:right+1])
                    break
        # mark used
        for i in range(left, right+1):
            used[i] = True
        chunks.append({
            'left': left,
            'right': right,
            'text': " ".join(sentences[left:right+1]),
            'is_summary': True
        })
    # as a fallback, if no chunks (e.g., very short doc), create one chunk of the whole text
    if not chunks and sentences:
        full = " ".join(sentences)
        chunks.append({'left': 0, 'right': n-1, 'text': full, 'is_summary': True})
        for i in range(0, n):
            used[i] = True
    return chunks, used

# New helper: map sentences to character offsets in cleaned text
def sentence_char_ranges(cleaned_text: str, sentences):
    ranges = []
    pos = 0
    for s in sentences:
        # find next occurrence of s starting at pos
        idx = cleaned_text.find(s, pos)
        if idx == -1:
            # fallback: try relaxed search (strip) or approximate by advancing pos
            idx = cleaned_text.find(s.strip(), pos)
        if idx == -1:
            # can't find exact match: mark -1 and attempt to set approximate positions
            start = -1
            end = -1
        else:
            start = idx
            end = idx + len(s)
            pos = end
        ranges.append((start, end))
    return ranges

# New helper: create leftover chunks covering unused sentence spans
def create_leftover_chunks(sentences, used_flags, sentence_ranges, min_chars=100, max_chars=1200):
    n = len(sentences)
    leftover_chunks = []
    i = 0
    while i < n:
        if used_flags[i]:
            i += 1
            continue
        j = i
        current_text_parts = []
        while j < n and not used_flags[j]:
            current_text_parts.append(sentences[j])
            j += 1
            # if we exceed max_chars, break to keep chunks bounded
            if sum(len(p) for p in current_text_parts) >= max_chars:
                break
        chunk_text = " ".join(current_text_parts).strip()
        if len(chunk_text) < min_chars and j < n:
            # try to extend into next used sentences if that helps (rare)
            pass
        # compute char range from sentence_ranges if available
        start_char = sentence_ranges[i][0] if i < len(sentence_ranges) else -1
        end_char = sentence_ranges[j-1][1] if (j-1) < len(sentence_ranges) else -1
        leftover_chunks.append({
            'left': i,
            'right': j-1,
            'text': chunk_text,
            'start_char': start_char,
            'end_char': end_char,
            'is_summary': False
        })
        # mark them used
        for k in range(i, j):
            used_flags[k] = True
        i = j
    return leftover_chunks

def process_pending_notices():
    # fetch notices with OCR done but not embedded
    response = supabase.table("notices") \
        .select("id, ocr_text, notice_title") \
        .eq("status", "processing") \
        .is_("clean_text", "null") \
        .execute()

    for record in response.data:
        notice_id = record["id"]
        ocr_text = record["ocr_text"]
        notice_title = record.get("notice_title")

        try:
            # Step 1: Clean text
            cleaned = clean_text(ocr_text)

            # Step 2: Split into sentences
            sentences = split_sentences(cleaned)

            # Step 2.1: compute char ranges for sentences (used later for precise chunk offsets)
            sentence_ranges = sentence_char_ranges(cleaned, sentences)

            # Step 3: Compute embeddings (document + sentences)
            doc_emb = model.encode(cleaned).astype(np.float32)
            if sentences:
                sent_embs = model.encode(sentences)
                sent_embs = np.asarray(sent_embs, dtype=np.float32)
            else:
                sent_embs = np.zeros((0, doc_emb.shape[0]), dtype=np.float32)

            # Step 4: Select important sentences via MMR
            top_k = min(10, max(1, len(sentences)))
            selected_indices = mmr_select(sent_embs, doc_emb, top_k=top_k, diversity=0.7) if len(sentences) > 0 else []

            # Compose important_text
            selected_indices_sorted = sorted(selected_indices)
            important_sentences = [sentences[i] for i in selected_indices_sorted]
            important_text = " ".join(important_sentences).strip()

            # Step 5: Build semantic chunks (summary chunks) and get used flags
            summary_chunks, used_flags = build_chunks(
                sentences,
                selected_indices,
                min_chars=250,
                max_chars=1200
            )

            # Step 5b: Build leftover chunks to cover any unused sentences (guarantee no sentence omitted)
            leftover_chunks = []
            if sentences:
                leftover_chunks = create_leftover_chunks(
                    sentences,
                    used_flags,
                    sentence_ranges,
                    min_chars=80,
                    max_chars=1200
                )

            # Combine chunks (summary first, then leftovers)
            all_chunks = []
            # enrich summary_chunks with char ranges
            for c in summary_chunks:
                left = c['left']
                right = c['right']
                start_char = sentence_ranges[left][0] if left < len(sentence_ranges) else -1
                end_char = sentence_ranges[right][1] if right < len(sentence_ranges) else -1
                all_chunks.append({
                    'left': left,
                    'right': right,
                    'text': c['text'],
                    'start_char': start_char,
                    'end_char': end_char,
                    'is_summary': True
                })
            # append leftovers
            all_chunks.extend(leftover_chunks)

            # If still empty (rare), insert full document as single chunk
            if not all_chunks and cleaned:
                all_chunks = [{
                    'left': 0,
                    'right': len(sentences)-1 if sentences else 0,
                    'text': cleaned,
                    'start_char': 0,
                    'end_char': len(cleaned),
                    'is_summary': False
                }]

            # Step 6: Insert chunks (idempotent)
            try:
                supabase.table("notice_chunks").delete().eq("notice_id", notice_id).execute()
            except Exception:
                pass

            for idx, chunk in enumerate(all_chunks):
                chunk_text = chunk['text']
                if not chunk_text:
                    continue
                chunk_emb = model.encode(chunk_text).astype(np.float32).tolist()
                row = {
                    "notice_id": notice_id,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "chunk_embedding": chunk_emb,
                    "left_sentence": chunk.get('left'),
                    "right_sentence": chunk.get('right'),
                    "start_char": chunk.get('start_char', -1),
                    "end_char": chunk.get('end_char', -1),
                    "is_summary": chunk.get('is_summary', False),
                    "notice_title": notice_title,
                    "created_at": datetime.datetime.utcnow().isoformat()
                }
                supabase.table("notice_chunks").insert(row).execute()

            # Step 7: Update notice row with summary & metadata
            supabase.table("notices").update({
                "clean_text": cleaned,
                "embedding": doc_emb.tolist(),
                "embedding_model": "all-MiniLM-L6-v2",
                "important_text": important_text,
                "chunk_count": len(all_chunks),
                "processed_at": datetime.datetime.utcnow().isoformat(),
                "status": "processed"
            }).eq("id", notice_id).execute()

        except Exception as e:
            supabase.table("notices").update({
                "status": "failed",
                "error_msg": str(e)
            }).eq("id", notice_id).execute()

if __name__ == "__main__":
    process_pending_notices()
