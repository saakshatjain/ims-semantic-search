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

def process_pending_notices():
    # fetch notices with OCR done but not embedded
    response = supabase.table("notices") \
        .select("id, ocr_text") \
        .eq("status", "processing") \
        .is_("clean_text", "null") \
        .execute()

    for record in response.data:
        notice_id = record["id"]
        ocr_text = record["ocr_text"]

        try:
            # Step 1: Clean text
            cleaned = clean_text(ocr_text)

            # Step 2: Generate embedding
            embedding_vector = model.encode(cleaned)
            embedding_vector = embedding_vector.astype(np.float32).tolist()

            # Step 3: Update row in Supabase
            supabase.table("notices").update({
                "clean_text": cleaned,
                "embedding": embedding_vector,
                "embedding_model": "all-MiniLM-L6-v2",
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
