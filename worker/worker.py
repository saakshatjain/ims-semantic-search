import os, io, tempfile, json
from datetime import datetime
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pytesseract
from PIL import Image
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# --- Setup Supabase ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Setup local embedding model ---
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------- #
#   PDF Fetch + Extraction  #
# ------------------------- #

def fetch_pdf(file_path: str) -> bytes:
    """Download PDF from Supabase storage"""
    res = supabase.storage.from_("notices").download(file_path)
    return res

def extract_text_and_tables(pdf_bytes: bytes):
    """Extract text + tables from PDF"""
    text_content = []
    table_data = []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        pdf_path = tmp.name

    # --- Extract embedded text ---
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_content.append(txt)

    # --- OCR fallback if no embedded text ---
    if not text_content:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            txt = pytesseract.image_to_string(img)
            if txt.strip():
                text_content.append(txt)

    # --- Extract tables ---
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        if not tables or tables.n == 0:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")

        for t in tables:
            table_data.append(t.df.values.tolist())
    except Exception as e:
        print("⚠️ Table extraction failed:", e)

    return "\n\n".join(text_content), table_data if table_data else None

# ------------------------- #
#     Embedding Utility     #
# ------------------------- #

def create_embedding(text: str, tables: list):
    """Generate embeddings from text + table content"""
    combined_content = text if text else ""

    # Flatten table content into text for semantic search
    if tables:
        table_texts = []
        for tbl in tables:
            rows = [" | ".join(row) for row in tbl]
            table_texts.append("\n".join(rows))
        combined_content += "\n\n" + "\n\n".join(table_texts)

    if not combined_content.strip():
        return None

    try:
        vector = embed_model.encode(combined_content, convert_to_numpy=True).tolist()
        return vector
    except Exception as e:
        print("⚠️ Embedding failed:", e)
        return None

# ------------------------- #
#     Worker Main Loop      #
# ------------------------- #

def process_pending():
    notices = supabase.table("notices").select("*").eq("status", "pending").limit(5).execute()
    for n in notices.data:
        print("📄 Processing:", n["filename"])

        # mark as processing
        supabase.table("notices").update({
            "status": "processing"
        }).eq("id", n["id"]).execute()

        try:
            pdf_bytes = fetch_pdf(n["file_path"])
            text, tables = extract_text_and_tables(pdf_bytes)

            embedding = create_embedding(text, tables)

            supabase.table("notices").update({
                "status": "processed",
                "ocr_text": text,
                "ocr_tables": json.dumps(tables) if tables else None,
                "embedding": embedding,
                "processed_at": datetime.utcnow().isoformat()
            }).eq("id", n["id"]).execute()

            print("✅ Processed:", n["filename"])
        except Exception as e:
            supabase.table("notices").update({
                "status": "failed",
                "error_msg": str(e)
            }).eq("id", n["id"]).execute()
            print("❌ Failed:", n["filename"], str(e))

if __name__ == "__main__":
    process_pending()
