import os, io, tempfile
from datetime import datetime
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pytesseract
from PIL import Image
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_pdf(file_path: str) -> bytes:
    """Download PDF from Supabase storage"""
    res = supabase.storage.from_("notices").download(file_path)
    return res.read()

def extract_text_and_tables(pdf_bytes: bytes):
    text_content = []
    table_data = []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        pdf_path = tmp.name

    # --- Extract text (if embedded) ---
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_content.append(txt)

    # --- If no text → OCR fallback ---
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

            supabase.table("notices").update({
                "status": "processed",
                "ocr_text": text,
                "ocr_tables": tables,
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
