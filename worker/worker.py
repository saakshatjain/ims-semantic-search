import os
import sys
from paddleocr import PaddleOCR
import pdfplumber
import textract
from PIL import Image

# Remove utils import, merge relevant functions here

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if not text.strip():
            # If no text found, try OCR
            text = extract_text_from_pdf_with_ocr(pdf_path)
    except Exception as e:
        text = extract_text_from_pdf_with_ocr(pdf_path)
    return text

def extract_text_from_pdf_with_ocr(pdf_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    images = pdf_to_images(pdf_path)
    text = ""
    for img in images:
        result = ocr.ocr(img, cls=True)
        for line in result:
            for word_info in line:
                text += word_info[1][0] + " "
    return text

def pdf_to_images(pdf_path):
    from pdf2image import convert_from_path
    images = convert_from_path(pdf_path)
    return images

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
    """Extract text from PDF (no table extraction)"""
    text_content = []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        pdf_path = tmp.name

    # --- Extract embedded text ---
    text = extract_text_from_pdf(pdf_path)
    if text:
        text_content.append(text)

    # --- OCR fallback if no embedded text ---
    if not text_content:
        ocr_text = ""
        try:
            ocr_text = textract.process(pdf_path, method='tesseract').decode('utf-8')
            print("Textract (tesseract) output:", repr(ocr_text[:200]))
        except Exception as e:
            print("⚠️ Textract tesseract failed:", e)
        if ocr_text.strip():
            text_content.append(ocr_text)

    return "\n\n".join(text_content), None

# ------------------------- #
#     Embedding Utility     #
# ------------------------- #

def create_embedding(text: str, tables: list):
    """Generate embeddings from text content only"""
    combined_content = text if text else ""
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
