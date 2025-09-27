import os, re, hashlib, requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
from datetime import datetime, timezone

# --- CONFIG ---
BASE = "https://www.imsnsit.org/imsnsit"
NOTICES = f"{BASE}/notifications.php"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- HELPERS ---
def notice_exists(notice_id: str) -> bool:
    res = supabase.table("notices").select("id").eq("id", notice_id).execute()
    return len(res.data) > 0

def save_to_supabase(file_bytes, url):
    """Insert new notice PDF into Supabase"""
    notice_id = hashlib.sha256(file_bytes).hexdigest()[:32]  # 32-char hash

    if notice_exists(notice_id):
        print(f"⚠️ Already exists in Supabase: {url}")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = notice_id[:8]
    filename = f"notice_{timestamp}_{short_hash}.pdf"
    path_in_bucket = f"notices/{filename}"

    # Upload to storage
    supabase.storage.from_("notices").upload(path_in_bucket, file_bytes)

    # Insert metadata in DB
    supabase.table("notices").insert({
        "id": notice_id,
        "url": url,
        "filename": filename,
        "status": "pending",
        "uploaded_at": datetime.now(timezone.utc).isoformat()
    }).execute()

    print(f"✅ Stored in Supabase + table: {filename}")
    return True

def _get_drive_file_id(url: str):
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def _get_confirm_token(resp):
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    m = re.search(r"confirm=([0-9A-Za-z_\-]+)", resp.text)
    if m:
        return m.group(1)
    return None


def _download_drive_file(session, file_id: str) -> bytes:
    URL = "https://docs.google.com/uc?export=download"
    resp = session.get(URL, params={"id": file_id}, stream=True, timeout=60)
    token = _get_confirm_token(resp)
    if token:
        resp = session.get(URL, params={"id": file_id, "confirm": token}, stream=True, timeout=60)
    resp.raise_for_status()
    return resp.content

# --- SCRAPER ---
def run_scraper():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    resp = session.get(NOTICES)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    raw_links = [a["href"] for a in soup.find_all("a", href=True)]

    links = []
    for href in raw_links:
        url = href if href.startswith("http") else f"{BASE}/{href}"
        if url not in links:
            links.append(url)

    links = links[:10]  # last 10 notices
    new_inserted = False  # flag to track new notices

    for i, url in enumerate(links, 1):
        url = url.replace("/view", "/edit")
        print(f"\n[{i}] Processing: {url}")

        try:
            # Google Sheets
            if "docs.google.com/spreadsheets" in url:
                sheet_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
                gid_match = re.search(r"gid=(\d+)", url)
                gid = gid_match.group(1) if gid_match else "0"
                pdf_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=pdf&gid={gid}"
                r = session.get(pdf_url)
                if r.headers.get("Content-Type", "").startswith("application/pdf"):
                    if save_to_supabase(r.content, url):
                        new_inserted = True
                continue

            # Google Docs
            if "docs.google.com/document" in url:
                doc_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
                pdf_url = f"https://docs.google.com/document/d/{doc_id}/export?format=pdf"
                r = session.get(pdf_url)
                if r.headers.get("Content-Type", "").startswith("application/pdf"):
                    if save_to_supabase(r.content, url):
                        new_inserted = True
                continue

            # Google Drive file
            if "drive.google.com" in url or "docs.google.com" in url:
                file_id = _get_drive_file_id(url)
                if not file_id:
                    print("⚠️ Could not extract Drive file id:", url)
                    continue
                try:
                    content = _download_drive_file(session, file_id)
                except Exception as e:
                    print("❌ Drive download failed:", url, e)
                    continue

                # basic validation: PDF header
                if content and content[:4] == b"%PDF":
                    if save_to_supabase(content, url):
                        new_inserted = True
                    continue

                # fallback: try direct uc download and check headers
                dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                r = session.get(dl_url)
                if r.headers.get("Content-Type", "").startswith("application/pdf"):
                    if save_to_supabase(r.content, url):
                        new_inserted = True
                else:
                    print("⚠️ Drive link didn't return a PDF:", url)
                continue

            # Default IMSNSIT PDFs
            r = session.get(url, headers={"Referer": NOTICES})
            if "application/pdf" in r.headers.get("Content-Type", ""):
                if save_to_supabase(r.content, url):
                    new_inserted = True
            else:
                print("⚠️ Not a PDF:", url)
        except Exception as e:
            print("❌ Failed:", url, e)

    return new_inserted

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    added_new = run_scraper()

    if added_new:
        print("✅ New notices added. Running OCR worker...")
        os.system("python worker/worker.py")  # OCR extraction

        print("✅ OCR done. Running processor/embedding...")
        os.system("python worker/processor.py")  # embeddings
    else:
        print("⚠️ No new notices. Skipping OCR and embeddings.")
