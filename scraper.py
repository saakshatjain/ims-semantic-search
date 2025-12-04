import os, re, hashlib, requests
import sys
from bs4 import BeautifulSoup
from supabase import create_client, Client
from datetime import datetime, timezone

# --- CONFIG ---
BASE = "https://www.imsnsit.org/imsnsit"
NOTICES = f"{BASE}/notifications.php"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "notices_new"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: SUPABASE_URL and SUPABASE_KEY environment variables are missing.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- HELPERS ---
def ensure_bucket_exists():
    """Checks if the storage bucket exists, creates it if not."""
    try:
        buckets = supabase.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        if BUCKET_NAME not in bucket_names:
            print(f"⚠️ Bucket '{BUCKET_NAME}' not found. Attempting to create...")
            supabase.storage.create_bucket(BUCKET_NAME, options={"public": False})
            print(f"✅ Bucket '{BUCKET_NAME}' created.")
        else:
            print(f"✅ Bucket '{BUCKET_NAME}' exists.")
    except Exception as e:
        print(f"⚠️ Warning: Could not verify bucket existence (Permissions issue?): {e}")

def notice_exists(notice_id: str) -> bool:
    try:
        res = supabase.table("notices_new").select("id").eq("id", notice_id).execute()
        return len(res.data) > 0
    except Exception as e:
        print(f"❌ DB Check Error: {e}")
        return False

def clean_notice_title(anchor_text: str | None) -> str | None:
    if not anchor_text:
        return None
    title = re.sub(r"\s+", " ", anchor_text).strip()
    return title

def save_to_supabase(file_bytes, url, notice_title: str | None):
    notice_id = hashlib.sha256(file_bytes).hexdigest()[:32]

    if notice_exists(notice_id):
        print(f"⚠️ Already exists in DB: {url}")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = notice_id[:8]
    filename = f"notice_{timestamp}_{short_hash}.pdf"
    
    # Upload to storage
    try:
        supabase.storage.from_(BUCKET_NAME).upload(filename, file_bytes)
    except Exception as e:
        print(f"❌ Storage Upload Failed for {filename}: {e}")
        # If upload fails, do NOT insert into DB, otherwise Worker will fail with 'Object not found'
        return False

    # Insert metadata in DB
    try:
        supabase.table("notices_new").insert({
            "id": notice_id,
            "url": url,
            "filename": filename,
            "status": "pending",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "notice_title": notice_title,
        }).execute()
    except Exception as e:
        print(f"❌ DB Insert Failed: {e}")
        return False

    print(f"✅ Stored: {filename}")
    return True

# [The rest of your scraper logic remains the same, skipping to _download_drive_file for brevity]
def _get_drive_file_id(url: str):
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m: return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m: return m.group(1)
    return None

def _get_confirm_token(resp):
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    m = re.search(r"confirm=([0-9A-Za-z_\-]+)", resp.text)
    if m: return m.group(1)
    return None

def _download_drive_file(session, file_id: str) -> bytes:
    URL = "https://docs.google.com/uc?export=download"
    resp = session.get(URL, params={"id": file_id}, stream=True, timeout=60)
    token = _get_confirm_token(resp)
    if token:
        resp = session.get(URL, params={"id": file_id, "confirm": token}, stream=True, timeout=60)
    resp.raise_for_status()
    return resp.content

def run_scraper():
    # 1. Ensure Bucket exists before starting
    ensure_bucket_exists()
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    try:
        resp = session.get(NOTICES, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"❌ Failed to fetch main page: {e}")
        return False

    raw_entries = []
    for a in soup.find_all("a", href=True):
        raw_entries.append((a["href"], a.get_text(strip=True) or None))

    # Archive notices
    try:
        resp_arch = session.post(
            NOTICES,
            data={"branch": "All", "olddata": "Archive: Click to View Old Notices / Circulars"},
            timeout=60,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp_arch.raise_for_status()
        soup_arch = BeautifulSoup(resp_arch.text, "html.parser")
        for a in soup_arch.find_all("a", href=True):
            raw_entries.append((a["href"], a.get_text(strip=True) or None))
    except Exception as e:
        print(f"⚠️ Failed to fetch archive (continuing with recent only): {e}")

    seen = set()
    links = []
    for href, text in raw_entries:
        url = href if href.startswith("http") else f"{BASE}/{href}"
        if url not in seen:
            seen.add(url)
            links.append((url, text))

    links = links[:200]
    new_inserted = False

    for i, (url, anchor_text) in enumerate(links, 1):
        url = url.replace("/view", "/edit")
        notice_title = clean_notice_title(anchor_text)
        
        # Skip logic if needed
        print(f"[{i}] Checking: {notice_title[:30]}...")

        try:
            content = None
            
            # 1. Google Sheets
            if "docs.google.com/spreadsheets" in url:
                sheet_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
                gid_match = re.search(r"gid=(\d+)", url)
                gid = gid_match.group(1) if gid_match else "0"
                pdf_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=pdf&gid={gid}"
                r = session.get(pdf_url, timeout=60)
                if r.headers.get("Content-Type", "").startswith("application/pdf"):
                    content = r.content

            # 2. Google Docs
            elif "docs.google.com/document" in url:
                doc_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
                pdf_url = f"https://docs.google.com/document/d/{doc_id}/export?format=pdf"
                r = session.get(pdf_url, timeout=60)
                if r.headers.get("Content-Type", "").startswith("application/pdf"):
                    content = r.content

            # 3. Google Drive
            elif "drive.google.com" in url or "docs.google.com" in url:
                file_id = _get_drive_file_id(url)
                if file_id:
                    try:
                        c = _download_drive_file(session, file_id)
                        if c and c[:4] == b"%PDF":
                            content = c
                        else:
                            # Try export link
                            dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                            r = session.get(dl_url, timeout=60)
                            if r.headers.get("Content-Type", "").startswith("application/pdf"):
                                content = r.content
                    except:
                        pass

            # 4. Standard Link
            else:
                r = session.get(url, headers={"Referer": NOTICES}, timeout=30)
                if "application/pdf" in r.headers.get("Content-Type", "") or (r.content and r.content[:4] == b"%PDF"):
                    content = r.content

            # Save if we got content
            if content:
                if save_to_supabase(content, url, notice_title):
                    new_inserted = True
            
        except Exception as e:
            print(f"❌ Error processing {url}: {e}")

    return new_inserted

if __name__ == "__main__":
    added_new = run_scraper()

    if added_new:
        print("✅ New notices added. Triggering worker...")
        
        # Check if worker file exists to avoid confusing errors
        if os.path.exists("worker/worker.py"):
            exit_code = os.system("python worker/worker.py")
            if exit_code != 0:
                print("❌ Worker failed.")
        else:
            print("⚠️ worker/worker.py not found. Skipping OCR.")

        # You did not share processor.py, so I added a check here too
        if os.path.exists("worker/processor.py"):
             os.system("python worker/processor.py")
    else:
        print("⚠️ No new notices found.")