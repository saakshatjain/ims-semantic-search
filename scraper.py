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
    res = supabase.table("notices_new").select("id").eq("id", notice_id).execute()
    return len(res.data) > 0

def clean_notice_title(anchor_text: str | None) -> str | None:
    """Normalize whitespace etc. You can add more cleanup rules here if needed."""
    if not anchor_text:
        return None
    title = re.sub(r"\s+", " ", anchor_text).strip()
    return title

def save_to_supabase(file_bytes, url, notice_title: str | None):
    """Insert new notice PDF into Supabase with notice_title."""
    notice_id = hashlib.sha256(file_bytes).hexdigest()[:32]  # 32-char hash

    if notice_exists(notice_id):
        print(f"⚠️ Already exists in Supabase: {url}")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = notice_id[:8]
    filename = f"notice_{timestamp}_{short_hash}.pdf"
    path_in_bucket = f"notices_new/{filename}"

    # Upload to storage
    supabase.storage.from_("notices_new").upload(path_in_bucket, file_bytes)

    # Insert metadata in DB (only notice_title + existing fields)
    supabase.table("notices_new").insert({
        "id": notice_id,
        "url": url,
        "filename": filename,
        "file_path": path_in_bucket,
        "status": "pending",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "notice_title": notice_title,
    }).execute()

    print(f"✅ Stored in Supabase + table: {filename}")
    print(f"   → notice_title: {notice_title}")
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

    # 1) Current notices
    resp = session.get(NOTICES, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    raw_entries = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True) or None
        raw_entries.append((href, text))

    # 2) Archive / old notices – replicate POST
    archive_payload = {
        "branch": "All",
        "olddata": "Archive: Click to View Old Notices / Circulars",
    }
    resp_arch = session.post(
        NOTICES,
        data=archive_payload,
        timeout=60,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp_arch.raise_for_status()
    soup_arch = BeautifulSoup(resp_arch.text, "html.parser")

    for a in soup_arch.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True) or None
        raw_entries.append((href, text))

    # Deduplicate while keeping anchor text
    seen = set()
    links: list[tuple[str, str | None]] = []
    for href, text in raw_entries:
        url = href if href.startswith("http") else f"{BASE}/{href}"
        if url not in seen:
            seen.add(url)
            links.append((url, text))

    # Optional: limit to 20
    links = links[:20]

    new_inserted = False

    for i, (url, anchor_text) in enumerate(links, 1):
        url = url.replace("/view", "/edit")
        notice_title = clean_notice_title(anchor_text)

        print(f"\n[{i}] Processing: {url}")
        print(f"   anchor: {anchor_text}")
        print(f"   notice_title (cleaned): {notice_title}")

        try:
            # Google Sheets
            if "docs.google.com/spreadsheets" in url:
                sheet_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
                gid_match = re.search(r"gid=(\d+)", url)
                gid = gid_match.group(1) if gid_match else "0"
                pdf_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=pdf&gid={gid}"
                r = session.get(pdf_url, timeout=60)
                if r.headers.get("Content-Type", "").startswith("application/pdf"):
                    if save_to_supabase(r.content, url, notice_title):
                        new_inserted = True
                else:
                    print("⚠️ Google Sheets export didn't return PDF:", url)
                continue

            # Google Docs
            if "docs.google.com/document" in url:
                doc_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
                pdf_url = f"https://docs.google.com/document/d/{doc_id}/export?format=pdf"
                r = session.get(pdf_url, timeout=60)
                if r.headers.get("Content-Type", "").startswith("application/pdf"):
                    if save_to_supabase(r.content, url, notice_title):
                        new_inserted = True
                else:
                    print("⚠️ Google Docs export didn't return PDF:", url)
                continue

            # Google Drive file (generic)
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

                if content and content[:4] == b"%PDF":
                    if save_to_supabase(content, url, notice_title):
                        new_inserted = True
                    continue

                dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                r = session.get(dl_url, timeout=60)
                if r.headers.get("Content-Type", "").startswith("application/pdf"):
                    if save_to_supabase(r.content, url, notice_title):
                        new_inserted = True
                else:
                    print("⚠️ Drive link didn't return a PDF:", url)
                continue

            # Default IMSNSIT PDFs
            r = session.get(url, headers={"Referer": NOTICES}, timeout=30)
            if "application/pdf" in r.headers.get("Content-Type", "") or (r.content and r.content[:4] == b"%PDF"):
                if save_to_supabase(r.content, url, notice_title):
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
