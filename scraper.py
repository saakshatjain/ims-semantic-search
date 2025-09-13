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
def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def notice_exists(notice_id: str) -> bool:
    res = supabase.table("notices").select("id").eq("id", notice_id).execute()
    return len(res.data) > 0


def save_to_supabase(file_bytes, filename, url):
    notice_id = sha256_of_bytes(file_bytes)

    # Check if already exists
    if notice_exists(notice_id):
        print(f"⚠️ Already exists in Supabase: {filename}")
        return False

    # 1. Upload file to storage
    path_in_bucket = f"notices/{filename}"
    supabase.storage.from_("notices").upload(path_in_bucket, file_bytes)

    # 2. Insert metadata row (aligned with schema)
    supabase.table("notices").insert({
        "id": notice_id,
        "url": url,
        "filename": filename,
        "status": "pending",                       # for worker
        "uploaded_at": datetime.now(timezone.utc).isoformat()
    }).execute()

    print(f"✅ Stored in Supabase + table: {filename}")
    return True

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

    # Take last 10 notices only
    links = links[-10:]

    for i, url in enumerate(links, 1):
        url = url.replace("/view", "/edit")
        print(f"\n[{i}] Processing: {url}")

        # Google Docs/Sheets/Drive export handling
        if "docs.google.com/spreadsheets" in url:
            sheet_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
            gid_match = re.search(r"gid=(\d+)", url)
            gid = gid_match.group(1) if gid_match else "0"
            pdf_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=pdf&gid={gid}"
            r = session.get(pdf_url)
            if r.headers.get("Content-Type", "").startswith("application/pdf"):
                save_to_supabase(r.content, f"notice_{i:02d}.pdf", url)
            continue

        if "docs.google.com/document" in url:
            doc_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
            pdf_url = f"https://docs.google.com/document/d/{doc_id}/export?format=pdf"
            r = session.get(pdf_url)
            if r.headers.get("Content-Type", "").startswith("application/pdf"):
                save_to_supabase(r.content, f"notice_{i:02d}.pdf", url)
            continue

        if "drive.google.com/file" in url:
            file_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
            dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            r = session.get(dl_url)
            if r.headers.get("Content-Type", "").startswith("application/pdf"):
                save_to_supabase(r.content, f"notice_{i:02d}.pdf", url)
            continue

        # Default IMSNSIT PDFs
        r = session.get(url, headers={"Referer": NOTICES})
        if "application/pdf" in r.headers.get("Content-Type", ""):
            save_to_supabase(r.content, f"notice_{i:02d}.pdf", url)
        else:
            print("⚠️ Not a PDF:", url)

if __name__ == "__main__":
    run_scraper()
