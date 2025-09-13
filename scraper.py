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
    # 1️⃣ generate a unique ID for DB
    notice_id = hashlib.sha256(file_bytes).hexdigest()[:32]  # 32-char hash

    if notice_exists(notice_id):
        print(f"⚠️ Already exists in Supabase: {url}")
        return False

    # 2️⃣ generate unique filename using timestamp + hash
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = notice_id[:8]
    filename = f"notice_{timestamp}_{short_hash}.pdf"

    path_in_bucket = f"notices/{filename}"

    # 3️⃣ upload to storage (no upsert)
    supabase.storage.from_("notices").upload(
        path_in_bucket,
        file_bytes
    )

    # 4️⃣ insert metadata in DB
    supabase.table("notices").insert({
        "id": notice_id,
        "url": url,
        "filename": filename,
        "status": "pending",
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
    links = links[:10]

    for i, url in enumerate(links, 1):
        url = url.replace("/view", "/edit")
        print(f"\n[{i}] Processing: {url}")

        try:
            # Google Docs/Sheets/Drive export handling
            if "docs.google.com/spreadsheets" in url:
                sheet_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
                gid_match = re.search(r"gid=(\d+)", url)
                gid = gid_match.group(1) if gid_match else "0"
                pdf_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=pdf&gid={gid}"
                r = session.get(pdf_url)
                if r.headers.get("Content-Type", "").startswith("application/pdf"):
                    save_to_supabase(r.content, url)
                continue

            if "docs.google.com/document" in url:
                doc_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
                pdf_url = f"https://docs.google.com/document/d/{doc_id}/export?format=pdf"
                r = session.get(pdf_url)
                if r.headers.get("Content-Type", "").startswith("application/pdf"):
                    save_to_supabase(r.content, url)
                continue

            if "drive.google.com/file" in url:
                file_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
                dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                r = session.get(dl_url)
                if r.headers.get("Content-Type", "").startswith("application/pdf"):
                    save_to_supabase(r.content, url)
                continue

            # Default IMSNSIT PDFs
            r = session.get(url, headers={"Referer": NOTICES})
            if "application/pdf" in r.headers.get("Content-Type", ""):
                save_to_supabase(r.content, url)
            else:
                print("⚠️ Not a PDF:", url)
        except Exception as e:
            print("❌ Failed:", url, e)

if __name__ == "__main__":
    run_scraper()
