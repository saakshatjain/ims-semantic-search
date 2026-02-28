import schedule
import time
import subprocess
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

def job():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Scraper and Worker pipeline...", flush=True)
    
    # 1. Run Scraper
    print(">>> Running scraper.py", flush=True)
    try:
        subprocess.run([sys.executable, "-u", "scraper.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Scraper failed with exit code: {e.returncode}", flush=True)
        return # Skip worker if scraper critically fails right away

    # 2. Run Worker
    print(">>> Running worker/worker.py", flush=True)
    try:
        subprocess.run([sys.executable, "-u", "worker/worker.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Worker failed with exit code: {e.returncode}", flush=True)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pipeline finished successfully.", flush=True)

# Run the job immediately on startup to catch up
job()

# Schedule the job to run every 15 minutes.
# This ensures notices are fetched quickly AND inherently acts as activity 
# to help keep the Hugging Face Space from freezing.
schedule.every(15).minutes.do(job)

class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Scheduler is running!")

def start_server():
    server = HTTPServer(('0.0.0.0', 7860), DummyHandler)
    print("Starting dummy server on port 7860 for health checks...", flush=True)
    server.serve_forever()

# Start the web server in a background thread
threading.Thread(target=start_server, daemon=True).start()

print("🚀 Background scheduler started! Running scraper & worker every 15 minutes.", flush=True)

while True:
    schedule.run_pending()
    time.sleep(60) # Wait one minute
