import schedule
import time
import subprocess
import os
import sys

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

# Schedule the job to run between 9:30 AM and 5:30 PM IST (which is roughly every hour)
# If you want it to mirror the previous cron ('0 4-12 * * *' in UTC), you can just run every hour:
schedule.every(1).hours.do(job)

print("🚀 Background scheduler started! Running scraper & worker every hour.", flush=True)

while True:
    schedule.run_pending()
    time.sleep(60) # Wait one minute
