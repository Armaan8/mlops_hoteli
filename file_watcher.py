# file_watcher.py
import os
import time
import subprocess

WATCH_FILE = "data/new_booking.xlsx"
CHECK_INTERVAL = 5  # seconds

def get_mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0

def run_pipeline():
    print("🔁 New booking detected. Running pipeline...")
    # Use the venv Python if inside Docker it'll be default
    subprocess.run(["python", "main.py"], check=True)

if __name__ == "__main__":
    print(f"👀 Watching {WATCH_FILE} for changes...")
    last = get_mtime(WATCH_FILE)
    while True:
        try:
            time.sleep(CHECK_INTERVAL)
            curr = get_mtime(WATCH_FILE)
            if curr != last:
                last = curr
                run_pipeline()
        except KeyboardInterrupt:
            print("🛑 Watcher stopped.")
            break
