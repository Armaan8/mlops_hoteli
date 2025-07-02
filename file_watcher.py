import os
import time
import subprocess

WATCH_FILE = "data/new_booking.xlsx"
CHECK_INTERVAL = 5  # seconds

def get_last_modified_time(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0

def run_pipeline():
    print("🔁 Detected new booking. Running pipeline...")
    subprocess.run(["python", "main.py"])

if __name__ == "__main__":
    print(f"👀 Watching for changes in {WATCH_FILE} ...")
    last_mtime = get_last_modified_time(WATCH_FILE)

    while True:
        try:
            time.sleep(CHECK_INTERVAL)
            current_mtime = get_last_modified_time(WATCH_FILE)

            if current_mtime != last_mtime:
                last_mtime = current_mtime
                run_pipeline()
        except KeyboardInterrupt:
            print("🛑 File watcher stopped.")
            break
