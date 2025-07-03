# main.py
import sys, os
# ensure src/ is discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pipeline_runner import run_pipeline

if __name__ == "__main__":
    print("🔔 [Main] Launching pipeline…")
    run_pipeline()
    print("🔔 [Main] Done.")
