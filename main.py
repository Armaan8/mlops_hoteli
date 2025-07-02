# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

print("🔄 Starting MLOps Pipeline...")
import pipeline_runner
print("✅ Pipeline Execution Completed.")
