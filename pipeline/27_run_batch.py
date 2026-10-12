"""
27_run_batch.py
===============
Step 3: Upload batch_input.jsonl to OpenAI Files API and submit the batch job.

Usage:
    export OPENAI_API_KEY=sk-...
    python 27_run_batch.py

Saves batch_id.txt so you can check status and retrieve results later
without re-running this script.
"""

import os
import json
from pathlib import Path
from openai import OpenAI

INPUT_FILE = Path("batch_input.jsonl")
BATCH_ID_FILE = Path("batch_id.txt")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Step 1: Upload the JSONL file ─────────────────────────────────────────────
print(f"Uploading {INPUT_FILE} ({INPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB)...")
with open(INPUT_FILE, "rb") as f:
    batch_file = client.files.create(file=f, purpose="batch")

print(f"File uploaded successfully.")
print(f"  File ID: {batch_file.id}")
print(f"  Status:  {batch_file.status}")

# ── Step 2: Create the batch job ──────────────────────────────────────────────
print("\nSubmitting batch job...")
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "experiment": "multipico_persona_prompting",
        "model":      "gpt-4.1-mini",
        "conditions": "p0,p1,p2,p3",
        "languages":  "ar,de,en,es,fr,hi,it,nl,pt",
        "n_requests": "25200",
    },
)

print(f"Batch submitted successfully.")
print(f"  Batch ID: {batch.id}")
print(f"  Status:   {batch.status}")
print(f"  Window:   {batch.completion_window}")

# ── Step 3: Save batch ID ─────────────────────────────────────────────────────
with open(BATCH_ID_FILE, "w") as f:
    f.write(batch.id)

# Also save full submission info for records
submission_log = {
    "batch_id":       batch.id,
    "file_id":        batch_file.id,
    "status":         batch.status,
    "n_requests":     25200,
    "completion_window": batch.completion_window,
}
with open("batch_submission.json", "w") as f:
    json.dump(submission_log, f, indent=2)

print(f"\nBatch ID saved to: {BATCH_ID_FILE}")
print("Submission log saved to: batch_submission.json")
print()
print("Next step: run  python 27b_retrieve_batch.py  to check status.")
print("The batch typically completes within 1–2 hours (up to 24h maximum).")
