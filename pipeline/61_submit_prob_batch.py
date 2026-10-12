"""
61_submit_prob_batch.py
=======================
Upload prob_batch_input.jsonl and submit the Batch API job.

Usage:
    export OPENAI_API_KEY=sk-...
    python 61_submit_prob_batch.py

Saves prob_batch_id.txt for retrieval.
"""

import os
import json
from pathlib import Path
from openai import OpenAI

INPUT_FILE    = Path("prob_batch_input.jsonl")
BATCH_ID_FILE = Path("prob_batch_id.txt")

if not INPUT_FILE.exists():
    print("ERROR: prob_batch_input.jsonl not found. Run 60_build_prob_batch.py first.")
    exit(1)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

print(f"Uploading {INPUT_FILE} ({INPUT_FILE.stat().st_size / 1024:.1f} KB)...")
with open(INPUT_FILE, "rb") as f:
    batch_file = client.files.create(file=f, purpose="batch")
print(f"  File ID: {batch_file.id}")

n_lines = sum(1 for _ in open(INPUT_FILE, encoding="utf-8"))
print(f"  Requests: {n_lines:,}")

print("\nSubmitting batch job...")
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "experiment": "multipico_prob_elicitation",
        "model":      "gpt-4.1-mini",
        "conditions": "p0,p1,p2,p3",
        "languages":  "ar,es",
        "n_requests": str(n_lines),
        "format":     "probability_0_1",
    },
)

print(f"  Batch ID: {batch.id}")
print(f"  Status:   {batch.status}")
print(f"  Window:   {batch.completion_window}")

BATCH_ID_FILE.write_text(batch.id)
with open("prob_batch_submission.json", "w") as f:
    json.dump({
        "batch_id":  batch.id,
        "file_id":   batch_file.id,
        "status":    batch.status,
        "n_requests": n_lines,
    }, f, indent=2)

print(f"\nBatch ID saved to: {BATCH_ID_FILE}")
print("Next step: python 62_retrieve_prob_batch.py  (check status / download results)")
