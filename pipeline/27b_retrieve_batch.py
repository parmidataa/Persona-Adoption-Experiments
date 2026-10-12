"""
27b_retrieve_batch.py
=====================
Step 3b: Check batch status and download results when complete.

Usage:
    export OPENAI_API_KEY=sk-...
    python 27b_retrieve_batch.py

Run this repeatedly to check status. When status == "completed",
it downloads batch_output.jsonl (and batch_errors.jsonl if there are failures).
"""

import os
import json
from pathlib import Path
from openai import OpenAI

BATCH_ID_FILE  = Path("batch_id.txt")
OUTPUT_FILE    = Path("batch_output.jsonl")
ERRORS_FILE    = Path("batch_errors.jsonl")

if not BATCH_ID_FILE.exists():
    print("ERROR: batch_id.txt not found. Run 27_run_batch.py first.")
    exit(1)

client   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
batch_id = BATCH_ID_FILE.read_text().strip()

# ── Check status ──────────────────────────────────────────────────────────────
batch = client.batches.retrieve(batch_id)

print(f"Batch ID:  {batch.id}")
print(f"Status:    {batch.status}")
print(f"Completed: {batch.request_counts.completed:,}")
print(f"Failed:    {batch.request_counts.failed:,}")
print(f"Total:     {batch.request_counts.total:,}")

if batch.request_counts.total > 0:
    pct = batch.request_counts.completed / batch.request_counts.total * 100
    print(f"Progress:  {pct:.1f}%")

# ── Download results if complete ──────────────────────────────────────────────
if batch.status == "completed":
    print(f"\nBatch completed! Downloading results...")

    content = client.files.content(batch.output_file_id).read()
    with open(OUTPUT_FILE, "wb") as f:
        f.write(content)
    n_lines = content.count(b"\n")
    print(f"Results saved: {OUTPUT_FILE}  ({n_lines:,} lines, {len(content)/1024/1024:.1f} MB)")

    if batch.error_file_id:
        errors = client.files.content(batch.error_file_id).read()
        with open(ERRORS_FILE, "wb") as f:
            f.write(errors)
        n_errs = errors.count(b"\n")
        print(f"Errors saved:  {ERRORS_FILE}  ({n_errs:,} lines)")
    else:
        print("No error file — all requests succeeded.")

    print("\nNext step: run  python 28_parse_results.py")

elif batch.status in ("failed", "expired", "cancelled"):
    print(f"\nBatch did not complete successfully: {batch.status}")
    print("Check batch_submission.json and the OpenAI dashboard for details.")

else:
    print(f"\nStill processing ({batch.status}). Re-run this script to check again.")
    print("Typical completion time: 1–2 hours.")
