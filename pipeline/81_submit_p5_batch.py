"""
81_submit_p5_batch.py
=====================
Upload p5_batch_input.jsonl and submit the Batch API job for P5.

Usage:
    export OPENAI_API_KEY=sk-...
    python 81_submit_p5_batch.py
"""

import os, json
from pathlib import Path
from openai import OpenAI

INPUT_FILE    = Path("p5_batch_input.jsonl")
BATCH_ID_FILE = Path("p5_batch_id.txt")
LOG_FILE      = Path("p5_batch_submission.json")

if not INPUT_FILE.exists():
    print("ERROR: p5_batch_input.jsonl not found. Run 80_build_p5_batch.py first.")
    exit(1)

client  = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
n_lines = sum(1 for _ in open(INPUT_FILE, encoding="utf-8"))
size_kb = INPUT_FILE.stat().st_size / 1024

print(f"Uploading {INPUT_FILE}  ({size_kb:.1f} KB, {n_lines:,} requests) ...")
with open(INPUT_FILE, "rb") as f:
    bf = client.files.create(file=f, purpose="batch")
print(f"  File ID : {bf.id}")

print("\nSubmitting batch job ...")
batch = client.batches.create(
    input_file_id=bf.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "experiment": "multipico_p5_relative_cot",
        "model":      "gpt-4.1-mini",
        "condition":  "p5",
        "languages":  "ar,es",
        "format":     "signed_deviation_cot",
        "n_requests": str(n_lines),
    },
)
print(f"  Batch ID : {batch.id}")
print(f"  Status   : {batch.status}")

BATCH_ID_FILE.write_text(batch.id)
with open(LOG_FILE, "w") as f:
    json.dump({"batch_id": batch.id, "file_id": bf.id,
               "status": batch.status, "n_requests": n_lines}, f, indent=2)

print(f"\nBatch ID saved to: {BATCH_ID_FILE}")
print("Next step: python 82_retrieve_p5_batch.py")
