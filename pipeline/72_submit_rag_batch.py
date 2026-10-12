"""
72_submit_rag_batch.py
======================
Upload rag_batch_input.jsonl and submit the OpenAI Batch API job
for the RAG experiment (P4 condition).

Saves rag_batch_id.txt for retrieval in 73_retrieve_rag_batch.py.

Usage:
    export OPENAI_API_KEY=sk-...
    python 72_submit_rag_batch.py
    (Run AFTER 71_build_rag_batch.py)
"""

import os
import json
from pathlib import Path
from openai import OpenAI

INPUT_FILE    = Path("rag_batch_input.jsonl")
BATCH_ID_FILE = Path("rag_batch_id.txt")
SUBMISSION_LOG = Path("rag_batch_submission.json")

if not INPUT_FILE.exists():
    print("ERROR: rag_batch_input.jsonl not found. Run 71_build_rag_batch.py first.")
    exit(1)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

n_lines = sum(1 for _ in open(INPUT_FILE, encoding="utf-8"))
size_kb = INPUT_FILE.stat().st_size / 1024

print(f"Uploading {INPUT_FILE}  ({size_kb:.1f} KB, {n_lines:,} requests) ...")
with open(INPUT_FILE, "rb") as f:
    batch_file = client.files.create(file=f, purpose="batch")
print(f"  File ID : {batch_file.id}")

print("\nSubmitting batch job ...")
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "experiment":  "multipico_rag_p4",
        "model":       "gpt-4.1-mini",
        "condition":   "p4",
        "languages":   "ar,es",
        "n_requests":  str(n_lines),
        "format":      "probability_0_1",
        "retrieval":   "text-embedding-3-small_K3_demographic_weighted",
    },
)

print(f"  Batch ID : {batch.id}")
print(f"  Status   : {batch.status}")
print(f"  Window   : {batch.completion_window}")

BATCH_ID_FILE.write_text(batch.id)
with open(SUBMISSION_LOG, "w") as f:
    json.dump({
        "batch_id":   batch.id,
        "file_id":    batch_file.id,
        "status":     batch.status,
        "n_requests": n_lines,
    }, f, indent=2)

print(f"\nBatch ID saved to  : {BATCH_ID_FILE}")
print(f"Submission log     : {SUBMISSION_LOG}")
print("\nNext step: python 73_retrieve_rag_batch.py  (re-run until status == 'completed')")
