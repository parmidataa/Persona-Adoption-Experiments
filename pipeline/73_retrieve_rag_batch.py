"""
73_retrieve_rag_batch.py
========================
Check batch status and download rag_batch_output.jsonl when complete.

Re-run this script until status == "completed", then proceed to
74_parse_rag_results.py.

Usage:
    export OPENAI_API_KEY=sk-...
    python 73_retrieve_rag_batch.py
    (Run AFTER 72_submit_rag_batch.py)
"""

import os
from pathlib import Path
from openai import OpenAI

BATCH_ID_FILE = Path("rag_batch_id.txt")
OUTPUT_FILE   = Path("rag_batch_output.jsonl")
ERRORS_FILE   = Path("rag_batch_errors.jsonl")

if not BATCH_ID_FILE.exists():
    print("ERROR: rag_batch_id.txt not found. Run 72_submit_rag_batch.py first.")
    exit(1)

client   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
batch_id = BATCH_ID_FILE.read_text().strip()

batch = client.batches.retrieve(batch_id)

print(f"Batch ID   : {batch.id}")
print(f"Status     : {batch.status}")
print(f"Completed  : {batch.request_counts.completed:,}")
print(f"Failed     : {batch.request_counts.failed:,}")
print(f"Total      : {batch.request_counts.total:,}")

if batch.request_counts.total > 0:
    pct = batch.request_counts.completed / batch.request_counts.total * 100
    print(f"Progress   : {pct:.1f}%")

if batch.status == "completed":
    print("\nBatch completed!  Downloading results ...")
    content = client.files.content(batch.output_file_id).read()
    with open(OUTPUT_FILE, "wb") as f:
        f.write(content)
    n_lines = content.count(b"\n")
    print(f"Results saved  : {OUTPUT_FILE}  ({n_lines:,} lines, "
          f"{len(content)/1024:.1f} KB)")

    if batch.error_file_id:
        errors = client.files.content(batch.error_file_id).read()
        with open(ERRORS_FILE, "wb") as f:
            f.write(errors)
        print(f"Errors saved   : {ERRORS_FILE}")
    else:
        print("No error file — all requests succeeded.")

    print("\nNext step: python 74_parse_rag_results.py")

elif batch.status in ("failed", "expired", "cancelled"):
    print(f"\nBatch did not complete: {batch.status}")
    print("Check the submission log (rag_batch_submission.json) and resubmit if needed.")

else:
    print(f"\nStill processing.  Re-run this script to check again.")
    print("Typical completion time: 1–2 hours.")
