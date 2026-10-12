"""
82_retrieve_p5_batch.py
=======================
Check batch status and download p5_batch_output.jsonl when complete.
Re-run until status == "completed", then run 83_parse_p5_results.py.

Usage:
    export OPENAI_API_KEY=sk-...
    python 82_retrieve_p5_batch.py
"""

import os
from pathlib import Path
from openai import OpenAI

BATCH_ID_FILE = Path("p5_batch_id.txt")
OUTPUT_FILE   = Path("p5_batch_output.jsonl")
ERRORS_FILE   = Path("p5_batch_errors.jsonl")

if not BATCH_ID_FILE.exists():
    print("ERROR: p5_batch_id.txt not found. Run 81_submit_p5_batch.py first.")
    exit(1)

client   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
batch_id = BATCH_ID_FILE.read_text().strip()
batch    = client.batches.retrieve(batch_id)

print(f"Batch ID   : {batch.id}")
print(f"Status     : {batch.status}")
print(f"Completed  : {batch.request_counts.completed:,}")
print(f"Failed     : {batch.request_counts.failed:,}")
print(f"Total      : {batch.request_counts.total:,}")
if batch.request_counts.total > 0:
    pct = batch.request_counts.completed / batch.request_counts.total * 100
    print(f"Progress   : {pct:.1f}%")

if batch.status == "completed":
    print("\nBatch completed! Downloading results ...")
    content = client.files.content(batch.output_file_id).read()
    with open(OUTPUT_FILE, "wb") as f:
        f.write(content)
    print(f"Results saved  : {OUTPUT_FILE}  ({content.count(b'\\n'):,} lines)")
    if batch.error_file_id:
        errors = client.files.content(batch.error_file_id).read()
        with open(ERRORS_FILE, "wb") as f:
            f.write(errors)
        print(f"Errors saved   : {ERRORS_FILE}")
    else:
        print("No error file — all requests succeeded.")
    print("\nNext step: python 83_parse_p5_results.py")
elif batch.status in ("failed", "expired", "cancelled"):
    print(f"\nBatch did not complete: {batch.status}")
else:
    print("\nStill processing. Re-run to check again.")
    print("Typical completion time: 1–2 hours.")
