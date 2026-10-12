"""
33_retrieve_extended_batch.py
=============================
Checks status of the extended batch.  When complete, downloads results to
extended_batch_output.jsonl and runs the parse step automatically.
"""

import os, json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("extended_batch_id.txt") as f:
    batch_id = f.read().strip()

batch = client.batches.retrieve(batch_id)
print(f"Batch ID : {batch_id}")
print(f"Status   : {batch.status}")
print(f"Progress : {batch.request_counts.completed} / {batch.request_counts.total}")

if batch.status == "completed":
    content = client.files.content(batch.output_file_id)
    with open("extended_batch_output.jsonl", "wb") as f:
        f.write(content.content)
    print(f"Downloaded → extended_batch_output.jsonl")
    print("Run 34_parse_extended_results.py to parse and analyse.")
elif batch.status in ("failed", "expired", "cancelled"):
    print(f"Batch failed: {batch.errors}")
else:
    print("Batch still running — re-run this script to check again.")
