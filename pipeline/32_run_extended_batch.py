"""
32_run_extended_batch.py
========================
Submits extended_batch_input.jsonl to the OpenAI Batch API.
Saves batch ID to extended_batch_id.txt for retrieval.
"""

import os, json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("extended_batch_input.jsonl", "rb") as f:
    uploaded = client.files.create(file=f, purpose="batch")

print(f"Uploaded file: {uploaded.id}")

batch = client.batches.create(
    input_file_id    = uploaded.id,
    endpoint         = "/v1/chat/completions",
    completion_window= "24h",
)

print(f"Batch submitted: {batch.id}")
print(f"Status: {batch.status}")

with open("extended_batch_id.txt", "w") as f:
    f.write(batch.id)

print(f"Saved batch ID to extended_batch_id.txt")
print(f"Run 33_retrieve_extended_batch.py to check status and download results.")
