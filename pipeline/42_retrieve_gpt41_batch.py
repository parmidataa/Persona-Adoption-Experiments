"""
42_retrieve_gpt41_batch.py  — check status / download gpt-4.1 results
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("gpt41_batch_id.txt") as f:
    batch_id = f.read().strip()

batch = client.batches.retrieve(batch_id)
print(f"Batch ID : {batch_id}")
print(f"Status   : {batch.status}")
print(f"Progress : {batch.request_counts.completed} / {batch.request_counts.total}")

if batch.status == "completed":
    content = client.files.content(batch.output_file_id)
    with open("gpt41_batch_output.jsonl", "wb") as f:
        f.write(content.content)
    print("Downloaded → gpt41_batch_output.jsonl")
    print("Next: python3 43_analyze_gpt41.py")
elif batch.status in ("failed", "expired", "cancelled"):
    print(f"Batch {batch.status}: {batch.errors}")
else:
    print("Batch still running — re-run this script to check again.")
