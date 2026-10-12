"""
41_run_gpt41_batch.py  — submit the gpt-4.1 comparison batch
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("gpt41_batch_input.jsonl", "rb") as f:
    uploaded = client.files.create(file=f, purpose="batch")
print(f"Uploaded file: {uploaded.id}")

batch = client.batches.create(
    input_file_id     = uploaded.id,
    endpoint          = "/v1/chat/completions",
    completion_window = "24h",
)
print(f"Batch submitted: {batch.id}")
print(f"Status: {batch.status}")

with open("gpt41_batch_id.txt", "w") as f:
    f.write(batch.id)
print("Saved batch ID to gpt41_batch_id.txt")
print("Next: python3 42_retrieve_gpt41_batch.py")
