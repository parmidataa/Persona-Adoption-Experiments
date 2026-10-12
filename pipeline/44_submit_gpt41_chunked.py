"""
44_submit_gpt41_chunked.py  — sequential chunked submission for gpt-4.1
=======================================================================
gpt-4.1 has a 900,000 enqueued-token batch limit. Our full batch is ~2.09M
input tokens, so we split it into chunks UNDER the limit and submit them one
at a time (waiting for each to finish before the next), then concatenate all
responses into gpt41_batch_output.jsonl.

Chunk sizing uses the ACTUAL prompt_tokens from the gpt-4.1-mini run
(identical tokenizer → identical counts), so each chunk is precisely under
budget.

Resumable: records each chunk's batch id in gpt41_chunks_state.json.
"""

import os, json, time
from pathlib import Path
import pandas as pd
from openai import OpenAI

TOKEN_BUDGET = 800_000          # safety margin under the 900k hard limit
POLL_SECONDS = 30
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Load actual token counts per custom_id (from the mini run) ────────────────
df_tok = pd.read_parquet("data_predictions_clean.parquet")[["custom_id", "prompt_tokens"]]
tok_map = dict(zip(df_tok["custom_id"], df_tok["prompt_tokens"]))

# ── Read all gpt-4.1 requests, assign each to a chunk under TOKEN_BUDGET ──────
requests = [json.loads(l) for l in open("gpt41_batch_input.jsonl")]
chunks, cur, cur_tok = [], [], 0
for req in requests:
    base_id = req["custom_id"].replace("g41_", "", 1)
    t = int(tok_map.get(base_id, 100)) + 5   # +5 output headroom
    if cur and cur_tok + t > TOKEN_BUDGET:
        chunks.append(cur); cur, cur_tok = [], 0
    cur.append(req); cur_tok += t
if cur:
    chunks.append(cur)

print(f"Split {len(requests):,} requests into {len(chunks)} chunks "
      f"(budget {TOKEN_BUDGET:,} tok/chunk):")
for i, ch in enumerate(chunks):
    ct = sum(int(tok_map.get(r['custom_id'].replace('g41_','',1), 100)) + 5 for r in ch)
    print(f"  chunk {i}: {len(ch):,} calls, ~{ct:,} tokens")

# ── Resumable state ───────────────────────────────────────────────────────────
state_path = Path("gpt41_chunks_state.json")
state = json.loads(state_path.read_text()) if state_path.exists() else {"chunks": {}}

def save_state():
    state_path.write_text(json.dumps(state, indent=2))

# ── Submit + wait + download each chunk sequentially ──────────────────────────
combined = []
for i, ch in enumerate(chunks):
    key = str(i)
    out_path = Path(f"gpt41_out_{i}.jsonl")

    # Already downloaded?
    if out_path.exists() and state["chunks"].get(key, {}).get("status") == "completed":
        print(f"\n[chunk {i}] already completed, reusing {out_path}")
        combined.extend(out_path.read_text().splitlines())
        continue

    # Write chunk input
    chunk_in = Path(f"gpt41_chunk_{i}.jsonl")
    with open(chunk_in, "w") as f:
        for r in ch:
            f.write(json.dumps(r) + "\n")

    # Submit (or reuse existing batch id)
    bid = state["chunks"].get(key, {}).get("batch_id")
    if not bid:
        up = client.files.create(file=open(chunk_in, "rb"), purpose="batch")
        b  = client.batches.create(input_file_id=up.id,
                                   endpoint="/v1/chat/completions",
                                   completion_window="24h")
        bid = b.id
        state["chunks"][key] = {"batch_id": bid, "status": b.status}
        save_state()
        print(f"\n[chunk {i}] submitted batch {bid}")
    else:
        print(f"\n[chunk {i}] resuming batch {bid}")

    # Poll until done
    while True:
        b = client.batches.retrieve(bid)
        done = b.request_counts.completed
        total = b.request_counts.total
        print(f"  [chunk {i}] {b.status}  {done}/{total}")
        if b.status == "completed":
            content = client.files.content(b.output_file_id)
            out_path.write_bytes(content.content)
            state["chunks"][key]["status"] = "completed"
            save_state()
            combined.extend(out_path.read_text().splitlines())
            break
        if b.status in ("failed", "expired", "cancelled"):
            print(f"  [chunk {i}] FAILED: {b.errors}")
            raise SystemExit(1)
        time.sleep(POLL_SECONDS)

# ── Concatenate all chunk outputs ─────────────────────────────────────────────
with open("gpt41_batch_output.jsonl", "w") as f:
    for line in combined:
        if line.strip():
            f.write(line + "\n")

print(f"\nAll {len(chunks)} chunks complete.")
print(f"Combined → gpt41_batch_output.jsonl ({len(combined):,} responses)")
print("Next: python3 43_analyze_gpt41.py")
