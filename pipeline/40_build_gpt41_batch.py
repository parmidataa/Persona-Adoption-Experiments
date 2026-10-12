"""
40_build_gpt41_batch.py  — Approach B (model comparison)
========================================================
Builds a batch identical to the original gpt-4.1-mini experiment, but using
the larger gpt-4.1 model. Every prompt (system + user message), temperature,
token cap, condition, and row is byte-for-byte identical to the baseline run,
so the ONLY variable that changes is model scale within the GPT-4.1 generation.

custom_ids are prefixed with "g41_" to keep outputs unambiguous; the original
suffix (e.g. p0_1400) still maps 1:1 to batch_index.jsonl for metadata.

Output: gpt41_batch_input.jsonl  (25,200 calls, ~$2.20 via Batch API)
"""

import json

# Pin the dated snapshot for reproducibility (see findings.md methodology note).
MODEL = "gpt-4.1-2025-04-14"

n_in, n_out = 0, 0
with open("batch_input.jsonl") as fin, open("gpt41_batch_input.jsonl", "w") as fout:
    for line in fin:
        req = json.loads(line)
        n_in += 1
        # Only change: model id, and a prefixed custom_id. Everything else identical.
        req["body"]["model"] = MODEL
        req["custom_id"] = "g41_" + req["custom_id"]
        fout.write(json.dumps(req) + "\n")
        n_out += 1

print(f"Read  {n_in:,} requests from batch_input.jsonl (model gpt-4.1-mini)")
print(f"Wrote {n_out:,} requests to gpt41_batch_input.jsonl (model {MODEL})")
print("Prompts, temperature, token cap, conditions, rows: IDENTICAL to baseline.")
print("Only model id changed → clean scale-within-generation comparison.")
