"""
90_build_opus5_batch.py  — Cross-family comparison (Claude Opus 5)
==================================================================
Converts the existing OpenAI-format main batch (batch_input.jsonl) into the
Anthropic Message Batches format, byte-for-byte identical in prompt content,
temperature, and token cap. The ONLY variable that changes relative to the
gpt-4.1-mini / gpt-4.1 runs is the model family.

This is the Priority-1 cross-model run: all 9 languages x 4 persona conditions
(P0-P3) x 700 rows = 25,200 requests, on Claude Opus 5.

Key format differences (OpenAI -> Anthropic):
  * OpenAI puts the system prompt as a message with role="system".
    Anthropic takes it as a top-level `system` string, NOT a message.
  * Anthropic requires `max_tokens` (mandatory); we mirror the OpenAI
    max_completion_tokens value (3 -> we use 5 for a small safety margin).
  * Each request is {"custom_id": ..., "params": {model, max_tokens,
    temperature, system, messages}}.
  * custom_id is kept identical to the OpenAI run (e.g. "p2_1400") so it maps
    1:1 to batch_index.jsonl for downstream analysis. It satisfies Anthropic's
    ^[a-zA-Z0-9_-]{1,64}$ constraint.

Thinking is deliberately DISABLED (`thinking: {"type": "disabled"}`).
Rationale — scientific validity:
  * Opus 5 ships with adaptive thinking ON by default (effort "high").
  * The gpt-4.1-mini and gpt-4.1 comparison runs used NO reasoning.
  * To isolate MODEL FAMILY as the only manipulated variable in the
    magnitude-direction comparison, we must not confound it with test-time
    compute. Disabling thinking makes the manipulation clean.
  * It also guarantees a single-digit output, avoiding truncation of the
    answer by thinking tokens under a small max_tokens, and holds cost minimal.
  * Disabling thinking is permitted on Opus 5 at effort <= "high" (the default),
    so no effort override is required.

Output: opus5_batch_input.jsonl  (25,200 Anthropic-format requests)
"""

import json
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
# Anthropic model id for Claude Opus 5. VERIFY against `client.models.list()`
# in the pre-flight step before submitting (91 prints the available ids).
MODEL       = "claude-opus-5"
MAX_TOKENS  = 5          # binary output "0"/"1" is 1 token; 5 is a safe margin
INPUT_PATH  = Path("batch_input.jsonl")
OUTPUT_PATH = Path("opus5_batch_input.jsonl")


def convert(req: dict) -> dict:
    """Convert one OpenAI /v1/chat/completions request → Anthropic batch request."""
    body = req["body"]
    messages_oa = body["messages"]

    system_txt = None
    messages_an = []
    for m in messages_oa:
        if m["role"] == "system":
            # Anthropic: system is a top-level param, not a message.
            system_txt = m["content"] if system_txt is None else system_txt + "\n" + m["content"]
        else:
            messages_an.append({"role": m["role"], "content": m["content"]})

    # NOTE: `temperature` is deprecated on Opus 5 and rejected (400). We omit it.
    # With thinking disabled the model decodes greedily, so output is effectively
    # deterministic for this single-digit classification — matching temperature=0.
    params = {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking":   {"type": "disabled"},   # clean cross-family comparison
        "messages":   messages_an,
    }
    if system_txt is not None:
        params["system"] = system_txt

    return {"custom_id": req["custom_id"], "params": params}


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"ERROR: {INPUT_PATH} not found (run 26_build_batch.py first).")

    n_in = n_out = 0
    seen = set()
    with open(INPUT_PATH, encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            req = json.loads(line)
            n_in += 1
            cid = req["custom_id"]
            # Anthropic requires unique custom_ids matching ^[a-zA-Z0-9_-]{1,64}$
            assert cid not in seen, f"duplicate custom_id {cid}"
            assert len(cid) <= 64 and all(c.isalnum() or c in "_-" for c in cid), \
                f"invalid custom_id {cid}"
            seen.add(cid)
            fout.write(json.dumps(convert(req), ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Read  {n_in:,} OpenAI requests from {INPUT_PATH}")
    print(f"Wrote {n_out:,} Anthropic requests to {OUTPUT_PATH}")
    print(f"Model : {MODEL}   max_tokens={MAX_TOKENS}   temperature=0")
    print("System prompt moved to top-level `system`; prompts otherwise identical.")
    print("\nNext: python 91_submit_opus5_batch.py")


if __name__ == "__main__":
    main()
