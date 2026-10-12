"""
91_submit_opus5_batch.py  — Submit the Opus 5 Message Batch
===========================================================
Pre-flight + submission for the Priority-1 Claude Opus 5 run.

Pre-flight (always runs, no charge):
  1. Reads ANTHROPIC_API_KEY from the environment.
  2. Verifies the target model id exists via client.models.list().
  3. Estimates cost from the actual input file (character-based token proxy)
     using Opus 5 batch pricing ($2.50 / MTok in, $12.50 / MTok out).

Submission (only if not a dry run):
  4. Loads opus5_batch_input.jsonl and submits it as ONE Message Batch.
  5. Saves opus5_batch_id.txt and opus5_batch_submission.json.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python 91_submit_opus5_batch.py            # pre-flight + submit
    DRY_RUN=1 python 91_submit_opus5_batch.py  # pre-flight only, no submit
"""

import os
import json
from pathlib import Path

import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
MODEL          = "claude-opus-5"
INPUT_PATH     = Path("opus5_batch_input.jsonl")
BATCH_ID_PATH  = Path("opus5_batch_id.txt")
SUBMISSION_LOG = Path("opus5_batch_submission.json")

# Opus 5 batch pricing (50% off the $5 / $25 base), USD per million tokens
PRICE_IN_PER_MTOK  = 2.50
PRICE_OUT_PER_MTOK = 12.50
# Character -> token proxy. Anthropic's tokenizer averages ~3.5 chars/token on
# mixed multilingual text (Arabic/Hindi push this lower); 3.5 is conservative.
CHARS_PER_TOKEN = 3.5
EST_OUT_TOKENS_PER_REQ = 1     # thinking disabled -> single-digit answer


def load_requests(path: Path) -> list[dict]:
    reqs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            reqs.append(json.loads(line))
    return reqs


def estimate_cost(reqs: list[dict]) -> None:
    total_chars = 0
    for r in reqs:
        p = r["params"]
        total_chars += len(p.get("system", ""))
        for m in p["messages"]:
            total_chars += len(m["content"])
    est_in_tok  = total_chars / CHARS_PER_TOKEN
    est_out_tok = len(reqs) * EST_OUT_TOKENS_PER_REQ
    cost_in  = est_in_tok  / 1_000_000 * PRICE_IN_PER_MTOK
    cost_out = est_out_tok / 1_000_000 * PRICE_OUT_PER_MTOK

    print("  ── Cost pre-flight (Opus 5, Batch API 50% off) ──")
    print(f"  Requests            : {len(reqs):,}")
    print(f"  Input chars         : {total_chars:,}")
    print(f"  Est. input tokens   : {est_in_tok:,.0f}  (~{CHARS_PER_TOKEN} chars/tok)")
    print(f"  Est. output tokens  : {est_out_tok:,}   (thinking disabled)")
    print(f"  Est. input cost     : ${cost_in:,.2f}")
    print(f"  Est. output cost    : ${cost_out:,.2f}")
    print(f"  Est. TOTAL          : ${cost_in + cost_out:,.2f}")
    print(f"  (Actual billed usage is reported by the batch; this is an estimate.)")


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"ERROR: {INPUT_PATH} not found (run 90_build_opus5_batch.py).")

    dry_run = bool(os.environ.get("DRY_RUN"))
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Cost estimate needs no key; allow a keyless dry run.
    if not api_key and dry_run:
        print("No ANTHROPIC_API_KEY set — keyless cost pre-flight only.\n")
        reqs = load_requests(INPUT_PATH)
        print(f"Loaded {len(reqs):,} requests from {INPUT_PATH}")
        sample = reqs[0]["params"]
        print(f"  Sample params: model={sample['model']}, max_tokens={sample['max_tokens']}, "
              f"thinking={sample.get('thinking')}, temperature={sample.get('temperature')}\n")
        estimate_cost(reqs)
        print("\nDRY_RUN with no key — set ANTHROPIC_API_KEY and remove DRY_RUN to submit.")
        return

    if not api_key:
        raise SystemExit("ERROR: set ANTHROPIC_API_KEY in the environment first.")

    client = anthropic.Anthropic(api_key=api_key)

    # 1. Verify model id
    print("Verifying model availability ...")
    try:
        models = [m.id for m in client.models.list(limit=100).data]
        opus_ids = [m for m in models if "opus" in m.lower()]
        print(f"  Opus models visible to this key: {opus_ids}")
        if MODEL not in models:
            print(f"  WARNING: '{MODEL}' not in the visible model list. "
                  f"Submission may fail. Available: {models}")
        else:
            print(f"  OK: '{MODEL}' is available.")
    except Exception as e:
        print(f"  (Could not list models: {e}. Proceeding to cost estimate.)")

    # 2. Load + cost estimate
    print("\nLoading batch requests ...")
    reqs = load_requests(INPUT_PATH)
    print(f"  Loaded {len(reqs):,} requests from {INPUT_PATH}")
    # Sanity: confirm thinking disabled + model set
    sample = reqs[0]["params"]
    print(f"  Sample params: model={sample['model']}, max_tokens={sample['max_tokens']}, "
          f"thinking={sample.get('thinking')}, temperature={sample.get('temperature')}")
    print()
    estimate_cost(reqs)

    if dry_run:
        print("\nDRY_RUN set — not submitting. Remove DRY_RUN to submit.")
        return

    # 3. Submit
    print(f"\nSubmitting {len(reqs):,} requests as one Message Batch ...")
    batch = client.messages.batches.create(requests=reqs)
    print(f"  Batch id           : {batch.id}")
    print(f"  Processing status  : {batch.processing_status}")
    print(f"  Request counts     : {batch.request_counts}")

    BATCH_ID_PATH.write_text(batch.id)
    SUBMISSION_LOG.write_text(json.dumps({
        "batch_id":          batch.id,
        "model":             MODEL,
        "n_requests":        len(reqs),
        "processing_status": batch.processing_status,
        "created_at":        str(getattr(batch, "created_at", "")),
    }, indent=2))
    print(f"\n  Saved batch id -> {BATCH_ID_PATH}")
    print(f"  Saved log      -> {SUBMISSION_LOG}")
    print("\nNext: python 92_retrieve_opus5_batch.py")


if __name__ == "__main__":
    main()
