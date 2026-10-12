"""
96_submit_claude_batch.py  — Generic Anthropic Message Batch submitter
=======================================================================
Pre-flight + submission for any Anthropic-format batch file.  Used for both
the RAG and P5 Claude follow-up runs (and reusable for any future batch).

Pre-flight (always runs, no charge):
  1. Reads ANTHROPIC_API_KEY from the environment.
  2. Verifies the target model id exists via client.models.list().
  3. Estimates cost from the actual input file (character-based token proxy)
     using Opus 5 batch pricing ($2.50 / MTok in, $12.50 / MTok out).
     Output tokens are estimated from each request's own max_tokens (ceiling).

Submission (only if not a dry run):
  4. Loads <input.jsonl> and submits it as ONE Message Batch.
  5. Saves <id_out.txt> and <id_out>.submission.json.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python 96_submit_claude_batch.py <input.jsonl> <id_out.txt>
    DRY_RUN=1 python 96_submit_claude_batch.py <input.jsonl> <id_out.txt>

Example:
    python 96_submit_claude_batch.py rag_claude_batch_input.jsonl rag_claude_batch_id.txt
    python 96_submit_claude_batch.py p5_claude_batch_input.jsonl  p5_claude_batch_id.txt
"""

import os
import sys
import json
from pathlib import Path

import anthropic

# Opus 5 batch pricing (50% off the $5 / $25 base), USD per million tokens
PRICE_IN_PER_MTOK  = 2.50
PRICE_OUT_PER_MTOK = 12.50
# Char -> token proxy. Anthropic averages ~3.5 chars/token on mixed multilingual
# text (Arabic/Hindi push it lower); 3.5 is conservative for a cost ceiling.
CHARS_PER_TOKEN = 3.5


def load_requests(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def estimate_cost(reqs: list[dict]) -> None:
    total_chars = 0
    est_out_tok = 0
    for r in reqs:
        p = r["params"]
        total_chars += len(p.get("system", ""))
        for m in p["messages"]:
            total_chars += len(m["content"])
        # Output ceiling = this request's max_tokens (real usage is usually less).
        est_out_tok += int(p.get("max_tokens", 1))
    est_in_tok = total_chars / CHARS_PER_TOKEN
    cost_in  = est_in_tok  / 1_000_000 * PRICE_IN_PER_MTOK
    cost_out = est_out_tok / 1_000_000 * PRICE_OUT_PER_MTOK

    print("  ── Cost pre-flight (Opus 5, Batch API 50% off) ──")
    print(f"  Requests             : {len(reqs):,}")
    print(f"  Input chars          : {total_chars:,}")
    print(f"  Est. input tokens    : {est_in_tok:,.0f}  (~{CHARS_PER_TOKEN} chars/tok)")
    print(f"  Est. output tokens   : {est_out_tok:,}   (max_tokens ceiling)")
    print(f"  Est. input cost      : ${cost_in:,.2f}")
    print(f"  Est. output cost     : ${cost_out:,.2f}  (upper bound)")
    print(f"  Est. TOTAL (ceiling) : ${cost_in + cost_out:,.2f}")
    print(f"  (Actual billed usage is reported by the batch; this is an estimate.)")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python 96_submit_claude_batch.py <input.jsonl> <id_out.txt>")

    input_path    = Path(sys.argv[1])
    batch_id_path = Path(sys.argv[2])
    submission_log = Path(str(batch_id_path).replace(".txt", "") + ".submission.json")

    if not input_path.exists():
        raise SystemExit(f"ERROR: {input_path} not found (run 95_build_claude_followups.py).")

    dry_run = bool(os.environ.get("DRY_RUN"))
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    reqs = load_requests(input_path)
    sample = reqs[0]["params"]
    model  = sample["model"]

    # Cost estimate needs no key; allow a keyless dry run.
    if not api_key and dry_run:
        print("No ANTHROPIC_API_KEY set — keyless cost pre-flight only.\n")
        print(f"Loaded {len(reqs):,} requests from {input_path}")
        print(f"  Sample params: model={model}, max_tokens={sample['max_tokens']}, "
              f"thinking={sample.get('thinking')}\n")
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
        if model not in models:
            print(f"  WARNING: '{model}' not in the visible model list. "
                  f"Submission may fail. Available: {models}")
        else:
            print(f"  OK: '{model}' is available.")
    except Exception as e:
        print(f"  (Could not list models: {e}. Proceeding to cost estimate.)")

    # 2. Cost estimate
    print(f"\nLoaded {len(reqs):,} requests from {input_path}")
    print(f"  Sample params: model={model}, max_tokens={sample['max_tokens']}, "
          f"thinking={sample.get('thinking')}")
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

    batch_id_path.write_text(batch.id)
    submission_log.write_text(json.dumps({
        "batch_id":          batch.id,
        "model":             model,
        "input_file":        str(input_path),
        "n_requests":        len(reqs),
        "processing_status": batch.processing_status,
        "created_at":        str(getattr(batch, "created_at", "")),
    }, indent=2))
    print(f"\n  Saved batch id -> {batch_id_path}")
    print(f"  Saved log      -> {submission_log}")
    print(f"\nNext: python 97_retrieve_claude_batch.py {batch_id_path} "
          f"{str(input_path).replace('_input.jsonl', '_output.jsonl')}")


if __name__ == "__main__":
    main()
