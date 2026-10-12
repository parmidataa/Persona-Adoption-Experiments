"""
92_retrieve_opus5_batch.py  — Poll + download the Opus 5 batch results
======================================================================
Checks the batch processing status. When it has "ended", streams every
individual result and writes a normalized JSONL that 93_parse_opus5_results.py
can read without depending on SDK object shapes.

Normalized output record (one per line in opus5_batch_output.jsonl):
    {
      "custom_id":     "p2_1400",
      "status":        "succeeded" | "errored" | "canceled" | "expired",
      "text":          "1",              # assistant text (concatenated text blocks)
      "stop_reason":   "end_turn",
      "input_tokens":  123,
      "output_tokens": 1,
      "error":         null | "..."      # error string if not succeeded
    }

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python 92_retrieve_opus5_batch.py
"""

import os
import json
from pathlib import Path

import anthropic

BATCH_ID_PATH = Path("opus5_batch_id.txt")
OUTPUT_PATH   = Path("opus5_batch_output.jsonl")


def extract_text(message) -> tuple[str, str]:
    """Concatenate text blocks from a Message; return (text, stop_reason)."""
    parts = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts), getattr(message, "stop_reason", None)


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: set ANTHROPIC_API_KEY in the environment first.")
    if not BATCH_ID_PATH.exists():
        raise SystemExit(f"ERROR: {BATCH_ID_PATH} not found (run 91 first).")

    batch_id = BATCH_ID_PATH.read_text().strip()
    client = anthropic.Anthropic(api_key=api_key)

    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch {batch_id}")
    print(f"  processing_status : {batch.processing_status}")
    print(f"  request_counts    : {batch.request_counts}")

    if batch.processing_status != "ended":
        print("\nBatch is not finished yet. Re-run this script later.")
        return

    print("\nBatch ended — downloading results ...")
    n_ok = n_bad = 0
    total_in = total_out = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for result in client.messages.batches.results(batch_id):
            cid = result.custom_id
            rtype = result.result.type
            rec = {
                "custom_id":     cid,
                "status":        rtype,
                "text":          None,
                "stop_reason":   None,
                "input_tokens":  0,
                "output_tokens": 0,
                "error":         None,
            }
            if rtype == "succeeded":
                msg = result.result.message
                text, stop = extract_text(msg)
                usage = getattr(msg, "usage", None)
                rec["text"] = text
                rec["stop_reason"] = stop
                if usage is not None:
                    rec["input_tokens"]  = getattr(usage, "input_tokens", 0) or 0
                    rec["output_tokens"] = getattr(usage, "output_tokens", 0) or 0
                    total_in  += rec["input_tokens"]
                    total_out += rec["output_tokens"]
                n_ok += 1
            else:
                # errored / canceled / expired
                err = getattr(result.result, "error", None)
                rec["error"] = str(err) if err is not None else rtype
                n_bad += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Wrote {n_ok + n_bad:,} results -> {OUTPUT_PATH}")
    print(f"  Succeeded : {n_ok:,}")
    print(f"  Non-OK    : {n_bad:,}")
    print(f"  Tokens    : input={total_in:,}  output={total_out:,}")
    cost = total_in / 1_000_000 * 2.50 + total_out / 1_000_000 * 12.50
    print(f"  Actual billed cost (batch) : ~${cost:,.2f}")
    print("\nNext: python 93_parse_opus5_results.py")


if __name__ == "__main__":
    main()
