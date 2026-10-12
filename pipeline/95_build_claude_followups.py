"""
95_build_claude_followups.py  — Repeat RAG (P4) and P5 on Claude Opus 5
=======================================================================
Converts the two existing OpenAI-format follow-up batches into the Anthropic
Message Batches format, byte-for-byte identical in PROMPT CONTENT.  The only
variable that changes relative to the gpt-4.1-mini runs is the model family.

Two batches are produced:
  * rag_batch_input.jsonl  ->  rag_claude_batch_input.jsonl   (P4 / RAG, prob output)
  * p5_batch_input.jsonl   ->  p5_claude_batch_input.jsonl    (P5 / Relative+CoT)

Format differences (OpenAI -> Anthropic), identical to 90_build_opus5_batch.py:
  * OpenAI system message  -> Anthropic top-level `system` string (not a message).
  * Anthropic requires `max_tokens` (mandatory).
  * `temperature` is DEPRECATED on Opus 5 (rejected with 400) -> omitted.
    With thinking disabled the model decodes greedily, which is deterministic
    for these short outputs and matches the temperature=0 GPT runs.
  * custom_id is kept identical to the OpenAI run so it maps 1:1 to the existing
    *_batch_index.jsonl for downstream analysis.

max_tokens differs per experiment because the OUTPUT differs:
  * RAG returns a probability like "0.73"          -> a few tokens; we use 10.
  * P5  returns "Reasoning: <sentence>\nDeviation: +0.12"
    (~30-word sentence + a signed decimal)          -> we use 200 for safety.

Thinking is DISABLED for BOTH (`thinking: {"type": "disabled"}`).
Rationale (scientific validity):
  * Opus 5 ships with adaptive thinking ON by default (effort "high").
  * The gpt-4.1-mini RAG and P5 runs used NO test-time reasoning.
  * To isolate MODEL FAMILY as the only manipulated variable we must not
    confound it with test-time compute, so thinking is turned off.
  * For P5 specifically, the chain-of-thought is the in-text "Reasoning:" line
    the prompt already requests; disabling the separate thinking channel keeps
    the run directly comparable to the mini P5 experiment.

Output: rag_claude_batch_input.jsonl, p5_claude_batch_input.jsonl

Usage:
    python 95_build_claude_followups.py
"""

import json
from pathlib import Path

MODEL = "claude-opus-5"

# (openai_input, anthropic_output, max_tokens, human-readable label)
JOBS = [
    ("rag_batch_input.jsonl", "rag_claude_batch_input.jsonl", 10,  "RAG / P4 (probability output)"),
    ("p5_batch_input.jsonl",  "p5_claude_batch_input.jsonl",  200, "P5 / Relative framing + CoT"),
]


def convert(req: dict, max_tokens: int) -> dict:
    """Convert one OpenAI /v1/chat/completions request -> Anthropic batch request."""
    body = req["body"]

    system_txt = None
    messages_an = []
    for m in body["messages"]:
        if m["role"] == "system":
            # Anthropic: system is a top-level param, not a message.
            system_txt = m["content"] if system_txt is None else system_txt + "\n" + m["content"]
        else:
            messages_an.append({"role": m["role"], "content": m["content"]})

    params = {
        "model":      MODEL,
        "max_tokens": max_tokens,
        "thinking":   {"type": "disabled"},   # clean cross-family comparison
        "messages":   messages_an,
    }
    if system_txt is not None:
        params["system"] = system_txt

    return {"custom_id": req["custom_id"], "params": params}


def build_one(in_path: Path, out_path: Path, max_tokens: int, label: str) -> int:
    if not in_path.exists():
        raise SystemExit(f"ERROR: {in_path} not found.")

    n_in = 0
    seen = set()
    with open(in_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            req = json.loads(line)
            n_in += 1
            cid = req["custom_id"]
            # Anthropic requires unique custom_ids matching ^[a-zA-Z0-9_-]{1,64}$
            assert cid not in seen, f"duplicate custom_id {cid}"
            assert len(cid) <= 64 and all(c.isalnum() or c in "_-" for c in cid), \
                f"invalid custom_id {cid}"
            seen.add(cid)
            fout.write(json.dumps(convert(req, max_tokens), ensure_ascii=False) + "\n")

    print(f"  [{label}]")
    print(f"    {in_path}  ->  {out_path}")
    print(f"    requests={n_in:,}   max_tokens={max_tokens}   thinking=disabled")
    return n_in


def main():
    print(f"Building Claude Opus 5 follow-up batches (model={MODEL})\n")
    total = 0
    for in_name, out_name, max_tok, label in JOBS:
        total += build_one(Path(in_name), Path(out_name), max_tok, label)
        print()

    print(f"Total requests written: {total:,}")
    print("System prompt moved to top-level `system`; prompts otherwise identical.")
    print("\nNext:")
    print("  python 96_submit_claude_batch.py rag_claude_batch_input.jsonl rag_claude_batch_id.txt")
    print("  python 96_submit_claude_batch.py p5_claude_batch_input.jsonl  p5_claude_batch_id.txt")


if __name__ == "__main__":
    main()
