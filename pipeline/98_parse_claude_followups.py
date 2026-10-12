"""
98_parse_claude_followups.py  — Parse Claude RAG (P4) and P5 batch outputs
===========================================================================
Reads the normalized batch outputs produced by 97_retrieve_claude_batch.py
and writes clean predictions parquets in the SAME schema the existing analysis
scripts use, but with a `_claude_` suffix so the gpt-4.1-mini parquets are
never overwritten:

    rag_claude_batch_output.jsonl + rag_batch_index.jsonl
        -> rag_predictions_claude_clean.parquet     (llm_prob,  condition=p4)

    p5_claude_batch_output.jsonl  + p5_batch_index.jsonl
        -> p5_predictions_claude_clean.parquet      (llm_deviation, llm_prob)

Parsing logic mirrors 74_parse_rag_results.py (probability) and
83_parse_p5_results.py (signed deviation), but reads the normalized Anthropic
record shape ({"status","text",...}) rather than the OpenAI response shape.

Usage:
    python 98_parse_claude_followups.py
    (Run AFTER 97_retrieve_claude_batch.py for BOTH batches)
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ── Parsers ───────────────────────────────────────────────────────────────────
_FLOAT_RE = re.compile(r"\b(1(?:\.0+)?|0(?:\.\d+)?)\b")


def parse_prob(raw):
    """Extract the first valid probability in [0, 1] (mirrors 74)."""
    if raw is None:
        return None
    raw = str(raw).strip()
    for m in _FLOAT_RE.findall(raw):
        try:
            v = float(m)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    try:
        v = float(raw)
        if 0.0 <= v <= 1.0:
            return v
    except ValueError:
        pass
    return None


_DEV_LINE_RE     = re.compile(r"[Dd]eviation\s*:\s*([+-]?\d*\.?\d+)")
_LAST_DECIMAL_RE = re.compile(r"([+-]?\d+\.?\d*)\s*$")
_REASONING_RE    = re.compile(r"[Rr]easoning\s*:\s*(.+)")


def parse_deviation(raw):
    """Return (deviation in [-1,1], reasoning) or (None, reasoning) (mirrors 83)."""
    if raw is None:
        return None, None
    raw = str(raw)
    reasoning = None
    m_r = _REASONING_RE.search(raw)
    if m_r:
        reasoning = m_r.group(1).strip()
    m = _DEV_LINE_RE.search(raw)
    if m:
        try:
            v = float(m.group(1))
            if -1.0 <= v <= 1.0:
                return v, reasoning
        except ValueError:
            pass
    m2 = _LAST_DECIMAL_RE.search(raw.strip())
    if m2:
        try:
            v = float(m2.group(1))
            if -1.0 <= v <= 1.0:
                return v, reasoning
        except ValueError:
            pass
    return None, reasoning


def load_output(path: Path) -> pd.DataFrame:
    """Load a normalized Anthropic batch output into a DataFrame."""
    rows, n_non_ok, tin, tout = [], 0, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("status") != "succeeded":
                n_non_ok += 1
            tin  += item.get("input_tokens", 0) or 0
            tout += item.get("output_tokens", 0) or 0
            rows.append(item)
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df):,} results from {path}  "
          f"(non-succeeded: {n_non_ok}, tokens in={tin:,} out={tout:,})")
    cost = tin / 1_000_000 * 2.50 + tout / 1_000_000 * 12.50
    print(f"  Actual billed cost (batch): ~${cost:.4f}")
    return df


def load_index(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(l) for l in f])


# ── RAG (P4) ──────────────────────────────────────────────────────────────────
def parse_rag():
    out_path = Path("rag_claude_batch_output.jsonl")
    idx_path = Path("rag_batch_index.jsonl")
    if not out_path.exists():
        print(f"SKIP RAG: {out_path} not found.")
        return
    print("\n" + "=" * 65)
    print("  RAG / P4  (Claude Opus 5)")
    print("=" * 65)

    df_out = load_output(out_path)
    df_out["llm_prob"] = df_out["text"].apply(parse_prob)
    df_out = df_out[["custom_id", "text", "llm_prob"]].rename(columns={"text": "raw_output"})

    df_idx = load_index(idx_path)
    df = df_out.merge(df_idx, on="custom_id", how="left")

    n_fail = df["llm_prob"].isna().sum()
    print(f"  Merged {len(df):,} rows | parse failures: {n_fail} "
          f"({n_fail/len(df)*100:.2f}%)")
    print("  Mean LLM probability by language:")
    print(df[df["llm_prob"].notna()].groupby("language")["llm_prob"].mean().round(3).to_string())

    df_clean = df[df["llm_prob"].notna()].copy()
    df_clean.to_parquet("rag_predictions_claude_clean.parquet", index=False)
    print(f"  Saved rag_predictions_claude_clean.parquet  {df_clean.shape}")
    if n_fail:
        df[df["llm_prob"].isna()][["custom_id", "language", "raw_output"]].to_csv(
            "rag_claude_parse_failures.csv", index=False)
        print("  Wrote rag_claude_parse_failures.csv")


# ── P5 (Relative + CoT) ───────────────────────────────────────────────────────
def parse_p5():
    out_path = Path("p5_claude_batch_output.jsonl")
    idx_path = Path("p5_batch_index.jsonl")
    if not out_path.exists():
        print(f"SKIP P5: {out_path} not found.")
        return
    print("\n" + "=" * 65)
    print("  P5 / Relative framing + CoT  (Claude Opus 5)")
    print("=" * 65)

    df_out = load_output(out_path)
    parsed = df_out["text"].apply(parse_deviation)
    df_out["llm_deviation"] = parsed.apply(lambda t: t[0])
    df_out["llm_reasoning"] = parsed.apply(lambda t: t[1])
    df_out = df_out[["custom_id", "text", "llm_deviation", "llm_reasoning"]].rename(
        columns={"text": "raw_output"})

    df_idx = load_index(idx_path)
    df = df_out.merge(df_idx, on="custom_id", how="left")

    # Reconstruct probability = baseline + deviation, clipped to [0, 1] (mirrors 83)
    df["llm_prob"] = np.clip(
        df["baseline_rate"].astype(float) + df["llm_deviation"].astype(float), 0.0, 1.0)

    n_fail = df["llm_deviation"].isna().sum()
    print(f"  Merged {len(df):,} rows | parse failures: {n_fail} "
          f"({n_fail/len(df)*100:.2f}%)")
    for lang in ["ar", "es"]:
        sub = df[(df["language"] == lang) & df["llm_deviation"].notna()]
        if len(sub):
            print(f"  {lang.upper()}: deviation mean={sub['llm_deviation'].mean():+.3f} "
                  f"std={sub['llm_deviation'].std():.3f}  n={len(sub)}")

    df_clean = df[df["llm_deviation"].notna()].copy()
    df_clean.to_parquet("p5_predictions_claude_clean.parquet", index=False)
    print(f"  Saved p5_predictions_claude_clean.parquet  {df_clean.shape}")

    # Reasoning sample for qualitative inspection
    df[df["llm_reasoning"].notna()][["language", "llm_deviation", "llm_reasoning"]].head(20)\
        .to_csv("p5_claude_reasoning_sample.csv", index=False)
    print("  Wrote p5_claude_reasoning_sample.csv")
    if n_fail:
        df[df["llm_deviation"].isna()][["custom_id", "language", "raw_output"]].to_csv(
            "p5_claude_parse_failures.csv", index=False)
        print("  Wrote p5_claude_parse_failures.csv")


def main():
    parse_rag()
    parse_p5()
    print("\nNext: python 99_analyze_claude_followups.py")


if __name__ == "__main__":
    main()
