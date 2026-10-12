"""
74_parse_rag_results.py
=======================
Parse rag_batch_output.jsonl into rag_predictions_clean.parquet.

Identical parsing logic to 63_parse_prob_results.py: extracts the first
float in [0, 1] from the model's response string.  Responses that cannot
be parsed are flagged in rag_parse_failures.csv.

Outputs:
    rag_predictions_raw.parquet    — all rows including parse failures (NaN)
    rag_predictions_clean.parquet  — only rows with a valid probability
    rag_parse_failures.csv         — failed rows for inspection

Usage:
    python 74_parse_rag_results.py
    (Run AFTER 73_retrieve_rag_batch.py)
"""

import json
import re
import pandas as pd
from pathlib import Path

OUTPUT_FILE   = Path("rag_batch_output.jsonl")
INDEX_FILE    = Path("rag_batch_index.jsonl")
RAW_PARQUET   = Path("rag_predictions_raw.parquet")
CLEAN_PARQUET = Path("rag_predictions_clean.parquet")
FAILURES_CSV  = Path("rag_parse_failures.csv")

# ── Float parser (identical to 63_parse_prob_results.py) ─────────────────────
_FLOAT_RE = re.compile(r"\b(1(?:\.0+)?|0(?:\.\d+)?)\b")

def parse_prob(raw: str | None) -> float | None:
    """
    Extract the first valid probability in [0, 1] from the model's response.
    Returns None if no valid value is found.
    """
    if raw is None:
        return None
    raw = raw.strip()
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


# ── Load batch output ─────────────────────────────────────────────────────────
if not OUTPUT_FILE.exists():
    print(f"ERROR: {OUTPUT_FILE} not found. Run 73_retrieve_rag_batch.py first.")
    exit(1)

print("Loading batch output ...")
output_rows  = []
total_tokens = 0

with open(OUTPUT_FILE, encoding="utf-8") as f:
    for line in f:
        item      = json.loads(line)
        custom_id = item["custom_id"]

        try:
            choices = item["response"]["body"]["choices"]
            raw     = choices[0]["message"]["content"] if choices else None
        except (KeyError, IndexError, TypeError):
            raw = None

        try:
            usage      = item["response"]["body"]["usage"]
            prompt_tok = usage.get("prompt_tokens", 0)
            output_tok = usage.get("completion_tokens", 0)
            total_tokens += prompt_tok + output_tok
        except (KeyError, TypeError):
            prompt_tok = output_tok = 0

        prob = parse_prob(raw)
        output_rows.append({
            "custom_id":     custom_id,
            "raw_output":    raw,
            "llm_prob":      prob,
            "prompt_tokens": prompt_tok,
            "output_tokens": output_tok,
        })

df_out = pd.DataFrame(output_rows)
print(f"  Loaded {len(df_out):,} response rows")
print(f"  Total tokens used : {total_tokens:,}")
cost_est = (total_tokens * 0.40 / 1_000_000) * 0.5
print(f"  Estimated cost    : ~${cost_est:.4f}  (Batch API)")

# ── Load index ────────────────────────────────────────────────────────────────
print("\nLoading batch index ...")
index_rows = []
with open(INDEX_FILE, encoding="utf-8") as f:
    for line in f:
        index_rows.append(json.loads(line))
df_idx = pd.DataFrame(index_rows)
print(f"  Loaded {len(df_idx):,} index rows")

# ── Merge ─────────────────────────────────────────────────────────────────────
df = df_out.merge(df_idx, on="custom_id", how="left")
print(f"\nMerged dataset: {len(df):,} rows")

# ── QC ────────────────────────────────────────────────────────────────────────
n_total  = len(df)
n_failed = df["llm_prob"].isna().sum()
fail_pct = n_failed / n_total * 100

print(f"\n{'═'*60}")
print("  QC REPORT")
print(f"{'═'*60}")
print(f"  Total requests   : {n_total:>6,}")
print(f"  Parse failures   : {n_failed:>6,}  ({fail_pct:.2f}%)")
print(f"  {'OK' if fail_pct < 5 else 'WARNING — >5% failures (inspect rag_parse_failures.csv)'}")

print("\n  Mean LLM probability by language (P4 condition):")
means = df[df["llm_prob"].notna()].groupby("language")["llm_prob"].mean().round(3)
print(means.to_string())

print("\n  Distribution of LLM probability outputs:")
valid  = df["llm_prob"].dropna()
bins   = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
labels = ["0.0-0.1","0.1-0.2","0.2-0.3","0.3-0.4","0.4-0.5",
          "0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-1.0"]
counts = pd.cut(valid, bins=bins, labels=labels, right=False).value_counts().sort_index()
for lbl, cnt in counts.items():
    bar = "█" * (cnt * 40 // max(counts))
    print(f"  {lbl}: {cnt:>5}  {bar}")

print(f"\n  Human label mean by language (irony base rate in sample):")
print(df.groupby("language")["human_label"].mean().round(3).to_string())

print(f"\n  Retrieval quality (best demographic score per row):")
if "best_demo_score" in df.columns:
    score_dist = df["best_demo_score"].value_counts().sort_index(ascending=False)
    for sc, cnt in score_dist.items():
        print(f"    Score {sc}: {cnt:>4} rows ({cnt/n_total*100:.1f}%)")

# Save failures
df_fail = df[df["llm_prob"].isna()][["custom_id", "language", "raw_output"]]
df_fail.to_csv(FAILURES_CSV, index=False)
if len(df_fail):
    print(f"\n  First 5 failures:")
    for _, r in df_fail.head(5).iterrows():
        print(f"    [{r['language'].upper()}] {repr(r['raw_output'])}")

print(f"\n{'═'*60}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(RAW_PARQUET, index=False)
print(f"\nSaved (all rows)  : {RAW_PARQUET}   {df.shape}")

df_clean = df[df["llm_prob"].notna()].copy()
df_clean.to_parquet(CLEAN_PARQUET, index=False)
print(f"Saved (clean)     : {CLEAN_PARQUET}   {df_clean.shape}")
print(f"Parse failures    : {FAILURES_CSV}   ({len(df_fail)} rows)")

print("\nNext step: python 75_rag_r2m.py")
