"""
28_parse_results.py
===================
Step 4: Parse batch_output.jsonl → data_predictions.parquet

For each response:
  1. Extract raw LLM content from choices[0].message.content
  2. Parse "0" or "1" (take first digit found)
  3. Join with batch_index.jsonl to attach all metadata
  4. QC report: parse failure rate, label distribution, token usage
  5. Save clean dataset to data_predictions.parquet

Output: data_predictions.parquet
"""

import json
import pandas as pd
from pathlib import Path

OUTPUT_FILE = Path("batch_output.jsonl")
INDEX_FILE  = Path("batch_index.jsonl")
RESULT_FILE = Path("data_predictions.parquet")

# ── Parse label from raw LLM output ─────────────────────────────────────────
def parse_label(raw: str | None) -> int | None:
    """Take the first '0' or '1' character in the string. Return None if not found."""
    if raw is None:
        return None
    for ch in raw.strip():
        if ch in ("0", "1"):
            return int(ch)
    return None

# ── Load batch output ─────────────────────────────────────────────────────────
print("Loading batch output...")
output_rows = []
total_tokens = 0

with open(OUTPUT_FILE, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        custom_id = item["custom_id"]

        # Extract raw content
        try:
            choices = item["response"]["body"]["choices"]
            raw = choices[0]["message"]["content"] if choices else None
        except (KeyError, IndexError, TypeError):
            raw = None

        # Extract token usage
        try:
            usage = item["response"]["body"]["usage"]
            prompt_tok = usage.get("prompt_tokens", 0)
            output_tok = usage.get("completion_tokens", 0)
            total_tokens += prompt_tok + output_tok
        except (KeyError, TypeError):
            prompt_tok = output_tok = 0

        output_rows.append({
            "custom_id":    custom_id,
            "raw_output":   raw,
            "llm_label":    parse_label(raw),
            "prompt_tokens": prompt_tok,
            "output_tokens": output_tok,
        })

df_out = pd.DataFrame(output_rows)
print(f"  Loaded {len(df_out):,} response rows")
print(f"  Total tokens used: {total_tokens:,}  "
      f"(≈ ${total_tokens * 0.075 / 1_000_000 + total_tokens * 0.3 / 1_000_000:.4f} est.)")

# ── Load batch index (metadata) ───────────────────────────────────────────────
print("\nLoading batch index...")
index_rows = []
with open(INDEX_FILE, encoding="utf-8") as f:
    for line in f:
        index_rows.append(json.loads(line))
df_idx = pd.DataFrame(index_rows)
print(f"  Loaded {len(df_idx):,} index rows")

# ── Merge ─────────────────────────────────────────────────────────────────────
df = df_out.merge(df_idx, on="custom_id", how="left")
print(f"\nMerged dataset: {len(df):,} rows")

# ── QC: Parse failures ────────────────────────────────────────────────────────
n_total       = len(df)
n_failed      = df["llm_label"].isna().sum()
fail_rate     = n_failed / n_total * 100

print(f"\n{'═'*55}")
print(f"  QC REPORT")
print(f"{'═'*55}")
print(f"  Total requests:    {n_total:>7,}")
print(f"  Parse failures:    {n_failed:>7,}  ({fail_rate:.2f}%)")
print(f"  {'OK' if fail_rate < 5 else 'WARNING — >5% failures, inspect raw outputs'}")

if n_failed > 0:
    print(f"\n  Failed raw outputs (first 10):")
    failed = df[df["llm_label"].isna()][["custom_id", "raw_output"]].head(10)
    for _, row in failed.iterrows():
        print(f"    {row['custom_id']}: {repr(row['raw_output'])}")

# ── QC: Failure rate per condition ─────────────────────────────────────────────
print(f"\n  Parse failure rate by condition:")
for cond in sorted(df["condition"].unique()):
    sub = df[df["condition"] == cond]
    n_fail = sub["llm_label"].isna().sum()
    print(f"    {cond.upper()}: {n_fail:>4} / {len(sub):>6}  ({n_fail/len(sub)*100:.2f}%)")

# ── QC: Label distribution by condition × language ───────────────────────────
print(f"\n  Mean LLM label (proportion ironic) by condition:")
pivot = (
    df[df["llm_label"].notna()]
    .groupby(["condition", "language"])["llm_label"]
    .mean()
    .unstack("language")
    .round(3)
)
print(pivot.to_string())

# ── QC: Compare to human labels ───────────────────────────────────────────────
print(f"\n  Human label mean (ground truth) by language:")
human_means = df.groupby("language")["human_label"].mean().round(3)
print(human_means.to_string())

print(f"\n{'═'*55}")

# ── Save ─────────────────────────────────────────────────────────────────────
# Keep all rows (including parse failures) — flag them with llm_label = NaN
# The analysis script drops NaN rows per condition
df.to_parquet(RESULT_FILE, index=False)
print(f"\nSaved: {RESULT_FILE}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Also save a clean version (no parse failures) for quick use
df_clean = df[df["llm_label"].notna()].copy()
df_clean["llm_label"] = df_clean["llm_label"].astype(int)
df_clean.to_parquet("data_predictions_clean.parquet", index=False)
print(f"\nClean version (no NaN labels): data_predictions_clean.parquet")
print(f"  {len(df_clean):,} rows ({n_total - len(df_clean)} dropped)")
