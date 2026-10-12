"""
83_parse_p5_results.py
======================
Parse p5_batch_output.jsonl into p5_predictions_clean.parquet.

The model responds in the format:
    Reasoning: [one sentence]
    Deviation: [signed decimal]

The parser extracts the signed decimal deviation from the "Deviation:" line.
The deviation represents percentage-point shift from the language baseline:
  +0.15 = model thinks this annotator would rate the text 15 pp MORE ironic
           than the 34 % / 38.3 % population average
  -0.08 = 8 pp LESS ironic than average

Derived columns added after parsing:
  llm_deviation  — the raw signed decimal from the model  ∈ [−1, 1]
  llm_prob       — baseline_rate + llm_deviation  (reconstructed probability)
                   clipped to [0, 1] for LMM compatibility
  llm_reasoning  — the reasoning sentence (kept for qualitative inspection)

The llm_prob column allows direct comparison with P2-binary, P2-prob, and
P4-RAG using the same R²m and concordance pipeline.

Outputs:
    p5_predictions_raw.parquet    — all rows (NaN for failed parses)
    p5_predictions_clean.parquet  — valid rows only
    p5_parse_failures.csv         — failed rows for inspection
    p5_reasoning_sample.csv       — first 20 reasoning sentences (sanity check)

Usage:
    python 83_parse_p5_results.py
    (Run AFTER 82_retrieve_p5_batch.py)
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_FILE   = Path("p5_batch_output.jsonl")
INDEX_FILE    = Path("p5_batch_index.jsonl")
RAW_PARQUET   = Path("p5_predictions_raw.parquet")
CLEAN_PARQUET = Path("p5_predictions_clean.parquet")
FAILURES_CSV  = Path("p5_parse_failures.csv")
REASONING_CSV = Path("p5_reasoning_sample.csv")

# ── Parsers ───────────────────────────────────────────────────────────────────
# Primary: look for "Deviation: +0.15" or "Deviation: -0.08" or "Deviation: 0.0"
_DEV_LINE_RE = re.compile(
    r"[Dd]eviation\s*:\s*([+-]?\d*\.?\d+)"
)
# Fallback: last signed or unsigned decimal in the response
_LAST_DECIMAL_RE = re.compile(
    r"([+-]?\d+\.?\d*)\s*$"
)
# Reasoning line
_REASONING_RE = re.compile(
    r"[Rr]easoning\s*:\s*(.+)"
)


def parse_deviation(raw: str | None) -> tuple[float | None, str | None]:
    """
    Returns (deviation, reasoning) or (None, reasoning) if parse fails.
    deviation is validated to be in [-1, 1].
    """
    if raw is None:
        return None, None

    # Extract reasoning (best-effort)
    reasoning = None
    m_r = _REASONING_RE.search(raw)
    if m_r:
        reasoning = m_r.group(1).strip()

    # Primary: "Deviation: X"
    m = _DEV_LINE_RE.search(raw)
    if m:
        try:
            v = float(m.group(1))
            if -1.0 <= v <= 1.0:
                return v, reasoning
        except ValueError:
            pass

    # Fallback: last decimal in the string
    m2 = _LAST_DECIMAL_RE.search(raw.strip())
    if m2:
        try:
            v = float(m2.group(1))
            if -1.0 <= v <= 1.0:
                return v, reasoning
        except ValueError:
            pass

    return None, reasoning


# ── Load batch output ─────────────────────────────────────────────────────────
if not OUTPUT_FILE.exists():
    print(f"ERROR: {OUTPUT_FILE} not found. Run 82_retrieve_p5_batch.py first.")
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

        dev, reasoning = parse_deviation(raw)
        output_rows.append({
            "custom_id":      custom_id,
            "raw_output":     raw,
            "llm_deviation":  dev,
            "llm_reasoning":  reasoning,
            "prompt_tokens":  prompt_tok,
            "output_tokens":  output_tok,
        })

df_out = pd.DataFrame(output_rows)
print(f"  Loaded {len(df_out):,} response rows")
print(f"  Total tokens used : {total_tokens:,}")
cost = (total_tokens * 0.40 / 1_000_000) * 0.5
print(f"  Estimated cost    : ~${cost:.4f}  (Batch API)")

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

# ── Reconstruct probability ───────────────────────────────────────────────────
# llm_prob = baseline_rate + llm_deviation, clipped to [0, 1]
# This allows direct comparison with P2-prob and P4-RAG using the same pipeline.
df["llm_prob"] = np.clip(
    df["baseline_rate"].astype(float) + df["llm_deviation"].astype(float),
    0.0, 1.0
)

# ── QC ────────────────────────────────────────────────────────────────────────
n_total  = len(df)
n_failed = df["llm_deviation"].isna().sum()
fail_pct = n_failed / n_total * 100

print(f"\n{'═'*65}")
print("  QC REPORT")
print(f"{'═'*65}")
print(f"  Total requests    : {n_total:>6,}")
print(f"  Parse failures    : {n_failed:>6,}  ({fail_pct:.2f}%)")
print(f"  {'OK' if fail_pct < 5 else 'WARNING — >5% failures (inspect p5_parse_failures.csv)'}")

print(f"\n  Deviation distribution by language (P5 condition):")
for lang in ["ar", "es"]:
    sub = df[(df["language"] == lang) & df["llm_deviation"].notna()]
    if len(sub) > 0:
        print(f"  {lang.upper()}: mean={sub['llm_deviation'].mean():+.3f}  "
              f"std={sub['llm_deviation'].std():.3f}  "
              f"min={sub['llm_deviation'].min():+.3f}  "
              f"max={sub['llm_deviation'].max():+.3f}  "
              f"n={len(sub)}")

print(f"\n  Distribution of signed deviations (all rows):")
valid = df["llm_deviation"].dropna()
bins   = [-1.01, -0.5, -0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3, 0.5, 1.01]
labels = ["≤-0.5","-0.5:-0.3","-0.3:-0.1","-0.1:-0.05","-0.05:0",
          "0:0.05","0.05:0.1","0.1:0.3","0.3:0.5","≥0.5"]
counts = pd.cut(valid, bins=bins, labels=labels, right=True).value_counts().sort_index()
for lbl, cnt in counts.items():
    bar = "█" * max(1, cnt * 40 // max(counts)) if cnt > 0 else ""
    print(f"  {lbl:>12}: {cnt:>5}  {bar}")

print(f"\n  Reconstructed llm_prob summary:")
prob_valid = df["llm_prob"].dropna()
print(f"  mean={prob_valid.mean():.3f}  std={prob_valid.std():.3f}  "
      f"min={prob_valid.min():.3f}  max={prob_valid.max():.3f}")

print(f"\n  Human label mean by language (reference):")
print(df.groupby("language")["human_label"].mean().round(3).to_string())

# ── Reasoning sample ─────────────────────────────────────────────────────────
df_reason = df[df["llm_reasoning"].notna()][
    ["language", "llm_deviation", "llm_reasoning"]
].head(20)
df_reason.to_csv(REASONING_CSV, index=False)
print(f"\n  Reasoning sample saved: {REASONING_CSV}  (first 20 rows)")

# Print a few examples for immediate inspection
print("\n  Sample reasonings:")
for _, r in df_reason.head(5).iterrows():
    print(f"  [{r['language'].upper()} {r['llm_deviation']:+.2f}] {r['llm_reasoning'][:120]}")

# ── Save failures ─────────────────────────────────────────────────────────────
df_fail = df[df["llm_deviation"].isna()][["custom_id", "language", "raw_output"]]
df_fail.to_csv(FAILURES_CSV, index=False)
if len(df_fail):
    print(f"\n  First 5 failures:")
    for _, r in df_fail.head(5).iterrows():
        print(f"  [{r['language'].upper()}] {repr(r['raw_output'][:200])}")

print(f"\n{'═'*65}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(RAW_PARQUET, index=False)
print(f"\nSaved (all rows)  : {RAW_PARQUET}   {df.shape}")

df_clean = df[df["llm_deviation"].notna()].copy()
df_clean.to_parquet(CLEAN_PARQUET, index=False)
print(f"Saved (clean)     : {CLEAN_PARQUET}   {df_clean.shape}")
print(f"Parse failures    : {FAILURES_CSV}   ({len(df_fail)} rows)")

print("\nNext step: python 84_p5_r2m.py")
