"""
25_prepare_data.py
==================
Step 1 of the LLM persona prompting experiment.

Selects 700 maximally informative, demographically diverse annotations
per language from the pre-processed per-language parquet files.

Pipeline per language:
  A. Text entropy filter      — keep intermediate-entropy texts (0.25–0.92)
  B. Annotator depth filter   — keep annotators with >= MIN_DEPTH texts
  C. Greedy diversity         — select annotators maximising demographic spread
  D. Stratified sample        — sample exactly TARGET_N rows, balanced by annotator

Output: data_selected.parquet  (6,300 rows x 9 languages)

Implementation notes:
  - annotator_id is stored as uint64 in parquets; converted to str immediately
  - All groupby / nunique operations avoided on ArrowDtype columns to prevent
    pandas 2.x bugs; replaced with explicit Python loops
  - groupby.apply avoided because in pandas 2.2+ it drops the grouping column
    from the result when ArrowDtype is used
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import LabelEncoder

# ── Parameters ────────────────────────────────────────────────────────────────
LANGUAGES         = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]
TARGET_N          = 700
MIN_DEPTH         = 5      # minimum texts per annotator after entropy filtering
ENTROPY_LOW       = 0.25
ENTROPY_HIGH      = 0.92
MIN_ANNOTATORS_PER_TEXT = 3
RANDOM_STATE      = 42

DATA_DIR    = Path(".")
OUTPUT_PATH = DATA_DIR / "data_selected.parquet"

DEMO_COLS = ["age_bin", "gender", "ethnicity", "nationality"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return scipy_entropy([p, 1 - p], base=2)


def safe_load(lang: str) -> pd.DataFrame:
    """
    Load per-language parquet and convert all columns to plain Python types.
    - annotator_id (uint64) → str
    - ArrowString columns   → object (Python str)
    - numeric columns       → kept as float/int numpy types
    Uses explicit column-by-column numpy conversion to avoid pandas ArrowDtype bugs.
    """
    df = pd.read_parquet(DATA_DIR / f"data_{lang}.parquet")

    for col in df.columns:
        arr = df[col].to_numpy()           # always gives numpy array, even for Arrow
        if col == "annotator_id":
            df[col] = arr.astype(str)      # uint64 → str
        elif arr.dtype.kind in ("U", "O"):  # unicode or object (str-like)
            df[col] = [str(v) for v in arr]
        elif "str" in str(df[col].dtype).lower() or "arrow" in str(df[col].dtype).lower():
            df[col] = [str(v) for v in arr]
        # otherwise leave numeric columns as-is

    # Ensure key numerics are float
    for col in ["label", "age", "age_centered"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure age_bin exists
    if "age_bin" not in df.columns:
        bins   = [0, 26, 36, 46, 200]
        labels = ["18-25", "26-35", "36-45", "46+"]
        df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=False).astype(str)

    return df


def unique_annotators(df: pd.DataFrame) -> list:
    """Return sorted list of unique annotator_id values."""
    return sorted(set(df["annotator_id"].tolist()))


def greedy_diverse_annotators(df_lang: pd.DataFrame, n_select: int) -> list:
    """
    Select n_select annotators that maximise demographic spread (greedy max-coverage).
    Returns a list of annotator_id strings.
    """
    all_anns = unique_annotators(df_lang)
    if n_select >= len(all_anns):
        return all_anns

    # Build per-annotator demographic profile
    profile_rows = {}
    for row in df_lang[["annotator_id"] + DEMO_COLS].drop_duplicates("annotator_id").itertuples(index=False):
        profile_rows[row.annotator_id] = {c: getattr(row, c) for c in DEMO_COLS}

    # Label-encode demographics for distance computation
    encoders = {}
    for col in DEMO_COLS:
        le = LabelEncoder()
        le.fit([str(profile_rows[a][col]) for a in all_anns])
        encoders[col] = le

    X = np.array([
        [encoders[col].transform([str(profile_rows[a][col])])[0] for col in DEMO_COLS]
        for a in all_anns
    ], dtype=float)

    centroid  = X.mean(axis=0)
    sel_idx   = [int(np.argmax(np.linalg.norm(X - centroid, axis=1)))]

    for _ in range(n_select - 1):
        sel_X = X[sel_idx]
        dists = np.min(
            np.linalg.norm(X[:, None, :] - sel_X[None, :, :], axis=2), axis=1
        )
        dists[sel_idx] = -np.inf
        sel_idx.append(int(np.argmax(dists)))

    return [all_anns[i] for i in sel_idx]


def stratified_sample(df_lang: pd.DataFrame, n_total: int, min_per_ann: int) -> pd.DataFrame:
    """
    Sample n_total rows from df_lang, stratified by annotator_id.
    Avoids groupby.apply (buggy with ArrowDtype in pandas 2.2+).
    """
    annotators = unique_annotators(df_lang)
    n_ann      = len(annotators)
    per_ann    = max(min_per_ann, n_total // n_ann)

    parts = []
    for ann in annotators:
        ann_df = df_lang[df_lang["annotator_id"] == ann]
        n_take = min(len(ann_df), per_ann)
        parts.append(ann_df.sample(n_take, random_state=RANDOM_STATE))

    sampled = pd.concat(parts, ignore_index=True)

    if len(sampled) < n_total:
        used_idx = set(sampled.index) if not sampled.index.is_unique else set(sampled.index.tolist())
        leftover = df_lang[~df_lang.index.isin(used_idx)]
        n_need   = min(n_total - len(sampled), len(leftover))
        if n_need > 0:
            topup  = leftover.sample(n_need, random_state=RANDOM_STATE)
            sampled = pd.concat([sampled, topup], ignore_index=True)

    return sampled.head(n_total).reset_index(drop=True)


# ── Main selection loop ───────────────────────────────────────────────────────
selected_frames = []

print("=" * 72)
print(f"{'Language':<6}  {'Input':>7}  {'After A':>8}  {'After B':>8}  "
      f"{'Annotators':>10}  {'Selected':>9}  {'MeanH':>6}")
print("-" * 72)

for lang in LANGUAGES:
    df = safe_load(lang)

    # ── Step A: text entropy filter ───────────────────────────────────────────
    # Compute per-text label mean and annotator count using Python dicts
    post_label_sum   = {}
    post_label_count = {}
    for pid, lbl in zip(df["post_id"].tolist(), df["label"].tolist()):
        if pd.isna(lbl):
            continue
        post_label_sum[pid]   = post_label_sum.get(pid, 0) + lbl
        post_label_count[pid] = post_label_count.get(pid, 0) + 1

    informative_posts = set()
    post_entropy = {}
    for pid, cnt in post_label_count.items():
        if cnt < MIN_ANNOTATORS_PER_TEXT:
            continue
        p = post_label_sum[pid] / cnt
        h = binary_entropy(p)
        post_entropy[pid] = h
        if ENTROPY_LOW < h < ENTROPY_HIGH:
            informative_posts.add(pid)

    df_a = df[[pid in informative_posts for pid in df["post_id"].tolist()]].copy()

    # ── Step B: annotator depth filter ───────────────────────────────────────
    ann_text_count = {}
    for ann in df_a["annotator_id"].tolist():
        ann_text_count[ann] = ann_text_count.get(ann, 0) + 1
    qualified = {a for a, c in ann_text_count.items() if c >= MIN_DEPTH}
    df_b = df_a[[a in qualified for a in df_a["annotator_id"].tolist()]].copy()

    # ── Step C: greedy demographic diversity ─────────────────────────────────
    n_ann_available = len(unique_annotators(df_b))
    n_to_select     = min(n_ann_available, max(10, int(0.8 * n_ann_available)))
    diverse_anns    = set(greedy_diverse_annotators(df_b, n_to_select))
    df_c            = df_b[[a in diverse_anns for a in df_b["annotator_id"].tolist()]].copy()

    # ── Step D: stratified sample ─────────────────────────────────────────────
    if len(df_c) < TARGET_N:
        sampled  = df_c.copy().reset_index(drop=True)
    else:
        sampled  = stratified_sample(df_c, TARGET_N, MIN_DEPTH)

    actual_n = len(sampled)

    # Attach text entropy
    sampled["text_entropy"] = [post_entropy.get(pid, 0.0) for pid in sampled["post_id"].tolist()]

    selected_frames.append(sampled)

    n_anns_final = len(unique_annotators(sampled))
    mean_h       = sampled["text_entropy"].mean()
    print(f"{lang.upper():<6}  {len(df):>7,}  {len(df_a):>8,}  {len(df_b):>8,}  "
          f"{n_ann_available:>10}  {actual_n:>9}  {mean_h:>6.3f}  "
          f"({n_anns_final} anns in sample)")

print("=" * 72)

# ── Combine ───────────────────────────────────────────────────────────────────
df_selected = pd.concat(selected_frames, ignore_index=True)

print(f"\nTotal: {len(df_selected):,} rows | {len(LANGUAGES)} languages\n")

# ── Validate ──────────────────────────────────────────────────────────────────
print(f"{'Lang':<6} {'Rows':>6}  {'Annotators':>11}  {'MeanH':>6}")
print(f"{'-'*6} {'-'*6}  {'-'*11}  {'-'*6}")
all_ok = True
for lang in sorted(LANGUAGES):
    lang_rows = df_selected[[v == lang for v in df_selected["language"].tolist()]]
    n_rows    = len(lang_rows)
    n_anns    = len(set(lang_rows["annotator_id"].tolist()))
    mean_h    = lang_rows["text_entropy"].mean() if n_rows > 0 else 0.0
    flag      = "" if n_anns >= MIN_DEPTH else "  <-- WARNING"
    if n_anns < MIN_DEPTH:
        all_ok = False
    print(f"{lang.upper():<6} {n_rows:>6}  {n_anns:>11}  {mean_h:>6.3f}{flag}")

assert all_ok, "Some languages have fewer than 5 annotators — check the selection pipeline."
assert df_selected["label"].isna().sum() == 0, "Null labels found"
assert df_selected["reply"].isna().sum() == 0, "Null reply text found"

# ── Save ─────────────────────────────────────────────────────────────────────
df_selected.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"Shape: {df_selected.shape}")
print(f"Columns: {df_selected.columns.tolist()}")
