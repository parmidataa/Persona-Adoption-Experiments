"""
Step 1: Load MultiPICo from HuggingFace and explore its structure.
Outputs summary statistics and demographic distributions per language.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset

print("=" * 70)
print("STEP 1: DATA EXPLORATION — MultiPICo Dataset")
print("=" * 70)

# --- Load dataset ---
print("\n[1] Loading MultiPICo from HuggingFace...")
ds = load_dataset("Multilingual-Perspectivist-NLU/MultiPICo")
df = ds["train"].to_pandas()
print(f"    Total rows: {len(df):,}")
print(f"    Columns ({len(df.columns)}): {list(df.columns)}")

# --- Basic info ---
print("\n[2] Column dtypes:")
print(df.dtypes.to_string())

print("\n[3] First 3 rows (transposed):")
print(df.head(3).T.to_string())

# --- Missing values ---
print("\n[4] Missing values per column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
print(missing_df[missing_df["missing_count"] > 0].to_string())
if missing_df["missing_count"].sum() == 0:
    print("    No missing values.")

# --- Identify key columns ---
print("\n[5] Unique value counts for key columns:")
key_cols = ["language", "language_variety", "annotator_id"]
for col in key_cols:
    if col in df.columns:
        print(f"    {col}: {df[col].nunique()} unique values")

# --- Identify the label column ---
print("\n[6] Identifying the irony label column...")
possible_labels = [c for c in df.columns if "iron" in c.lower() or "label" in c.lower() or "annotation" in c.lower()]
print(f"    Possible label columns: {possible_labels}")
for col in possible_labels:
    print(f"    {col} value counts:\n{df[col].value_counts().to_string()}\n")

# --- Identify demographic columns ---
print("\n[7] Demographic columns exploration:")
demo_candidates = ["age", "gender", "ethnicity", "nationality", "education",
                   "sex", "race", "country", "region"]
found_demos = [c for c in df.columns if any(d in c.lower() for d in demo_candidates)]
print(f"    Found demographic columns: {found_demos}")
for col in found_demos:
    print(f"\n    --- {col} ---")
    print(f"    dtype: {df[col].dtype}")
    print(f"    unique values: {df[col].nunique()}")
    if df[col].nunique() <= 30:
        print(f"    value counts:\n{df[col].value_counts().head(20).to_string()}")
    else:
        print(f"    top 20 values:\n{df[col].value_counts().head(20).to_string()}")
        print(f"    ... and {df[col].nunique() - 20} more")

# --- Per-language summary ---
print("\n[8] Per-language summary:")
lang_summary = df.groupby("language").agg(
    n_rows=("language", "size"),
    n_annotators=("annotator_id", "nunique"),
).reset_index()

# Count unique texts per language
text_col_candidates = ["post_id", "text_id", "id"]
text_col = None
for c in text_col_candidates:
    if c in df.columns:
        text_col = c
        break
if text_col:
    texts_per_lang = df.groupby("language")[text_col].nunique().reset_index()
    texts_per_lang.columns = ["language", "n_texts"]
    lang_summary = lang_summary.merge(texts_per_lang, on="language")

print(lang_summary.to_string(index=False))

# --- Per-language demographic distributions ---
print("\n[9] Demographic distributions per language:")
for lang in sorted(df["language"].unique()):
    df_lang = df[df["language"] == lang]
    print(f"\n{'='*50}")
    print(f"  Language: {lang} (n={len(df_lang):,}, annotators={df_lang['annotator_id'].nunique()})")
    print(f"{'='*50}")
    for col in found_demos:
        if col in df_lang.columns:
            vc = df_lang.drop_duplicates("annotator_id")[col].value_counts()
            print(f"  {col}:")
            print(f"    {vc.to_dict()}")

# --- Save cleaned dataframe ---
output_path = "data_raw.parquet"
df.to_parquet(output_path)
print(f"\n[10] Saved raw data to {output_path}")
print(f"     Shape: {df.shape}")

print("\n" + "=" * 70)
print("DATA EXPLORATION COMPLETE")
print("=" * 70)
