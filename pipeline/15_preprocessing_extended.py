"""
Step 15: Extended preprocessing for Tasks 1B, 2A, 2B, 3A1, 3A2.

Differences from 02_preprocessing.py:
  - Exposes country_birth, country_residence, student_status, employment_status,
    generation as modeling-ready columns (with rare-cat merging per language).
  - Builds an annotator-level leave-one-text-out base-rate proxy
    (`base_rate_loo`) for use as an "attitudinal proxy" fixed effect in Task 2B.
    Definition: for each (annotator, text_id), base_rate_loo is the annotator's
    mean label across all OTHER texts in their language.
  - Saves data_{lang}_v2.parquet and lang_metadata_v2.json.

Sanity checks logged at the end include row-count delta vs the v1 datasets and
a per-language summary of which extended fields have >= 2 categories.
"""

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 70)
print("STEP 15: EXTENDED PREPROCESSING")
print("=" * 70)

df = pd.read_parquet("data_raw.parquet")
print(f"Loaded {len(df):,} rows from data_raw.parquet")

# --- Standardize column names (mirror 02_preprocessing.py + extras) ---
df = df.rename(columns={
    "Age": "age",
    "Gender": "gender",
    "Ethnicity simplified": "ethnicity",
    "Nationality": "nationality",
    "Country of birth": "country_birth",
    "Country of residence": "country_residence",
    "Student status": "student_status",
    "Employment status": "employment_status",
    "GenerationAggregated": "generation",
})
df["text_id"] = df["post_id"].astype(str)

# --- Drop DATA_EXPIRED / missing across the EXTENDED set of fields ---
expired_mask = (
    (df["gender"] == "DATA_EXPIRED")
    | (df["ethnicity"] == "DATA_EXPIRED")
    | (df["nationality"] == "DATA_EXPIRED")
    | (df["country_birth"] == "DATA_EXPIRED")
    | (df["country_residence"] == "DATA_EXPIRED")
    | (df["student_status"] == "DATA_EXPIRED")
    | (df["employment_status"] == "DATA_EXPIRED")
    | df["age"].isna()
)
print(
    f"\nRows with DATA_EXPIRED in any extended field or missing age: "
    f"{expired_mask.sum():,} ({expired_mask.sum()/len(df)*100:.1f}%)"
)

df_clean = df[~expired_mask].copy()

# Drop rare gender bucket as in v1
prefer_not = df_clean["gender"] == "Prefer not to say"
df_clean = df_clean[~prefer_not].copy()
print(f"Rows after extended cleaning: {len(df_clean):,}")

# --- Centered age + binned age (binned version used by Task 3A2) ---
overall_mean_age = df_clean.drop_duplicates("annotator_id")["age"].mean()
df_clean["age_centered"] = df_clean["age"] - overall_mean_age
df_clean["age_bin"] = pd.cut(
    df_clean["age"],
    bins=[0, 25, 35, 45, 100],
    labels=["18-25", "26-35", "36-45", "46+"],
).astype(str)

# --- Leave-one-TEXT-out annotator base rate (attitudinal proxy) ----------
# For each (annotator, text_id) compute mean of `label` across that
# annotator's OTHER texts. Computed per-language to avoid cross-language
# leakage of irony rates.
print("\nComputing leave-one-text-out base rate per annotator...")

def loo_base_rate(group: pd.DataFrame) -> pd.Series:
    s = group["label"].astype(float)
    # If an annotator labelled the same text_id more than once (shouldn't
    # happen here but be defensive), average across rows so the LOO rate is
    # over distinct texts.
    by_text = s.groupby(group["text_id"]).mean()
    n_texts = len(by_text)
    total = by_text.sum()
    out = pd.Series(index=group.index, dtype=float)
    for idx, tid in zip(group.index, group["text_id"]):
        if n_texts <= 1:
            out.loc[idx] = np.nan
            continue
        rate = (total - by_text.loc[tid]) / (n_texts - 1)
        out.loc[idx] = float(rate)
    return out

base_rates = []
for (lang, ann_id), grp in df_clean.groupby(["language", "annotator_id"], sort=False):
    base_rates.append(loo_base_rate(grp))
df_clean["base_rate_loo"] = pd.concat(base_rates).sort_index()
n_nan = df_clean["base_rate_loo"].isna().sum()
print(
    f"  base_rate_loo NaN count (single-text annotators): {n_nan:,}"
    f" ({n_nan/len(df_clean)*100:.2f}%)"
)
df_clean = df_clean[df_clean["base_rate_loo"].notna()].copy()
print(f"  Rows after dropping single-text annotators: {len(df_clean):,}")

# --- Per-language datasets with rare-category merging on every cat field ---
CAT_FIELDS = [
    "ethnicity",
    "nationality",
    "country_birth",
    "country_residence",
    "student_status",
    "employment_status",
    "generation",
]

print("\n" + "=" * 70)
print("PER-LANGUAGE EXTENDED DATASETS")
print("=" * 70)

languages = sorted(df_clean["language"].unique())
lang_meta = {}

for lang in languages:
    df_lang = df_clean[df_clean["language"] == lang].copy()
    n_ann = df_lang["annotator_id"].nunique()
    n_texts = df_lang["text_id"].nunique()

    usable_cat_vars = []
    field_summary = {}
    for field in CAT_FIELDS:
        ann_level = df_lang.drop_duplicates("annotator_id")[field]
        counts = ann_level.value_counts()
        rare = counts[counts < 2].index.tolist()
        if rare:
            replacement = f"Other_{field}"
            df_lang.loc[df_lang[field].isin(rare), field] = replacement
        n_cats = df_lang.drop_duplicates("annotator_id")[field].nunique()
        field_summary[field] = {
            "n_categories": int(n_cats),
            "categories": (
                df_lang.drop_duplicates("annotator_id")[field]
                .value_counts()
                .to_dict()
            ),
        }
        if n_cats > 1:
            usable_cat_vars.append(field)

    # Always check gender (kept already, but record categories)
    field_summary["gender"] = {
        "n_categories": int(df_lang.drop_duplicates("annotator_id")["gender"].nunique()),
        "categories": (
            df_lang.drop_duplicates("annotator_id")["gender"].value_counts().to_dict()
        ),
    }

    df_lang.to_parquet(f"data_{lang}_v2.parquet")

    lang_meta[lang] = {
        "n_rows": int(len(df_lang)),
        "n_annotators": int(n_ann),
        "n_texts": int(n_texts),
        "irony_rate": float(df_lang["label"].mean()),
        "usable_cat_vars": usable_cat_vars,
        "fields": field_summary,
    }

    print(
        f"\n  {lang}: {len(df_lang):,} rows | {n_ann} annotators | {n_texts} texts"
    )
    print(f"    Usable categorical vars (>=2 cats): {usable_cat_vars}")

with open("lang_metadata_v2.json", "w") as fh:
    json.dump(lang_meta, fh, indent=2)
df_clean.to_parquet("data_preprocessed_v2.parquet")

# --- Compare row counts to v1 ---
print("\n" + "=" * 70)
print("ROW-COUNT DELTA vs v1 (data_{lang}.parquet)")
print("=" * 70)
try:
    with open("lang_metadata.json") as fh:
        v1_meta = json.load(fh)
    rows = []
    for lang in languages:
        v1n = v1_meta.get(lang, {}).get("n_rows")
        v2n = lang_meta[lang]["n_rows"]
        delta = v2n - (v1n or 0) if v1n is not None else None
        rows.append({
            "language": lang,
            "v1_rows": v1n,
            "v2_rows": v2n,
            "delta_v2_minus_v1": delta,
        })
    delta_df = pd.DataFrame(rows)
    print(delta_df.to_string(index=False))
    delta_df.to_csv("results_v2_row_count_delta.csv", index=False)
except FileNotFoundError:
    print("  lang_metadata.json (v1) not found; skipping delta table.")

print("\nSaved data_{lang}_v2.parquet for all languages.")
print("Saved lang_metadata_v2.json.")
print("\n" + "=" * 70)
print("STEP 15 COMPLETE")
print("=" * 70)
