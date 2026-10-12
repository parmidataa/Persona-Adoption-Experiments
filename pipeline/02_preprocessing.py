"""
Step 2: Preprocess MultiPICo data for mixed-effects regression.
- Clean demographic variables
- Handle DATA_EXPIRED and missing values
- Merge rare categories for sparse demographics
- Create per-language subsets ready for modeling
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 2: PREPROCESSING")
print("=" * 70)

df = pd.read_parquet("data_raw.parquet")
print(f"Loaded {len(df):,} rows")

# --- Standardize column names ---
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

# --- Use post_id as text_id (each post-reply pair is a unique text) ---
df["text_id"] = df["post_id"].astype(str)

# --- Handle DATA_EXPIRED and missing values ---
expired_mask = (
    (df["gender"] == "DATA_EXPIRED") |
    (df["ethnicity"] == "DATA_EXPIRED") |
    (df["nationality"] == "DATA_EXPIRED") |
    (df["country_birth"] == "DATA_EXPIRED") |
    df["age"].isna()
)
print(f"\nRows with DATA_EXPIRED or missing age: {expired_mask.sum():,} "
      f"({expired_mask.sum()/len(df)*100:.1f}%)")

# Count affected annotators
expired_annotators = df[expired_mask]["annotator_id"].unique()
print(f"Affected annotators: {len(expired_annotators)}")

df_clean = df[~expired_mask].copy()
print(f"Rows after removing expired/missing: {len(df_clean):,}")

# --- Clean gender ---
# Drop "Prefer not to say" (only 1 annotator in Arabic)
prefer_not = df_clean["gender"] == "Prefer not to say"
print(f"\nGender 'Prefer not to say' rows: {prefer_not.sum()}")
df_clean = df_clean[~prefer_not].copy()
print(f"Gender values: {df_clean['gender'].value_counts().to_dict()}")

# --- Clean ethnicity ---
# Merge rare categories per language
print("\nEthnicity distribution (overall, annotator-level):")
eth_per_ann = df_clean.drop_duplicates("annotator_id")["ethnicity"].value_counts()
print(eth_per_ann.to_string())

# For languages with very low ethnicity diversity, some categories will
# be empty. We keep the main categories and merge rare ones into "Other".
# Categories: White, Asian, Black, Mixed, Other
# This is already fairly clean. Keep as-is.

# --- Clean nationality ---
# Nationality has 33 unique values across languages.
# For per-language models, some languages have only 1-2 nationalities.
# We keep nationality as-is for diverse languages, and will skip it
# for languages where it's constant (Italian, Dutch, Hindi).

print("\nNationality unique values per language:")
for lang in sorted(df_clean["language"].unique()):
    n_nat = df_clean[df_clean["language"] == lang].drop_duplicates("annotator_id")["nationality"].nunique()
    n_ann = df_clean[df_clean["language"] == lang]["annotator_id"].nunique()
    print(f"  {lang}: {n_nat} nationalities among {n_ann} annotators")

# --- Standardize age ---
# Center age (subtract mean) so the intercept is interpretable
overall_mean_age = df_clean.drop_duplicates("annotator_id")["age"].mean()
print(f"\nOverall mean age (annotator-level): {overall_mean_age:.1f}")
df_clean["age_centered"] = df_clean["age"] - overall_mean_age

# Also create age bins for robustness check
df_clean["age_bin"] = pd.cut(
    df_clean["age"],
    bins=[0, 25, 35, 45, 100],
    labels=["18-25", "26-35", "36-45", "46+"]
)
print(f"Age bins (annotator-level):")
print(df_clean.drop_duplicates("annotator_id")["age_bin"].value_counts().sort_index().to_string())

# --- Create per-language datasets ---
print("\n" + "=" * 70)
print("PER-LANGUAGE DATASET SUMMARY")
print("=" * 70)

languages = sorted(df_clean["language"].unique())
lang_dfs = {}

for lang in languages:
    df_lang = df_clean[df_clean["language"] == lang].copy()
    n_ann = df_lang["annotator_id"].nunique()
    n_texts = df_lang["text_id"].nunique()
    irony_rate = df_lang["label"].mean()

    # Determine which variables have enough variance for modeling
    usable_vars = ["age_centered"]

    gender_cats = df_lang.drop_duplicates("annotator_id")["gender"].nunique()
    if gender_cats > 1:
        usable_vars.append("C(gender)")

    eth_cats = df_lang.drop_duplicates("annotator_id")["ethnicity"].nunique()
    if eth_cats > 1:
        # Merge rare ethnicity categories (< 2 annotators) into "Other"
        eth_counts = df_lang.drop_duplicates("annotator_id")["ethnicity"].value_counts()
        rare_eths = eth_counts[eth_counts < 2].index.tolist()
        if rare_eths:
            df_lang.loc[df_lang["ethnicity"].isin(rare_eths), "ethnicity"] = "Other"
            eth_cats = df_lang.drop_duplicates("annotator_id")["ethnicity"].nunique()
        if eth_cats > 1:
            usable_vars.append("C(ethnicity)")

    nat_cats = df_lang.drop_duplicates("annotator_id")["nationality"].nunique()
    if nat_cats > 1:
        # Merge rare nationalities (< 2 annotators) into "Other_nat"
        nat_counts = df_lang.drop_duplicates("annotator_id")["nationality"].value_counts()
        rare_nats = nat_counts[nat_counts < 2].index.tolist()
        if rare_nats:
            df_lang.loc[df_lang["nationality"].isin(rare_nats), "nationality"] = "Other_nat"
            nat_cats = df_lang.drop_duplicates("annotator_id")["nationality"].nunique()
        if nat_cats > 1:
            usable_vars.append("C(nationality)")

    lang_dfs[lang] = {
        "data": df_lang,
        "usable_vars": usable_vars,
        "n_annotators": n_ann,
        "n_texts": n_texts,
        "n_rows": len(df_lang),
        "irony_rate": irony_rate,
    }

    print(f"\n  {lang}: {len(df_lang):,} rows, {n_ann} annotators, {n_texts} texts, "
          f"irony_rate={irony_rate:.3f}")
    print(f"    Usable variables: {usable_vars}")
    if "C(ethnicity)" in usable_vars:
        print(f"    Ethnicity cats: {df_lang.drop_duplicates('annotator_id')['ethnicity'].value_counts().to_dict()}")
    if "C(nationality)" in usable_vars:
        print(f"    Nationality cats: {df_lang.drop_duplicates('annotator_id')['nationality'].value_counts().to_dict()}")

# --- Save preprocessed data ---
df_clean.to_parquet("data_preprocessed.parquet")

# Save per-language data
for lang, info in lang_dfs.items():
    info["data"].to_parquet(f"data_{lang}.parquet")

# Save metadata
import json
meta = {lang: {k: v for k, v in info.items() if k != "data"} for lang, info in lang_dfs.items()}
with open("lang_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved preprocessed data and {len(languages)} per-language files.")

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
