"""
Step 7: Undersample all languages to match the smallest language (Hindi).

Problem: Language sample sizes are highly imbalanced:
  Hindi:      4,711  (smallest)
  Spanish:   23,441  (largest, ~5x Hindi)

Strategy: For each language, randomly sample N_target rows (= Hindi's count),
stratified by annotator_id so every annotator retains representation.
This produces balanced per-language datasets for a fairer cross-language comparison.
"""

import pandas as pd
import numpy as np
import json

SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("STEP 7: UNDERSAMPLED PREPROCESSING")
print("=" * 70)

with open("lang_metadata.json") as f:
    meta = json.load(f)

# Determine the target sample size (smallest language)
lang_sizes = {lang: info["n_rows"] for lang, info in meta.items()}
min_lang = min(lang_sizes, key=lang_sizes.get)
n_target = lang_sizes[min_lang]
print(f"\nSmallest language: {min_lang} with {n_target:,} rows")
print(f"All languages will be undersampled to {n_target:,} rows.\n")

print("Original sample sizes:")
for lang in sorted(lang_sizes, key=lang_sizes.get, reverse=True):
    print(f"  {lang}: {lang_sizes[lang]:,} rows")

# --- Undersample each language ---
undersampled_dfs = {}
undersampled_meta = {}

for lang in sorted(meta.keys()):
    df_lang = pd.read_parquet(f"data_{lang}.parquet")
    info = meta[lang]
    original_n = len(df_lang)

    if original_n <= n_target:
        df_sampled = df_lang.copy()
        print(f"\n  {lang}: {original_n:,} rows — NO undersampling needed (already ≤ target)")
    else:
        # Stratified sampling by annotator_id: each annotator contributes
        # proportionally to their share in the original dataset.
        annotator_counts = df_lang["annotator_id"].value_counts()
        annotator_fractions = annotator_counts / annotator_counts.sum()

        # Allocate rows per annotator (at least 1 per annotator)
        rows_per_annotator = (annotator_fractions * n_target).astype(int)
        rows_per_annotator = rows_per_annotator.clip(lower=1)

        # Adjust to hit exact target: distribute remainder or trim excess
        diff = n_target - rows_per_annotator.sum()
        if diff > 0:
            # Add extra rows to the largest annotators
            candidates = rows_per_annotator.index.tolist()
            for i in range(diff):
                ann = candidates[i % len(candidates)]
                max_possible = annotator_counts[ann]
                if rows_per_annotator[ann] < max_possible:
                    rows_per_annotator[ann] += 1
        elif diff < 0:
            # Remove rows from the largest allocations
            sorted_anns = rows_per_annotator.sort_values(ascending=False).index.tolist()
            for i in range(-diff):
                ann = sorted_anns[i % len(sorted_anns)]
                if rows_per_annotator[ann] > 1:
                    rows_per_annotator[ann] -= 1

        # Sample from each annotator
        sampled_parts = []
        for ann_id, n_sample in rows_per_annotator.items():
            ann_data = df_lang[df_lang["annotator_id"] == ann_id]
            n_sample = min(n_sample, len(ann_data))
            sampled_parts.append(ann_data.sample(n=n_sample, random_state=SEED))

        df_sampled = pd.concat(sampled_parts, ignore_index=True)
        print(f"\n  {lang}: {original_n:,} → {len(df_sampled):,} rows "
              f"(target={n_target:,}, undersampled {original_n - len(df_sampled):,} rows)")

    n_ann = df_sampled["annotator_id"].nunique()
    n_texts = df_sampled["text_id"].nunique()
    irony_rate = df_sampled["label"].mean()

    print(f"    Annotators: {info['n_annotators']} → {n_ann} "
          f"| Texts: {info['n_texts']} → {n_texts} "
          f"| Irony rate: {info['irony_rate']:.3f} → {irony_rate:.3f}")

    undersampled_dfs[lang] = df_sampled
    undersampled_meta[lang] = {
        "usable_vars": info["usable_vars"],
        "n_annotators": n_ann,
        "n_texts": n_texts,
        "n_rows": len(df_sampled),
        "irony_rate": irony_rate,
        "original_n_rows": original_n,
        "original_n_annotators": info["n_annotators"],
        "original_n_texts": info["n_texts"],
    }

# --- Save undersampled data ---
for lang, df_sampled in undersampled_dfs.items():
    df_sampled.to_parquet(f"data_{lang}_undersampled.parquet")

with open("lang_metadata_undersampled.json", "w") as f:
    json.dump(undersampled_meta, f, indent=2)

# --- Summary comparison table ---
print("\n\n" + "=" * 70)
print("UNDERSAMPLING SUMMARY")
print("=" * 70)
print(f"\n{'Language':<12} {'Original':>10} {'Undersampled':>14} {'Reduction':>10} "
      f"{'Ann(orig)':>10} {'Ann(new)':>10} {'Texts(orig)':>12} {'Texts(new)':>12}")
print("-" * 92)
for lang in sorted(undersampled_meta.keys()):
    m = undersampled_meta[lang]
    reduction = (1 - m["n_rows"] / m["original_n_rows"]) * 100
    print(f"  {lang:<10} {m['original_n_rows']:>10,} {m['n_rows']:>14,} "
          f"{reduction:>9.1f}% {m['original_n_annotators']:>10} {m['n_annotators']:>10} "
          f"{m['original_n_texts']:>12} {m['n_texts']:>12}")

print(f"\nTarget sample size: {n_target:,} (from {min_lang})")
print(f"Random seed: {SEED}")
print(f"\nSaved {len(undersampled_dfs)} undersampled parquet files and lang_metadata_undersampled.json")

print("\n" + "=" * 70)
print("UNDERSAMPLED PREPROCESSING COMPLETE")
print("=" * 70)
