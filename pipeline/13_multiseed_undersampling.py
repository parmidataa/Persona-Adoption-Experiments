"""
Step 13: Multi-seed undersampling stability analysis.

Repeats the undersampling procedure from Step 7 across 50 different
random seeds. For each seed, all languages are undersampled to match
Hindi's count (4,711 rows), and the full mixed-effects model is fitted.

This tests whether the undersampled R² estimates are stable or sensitive
to which specific rows happen to be selected.
"""

import sys
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import pandas as pd
import numpy as np
import json
import warnings
warnings.simplefilter("ignore")
import logging
logging.disable(logging.CRITICAL)
import time

from utils import fit_lmm

def log(msg):
    print(msg, flush=True)

N_SEEDS = 50
BASE_SEED = 100

log("=" * 70)
log("STEP 13: MULTI-SEED UNDERSAMPLING STABILITY ANALYSIS")
log(f"  Seeds: {N_SEEDS} (starting from {BASE_SEED})")
log("=" * 70)

with open("lang_metadata.json") as f:
    meta = json.load(f)

# Target = smallest language
lang_sizes = {lang: info["n_rows"] for lang, info in meta.items()}
min_lang = min(lang_sizes, key=lang_sizes.get)
n_target = lang_sizes[min_lang]
log(f"  Target sample size: {n_target:,} (from {min_lang})")

# Pre-load all language dataframes
lang_data = {}
for lang in sorted(meta.keys()):
    lang_data[lang] = pd.read_parquet(f"data_{lang}.parquet")


def undersample_language(df_lang, n_target, seed):
    """Undersample a language dataset to n_target rows, stratified by annotator."""
    rng = np.random.RandomState(seed)

    if len(df_lang) <= n_target:
        return df_lang.copy()

    annotator_counts = df_lang["annotator_id"].value_counts()
    annotator_fractions = annotator_counts / annotator_counts.sum()

    rows_per_annotator = (annotator_fractions * n_target).astype(int)
    rows_per_annotator = rows_per_annotator.clip(lower=1)

    diff = n_target - rows_per_annotator.sum()
    if diff > 0:
        candidates = rows_per_annotator.index.tolist()
        for i in range(diff):
            ann = candidates[i % len(candidates)]
            if rows_per_annotator[ann] < annotator_counts[ann]:
                rows_per_annotator[ann] += 1
    elif diff < 0:
        sorted_anns = rows_per_annotator.sort_values(ascending=False).index.tolist()
        for i in range(-diff):
            ann = sorted_anns[i % len(sorted_anns)]
            if rows_per_annotator[ann] > 1:
                rows_per_annotator[ann] -= 1

    sampled_parts = []
    for ann_id, n_sample in rows_per_annotator.items():
        ann_data = df_lang[df_lang["annotator_id"] == ann_id]
        n_sample = min(n_sample, len(ann_data))
        sampled_parts.append(ann_data.sample(n=n_sample, random_state=rng))

    return pd.concat(sampled_parts, ignore_index=True)


all_results = []
t0_total = time.time()

for seed_idx, seed in enumerate(range(BASE_SEED, BASE_SEED + N_SEEDS)):
    t0_seed = time.time()

    for lang in sorted(meta.keys()):
        info = meta[lang]
        df_lang = lang_data[lang]
        usable_vars = info["usable_vars"]
        formula = f"label ~ {' + '.join(usable_vars)}"

        df_sampled = undersample_language(df_lang, n_target, seed)

        try:
            _, r2 = fit_lmm(formula, df_sampled, groups_col="text_id")
            all_results.append({
                "seed": seed,
                "language": lang,
                "n_rows": len(df_sampled),
                "r2_marginal": r2["r2_marginal"],
                "r2_conditional": r2["r2_conditional"],
            })
        except Exception as e:
            all_results.append({
                "seed": seed,
                "language": lang,
                "n_rows": len(df_sampled),
                "error": str(e),
            })

    elapsed_seed = time.time() - t0_seed
    elapsed_total = time.time() - t0_total
    if (seed_idx + 1) % 5 == 0:
        rate = (seed_idx + 1) / elapsed_total
        remaining = (N_SEEDS - seed_idx - 1) / rate
        log(f"  [{seed_idx+1}/{N_SEEDS}] seed={seed}, "
            f"elapsed={elapsed_total:.0f}s, ~{remaining:.0f}s remaining")

total_time = time.time() - t0_total
log(f"\n  Total time: {total_time:.0f}s for {N_SEEDS} seeds × {len(meta)} languages")

# --- Analysis ---
results_df = pd.DataFrame(all_results)
results_df = results_df[results_df["r2_marginal"].notna()]

log("\n\n" + "=" * 70)
log("MULTI-SEED UNDERSAMPLING RESULTS")
log("=" * 70)

summary_rows = []
log(f"\n{'Language':<10} {'Mean R²_m':>12} {'SD':>10} {'Min':>10} {'Max':>10} "
    f"{'Range':>10} {'CV':>10}")
log("-" * 74)

for lang in sorted(results_df["language"].unique()):
    lang_r2 = results_df[results_df["language"] == lang]["r2_marginal"]
    mean_r2 = lang_r2.mean()
    std_r2 = lang_r2.std()
    min_r2 = lang_r2.min()
    max_r2 = lang_r2.max()
    range_r2 = max_r2 - min_r2
    cv = std_r2 / mean_r2 * 100 if mean_r2 > 0 else 0

    log(f"  {lang:<8} {mean_r2*100:>11.3f}% {std_r2*100:>9.4f}% "
        f"{min_r2*100:>9.3f}% {max_r2*100:>9.3f}% "
        f"{range_r2*100:>9.3f}% {cv:>9.1f}%")

    summary_rows.append({
        "language": lang,
        "r2m_mean": mean_r2,
        "r2m_std": std_r2,
        "r2m_min": min_r2,
        "r2m_max": max_r2,
        "r2m_range": range_r2,
        "r2m_cv": cv,
        "r2m_ci_lower": np.percentile(lang_r2, 2.5),
        "r2m_ci_upper": np.percentile(lang_r2, 97.5),
        "n_seeds": len(lang_r2),
    })

summary_df = pd.DataFrame(summary_rows)

# Ranking stability
log("\n\nRANKING STABILITY ACROSS SEEDS:")
rank_per_seed = results_df.pivot(index="seed", columns="language", values="r2_marginal").rank(
    axis=1, ascending=False
)
mean_rank = rank_per_seed.mean()
std_rank = rank_per_seed.std()
log(f"\n{'Language':<10} {'Mean rank':>12} {'SD rank':>10} {'Always top-3':>14} {'Always bot-2':>14}")
log("-" * 62)
for lang in mean_rank.sort_values().index:
    top3_pct = (rank_per_seed[lang] <= 3).mean() * 100
    bot2_pct = (rank_per_seed[lang] >= 8).mean() * 100
    log(f"  {lang:<8} {mean_rank[lang]:>11.1f} {std_rank[lang]:>9.2f} "
        f"{top3_pct:>13.0f}% {bot2_pct:>13.0f}%")

# Compare multi-seed mean with original and single-seed undersampled
try:
    orig_df = pd.read_csv("results_r2_per_language.csv")
    us_df = pd.read_csv("results_r2_per_language_undersampled.csv")

    log("\n\nCOMPARISON: Original vs. Single-Seed vs. Multi-Seed Mean:")
    log(f"  {'Language':<10} {'Original':>12} {'Seed=42':>12} {'50-seed mean':>14} {'50-seed SD':>12}")
    log("  " + "-" * 62)
    for lang in sorted(meta.keys()):
        orig_r2 = orig_df.loc[orig_df["language"] == lang, "r2_marginal"].values
        us_r2 = us_df.loc[us_df["language"] == lang, "r2_marginal"].values
        ms_row = summary_df[summary_df["language"] == lang].iloc[0]
        if len(orig_r2) > 0 and len(us_r2) > 0:
            log(f"  {lang:<10} {orig_r2[0]*100:>11.3f}% {us_r2[0]*100:>11.3f}% "
                f"{ms_row['r2m_mean']*100:>13.3f}% {ms_row['r2m_std']*100:>11.4f}%")
except FileNotFoundError:
    pass

# --- Save ---
results_df.to_csv("results_multiseed_undersampling.csv", index=False)
summary_df.to_csv("results_multiseed_summary.csv", index=False)
log(f"\nSaved results_multiseed_undersampling.csv and results_multiseed_summary.csv")

log("\n" + "=" * 70)
log("MULTI-SEED UNDERSAMPLING ANALYSIS COMPLETE")
log("=" * 70)
