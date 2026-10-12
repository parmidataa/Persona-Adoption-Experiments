"""
Step 12: Bootstrap confidence intervals for marginal R².

Uses cluster bootstrap resampling at the ANNOTATOR level:
  - For each iteration, sample annotators with replacement
  - Include all annotations from selected annotators
  - Refit the mixed-effects model and record R²_m

This respects the hierarchical structure (annotations nested in annotators)
and captures uncertainty in the persona-variable effects.

200 bootstrap iterations per language → 95% CIs via percentile method.
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

N_BOOTSTRAP = 200
SEED = 42
np.random.seed(SEED)

log("=" * 70)
log("STEP 12: BOOTSTRAP CONFIDENCE INTERVALS FOR R²_m")
log(f"  Iterations: {N_BOOTSTRAP} per language (cluster bootstrap on annotators)")
log("=" * 70)

with open("lang_metadata.json") as f:
    meta = json.load(f)

all_bootstrap_results = []
summary_rows = []

for lang in sorted(meta.keys()):
    info = meta[lang]
    df_lang = pd.read_parquet(f"data_{lang}.parquet")
    usable_vars = info["usable_vars"]
    formula = f"label ~ {' + '.join(usable_vars)}"

    annotator_ids = df_lang["annotator_id"].unique()
    n_annotators = len(annotator_ids)

    log(f"\n{'='*60}")
    log(f"  {lang.upper()}: {n_annotators} annotators, {len(df_lang):,} rows")
    log(f"  Formula: {formula}")
    log(f"  Running {N_BOOTSTRAP} bootstrap iterations...")
    log(f"{'='*60}")

    # Point estimate (original)
    try:
        _, r2_orig = fit_lmm(formula, df_lang, groups_col="text_id")
        r2m_point = r2_orig["r2_marginal"]
        r2c_point = r2_orig["r2_conditional"]
    except Exception as e:
        log(f"  ERROR on point estimate: {e}")
        continue

    boot_r2m = []
    boot_r2c = []
    n_failures = 0
    t0 = time.time()

    for i in range(N_BOOTSTRAP):
        # Cluster resample: draw annotators with replacement
        boot_annotators = np.random.choice(annotator_ids, size=n_annotators, replace=True)

        # Build bootstrap dataset (handling duplicate annotator draws)
        parts = []
        for idx, ann in enumerate(boot_annotators):
            ann_data = df_lang[df_lang["annotator_id"] == ann].copy()
            # Give duplicated annotators unique IDs so model treats them as separate
            ann_data["annotator_id_boot"] = f"{ann}_{idx}"
            parts.append(ann_data)
        df_boot = pd.concat(parts, ignore_index=True)

        try:
            _, r2_boot = fit_lmm(formula, df_boot, groups_col="text_id")
            boot_r2m.append(r2_boot["r2_marginal"])
            boot_r2c.append(r2_boot["r2_conditional"])
        except Exception:
            n_failures += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (N_BOOTSTRAP - i - 1) / rate
            log(f"    [{i+1}/{N_BOOTSTRAP}] "
                f"elapsed={elapsed:.0f}s, ~{remaining:.0f}s remaining, "
                f"failures={n_failures}")

    elapsed = time.time() - t0
    boot_r2m = np.array(boot_r2m)
    boot_r2c = np.array(boot_r2c)

    ci_lower_m = np.percentile(boot_r2m, 2.5)
    ci_upper_m = np.percentile(boot_r2m, 97.5)
    ci_lower_c = np.percentile(boot_r2c, 2.5)
    ci_upper_c = np.percentile(boot_r2c, 97.5)
    boot_mean_m = np.mean(boot_r2m)
    boot_std_m = np.std(boot_r2m)

    log(f"\n  Results ({len(boot_r2m)} successful iterations, {n_failures} failures, {elapsed:.0f}s):")
    log(f"  R²_m point estimate:  {r2m_point*100:.3f}%")
    log(f"  R²_m bootstrap mean:  {boot_mean_m*100:.3f}%")
    log(f"  R²_m 95% CI:          [{ci_lower_m*100:.3f}%, {ci_upper_m*100:.3f}%]")
    log(f"  R²_m bootstrap SD:    {boot_std_m*100:.4f}%")
    log(f"  R²_c 95% CI:          [{ci_lower_c*100:.3f}%, {ci_upper_c*100:.3f}%]")

    for val in boot_r2m:
        all_bootstrap_results.append({
            "language": lang,
            "r2_marginal": val,
            "r2_type": "marginal",
        })
    for val in boot_r2c:
        all_bootstrap_results.append({
            "language": lang,
            "r2_conditional": val,
            "r2_type": "conditional",
        })

    summary_rows.append({
        "language": lang,
        "n_annotators": n_annotators,
        "n_rows": len(df_lang),
        "r2m_point": r2m_point,
        "r2m_boot_mean": boot_mean_m,
        "r2m_boot_std": boot_std_m,
        "r2m_ci_lower": ci_lower_m,
        "r2m_ci_upper": ci_upper_m,
        "r2c_point": r2c_point,
        "r2c_ci_lower": ci_lower_c,
        "r2c_ci_upper": ci_upper_c,
        "ci_width": (ci_upper_m - ci_lower_m) * 100,
        "n_successful": len(boot_r2m),
        "n_failures": n_failures,
    })

# --- Save ---
boot_df = pd.DataFrame(all_bootstrap_results)
boot_df.to_csv("results_bootstrap_r2.csv", index=False)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results_bootstrap_summary.csv", index=False)

# --- Summary table ---
log("\n\n" + "=" * 70)
log("BOOTSTRAP CONFIDENCE INTERVALS — SUMMARY")
log("=" * 70)

summary_df_sorted = summary_df.sort_values("r2m_point", ascending=False)
log(f"\n{'Language':<10} {'R²_m point':>12} {'Boot mean':>12} {'95% CI':>24} {'CI width':>10}")
log("-" * 70)
for _, row in summary_df_sorted.iterrows():
    log(f"  {row['language']:<8} {row['r2m_point']*100:>11.3f}% {row['r2m_boot_mean']*100:>11.3f}% "
        f"  [{row['r2m_ci_lower']*100:>6.3f}%, {row['r2m_ci_upper']*100:>6.3f}%] "
        f"{row['ci_width']:>9.3f}%")

# Check for overlapping CIs
log("\n\nOVERLAPPING CI ANALYSIS:")
log("(Languages whose 95% CIs overlap cannot be considered significantly different)")
langs_sorted = summary_df_sorted["language"].tolist()
for i in range(len(langs_sorted)):
    for j in range(i + 1, len(langs_sorted)):
        l1, l2 = langs_sorted[i], langs_sorted[j]
        r1 = summary_df_sorted[summary_df_sorted["language"] == l1].iloc[0]
        r2 = summary_df_sorted[summary_df_sorted["language"] == l2].iloc[0]
        overlap = r1["r2m_ci_lower"] < r2["r2m_ci_upper"] and r2["r2m_ci_lower"] < r1["r2m_ci_upper"]
        if overlap:
            log(f"  {l1} [{r1['r2m_ci_lower']*100:.3f}%, {r1['r2m_ci_upper']*100:.3f}%]  ↔  "
                f"{l2} [{r2['r2m_ci_lower']*100:.3f}%, {r2['r2m_ci_upper']*100:.3f}%]  → OVERLAP")

log(f"\nSaved results_bootstrap_r2.csv and results_bootstrap_summary.csv")

log("\n" + "=" * 70)
log("BOOTSTRAP CONFIDENCE INTERVALS COMPLETE")
log("=" * 70)
