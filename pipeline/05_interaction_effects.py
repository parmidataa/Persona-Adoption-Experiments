"""
Step 5: Test interaction effects.
Use likelihood ratio tests to assess whether adding interaction terms
(gender×ethnicity, gender×nationality, age×gender) significantly
improves model fit.
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from utils import interaction_lr_test

print("=" * 70)
print("STEP 5: INTERACTION EFFECTS — Likelihood Ratio Tests")
print("=" * 70)

with open("lang_metadata.json") as f:
    meta = json.load(f)

# Define interactions to test, conditional on available variables
interaction_specs = [
    ("gender × ethnicity", "C(gender):C(ethnicity)"),
    ("age × gender", "age_centered:C(gender)"),
    ("age × ethnicity", "age_centered:C(ethnicity)"),
]

interaction_specs_with_nat = [
    ("gender × nationality", "C(gender):C(nationality)"),
    ("age × nationality", "age_centered:C(nationality)"),
    ("ethnicity × nationality", "C(ethnicity):C(nationality)"),
]

all_results = []

for lang in sorted(meta.keys()):
    info = meta[lang]
    df_lang = pd.read_parquet(f"data_{lang}.parquet")
    usable_vars = info["usable_vars"]
    base_formula = f"label ~ {' + '.join(usable_vars)}"

    has_nationality = "C(nationality)" in usable_vars

    print(f"\n{'='*60}")
    print(f"  Language: {lang.upper()}")
    print(f"  Base: {base_formula}")
    print(f"{'='*60}")

    tests = interaction_specs.copy()
    if has_nationality:
        tests.extend(interaction_specs_with_nat)

    for name, term in tests:
        # Check that both variables in the interaction are in the model
        parts = term.split(":")
        all_in_model = all(
            p in usable_vars or p.replace("C(", "").replace(")", "") + "_centered" in " ".join(usable_vars)
            for p in parts
        )
        if not all_in_model:
            continue

        try:
            result = interaction_lr_test(df_lang, base_formula, term, groups_col="text_id")
            sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
            print(f"  {name:30s}: LR={result['lr_statistic']:8.2f}, "
                  f"df={result['df']:3d}, p={result['p_value']:.4f} {sig}, "
                  f"ΔR²_m={result['r2_change']*100:+.4f}%")

            all_results.append({
                "language": lang,
                "interaction": name,
                "term": term,
                "lr_statistic": result["lr_statistic"],
                "df": result["df"],
                "p_value": result["p_value"],
                "significant": result["p_value"] < 0.05,
                "r2_base": result["r2_base"],
                "r2_full": result["r2_full"],
                "r2_change": result["r2_change"],
                "r2_change_pct": result["r2_change"] * 100,
            })
        except Exception as e:
            print(f"  {name:30s}: ERROR - {e}")

# --- Summary ---
print("\n\n" + "=" * 70)
print("INTERACTION EFFECTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(all_results)

# Count significant interactions per type
print("\nSignificant interactions (p < 0.05) by type:")
sig_df = results_df[results_df["significant"]]
if len(sig_df) > 0:
    sig_counts = sig_df.groupby("interaction").agg(
        n_significant=("language", "count"),
        languages=("language", lambda x: ", ".join(x)),
        avg_r2_change=("r2_change_pct", "mean"),
    )
    print(sig_counts.to_string())
else:
    print("  No significant interactions found.")

# Per language: best interaction
print("\nBest interaction per language (largest R² increase):")
for lang in sorted(results_df["language"].unique()):
    lang_df = results_df[results_df["language"] == lang]
    best = lang_df.loc[lang_df["r2_change"].idxmax()]
    sig_str = " (significant)" if best["significant"] else " (not significant)"
    print(f"  {lang}: {best['interaction']} "
          f"(ΔR²_m = {best['r2_change_pct']:+.4f}%, p={best['p_value']:.4f}){sig_str}")

# Bonferroni-corrected significance
n_tests = len(results_df)
bonferroni_threshold = 0.05 / n_tests
print(f"\nBonferroni-corrected threshold (α = 0.05 / {n_tests}): p < {bonferroni_threshold:.6f}")
bonf_sig = results_df[results_df["p_value"] < bonferroni_threshold]
if len(bonf_sig) > 0:
    print("Interactions surviving Bonferroni correction:")
    for _, row in bonf_sig.iterrows():
        print(f"  {row['language']}: {row['interaction']} "
              f"(p={row['p_value']:.6f}, ΔR²_m={row['r2_change_pct']:+.4f}%)")
else:
    print("No interactions survive Bonferroni correction.")

# Save
results_df.to_csv("results_interaction_effects.csv", index=False)
print(f"\nSaved to results_interaction_effects.csv")

print("\n" + "=" * 70)
print("INTERACTION EFFECTS ANALYSIS COMPLETE")
print("=" * 70)
