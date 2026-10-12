"""
Step 9: Leave-one-out variable importance on UNDERSAMPLED data.
Identical methodology to Step 4, but using balanced datasets.
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from utils import fit_lmm

print("=" * 70)
print("STEP 9: VARIABLE IMPORTANCE (UNDERSAMPLED) — Leave-One-Out Analysis")
print("=" * 70)

with open("lang_metadata_undersampled.json") as f:
    meta = json.load(f)

all_importance = []

for lang in sorted(meta.keys()):
    info = meta[lang]
    df_lang = pd.read_parquet(f"data_{lang}_undersampled.parquet")
    usable_vars = info["usable_vars"]

    print(f"\n{'='*60}")
    print(f"  Language: {lang.upper()} — Variables: {usable_vars}")
    print(f"{'='*60}")

    full_formula = f"label ~ {' + '.join(usable_vars)}"
    try:
        _, full_r2 = fit_lmm(full_formula, df_lang, groups_col="text_id")
        full_r2m = full_r2["r2_marginal"]
        print(f"  Full model R²_m = {full_r2m*100:.4f}%")
    except Exception as e:
        print(f"  ERROR fitting full model: {e}")
        continue

    for var in usable_vars:
        reduced_vars = [v for v in usable_vars if v != var]
        if not reduced_vars:
            reduced_formula = "label ~ 1"
        else:
            reduced_formula = f"label ~ {' + '.join(reduced_vars)}"

        var_name = var.replace("C(", "").replace(")", "").replace("age_centered", "age")

        try:
            _, reduced_r2 = fit_lmm(reduced_formula, df_lang, groups_col="text_id")
            delta = full_r2m - reduced_r2["r2_marginal"]
            print(f"  Drop {var_name:20s}: ΔR²_m = {delta*100:+.4f}% "
                  f"(reduced R²_m = {reduced_r2['r2_marginal']*100:.4f}%)")

            all_importance.append({
                "language": lang,
                "variable": var_name,
                "r2_full": full_r2m,
                "r2_reduced": reduced_r2["r2_marginal"],
                "delta_r2": delta,
                "delta_r2_pct": delta * 100,
            })
        except Exception as e:
            print(f"  Drop {var_name:20s}: ERROR - {e}")
            all_importance.append({
                "language": lang,
                "variable": var_name,
                "error": str(e),
            })

# --- Summary ---
print("\n\n" + "=" * 70)
print("VARIABLE IMPORTANCE SUMMARY TABLE — UNDERSAMPLED (ΔR²_m × 100)")
print("=" * 70)

imp_df = pd.DataFrame(all_importance)
imp_df = imp_df[imp_df["delta_r2"].notna()]

pivot = imp_df.pivot_table(
    index="language", columns="variable", values="delta_r2_pct", aggfunc="first"
)
col_order = ["age", "gender", "ethnicity", "nationality"]
cols = [c for c in col_order if c in pivot.columns]
pivot = pivot[cols]

print("\n(Values are ΔR²_m in percentage points — higher = more important)")
print(pivot.to_string(float_format="%.4f", na_rep="  n/a"))

print("\n\nVARIABLE RANKINGS PER LANGUAGE (most important first):")
for lang in sorted(pivot.index):
    row = pivot.loc[lang].dropna().sort_values(ascending=False)
    ranking = ", ".join([f"{var} ({val:.4f}%)" for var, val in row.items()])
    print(f"  {lang}: {ranking}")

print("\n\nOVERALL AVERAGE VARIABLE IMPORTANCE (UNDERSAMPLED):")
avg_imp = pivot.mean()
for var, val in avg_imp.sort_values(ascending=False).items():
    print(f"  {var}: {val:.4f}% (averaged across languages where available)")

# --- Compare with original ---
try:
    orig_imp = pd.read_csv("results_variable_importance.csv")
    orig_imp = orig_imp[orig_imp["delta_r2"].notna()]
    orig_pivot = orig_imp.pivot_table(
        index="language", columns="variable", values="delta_r2_pct", aggfunc="first"
    )
    orig_avg = orig_pivot.mean()

    print("\n\nCOMPARISON: Original vs. Undersampled Average Variable Importance:")
    print(f"  {'Variable':<15} {'Original':>10} {'Undersampled':>14} {'Δ':>10}")
    print("  " + "-" * 50)
    for var in col_order:
        if var in avg_imp.index and var in orig_avg.index:
            diff = avg_imp[var] - orig_avg[var]
            print(f"  {var:<15} {orig_avg[var]:>10.4f} {avg_imp[var]:>14.4f} {diff:>+10.4f}")
except FileNotFoundError:
    pass

# Save
imp_df.to_csv("results_variable_importance_undersampled.csv", index=False)
pivot.to_csv("results_variable_importance_pivot_undersampled.csv")
print(f"\nSaved to results_variable_importance_undersampled.csv")

print("\n" + "=" * 70)
print("VARIABLE IMPORTANCE ANALYSIS (UNDERSAMPLED) COMPLETE")
print("=" * 70)
