"""
Step 4: Leave-one-out variable importance analysis.
For each language, drop one persona variable at a time and measure
the decrease in marginal R-squared.
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from utils import fit_lmm

print("=" * 70)
print("STEP 4: VARIABLE IMPORTANCE — Leave-One-Out Analysis")
print("=" * 70)

with open("lang_metadata.json") as f:
    meta = json.load(f)

all_importance = []

for lang in sorted(meta.keys()):
    info = meta[lang]
    df_lang = pd.read_parquet(f"data_{lang}.parquet")
    usable_vars = info["usable_vars"]

    print(f"\n{'='*60}")
    print(f"  Language: {lang.upper()} — Variables: {usable_vars}")
    print(f"{'='*60}")

    # Fit full model
    full_formula = f"label ~ {' + '.join(usable_vars)}"
    try:
        _, full_r2 = fit_lmm(full_formula, df_lang, groups_col="text_id")
        full_r2m = full_r2["r2_marginal"]
        print(f"  Full model R²_m = {full_r2m*100:.4f}%")
    except Exception as e:
        print(f"  ERROR fitting full model: {e}")
        continue

    # Leave-one-out
    for var in usable_vars:
        reduced_vars = [v for v in usable_vars if v != var]
        if not reduced_vars:
            # Only one variable -- compare against intercept-only
            reduced_formula = "label ~ 1"
        else:
            reduced_formula = f"label ~ {' + '.join(reduced_vars)}"

        # Clean variable name for display
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
print("VARIABLE IMPORTANCE SUMMARY TABLE (ΔR²_m × 100)")
print("=" * 70)

imp_df = pd.DataFrame(all_importance)
imp_df = imp_df[imp_df["delta_r2"].notna()]

pivot = imp_df.pivot_table(
    index="language", columns="variable", values="delta_r2_pct", aggfunc="first"
)
# Reorder columns
col_order = ["age", "gender", "ethnicity", "nationality"]
cols = [c for c in col_order if c in pivot.columns]
pivot = pivot[cols]

print("\n(Values are ΔR²_m in percentage points — higher = more important)")
print(pivot.to_string(float_format="%.4f", na_rep="  n/a"))

# Rank variables per language
print("\n\nVARIABLE RANKINGS PER LANGUAGE (most important first):")
for lang in sorted(pivot.index):
    row = pivot.loc[lang].dropna().sort_values(ascending=False)
    ranking = ", ".join([f"{var} ({val:.4f}%)" for var, val in row.items()])
    print(f"  {lang}: {ranking}")

# Overall average importance
print("\n\nOVERALL AVERAGE VARIABLE IMPORTANCE:")
avg_imp = pivot.mean()
for var, val in avg_imp.sort_values(ascending=False).items():
    print(f"  {var}: {val:.4f}% (averaged across languages where available)")

# Save
imp_df.to_csv("results_variable_importance.csv", index=False)
pivot.to_csv("results_variable_importance_pivot.csv")
print(f"\nSaved to results_variable_importance.csv")

print("\n" + "=" * 70)
print("VARIABLE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 70)
