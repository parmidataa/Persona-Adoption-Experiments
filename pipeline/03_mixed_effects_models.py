"""
Step 3: Fit mixed-effects models per language.
Linear mixed model: label ~ persona_variables + (1 | text_id)
Computes marginal and conditional R-squared (Nakagawa-Schielzeth).
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from utils import fit_lmm, compute_r_squared_nakagawa

print("=" * 70)
print("STEP 3: MIXED-EFFECTS MODELS — Per-Language R-squared")
print("=" * 70)

with open("lang_metadata.json") as f:
    meta = json.load(f)

results = []
model_details = {}

for lang in sorted(meta.keys()):
    info = meta[lang]
    df_lang = pd.read_parquet(f"data_{lang}.parquet")
    usable_vars = info["usable_vars"]

    formula = f"label ~ {' + '.join(usable_vars)}"
    print(f"\n{'='*60}")
    print(f"  Language: {lang.upper()} (n={len(df_lang):,}, "
          f"annotators={info['n_annotators']}, texts={info['n_texts']})")
    print(f"  Formula: {formula}")
    print(f"{'='*60}")

    try:
        model, r2 = fit_lmm(formula, df_lang, groups_col="text_id")

        print(f"  Marginal R²:      {r2['r2_marginal']:.6f} ({r2['r2_marginal']*100:.3f}%)")
        print(f"  Conditional R²:   {r2['r2_conditional']:.6f} ({r2['r2_conditional']*100:.3f}%)")
        print(f"  Marginal R² (logistic): {r2['r2_marginal_logistic']:.6f} ({r2['r2_marginal_logistic']*100:.3f}%)")
        print(f"  Conditional R² (logistic): {r2['r2_conditional_logistic']:.6f} ({r2['r2_conditional_logistic']*100:.3f}%)")
        print(f"  Variance components:")
        print(f"    Fixed effects:  {r2['var_fixed']:.6f}")
        print(f"    Random (text):  {r2['var_random']:.6f}")
        print(f"    Residual:       {r2['var_residual']:.6f}")

        total = r2["var_fixed"] + r2["var_random"] + r2["var_residual"]
        pct_text = r2["var_random"] / total * 100
        pct_unexplained = r2["var_residual"] / total * 100
        print(f"  Variance decomposition:")
        print(f"    Persona vars:   {r2['r2_marginal']*100:.2f}%")
        print(f"    Text effects:   {pct_text:.2f}%")
        print(f"    Unexplained:    {pct_unexplained:.2f}%")

        # Fixed effects summary
        print(f"\n  Fixed effects (significant at p<0.05):")
        fe_params = model.fe_params
        bse = model.bse
        z_vals = fe_params / bse
        from scipy import stats as sp_stats
        p_vals = 2 * sp_stats.norm.sf(np.abs(z_vals))
        coef_df = pd.DataFrame({
            "coef": fe_params,
            "se": bse[:len(fe_params)],
            "z": z_vals,
            "p": p_vals,
        })
        sig = coef_df[coef_df["p"] < 0.05]
        for idx, row in sig.iterrows():
            print(f"    {idx}: coef={row['coef']:.4f}, z={row['z']:.2f}, p={row['p']:.4f}")

        results.append({
            "language": lang,
            "n_rows": len(df_lang),
            "n_annotators": info["n_annotators"],
            "n_texts": info["n_texts"],
            "irony_rate": info["irony_rate"],
            "n_persona_vars": len(usable_vars),
            "formula": formula,
            "r2_marginal": r2["r2_marginal"],
            "r2_conditional": r2["r2_conditional"],
            "r2_marginal_logistic": r2["r2_marginal_logistic"],
            "r2_conditional_logistic": r2["r2_conditional_logistic"],
            "var_fixed": r2["var_fixed"],
            "var_random": r2["var_random"],
            "var_residual": r2["var_residual"],
            "pct_persona": r2["r2_marginal"] * 100,
            "pct_text": pct_text,
            "pct_unexplained": pct_unexplained,
        })

        model_details[lang] = {
            "coefficients": coef_df.to_dict(),
            "r2": r2,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            "language": lang,
            "n_rows": len(df_lang),
            "n_annotators": info["n_annotators"],
            "n_texts": info["n_texts"],
            "error": str(e),
        })

# --- Also fit age+gender only models for direct comparison with Hu & Collier ---
print("\n\n" + "=" * 70)
print("COMPARISON: Age + Gender Only (Hu & Collier baseline)")
print("=" * 70)

ag_results = []
for lang in sorted(meta.keys()):
    df_lang = pd.read_parquet(f"data_{lang}.parquet")
    formula_ag = "label ~ age_centered + C(gender)"

    try:
        model_ag, r2_ag = fit_lmm(formula_ag, df_lang, groups_col="text_id")
        print(f"  {lang}: Marginal R² (age+gender only) = {r2_ag['r2_marginal']*100:.3f}%")
        ag_results.append({
            "language": lang,
            "r2_marginal_age_gender": r2_ag["r2_marginal"],
        })
    except Exception as e:
        print(f"  {lang}: ERROR - {e}")

# --- Summary table ---
print("\n\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

results_df = pd.DataFrame(results)
ag_df = pd.DataFrame(ag_results)
if len(ag_df) > 0 and "r2_marginal" in results_df.columns:
    results_df = results_df.merge(ag_df, on="language", how="left")
    results_df["r2_gain_from_extra_vars"] = (
        results_df["r2_marginal"] - results_df["r2_marginal_age_gender"]
    )

cols_to_show = ["language", "n_annotators", "n_texts", "irony_rate",
                "r2_marginal", "r2_conditional", "r2_marginal_age_gender",
                "r2_gain_from_extra_vars", "pct_persona", "pct_text", "pct_unexplained"]
available_cols = [c for c in cols_to_show if c in results_df.columns]
print(results_df[available_cols].to_string(index=False, float_format="%.4f"))

# Hu & Collier reference
print(f"\nHu & Collier EPI irony reference: Marginal R² = 4.6% (age+gender only)")
if "r2_marginal" in results_df.columns:
    mean_r2 = results_df["r2_marginal"].mean() * 100
    print(f"MultiPICo mean Marginal R² (all vars): {mean_r2:.3f}%")

# --- Save results ---
results_df.to_csv("results_r2_per_language.csv", index=False)
with open("model_details.json", "w") as f:
    json.dump({k: {kk: vv if not isinstance(vv, dict) else {kkk: (vvv if not isinstance(vvv, float) else round(vvv, 8)) for kkk, vvv in vv.items()} for kk, vv in v.items()} for k, v in model_details.items()}, f, indent=2, default=str)

print(f"\nSaved results to results_r2_per_language.csv")

print("\n" + "=" * 70)
print("MIXED-EFFECTS MODELS COMPLETE")
print("=" * 70)
