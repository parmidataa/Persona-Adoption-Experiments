"""
Step 18 (Task 2B): Extended fixed effects + attitudinal proxy.

Fits a nested sequence of models per language with a single random
intercept for text_id (matching the v1 §1 baseline so R^2_m values are
directly comparable to findings.md):

    M1: age + gender + ethnicity + nationality                 (current full)
    M2: M1 + country_birth + country_residence
    M3: M2 + student_status + employment_status
    M4: M3 + generation                                        (drop if VIF>10)
    M5: M4 + base_rate_loo                                     (attitudinal proxy)

Robustness strategy
-------------------
For each language and each model spec we:
  1. Build the dummy design matrix manually (numerically).
  2. Run a QR-pivoted rank check; drop linearly dependent dummy columns.
  3. Map the surviving dummies back to a *reduced term list* (a categorical
     term is kept only if at least one of its dummies survived).
  4. Fit using the statsmodels formula API on the reduced term list.
This avoids the manual MixedLM fallback path that was producing degenerate
variance estimates on rank-deficient designs.

Outputs:
  - results_extended_fixed_effects.csv
  - fig20_extended_fe_progression.png
"""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

print("=" * 70)
print("STEP 18 (Task 2B): EXTENDED FIXED EFFECTS")
print("=" * 70)

with open("lang_metadata_v2.json") as fh:
    meta = json.load(fh)


BASE_TERMS = {
    "M1": ["age_centered", "gender", "ethnicity", "nationality"],
    "M2": ["age_centered", "gender", "ethnicity", "nationality",
           "country_birth", "country_residence"],
    "M3": ["age_centered", "gender", "ethnicity", "nationality",
           "country_birth", "country_residence",
           "student_status", "employment_status"],
    "M4": ["age_centered", "gender", "ethnicity", "nationality",
           "country_birth", "country_residence",
           "student_status", "employment_status", "generation"],
    "M5": ["age_centered", "gender", "ethnicity", "nationality",
           "country_birth", "country_residence",
           "student_status", "employment_status", "generation",
           "base_rate_loo"],
}


def adapt_terms(terms, df_lang):
    out = []
    for t in terms:
        if t in ("age_centered", "base_rate_loo"):
            out.append(t)
            continue
        if df_lang.drop_duplicates("annotator_id")[t].nunique() > 1:
            out.append(t)
    return out


def build_design(df, base_terms):
    parts = []
    col_to_term = {}
    for term in base_terms:
        if term in ("age_centered", "base_rate_loo"):
            parts.append(df[[term]].astype(float))
            col_to_term[term] = term
        else:
            d = pd.get_dummies(df[term], prefix=term, drop_first=True).astype(float)
            for c in d.columns:
                col_to_term[c] = term
            parts.append(d)
    if not parts:
        return pd.DataFrame(index=df.index), {}
    X = pd.concat(parts, axis=1)
    nz = X.std(axis=0) > 0
    X = X.loc[:, nz]
    return X, {c: col_to_term[c] for c in X.columns}


def rank_prune_terms(df_lang, terms, rel_tol=1e-7):
    """Use a pivoted QR (scipy) to identify which dummy columns are
    linearly dependent on earlier ones; map kept dummies back to *terms*."""
    from scipy.linalg import qr as scipy_qr
    X, col_to_term = build_design(df_lang, terms)
    if X.shape[1] == 0:
        return [t for t in terms if t in ("age_centered", "base_rate_loo")]
    aug = np.column_stack([np.ones(len(X)), X.values.astype(float)])
    # pivoting=True returns column permutation
    _, r, piv = scipy_qr(aug, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if len(diag) == 0:
        return []
    cutoff = rel_tol * diag[0]
    rank = int((diag > cutoff).sum())
    # piv[:rank] are the indices (in `aug`) of the linearly-independent cols.
    # piv index 0 corresponds to the constant column, indices 1..N to X.columns.
    kept_aug_idx = set(piv[:rank].tolist())
    keep_cols = []
    for j in range(X.shape[1]):
        if (j + 1) in kept_aug_idx:
            keep_cols.append(X.columns[j])
    surviving_terms = []
    seen = set()
    for c in keep_cols:
        t = col_to_term[c]
        if t not in seen:
            seen.add(t)
            surviving_terms.append(t)
    cont = [t for t in ("age_centered", "base_rate_loo") if t in surviving_terms]
    cats = [t for t in surviving_terms if t not in ("age_centered", "base_rate_loo")]
    cats_ordered = [t for t in terms if t in cats]
    return cont + cats_ordered


def fit_lmm(df, base_terms, groups_col="text_id", start_params=None):
    """Fit `label ~ controls + base_terms + (1|text_id)` via the formula API."""
    formula_rhs = []
    for term in base_terms:
        if term in ("age_centered", "base_rate_loo"):
            formula_rhs.append(term)
        else:
            formula_rhs.append(f"C({term})")
    formula = "label ~ " + " + ".join(formula_rhs) if formula_rhs else "label ~ 1"
    model = smf.mixedlm(formula, df, groups=df[groups_col])
    fit_kwargs = {"reml": True, "method": ["lbfgs", "powell", "nm"]}
    if start_params is not None:
        fit_kwargs["start_params"] = start_params
    res = model.fit(**fit_kwargs)
    return res, formula


def ols_r2m_with_anchor_denom(df, base_terms, anchor_total_var):
    """Compute R^2_m via OLS using `anchor_total_var` (from a clean LMM fit
    such as M1) as the denominator. Returns (r2m, var_fixed, ols_var_resid)."""
    formula_rhs = []
    for term in base_terms:
        if term in ("age_centered", "base_rate_loo"):
            formula_rhs.append(term)
        else:
            formula_rhs.append(f"C({term})")
    formula = "label ~ " + " + ".join(formula_rhs) if formula_rhs else "label ~ 1"
    res = smf.ols(formula, df).fit()
    fixed_pred = np.asarray(res.model.exog) @ res.params
    var_fixed = float(np.var(fixed_pred))
    ols_var_resid = float(res.mse_resid)
    r2m = var_fixed / anchor_total_var if anchor_total_var > 0 else np.nan
    return r2m, var_fixed, ols_var_resid, formula


def r2_from_result(res):
    fixed_pred = np.asarray(res.model.exog) @ res.fe_params
    var_fixed = float(np.var(fixed_pred))
    cov_re = res.cov_re
    if hasattr(cov_re, "iloc"):
        var_random = float(cov_re.iloc[0, 0])
    else:
        var_random = float(np.asarray(cov_re).flatten()[0])
    var_residual = float(res.scale)
    total = var_fixed + var_random + var_residual
    return {
        "var_fixed": var_fixed,
        "var_random": var_random,
        "var_residual": var_residual,
        "r2_marginal": var_fixed / total if total > 0 else 0.0,
        "r2_conditional": (var_fixed + var_random) / total if total > 0 else 0.0,
    }


rows = []
languages = sorted(meta.keys())

for lang in languages:
    df_lang = pd.read_parquet(f"data_{lang}_v2.parquet")
    n_rows = len(df_lang)
    print(f"\n[{lang}] n={n_rows:,}")

    # --- VIF check on the full M4 design ---
    m4_full = adapt_terms(BASE_TERMS["M4"], df_lang)
    if "generation" in m4_full and "age_centered" in m4_full:
        X4, _ = build_design(df_lang, m4_full)
        X4_arr = sm.add_constant(X4, has_constant="add").values.astype(float)
        cols = ["const"] + list(X4.columns)
        try:
            j = cols.index("age_centered")
            vif_age = variance_inflation_factor(X4_arr, j)
        except Exception:
            vif_age = np.nan
        gen_dummies = [c for c in X4.columns if c.startswith("generation_")]
        vif_gen = np.nan
        if gen_dummies:
            try:
                j = cols.index(gen_dummies[0])
                vif_gen = variance_inflation_factor(X4_arr, j)
            except Exception:
                pass
        print(f"   VIF(age_centered)={vif_age:.2f}  "
              f"VIF(generation_dummy)={vif_gen:.2f}")
    else:
        vif_age = vif_gen = np.nan

    prev_r2m = None
    m1_total_var = None  # set after fitting M1; used as anchor for OLS fallback
    for model_name in ("M1", "M2", "M3", "M4", "M5"):
        terms_full = adapt_terms(BASE_TERMS[model_name], df_lang)
        if model_name == "M5" and "base_rate_loo" not in terms_full:
            terms_full.append("base_rate_loo")
        terms = rank_prune_terms(df_lang, terms_full)
        dropped = [t for t in terms_full if t not in terms]

        method = "lmm"
        boundary = False
        formula = None
        res = None
        r2 = None

        try:
            res, formula = fit_lmm(df_lang, terms)
            r2 = r2_from_result(res)
            tot = r2["var_fixed"] + r2["var_random"] + r2["var_residual"]
            # Boundary if any component is below 1% of the total OR R^2_c
            # is essentially 100% (residual variance collapsed) OR R^2_c
            # equals R^2_m (random-effect variance collapsed).
            boundary = (
                tot <= 0
                or r2["var_residual"] / tot < 0.01
                or r2["var_random"] / tot < 1e-6
                or r2["r2_conditional"] > 0.99
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            print(f"   {model_name}: LMM singular ({exc}); using OLS fallback")
            method = "ols_fallback"

        if method == "lmm" and boundary and m1_total_var is not None:
            method = "ols_fallback"

        if method == "ols_fallback":
            if m1_total_var is None:
                print(f"   {model_name}: SKIPPED (no M1 anchor available)")
                continue
            r2m_ols, var_fixed_ols, ols_var_resid, formula = (
                ols_r2m_with_anchor_denom(df_lang, terms, m1_total_var)
            )
            r2 = {
                "var_fixed": var_fixed_ols,
                "var_random": np.nan,
                "var_residual": ols_var_resid,
                "r2_marginal": r2m_ols,
                "r2_conditional": np.nan,
            }

        delta = (r2["r2_marginal"] - prev_r2m) if prev_r2m is not None else 0.0
        prev_r2m = r2["r2_marginal"]

        if model_name == "M1" and method == "lmm":
            m1_total_var = r2["var_fixed"] + r2["var_random"] + r2["var_residual"]

        flag = "  [OLS-fallback]" if method == "ols_fallback" else (
            "  BOUNDARY!" if boundary else ""
        )
        drop_msg = f"  (dropped: {dropped})" if dropped else ""
        rc = r2["r2_conditional"]
        rc_str = "n/a" if not np.isfinite(rc) else f"{rc*100:.3f}%"
        print(f"   {model_name}: R²_m={r2['r2_marginal']*100:.3f}%  "
              f"R²_c={rc_str}  "
              f"ΔvsPrev={delta*100:+.3f}pp{drop_msg}{flag}")

        rows.append({
            "language": lang,
            "model": model_name,
            "terms_requested": "+".join(terms_full),
            "terms_used": "+".join(terms),
            "n_terms": len(terms),
            "dropped_terms": ",".join(dropped),
            "r2_marginal": r2["r2_marginal"],
            "r2_conditional": r2["r2_conditional"],
            "var_fixed": r2["var_fixed"],
            "var_random": r2["var_random"],
            "var_residual": r2["var_residual"],
            "method": method,
            "boundary": boundary,
            "vif_age_in_M4": vif_age,
            "vif_generation_in_M4": vif_gen,
            "is_upper_bound": (model_name == "M5"),
        })


results_df = pd.DataFrame(rows)

# --- Add delta vs M1 ---
m1 = results_df[results_df["model"] == "M1"][["language", "r2_marginal"]].rename(
    columns={"r2_marginal": "r2_m_M1"}
)
results_df = results_df.merge(m1, on="language", how="left")
results_df["delta_vs_M1_pp"] = (
    results_df["r2_marginal"] - results_df["r2_m_M1"]
) * 100
results_df.to_csv("results_extended_fixed_effects.csv", index=False)
print("\nSaved results_extended_fixed_effects.csv")

# --- Summary table ---
print("\n" + "=" * 70)
print("R²_m TABLE (percentage; M5 is informational upper bound)")
print("=" * 70)
pivot = results_df.pivot(index="language", columns="model",
                         values="r2_marginal").reindex(
    columns=["M1", "M2", "M3", "M4", "M5"]) * 100
print(pivot.to_string(float_format="%.3f"))

fb = results_df[results_df["method"] == "ols_fallback"]
if len(fb):
    print("\nOLS-FALLBACK MODELS (LMM hit boundary or singular; "
          "R²_m computed via OLS using M1's total-variance anchor):")
    print(fb[["language", "model", "r2_marginal", "boundary"]]
          .to_string(index=False))

# --- Figure 20 ---
fig, ax = plt.subplots(figsize=(11, 6))
order = ["M1", "M2", "M3", "M4", "M5"]
for lang in pivot.index:
    vals = pivot.loc[lang, order].values
    ax.plot(order, vals, marker="o", label=lang)
ax.set_ylabel("R$^2_m$ (%)")
ax.set_xlabel("Model (fixed-effects coverage)")
ax.set_title("Task 2B: Extended fixed-effects R²_m progression\n"
             "M5 (rightmost) is an informational upper bound, not a "
             "demographic-only model")
ax.legend(loc="upper left", ncol=3, fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("fig20_extended_fe_progression.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved fig20_extended_fe_progression.png")

print("\n" + "=" * 70)
print("STEP 18 COMPLETE")
print("=" * 70)
