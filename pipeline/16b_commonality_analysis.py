"""
Step 16b (Task 1B): Commonality / variance partitioning analysis.

For each language, partition R^2_m for V = {ethnicity, country_birth,
country_residence, nationality} into 2^|V| - 1 unique + shared components,
controlling for age + gender as fixed effects (always in the model) and a
random intercept for text_id. Languages where a field is constant (e.g.
nationality in IT/NL/HI) reduce V to its non-constant subset.

Mathematical formulation (Nimon, Oswald & Roberts 2008; Mood 1969):

  Define f(W) = R^2_m(controls + W) - R^2_m(controls), for any W subseteq V.
  Define g(S) = f(V) - f(V \\ S) for any S subseteq V (additional R^2 lost
  when S is dropped from the full set V).

  Commonality coefficient for any non-empty T subseteq V:
      C(T) = sum_{S subseteq T} (-1)^{|T|-|S|} * g(S)

  This satisfies:  sum_{non-empty T subseteq V} C(T) = f(V).

Implementation notes:
  * fit_lmm() from utils.py provides R^2_m for each subset.
  * Some subsets are rank-deficient because pairs of nationality-adjacent
    fields are collinear (e.g. country_birth and nationality in [de], [es],
    [pt]). We handle this with a fallback that drops linearly dependent dummy
    columns via a QR pivot-rank check, then refits a manual LMM via
    statsmodels.MixedLM with the reduced design matrix.

Outputs:
  - results_commonality_subset_r2.csv     (per-subset R^2)
  - results_commonality_per_language.csv  (15 components per language)
  - results_commonality_summary.csv       (unique + shared rollup)
  - fig17_commonality_stacked.png         (per-language stacked bar)
"""

import json
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

from utils import fit_lmm

warnings.filterwarnings("ignore")

print("=" * 70)
print("STEP 16b (Task 1B): COMMONALITY ANALYSIS")
print("=" * 70)

with open("lang_metadata_v2.json") as fh:
    meta = json.load(fh)

V_FULL = ["ethnicity", "country_birth", "country_residence", "nationality"]
CONTROLS = ["age_centered", "C(gender)"]


def all_nonempty_subsets(items):
    out = []
    for k in range(1, len(items) + 1):
        out.extend([tuple(s) for s in combinations(items, k)])
    return out


def build_design_matrix(df, vars_list):
    """Build a numeric design matrix (no intercept here; LMM adds it)."""
    parts = []
    for v in vars_list:
        if v == "age_centered":
            parts.append(df[["age_centered"]].astype(float))
        elif v == "gender":
            d = pd.get_dummies(df["gender"], prefix="gender", drop_first=True)
            parts.append(d.astype(float))
        else:
            d = pd.get_dummies(df[v], prefix=v, drop_first=True)
            parts.append(d.astype(float))
    if not parts:
        return pd.DataFrame(index=df.index)
    X = pd.concat(parts, axis=1)
    # Drop zero-variance columns (rare) defensively
    nz = X.std(axis=0) > 0
    X = X.loc[:, nz]
    return X


def drop_collinear_columns(X: pd.DataFrame, tol: float = 1e-8) -> pd.DataFrame:
    """Iteratively drop columns whose addition makes the design matrix
    rank-deficient (using QR pivoting via NumPy)."""
    if X.shape[1] == 0:
        return X
    arr = X.values.astype(float)
    n_cols = arr.shape[1]
    # Augmented with a constant column so we measure rank against an intercept
    aug = np.column_stack([np.ones(len(arr)), arr])
    q, r = np.linalg.qr(aug)
    diag = np.abs(np.diag(r))
    keep = []
    for j in range(n_cols):
        # diag[0] is the intercept; columns of arr correspond to diag[1:]
        if diag[j + 1] > tol:
            keep.append(X.columns[j])
    return X[keep]


def fit_subset_r2_robust(df_lang, subset):
    """Try the formula API first; fall back to a manual MixedLM with rank
    reduction on singular designs.

    Returns (r2_marginal, formula_str).
    """
    base_vars = ["age_centered", "gender"] + list(subset)
    formula_rhs = ["age_centered", "C(gender)"] + [f"C({v})" for v in subset]
    formula = "label ~ " + " + ".join(formula_rhs)
    try:
        _, r2 = fit_lmm(formula, df_lang, groups_col="text_id")
        return float(r2["r2_marginal"]), formula
    except (np.linalg.LinAlgError, ValueError) as exc:  # noqa: BLE001
        # Rank-deficient design -> drop collinear dummies manually.
        X = build_design_matrix(df_lang, base_vars)
        X = drop_collinear_columns(X)
        if X.shape[1] == 0:
            # Only an intercept can be fit; R^2 marginal = 0
            return 0.0, formula + "  [reduced: intercept only]"
        X = sm.add_constant(X, has_constant="add")
        try:
            md = MixedLM(
                endog=df_lang["label"].astype(float).values,
                exog=X.values.astype(float),
                groups=df_lang["text_id"].values,
            )
            res = md.fit(reml=True, method="lbfgs")
        except Exception as inner:  # noqa: BLE001
            print(f"      WARNING: manual fit failed for {subset}: {inner}")
            return np.nan, formula + "  [FAILED]"

        # Manual Nakagawa-Schielzeth marginal R^2
        fixed_pred = X.values @ res.fe_params
        var_fixed = float(np.var(fixed_pred))
        cov_re = res.cov_re
        if hasattr(cov_re, "iloc"):
            var_random = float(cov_re.iloc[0, 0])
        else:
            var_random = float(np.asarray(cov_re).flatten()[0])
        var_residual = float(res.scale)
        total = var_fixed + var_random + var_residual
        r2m = var_fixed / total if total > 0 else 0.0
        return float(r2m), formula + f"  [manual, kept {X.shape[1]-1} cols]"


rows_components = []
rows_subset_r2 = []

languages = sorted(meta.keys())

for lang in languages:
    df_lang = pd.read_parquet(f"data_{lang}_v2.parquet")
    V = [
        v for v in V_FULL
        if df_lang.drop_duplicates("annotator_id")[v].nunique() > 1
    ]
    if not V:
        print(f"\n[{lang}] All 4 nationality-adjacent fields are constant. SKIP.")
        continue

    print(f"\n[{lang}] V = {V} ({len(V)} variables, "
          f"{2 ** len(V) - 1} non-empty subsets)")

    # --- Baseline R^2 with controls only ---
    base_formula = f"label ~ {' + '.join(CONTROLS)}"
    _, r2_base = fit_lmm(base_formula, df_lang, groups_col="text_id")
    r2_controls = float(r2_base["r2_marginal"])

    # --- f(W) = R^2_m(controls + W) - R^2_m(controls), W subseteq V ---
    f_lookup = {(): 0.0}
    for S in all_nonempty_subsets(V):
        r2_S, formula = fit_subset_r2_robust(df_lang, list(S))
        if np.isnan(r2_S):
            f_lookup[tuple(S)] = np.nan
        else:
            f_lookup[tuple(S)] = r2_S - r2_controls
        rows_subset_r2.append({
            "language": lang,
            "subset": "+".join(S),
            "r2_with_controls": r2_S,
            "r2_above_controls": (
                np.nan if np.isnan(r2_S) else r2_S - r2_controls
            ),
            "formula": formula,
        })
        flag = " [reduced]" if "[manual" in formula or "[reduced" in formula else ""
        print(f"   subset={S}  R^2_m={r2_S*100:.3f}%"
              f"  (above controls: {(r2_S-r2_controls)*100:.3f}%){flag}")

    if any(np.isnan(v) for v in f_lookup.values()):
        print(f"   WARNING: NaN R^2 in subset(s); commonality skipped for {lang}.")
        continue

    # --- g(S) = f(V) - f(V \\ S) ---
    f_V = f_lookup[tuple(V)]

    def f_of(W):
        # Always look up by canonical (sorted-by-V) tuple
        key = tuple(v for v in V if v in W)
        return f_lookup[key]

    def g_of(S):
        return f_V - f_of([v for v in V if v not in S])

    # --- C(T) = sum_{S subseteq T} (-1)^{|T|-|S|} * g(S) ---
    components = {}
    for T in all_nonempty_subsets(V):
        c_T = 0.0
        T_set = set(T)
        for k in range(0, len(T) + 1):
            for S in combinations(T, k):
                sign = (-1) ** (len(T) - len(S))
                c_T += sign * g_of(set(S))
        components[T] = c_T

    total_components = sum(components.values())
    print(f"   sum(components) = {total_components*100:.4f}%   "
          f"f(V) = {f_V*100:.4f}%   "
          f"|diff| = {abs(total_components - f_V)*1e6:.2f} ppm")

    for T, c in components.items():
        rows_components.append({
            "language": lang,
            "component_type": (
                "unique" if len(T) == 1 else f"shared_{len(T)}way"
            ),
            "variables": "+".join(T),
            "value": float(c),
        })

# --- Save outputs ---
components_df = pd.DataFrame(rows_components)
subset_r2_df = pd.DataFrame(rows_subset_r2)
components_df.to_csv("results_commonality_per_language.csv", index=False)
subset_r2_df.to_csv("results_commonality_subset_r2.csv", index=False)

# --- Per-language summary: unique component per variable + total shared ---
summary_rows = []
for lang in components_df["language"].unique():
    sub = components_df[components_df["language"] == lang]
    rec = {"language": lang}
    for v in V_FULL:
        u = sub[(sub["component_type"] == "unique") & (sub["variables"] == v)]
        rec[f"unique_{v}"] = float(u["value"].iloc[0]) if len(u) else 0.0
    shared = sub[sub["component_type"].str.startswith("shared")]["value"].sum()
    rec["total_shared"] = float(shared)
    rec["total_above_controls"] = float(sub["value"].sum())
    summary_rows.append(rec)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results_commonality_summary.csv", index=False)

print("\n" + "=" * 70)
print("UNIQUE-COMPONENT SUMMARY (percentage points above age+gender controls)")
print("=" * 70)
display = summary_df.copy()
for col in display.columns:
    if col == "language":
        continue
    display[col] = display[col] * 100
print(display.to_string(index=False, float_format="%.3f"))

# --- Figure 17: stacked bar ---
fig, ax = plt.subplots(figsize=(11, 6))
labels = list(summary_df["language"])
unique_cols = [f"unique_{v}" for v in V_FULL]
bottoms = np.zeros(len(labels))
colors = {
    "unique_ethnicity": "#4C72B0",
    "unique_country_birth": "#55A868",
    "unique_country_residence": "#C44E52",
    "unique_nationality": "#8172B2",
}
pretty = {
    "unique_ethnicity": "Unique(ethnicity)",
    "unique_country_birth": "Unique(country_birth)",
    "unique_country_residence": "Unique(country_residence)",
    "unique_nationality": "Unique(nationality)",
}
for col in unique_cols:
    vals = (summary_df[col] * 100).clip(lower=0).values
    ax.bar(labels, vals, bottom=bottoms, label=pretty[col],
           color=colors[col], edgecolor="white", linewidth=0.5)
    bottoms += vals
shared_vals = (summary_df["total_shared"] * 100).clip(lower=0).values
ax.bar(labels, shared_vals, bottom=bottoms, label="Total shared",
       color="#999999", edgecolor="white", linewidth=0.5)

ax.set_ylabel("R$^2$ component (pp above age+gender controls)")
ax.set_xlabel("Language")
ax.set_title("Task 1B: Commonality decomposition of nationality-adjacent fields\n"
             "(unique contributions + total shared variance)")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("fig17_commonality_stacked.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved fig17_commonality_stacked.png")

print("\n" + "=" * 70)
print("STEP 16b COMPLETE")
print("=" * 70)
