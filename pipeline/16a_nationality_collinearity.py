"""
Step 16a: Audit collinearity among the 4 nationality-adjacent fields.

Per language, computes:
  * 4x4 Cramer's V matrix on annotator-level values of
    {ethnicity, country_birth, country_residence, nationality}.
  * VIF on a design matrix encoding all four as dummies in
    `label ~ age + C(gender) + C(eth) + C(cb) + C(cr) + C(nat)` (no random
    effects -- VIF is a property of the design matrix).
  * Per-annotator concordance: number of distinct values across the 4 fields,
    and the share of annotators with concordance == 1.

Outputs:
  - results_nationality_collinearity.csv  (long format Cramer's V)
  - results_nationality_vif.csv           (per coefficient VIF per language)
  - results_nationality_concordance.csv   (annotator concordance summary)
  - fig_nationality_collinearity.png      (heatmap grid)
"""

import json
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

warnings.filterwarnings("ignore")

print("=" * 70)
print("STEP 16a: NATIONALITY-ADJACENT COLLINEARITY AUDIT")
print("=" * 70)

with open("lang_metadata_v2.json") as fh:
    meta = json.load(fh)

NAT_FIELDS = ["ethnicity", "country_birth", "country_residence", "nationality"]


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma & Wicher, 2013)."""
    confusion = pd.crosstab(x, y)
    if confusion.size == 0:
        return np.nan
    chi2 = chi2_contingency(confusion, correction=False)[0]
    n = confusion.values.sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion.shape
    if min(r, k) <= 1:
        return 0.0
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    r_corr = r - (r - 1) ** 2 / max(n - 1, 1)
    k_corr = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


cramer_rows = []
vif_rows = []
concord_rows = []

languages = sorted(meta.keys())

for lang in languages:
    df_lang = pd.read_parquet(f"data_{lang}_v2.parquet")
    ann = df_lang.drop_duplicates("annotator_id")[NAT_FIELDS + ["annotator_id"]].copy()

    # --- Cramer's V matrix ---
    print(f"\n[{lang}] {len(ann)} annotators")
    for f1, f2 in combinations(NAT_FIELDS, 2):
        v = cramers_v(ann[f1], ann[f2])
        cramer_rows.append(
            {"language": lang, "field_a": f1, "field_b": f2, "cramers_v": v}
        )
        print(f"   V({f1}, {f2}) = {v:.3f}")

    # --- Concordance: number of distinct values across the 4 fields ---
    distinct_counts = ann[NAT_FIELDS].nunique(axis=1)
    n_concord_1 = int((distinct_counts == 1).sum())
    n_concord_2 = int((distinct_counts == 2).sum())
    n_concord_ge3 = int((distinct_counts >= 3).sum())
    concord_rows.append({
        "language": lang,
        "n_annotators": int(len(ann)),
        "concord_eq_1": n_concord_1,
        "concord_eq_2": n_concord_2,
        "concord_ge_3": n_concord_ge3,
        "share_concord_1": n_concord_1 / len(ann) if len(ann) else np.nan,
    })

    # --- VIF: build a design matrix, drop redundant columns ---
    fields_for_vif = ["age_centered", "gender"] + [
        f for f in NAT_FIELDS
        if df_lang.drop_duplicates("annotator_id")[f].nunique() > 1
    ]
    cat_fields = [f for f in fields_for_vif if f != "age_centered"]
    X_parts = [df_lang[["age_centered"]].astype(float)]
    for f in cat_fields:
        dummies = pd.get_dummies(df_lang[f], prefix=f, drop_first=True).astype(float)
        X_parts.append(dummies)
    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X, has_constant="add")
    # Drop zero-variance columns (rare).
    X = X.loc[:, X.std(axis=0) > 0]
    X_arr = X.values.astype(float)

    for j, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            v = variance_inflation_factor(X_arr, j)
        except Exception:
            v = np.nan
        # Map column back to its source field for readability
        source = "age" if col == "age_centered" else col.split("_", 1)[0]
        vif_rows.append({
            "language": lang,
            "column": col,
            "source_field": source,
            "vif": float(v) if np.isfinite(v) else np.nan,
        })

# --- Save outputs ---
cramer_df = pd.DataFrame(cramer_rows)
vif_df = pd.DataFrame(vif_rows)
concord_df = pd.DataFrame(concord_rows)

cramer_df.to_csv("results_nationality_collinearity.csv", index=False)
vif_df.to_csv("results_nationality_vif.csv", index=False)
concord_df.to_csv("results_nationality_concordance.csv", index=False)

print("\n" + "=" * 70)
print("CONCORDANCE SUMMARY")
print("=" * 70)
print(concord_df.to_string(index=False, float_format="%.3f"))

# --- Per-language max VIF among nationality-adjacent fields ---
print("\nMax VIF among nationality-adjacent dummies per language:")
nat_keys = {"ethnicity", "country_birth", "country_residence", "nationality"}
for lang in languages:
    sub = vif_df[(vif_df["language"] == lang) & vif_df["source_field"].isin(nat_keys)]
    if len(sub):
        worst = sub.loc[sub["vif"].idxmax()]
        print(f"  {lang}: max VIF = {worst['vif']:.1f} ({worst['column']})")

# --- Heatmap grid: Cramer's V matrix per language ---
fig, axes = plt.subplots(3, 3, figsize=(12, 11))
short_labels = {
    "ethnicity": "eth",
    "country_birth": "cob",
    "country_residence": "cor",
    "nationality": "nat",
}
for ax, lang in zip(axes.flatten(), languages):
    mat = pd.DataFrame(
        np.eye(4), index=NAT_FIELDS, columns=NAT_FIELDS, dtype=float
    )
    sub = cramer_df[cramer_df["language"] == lang]
    for _, row in sub.iterrows():
        mat.loc[row["field_a"], row["field_b"]] = row["cramers_v"]
        mat.loc[row["field_b"], row["field_a"]] = row["cramers_v"]
    im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([short_labels[f] for f in NAT_FIELDS], rotation=45)
    ax.set_yticklabels([short_labels[f] for f in NAT_FIELDS])
    ax.set_title(lang.upper())
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat.values[i, j] < 0.6 else "black", fontsize=8)
fig.suptitle("Cramer's V among nationality-adjacent fields (annotator level)", y=1.0)
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Cramer's V")
fig.tight_layout(rect=(0, 0, 0.92, 0.97))
fig.savefig("fig_nationality_collinearity.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved fig_nationality_collinearity.png")

print("\n" + "=" * 70)
print("STEP 16a COMPLETE")
print("=" * 70)
