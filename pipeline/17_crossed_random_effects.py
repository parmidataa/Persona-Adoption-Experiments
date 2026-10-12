"""
Step 17 (Task 2A): Crossed random effects -- (1|text_id) + (1|annotator_id).

Fits per-language LMM with both text and annotator random intercepts and
decomposes total variance into:
   - var_fixed     (persona fixed effects)
   - var_text      (between-text variance)
   - var_annotator (between-annotator variance, captures stable individual
                    differences; the "ceiling" for any persona-based model)
   - var_residual  (within-annotator within-text noise)

Defines:
   icc_text       = var_text / total
   icc_annotator  = var_annotator / total
   r2_marginal    = var_fixed / total
   r2_conditional = (var_fixed + var_text + var_annotator) / total
   target_r2      = var_annotator / (var_fixed + var_annotator + var_residual)
                    -- the proportion of variance any annotator-level predictor
                       (incl. an oracle persona model) could ever capture.
                       Excludes var_text from the denominator because text
                       variance is a different axis.

Implementation: statsmodels MixedLM with `groups=np.ones(...)` and
`vc_formula={"text": "0 + C(text_id)", "ann": "0 + C(annotator_id)"}` --
this is the canonical pattern for crossed random intercepts in pure Python
(R is not installed in the environment, so pymer4 is unavailable).

Outputs:
  - results_crossed_r2_per_language.csv
  - fig18_four_way_variance_decomposition.png
  - fig19_saturation_per_language.png
"""

import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

print("=" * 70)
print("STEP 17 (Task 2A): CROSSED RANDOM EFFECTS")
print("=" * 70)

with open("lang_metadata_v2.json") as fh:
    meta = json.load(fh)


def fit_crossed_lmm(df_lang, fixed_rhs):
    """Fit `label ~ fixed_rhs + (1|text_id) + (1|annotator_id)`.

    Returns a dict of variance components and derived R^2 / ICC quantities.
    """
    df = df_lang.copy()
    df["_one"] = 1  # single grouping stub
    formula = "label ~ " + fixed_rhs
    vc = {
        "text": "0 + C(text_id)",
        "ann": "0 + C(annotator_id)",
    }
    md = smf.mixedlm(
        formula,
        df,
        groups=df["_one"],
        vc_formula=vc,
    )
    res = md.fit(reml=True, method="lbfgs")

    # Variance components: vcomp[0] = text, vcomp[1] = annotator (alphabetical)
    # Map by name to be safe.
    vc_names = list(res.vcomp_names) if hasattr(res, "vcomp_names") else None
    if vc_names is None:
        # Fallback: re.vc_names
        vc_names = res.model.exog_vc.names
    vcomp = np.asarray(res.vcomp)
    vc_map = dict(zip(vc_names, vcomp.tolist()))

    var_text = float(vc_map.get("text", 0.0))
    var_ann = float(vc_map.get("ann", 0.0))
    var_residual = float(res.scale)
    fixed_pred = res.model.exog @ res.fe_params
    var_fixed = float(np.var(fixed_pred))

    total = var_fixed + var_text + var_ann + var_residual
    r2_marginal = var_fixed / total if total > 0 else 0.0
    r2_conditional = (var_fixed + var_text + var_ann) / total if total > 0 else 0.0
    icc_text = var_text / total if total > 0 else 0.0
    icc_annotator = var_ann / total if total > 0 else 0.0
    target_denom = var_fixed + var_ann + var_residual
    target_r2 = var_ann / target_denom if target_denom > 0 else 0.0
    saturation = r2_marginal / target_r2 if target_r2 > 0 else np.nan

    return {
        "var_fixed": var_fixed,
        "var_text": var_text,
        "var_annotator": var_ann,
        "var_residual": var_residual,
        "total_var": total,
        "r2_marginal": r2_marginal,
        "r2_conditional": r2_conditional,
        "icc_text": icc_text,
        "icc_annotator": icc_annotator,
        "target_r2": target_r2,
        "saturation": saturation,
        "converged": bool(res.converged),
    }


rows = []
languages = sorted(meta.keys())

for lang in languages:
    df_lang = pd.read_parquet(f"data_{lang}_v2.parquet")
    info = meta[lang]
    cat_vars = info["usable_cat_vars"]

    # Match the v1 "full persona" RHS as closely as possible:
    # age + gender + ethnicity + nationality (only those 4, to be comparable
    # with §1 of findings.md).
    rhs_terms = ["age_centered", "C(gender)"]
    for v in ("ethnicity", "nationality"):
        if v in cat_vars:
            rhs_terms.append(f"C({v})")
    rhs = " + ".join(rhs_terms)

    print(f"\n[{lang}] n={len(df_lang):,} ann={info['n_annotators']} "
          f"texts={info['n_texts']} | fixed: {rhs}")
    t0 = time.time()
    try:
        out = fit_crossed_lmm(df_lang, rhs)
    except Exception as exc:  # noqa: BLE001
        print(f"   FAILED: {exc}")
        continue
    elapsed = time.time() - t0
    print(f"   converged={out['converged']}  ({elapsed:.1f}s)")
    print(f"   var_fixed={out['var_fixed']:.5f}  var_text={out['var_text']:.5f}"
          f"  var_ann={out['var_annotator']:.5f}  var_resid={out['var_residual']:.5f}")
    print(f"   r2_marg={out['r2_marginal']*100:.3f}%   "
          f"r2_cond={out['r2_conditional']*100:.3f}%")
    print(f"   icc_text={out['icc_text']*100:.2f}%   "
          f"icc_ann={out['icc_annotator']*100:.2f}%")
    print(f"   target_r2={out['target_r2']*100:.3f}%   "
          f"saturation={out['saturation']*100:.1f}%")

    rows.append({
        "language": lang,
        "n_rows": int(info["n_rows"]),
        "n_annotators": int(info["n_annotators"]),
        "n_texts": int(info["n_texts"]),
        "fixed_rhs": rhs,
        **{k: out[k] for k in [
            "var_fixed", "var_text", "var_annotator", "var_residual", "total_var",
            "r2_marginal", "r2_conditional", "icc_text", "icc_annotator",
            "target_r2", "saturation", "converged",
        ]},
        "fit_seconds": elapsed,
    })

results_df = pd.DataFrame(rows).sort_values("language").reset_index(drop=True)
results_df.to_csv("results_crossed_r2_per_language.csv", index=False)
print("\nSaved results_crossed_r2_per_language.csv")

# --- Summary table ---
print("\n" + "=" * 70)
print("SUMMARY (percentages)")
print("=" * 70)
disp = results_df[[
    "language", "icc_text", "icc_annotator", "r2_marginal", "target_r2",
    "saturation", "converged", "fit_seconds",
]].copy()
for c in ("icc_text", "icc_annotator", "r2_marginal", "target_r2", "saturation"):
    disp[c] = disp[c] * 100
print(disp.to_string(index=False, float_format="%.3f"))

# --- Figure 18: stacked variance decomposition (4 components) ---
fig, ax = plt.subplots(figsize=(11, 6))
labels = list(results_df["language"])
totals = results_df["total_var"].values
shares = {
    "Persona (fixed)": (results_df["var_fixed"] / totals * 100).values,
    "Text RE": (results_df["var_text"] / totals * 100).values,
    "Annotator RE": (results_df["var_annotator"] / totals * 100).values,
    "Residual": (results_df["var_residual"] / totals * 100).values,
}
colors_4 = {
    "Persona (fixed)": "#4C72B0",
    "Text RE": "#55A868",
    "Annotator RE": "#C44E52",
    "Residual": "#999999",
}
bottoms = np.zeros(len(labels))
for name, vals in shares.items():
    ax.bar(labels, vals, bottom=bottoms, label=name,
           color=colors_4[name], edgecolor="white", linewidth=0.5)
    bottoms += vals
ax.set_ylabel("% of total variance")
ax.set_xlabel("Language")
ax.set_title("Task 2A: Four-way variance decomposition\n"
             "Persona / Text RE / Annotator RE / Residual")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig("fig18_four_way_variance_decomposition.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print("\nSaved fig18_four_way_variance_decomposition.png")

# --- Figure 19: saturation per language ---
fig, ax = plt.subplots(figsize=(10, 5))
order = results_df.sort_values("saturation", ascending=False)
sat = order["saturation"].fillna(0).values * 100
ax.barh(order["language"].values, sat, color="#4C72B0", edgecolor="white")
for i, (lang, s, r2m, tgt) in enumerate(zip(
    order["language"].values, sat,
    order["r2_marginal"].values * 100,
    order["target_r2"].values * 100,
)):
    ax.text(s + 0.5, i,
            f"R²_m={r2m:.2f}% / Target={tgt:.2f}%",
            va="center", fontsize=8)
ax.set_xlabel("Saturation = R²_m / Target_R²  (%)")
ax.set_title("Task 2A: How much of the annotator-variance ceiling do current\n"
             "persona variables capture? (Higher = closer to oracle)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("fig19_saturation_per_language.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig19_saturation_per_language.png")

print("\n" + "=" * 70)
print("STEP 17 COMPLETE")
print("=" * 70)
