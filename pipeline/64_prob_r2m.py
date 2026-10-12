"""
64_prob_r2m.py
==============
R²m analysis on continuous probability labels (Variant A experiment).

Methodology is identical to 29_rmm_analysis.py with two differences:
  1. Outcome variable is llm_prob (continuous [0,1]) instead of llm_label (0/1).
  2. Sample Human (SH) baseline is also computed with continuous human proportions
     — the mean human_label per post_id — to keep the comparison apples-to-apples.

Outputs:
    results_prob_r2m.csv             — R²m (prob format) per lang × cond
    results_prob_steerability.csv    — steerability (prob format)
    results_prob_comparison.csv      — side-by-side: binary vs prob R²m per lang × cond
    fig_prob_r2m_comparison.png      — visual comparison
"""

import warnings, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANGUAGES  = ["ar", "es"]
CONDITIONS = ["p0", "p1", "p2", "p3"]

# ── Full-dataset Study 1 benchmarks (from 29_rmm_analysis.py) ────────────────
BENCHMARKS = {
    "ar": {"m1": 0.019017, "m4": 0.038825, "target": 0.076480},
    "es": {"m1": 0.010041, "m4": 0.032465, "target": 0.093269},
}

# ── Condition-matched formula definitions (identical to 29_rmm_analysis.py) ───
M1_BASE = {
    "ar": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "es": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
}
M4_EXTRAS = ["C(country_birth)", "C(country_residence)",
             "C(employment_status)", "C(student_status)"]
P3_TEMPLATE = {
    "ar": ["age_centered", "C(nationality)"],
    "es": ["age_centered", "C(country_birth)", "C(country_residence)"],
}
ALL_PRED_COLS = ["age_centered", "gender", "ethnicity", "nationality",
                 "country_birth", "country_residence",
                 "employment_status", "student_status", "Generation"]


def col_from_term(term):
    m = re.match(r"C\((\w+)\)", term)
    return m.group(1) if m else term


def build_formula(target, var_list, df_check):
    valid = []
    for term in var_list:
        col = col_from_term(term)
        if col not in df_check.columns:
            continue
        if col == "age_centered":
            valid.append(term)
            continue
        if df_check[col].dropna().nunique() >= 2:
            valid.append(term)
    if not valid:
        return None
    return f"{target} ~ " + " + ".join(valid)


def get_formula(target, lang, cond, df_check):
    if cond in ("p0", "p1"):
        return build_formula(target, M1_BASE[lang], df_check)
    elif cond == "p2":
        return build_formula(target, M1_BASE[lang] + M4_EXTRAS, df_check)
    else:
        return build_formula(target, P3_TEMPLATE[lang], df_check)


def needed_cols(formula, df, target):
    cols = [target, "post_id"]
    for col in ALL_PRED_COLS:
        if col in formula and col in df.columns:
            cols.append(col)
    return list(dict.fromkeys(cols))


def fit_r2m(df_sub, formula, method="bfgs"):
    target = formula.split("~")[0].strip()
    nc = needed_cols(formula, df_sub, target)
    df_fit = df_sub[[c for c in nc if c in df_sub.columns]].dropna()

    if len(df_fit) < 50 or df_fit[target].nunique() < 2:
        return {"r2m": np.nan, "n": len(df_fit), "converged": False}

    try:
        m = smf.mixedlm(formula, data=df_fit,
                        groups=df_fit["post_id"]).fit(
                            reml=False, method=method, disp=False)
        fp = np.dot(m.model.exog, m.fe_params)
        vf = np.var(fp)
        vr = float(m.cov_re.iloc[0, 0])
        r2m = vf / (vf + vr + m.scale)
        return {"r2m": r2m, "n": len(df_fit), "converged": m.converged}
    except Exception as e:
        return {"r2m": np.nan, "n": 0, "converged": False, "error": str(e)}


# ── Load probability predictions ──────────────────────────────────────────────
df_prob = pd.read_parquet("prob_predictions_clean.parquet")
df_prob["age"] = pd.to_numeric(df_prob["age"], errors="coerce")
lang_age_mean  = df_prob.groupby("language")["age"].transform("mean")
df_prob["age_centered"] = df_prob["age"] - lang_age_mean
for col in ["gender","ethnicity","nationality","country_birth",
            "country_residence","employment_status","student_status"]:
    if col in df_prob.columns:
        df_prob[col] = df_prob[col].replace("unknown", np.nan)

print(f"Loaded {len(df_prob):,} probability prediction rows")
print(f"Languages: {sorted(df_prob['language'].unique())}")
print(f"Conditions: {sorted(df_prob['condition'].unique())}")

# ── Compute mean human_label per post_id per language for the SH baseline ────
# This gives the annotation proportion (e.g. 3/5 annotators = 0.6) per text.
human_prop = (
    df_prob[df_prob["condition"]=="p0"]
    .groupby(["language","post_id"])["human_label"]
    .mean()
    .reset_index()
    .rename(columns={"human_label": "human_prop"})
)
df_prob = df_prob.merge(human_prop, on=["language","post_id"], how="left")

# ── Sample Human (SH) baseline — continuous human proportion ─────────────────
print(f"\n{'═'*65}")
print("  SAMPLE HUMAN R²m (condition-matched formula, human_prop outcome)")
print(f"{'═'*65}")

SAMPLE_HUMAN_PROB = {}
for lang in LANGUAGES:
    mask_h = (df_prob["language"] == lang) & (df_prob["condition"] == "p0")
    df_h = df_prob[mask_h].copy()
    for cond in CONDITIONS:
        hf = get_formula("human_prop", lang, cond, df_h)
        if hf is None:
            SAMPLE_HUMAN_PROB[(lang, cond)] = np.nan
            continue
        res = fit_r2m(df_h, hf)
        SAMPLE_HUMAN_PROB[(lang, cond)] = res["r2m"]

print(f"{'Lang':<5} {'SH_P0/P1':>10} {'SH_P2':>10} {'SH_P3':>10}")
print("-"*40)
for lang in LANGUAGES:
    print(f"{lang.upper():<5} "
          f"{SAMPLE_HUMAN_PROB[(lang,'p0')]*100:>9.3f}%  "
          f"{SAMPLE_HUMAN_PROB[(lang,'p2')]*100:>9.3f}%  "
          f"{SAMPLE_HUMAN_PROB[(lang,'p3')]*100:>9.3f}%")

# ── Main R²m loop on llm_prob ─────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("  R²m ANALYSIS — continuous llm_prob outcome, condition-matched formulas")
print(f"{'═'*65}")

results = []
for lang in LANGUAGES:
    bench = BENCHMARKS[lang]
    print(f"\n{lang.upper()}  Dir1 M1={bench['m1']*100:.3f}%  M4={bench['m4']*100:.3f}%")
    for cond in CONDITIONS:
        mask = (df_prob["language"] == lang) & (df_prob["condition"] == cond)
        df_sub = df_prob[mask].copy()
        formula = get_formula("llm_prob", lang, cond, df_sub)
        if formula is None:
            print(f"  {cond.upper()}: SKIPPED")
            results.append({"language": lang, "condition": cond,
                            "r2m_prob": np.nan, "pct_sh_prob": np.nan})
            continue

        res = fit_r2m(df_sub, formula)
        r2m = res["r2m"]
        sh  = SAMPLE_HUMAN_PROB[(lang, cond)]
        pct = (r2m / sh * 100) if (not np.isnan(r2m) and sh and sh > 0) else np.nan

        print(f"  {cond.upper()} R²m(prob)={r2m*100:.3f}%  SH(prop)={sh*100:.3f}%  %SH={pct:.1f}%"
              f"  n={res['n']}")
        results.append({
            "language":    lang,
            "condition":   cond,
            "r2m_prob":    r2m,
            "r2m_prob_pct": r2m*100 if not np.isnan(r2m) else np.nan,
            "r2m_sh_prob": sh,
            "pct_sh_prob": pct,
        })

df_res = pd.DataFrame(results)

# ── Load binary results for side-by-side comparison ──────────────────────────
try:
    df_bin = pd.read_csv("results_r2m_llm.csv")
    df_bin_filt = df_bin[df_bin["language"].isin(LANGUAGES)][
        ["language","condition","r2m_llm_pct","pct_of_sample_human"]
    ].rename(columns={
        "r2m_llm_pct":         "r2m_binary_pct",
        "pct_of_sample_human": "pct_sh_binary",
    })
    df_comp = df_res.merge(df_bin_filt, on=["language","condition"], how="left")
    print(f"\n{'═'*65}")
    print("  SIDE-BY-SIDE COMPARISON: Binary vs Probability format")
    print(f"{'═'*65}")
    for lang in LANGUAGES:
        print(f"\n{lang.upper()}:")
        sub = df_comp[df_comp["language"]==lang][
            ["condition","r2m_binary_pct","r2m_prob_pct","pct_sh_binary","pct_sh_prob"]
        ]
        print(sub.to_string(index=False))
    df_comp.to_csv("results_prob_comparison.csv", index=False)
    print("\nSaved: results_prob_comparison.csv")
except FileNotFoundError:
    df_comp = df_res
    print("\nNote: results_r2m_llm.csv not found — binary comparison skipped.")

# ── Steerability ──────────────────────────────────────────────────────────────
steer_rows = []
for lang in LANGUAGES:
    bench  = BENCHMARKS[lang]
    lang_df = df_res[df_res["language"]==lang]
    r2m_p0  = lang_df[lang_df["condition"]=="p0"]["r2m_prob"].values[0]
    idx_max = lang_df["r2m_prob"].idxmax()
    r2m_best = lang_df.loc[idx_max, "r2m_prob"]
    best_cond = lang_df.loc[idx_max, "condition"]
    gap   = bench["m4"] - bench["m1"]
    steer = (r2m_best - r2m_p0) / gap if gap > 0 else np.nan
    sh_best = SAMPLE_HUMAN_PROB[(lang, best_cond)]
    pct_sh  = (r2m_best / sh_best * 100) if sh_best and sh_best > 0 else np.nan
    steer_rows.append({
        "language": lang, "best_condition": best_cond,
        "r2m_p0": r2m_p0, "r2m_best": r2m_best,
        "pct_sh": pct_sh, "steerability": steer,
    })

df_steer = pd.DataFrame(steer_rows)
print(f"\n{'═'*65}")
print("  STEERABILITY (probability format)")
print(f"{'═'*65}")
print(df_steer.to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
df_res.to_csv("results_prob_r2m.csv", index=False)
df_steer.to_csv("results_prob_steerability.csv", index=False)
print("\nSaved: results_prob_r2m.csv, results_prob_steerability.csv")

# ── Figure: Binary vs Prob R²m per condition ─────────────────────────────────
if "r2m_binary_pct" in df_comp.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"p0":"#aec6e8","p1":"#4c9be8","p2":"#1a5fa8","p3":"#e85c4c"}
    x = np.arange(len(LANGUAGES))
    w = 0.15

    for ax, metric, title in zip(
        axes,
        [("r2m_binary_pct","r2m_prob_pct"), ("pct_sh_binary","pct_sh_prob")],
        ["R²m (%) — Binary vs Probability format",
         "% of Sample Human R²m — Binary vs Probability"]
    ):
        for i, cond in enumerate(CONDITIONS):
            sub = df_comp[df_comp["condition"]==cond]
            bin_vals  = sub.set_index("language").reindex(LANGUAGES)[metric[0]].values
            prob_vals = sub.set_index("language").reindex(LANGUAGES)[metric[1]].values
            pos = x + i * w * 2
            ax.bar(pos,       bin_vals,  w, color=colors[cond], alpha=0.6,
                   label=f"{cond.upper()} binary" if i==0 else None,
                   hatch="")
            ax.bar(pos + w,   prob_vals, w, color=colors[cond], alpha=1.0,
                   label=f"{cond.upper()} prob" if i==0 else None,
                   hatch="//")
        ax.set_xticks(x + w * 3.5)
        ax.set_xticklabels([l.upper() for l in LANGUAGES])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("%")
        ax.grid(axis="y", alpha=0.3)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elem = (
        [Patch(facecolor=colors[c], alpha=0.6, label=f"{c.upper()} binary") for c in CONDITIONS] +
        [Patch(facecolor=colors[c], alpha=1.0, hatch="//", label=f"{c.upper()} prob") for c in CONDITIONS]
    )
    axes[0].legend(handles=legend_elem, fontsize=7, ncol=2)
    fig.suptitle("Probability Elicitation vs Binary: AR and ES", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig_prob_r2m_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig_prob_r2m_comparison.png")

print(f"\n{'═'*65}")
print("  DONE — next step: python 65_prob_concordance.py")
print(f"{'═'*65}")
