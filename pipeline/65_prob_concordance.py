"""
65_prob_concordance.py
======================
Directional concordance analysis for the probability-elicitation experiment.

Methodology is identical to 35_directional_concordance.py, with the single
difference that the LLM outcome is llm_prob (continuous) instead of
llm_label (binary).  Because both coefficient vectors are derived from the
same design matrix, their signs and ranks remain directly comparable despite
the different scales of the two outcome variables.

Compares:
  - Human model  : binary human_label ~ demographics + (1|post_id)
  - LLM model    : continuous llm_prob  ~ demographics + (1|post_id)

For each language × best condition:
  - Sign agreement (fraction of matching signs)
  - Spearman ρ (rank correlation between coefficient vectors)
  - Pearson r and cosine similarity (secondary)
  - Model-free group-deviation cross-check (same as 35_directional_concordance.py)

Prints a direct comparison: binary format vs probability format.

Outputs:
    results_prob_concordance.csv
    fig_prob_concordance_comparison.png
"""

import warnings, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_SUPPORT = 15
LANGUAGES   = ["ar", "es"]
BEST_COND   = {"ar": "p2", "es": "p2"}   # from original binary experiment

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


def col_from_term(t):
    m = re.match(r"C\((\w+)\)", t)
    return m.group(1) if m else t


def build_formula(target, var_list, df_check):
    valid = []
    for term in var_list:
        col = col_from_term(term)
        if col not in df_check.columns:
            continue
        if col == "age_centered":
            valid.append(term); continue
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


def coef_support(df_fit, term):
    """Number of observations for a categorical level (or total for continuous)."""
    col = col_from_term(term)
    if col == "age_centered":
        return len(df_fit)
    if col not in df_fit.columns:
        return 0
    # For C(col)[T.level], the support is the count of that level
    return df_fit[col].value_counts().min()  # use min level as conservative


def fit_and_coefs(df_sub, formula):
    """Fit LMM and return a dict {term: coef} for terms passing MIN_SUPPORT."""
    target = formula.split("~")[0].strip()
    nc = [target, "post_id"] + [col_from_term(t) for t in
          re.findall(r"C\(\w+\)|\w+", formula.split("~")[1])]
    nc = [c for c in dict.fromkeys(nc) if c in df_sub.columns]
    df_fit = df_sub[nc].dropna()
    if len(df_fit) < 50:
        return {}
    try:
        m = smf.mixedlm(formula, data=df_fit,
                        groups=df_fit["post_id"]).fit(reml=False, disp=False)
        coefs = {}
        for name, val in m.fe_params.items():
            if name == "Intercept":
                continue
            # Check support for each categorical level
            # Identify which column this param came from
            col_match = re.search(r"C\((\w+)\)", name)
            if col_match:
                col = col_match.group(1)
                level_match = re.search(r"\[T\.(.+)\]", name)
                if level_match and col in df_fit.columns:
                    level = level_match.group(1)
                    support = (df_fit[col] == level).sum()
                    if support >= MIN_SUPPORT:
                        coefs[name] = val
            else:
                # Continuous predictor: always include
                coefs[name] = val
        return coefs
    except Exception:
        return {}


def concordance_metrics(h_coefs, l_coefs):
    """Compute sign agreement, Spearman ρ, Pearson r, cosine similarity."""
    common = sorted(set(h_coefs) & set(l_coefs))
    if len(common) < 2:
        return None
    hv = np.array([h_coefs[k] for k in common])
    lv = np.array([l_coefs[k] for k in common])
    sign_agr = np.mean(np.sign(hv) == np.sign(lv))
    rho, _ = spearmanr(hv, lv)
    try:
        r, _ = pearsonr(hv, lv)
    except Exception:
        r = np.nan
    # cosine
    denom = np.linalg.norm(hv) * np.linalg.norm(lv)
    cos = float(np.dot(hv, lv) / denom) if denom > 0 else np.nan
    return {
        "n_coefs": len(common),
        "sign_agreement": sign_agr,
        "spearman_rho":   rho,
        "pearson_r":      r,
        "cosine":         cos,
        "terms":          common,
    }


def group_dev_concordance(df_h, df_l, formula, min_support=MIN_SUPPORT):
    """Model-free group-mean-deviation cross-check (matches 35_directional_concordance.py)."""
    # Identify categorical columns in formula
    cat_cols = re.findall(r"C\((\w+)\)", formula)
    h_target = "human_label"
    l_target = "llm_prob"
    overall_h = df_h[h_target].mean()
    overall_l = df_l[l_target].mean()

    h_devs, l_devs = [], []
    for col in cat_cols:
        if col not in df_h.columns or col not in df_l.columns:
            continue
        for level in set(df_h[col].dropna()):
            mask_h = df_h[col] == level
            mask_l = df_l[col] == level
            if mask_h.sum() < min_support or mask_l.sum() < min_support:
                continue
            h_devs.append(df_h.loc[mask_h, h_target].mean() - overall_h)
            l_devs.append(df_l.loc[mask_l, l_target].mean() - overall_l)

    if len(h_devs) < 2:
        return {"groupdev_sign": np.nan, "groupdev_rho": np.nan, "groupdev_n": 0}
    hv = np.array(h_devs); lv = np.array(l_devs)
    sign_agr = np.mean(np.sign(hv) == np.sign(lv))
    rho, _ = spearmanr(hv, lv)
    return {"groupdev_sign": sign_agr, "groupdev_rho": rho, "groupdev_n": len(h_devs)}


# ── Load data ─────────────────────────────────────────────────────────────────
df_prob = pd.read_parquet("prob_predictions_clean.parquet")
df_prob["age"] = pd.to_numeric(df_prob["age"], errors="coerce")
lang_age_mean  = df_prob.groupby("language")["age"].transform("mean")
df_prob["age_centered"] = df_prob["age"] - lang_age_mean
for col in ["gender","ethnicity","nationality","country_birth",
            "country_residence","employment_status","student_status"]:
    if col in df_prob.columns:
        df_prob[col] = df_prob[col].replace("unknown", np.nan)

print(f"Loaded {len(df_prob):,} probability prediction rows")

# ── Main concordance loop ──────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print(f"  DIRECTIONAL CONCORDANCE — probability format (MIN_SUPPORT={MIN_SUPPORT})")
print(f"{'═'*70}")

rows = []
for lang in LANGUAGES:
    cond = BEST_COND[lang]
    mask = (df_prob["language"] == lang) & (df_prob["condition"] == cond)
    df_sub = df_prob[mask].copy()

    formula_h = get_formula("human_label", lang, cond, df_sub)
    formula_l = get_formula("llm_prob",    lang, cond, df_sub)

    if formula_h is None or formula_l is None:
        print(f"\n{lang.upper()} {cond.upper()}: formula could not be built — skipped")
        continue

    h_coefs = fit_and_coefs(df_sub, formula_h)
    l_coefs = fit_and_coefs(df_sub, formula_l)

    metrics = concordance_metrics(h_coefs, l_coefs)
    if metrics is None:
        print(f"\n{lang.upper()} {cond.upper()}: fewer than 2 common coefficients — skipped")
        continue

    # Group-deviation cross-check
    df_h_sub = df_sub.copy()  # both human and LLM on same rows
    gd = group_dev_concordance(df_h_sub, df_h_sub, formula_h)

    row = {
        "language":       lang,
        "condition":      cond,
        "n_coefs":        metrics["n_coefs"],
        "sign_agreement": metrics["sign_agreement"],
        "spearman_rho":   metrics["spearman_rho"],
        "pearson_r":      metrics["pearson_r"],
        "cosine":         metrics["cosine"],
        **gd,
    }
    rows.append(row)

    print(f"\n{lang.upper()} {cond.upper()} (prob format):")
    print(f"  n_coefs       = {metrics['n_coefs']}")
    print(f"  sign_agreement= {metrics['sign_agreement']*100:.1f}%")
    print(f"  Spearman ρ    = {metrics['spearman_rho']:.3f}")
    print(f"  Pearson r     = {metrics['pearson_r']:.3f}")
    print(f"  cosine        = {metrics['cosine']:.3f}")
    print(f"  group-dev sign= {gd['groupdev_sign']*100:.1f}%  (n={gd['groupdev_n']})")

df_conc = pd.DataFrame(rows)

# ── Load binary concordance for comparison ─────────────────────────────────────
print(f"\n{'═'*70}")
print("  COMPARISON: Binary vs Probability concordance")
print(f"{'═'*70}")
try:
    df_bin_conc = pd.read_csv("results_directional_concordance.csv")
    df_bin_best = df_bin_conc[
        df_bin_conc["language"].isin(LANGUAGES) &
        df_bin_conc["is_best_condition"]
    ][["language","sign_agreement","spearman_rho","groupdev_sign_agreement"]].rename(columns={
        "sign_agreement":          "binary_sign",
        "spearman_rho":            "binary_rho",
        "groupdev_sign_agreement": "binary_groupdev_sign",
    })
    df_comp = df_conc.merge(df_bin_best, on="language", how="left")
    print(f"\n{'Lang':<5} {'Binary sign':>12} {'Prob sign':>10} {'Δ':>8}")
    print("-"*40)
    for _, r in df_comp.iterrows():
        delta = r["sign_agreement"] - r["binary_sign"]
        arrow = "▲" if delta > 0.02 else ("▼" if delta < -0.02 else "≈")
        print(f"{r['language'].upper():<5} "
              f"{r['binary_sign']*100:>10.1f}%  "
              f"{r['sign_agreement']*100:>8.1f}%  "
              f"{arrow} {delta*100:+.1f} pp")
    df_comp.to_csv("results_prob_concordance_comparison.csv", index=False)
    print("\nSaved: results_prob_concordance_comparison.csv")
except FileNotFoundError:
    print("results_directional_concordance.csv not found — binary comparison skipped.")

# ── Save ──────────────────────────────────────────────────────────────────────
df_conc.to_csv("results_prob_concordance.csv", index=False)
print(f"\nSaved: results_prob_concordance.csv")

# ── Figure ────────────────────────────────────────────────────────────────────
if "binary_sign" in df_comp.columns:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(LANGUAGES))
    w = 0.3
    ax.bar(x - w/2, df_comp["binary_sign"]*100, w, color="#4c9be8", label="Binary (original)")
    ax.bar(x + w/2, df_comp["sign_agreement"]*100, w, color="#e85c4c", label="Probability (new)")
    ax.axhline(50, color="grey", linestyle="--", linewidth=1, label="50% chance")
    ax.axhline(60, color="green", linestyle=":", linewidth=0.8, label="60% aligned threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in LANGUAGES])
    ax.set_ylabel("Sign agreement (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Directional sign agreement: Binary vs Probability format\n"
                 f"(best condition per language, MIN_SUPPORT={MIN_SUPPORT})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_prob_concordance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig_prob_concordance_comparison.png")

print(f"\n{'═'*70}")
print("  DONE. Inspect results_prob_concordance_comparison.csv for the key verdict.")
print(f"{'═'*70}")
