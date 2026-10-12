"""
76_rag_concordance.py
=====================
Directional concordance analysis for the RAG experiment (P4 condition).

Identical methodology to 65_prob_concordance.py, with llm_prob (continuous)
as the LLM outcome and the P2 formula applied to the P4 condition.

Produces a three-way comparison table — the central scientific result
of the RAG experiment:

  | Condition          | AR sign agr. | ES sign agr. | AR group-dev | ES group-dev |
  |--------------------|--------------|--------------|--------------|--------------|
  | P2 binary  (Ch.4)  |    67.7%     |    34.8%     |      —       |      —       |
  | P2 prob.   (Exp 1) |    64.5%     |    60.9%     |    79.2%     |    70.0%     |
  | P4 RAG     (Exp 2) |      ?       |      ?       |      ?       |      ?       |

For Arabic: does P4 stay above 60%?
For Spanish: does P4 exceed 60.9% (the probability-elicitation ceiling)?
  → YES: knowledge-gap hypothesis supported (RAG provides the missing behavioural signal)
  → NO:  the directional failure is fundamental, not fixable by retrieval alone

Outputs:
    results_rag_concordance.csv           — P4 concordance metrics per language
    results_rag_concordance_comparison.csv — three-way comparison table
    fig_rag_concordance_comparison.png     — bar chart: binary vs prob vs RAG

Usage:
    python 76_rag_concordance.py
    (Run AFTER 74_parse_rag_results.py)
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
from pathlib import Path

MIN_SUPPORT = 15
LANGUAGES   = ["ar", "es"]
CONDITION   = "p4"     # P4 uses the P2 formula

M1_BASE = {
    "ar": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "es": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
}
M4_EXTRAS = [
    "C(country_birth)", "C(country_residence)",
    "C(employment_status)", "C(student_status)",
]
ALL_PRED_COLS = [
    "age_centered", "gender", "ethnicity", "nationality",
    "country_birth", "country_residence",
    "employment_status", "student_status", "Generation",
]


def col_from_term(t: str) -> str:
    m = re.match(r"C\((\w+)\)", t)
    return m.group(1) if m else t


def build_formula(target: str, var_list: list, df_check: pd.DataFrame) -> str | None:
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


def get_formula_p4(target: str, lang: str, df_check: pd.DataFrame) -> str | None:
    """P4 always uses the full P2 (M1_BASE + M4_EXTRAS) formula."""
    return build_formula(target, M1_BASE[lang] + M4_EXTRAS, df_check)


def fit_and_coefs(df_sub: pd.DataFrame, formula: str) -> dict:
    """Fit LMM and return {term: coef} for terms passing MIN_SUPPORT."""
    target = formula.split("~")[0].strip()
    nc = [target, "post_id"] + [
        col_from_term(t)
        for t in re.findall(r"C\(\w+\)|\w+", formula.split("~")[1])
    ]
    nc     = [c for c in dict.fromkeys(nc) if c in df_sub.columns]
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
            col_match = re.search(r"C\((\w+)\)", name)
            if col_match:
                col = col_match.group(1)
                level_match = re.search(r"\[T\.(.+)\]", name)
                if level_match and col in df_fit.columns:
                    level   = level_match.group(1)
                    support = (df_fit[col] == level).sum()
                    if support >= MIN_SUPPORT:
                        coefs[name] = val
            else:
                coefs[name] = val   # continuous predictor always included
        return coefs
    except Exception:
        return {}


def concordance_metrics(h_coefs: dict, l_coefs: dict) -> dict | None:
    """Sign agreement, Spearman ρ, Pearson r, cosine similarity."""
    common = sorted(set(h_coefs) & set(l_coefs))
    if len(common) < 2:
        return None
    hv = np.array([h_coefs[k] for k in common])
    lv = np.array([l_coefs[k] for k in common])
    sign_agr = float(np.mean(np.sign(hv) == np.sign(lv)))
    rho, _   = spearmanr(hv, lv)
    try:
        r, _ = pearsonr(hv, lv)
    except Exception:
        r = np.nan
    denom = np.linalg.norm(hv) * np.linalg.norm(lv)
    cos   = float(np.dot(hv, lv) / denom) if denom > 0 else np.nan
    return {
        "n_coefs":       len(common),
        "sign_agreement": sign_agr,
        "spearman_rho":  float(rho),
        "pearson_r":     float(r),
        "cosine":        cos,
        "terms":         common,
    }


def group_dev_concordance(df_sub: pd.DataFrame, formula: str,
                          min_support: int = MIN_SUPPORT) -> dict:
    """
    Model-free group-mean-deviation cross-check (matches 35 / 65 scripts).
    Compares per-group mean deviations in human_label vs llm_prob.
    """
    cat_cols  = re.findall(r"C\((\w+)\)", formula)
    overall_h = df_sub["human_label"].mean()
    overall_l = df_sub["llm_prob"].mean()
    h_devs, l_devs = [], []
    for col in cat_cols:
        if col not in df_sub.columns:
            continue
        for level in df_sub[col].dropna().unique():
            mask = df_sub[col] == level
            if mask.sum() < min_support:
                continue
            h_devs.append(df_sub.loc[mask, "human_label"].mean() - overall_h)
            l_devs.append(df_sub.loc[mask, "llm_prob"].mean() - overall_l)
    if len(h_devs) < 2:
        return {"groupdev_sign": np.nan, "groupdev_rho": np.nan, "groupdev_n": 0}
    hv  = np.array(h_devs)
    lv  = np.array(l_devs)
    rho, _ = spearmanr(hv, lv)
    return {
        "groupdev_sign": float(np.mean(np.sign(hv) == np.sign(lv))),
        "groupdev_rho":  float(rho),
        "groupdev_n":    len(h_devs),
    }


# ── Load RAG predictions ───────────────────────────────────────────────────────
rag_path = Path("rag_predictions_clean.parquet")
if not rag_path.exists():
    print("ERROR: rag_predictions_clean.parquet not found. Run 74_parse_rag_results.py first.")
    exit(1)

df_rag = pd.read_parquet(rag_path)
df_rag["age"] = pd.to_numeric(df_rag["age"], errors="coerce")
lang_age_mean = df_rag.groupby("language")["age"].transform("mean")
df_rag["age_centered"] = df_rag["age"] - lang_age_mean
for col in ["gender", "ethnicity", "nationality", "country_birth",
            "country_residence", "employment_status", "student_status"]:
    if col in df_rag.columns:
        df_rag[col] = df_rag[col].replace("unknown", np.nan)

print(f"Loaded {len(df_rag):,} RAG prediction rows")

# ── Concordance for P4 ────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print(f"  DIRECTIONAL CONCORDANCE — P4 / RAG (MIN_SUPPORT={MIN_SUPPORT})")
print(f"{'═'*70}")

rows = []
for lang in LANGUAGES:
    mask   = (df_rag["language"] == lang) & (df_rag["condition"] == CONDITION)
    df_sub = df_rag[mask].copy()

    formula_h = get_formula_p4("human_label", lang, df_sub)
    formula_l = get_formula_p4("llm_prob",    lang, df_sub)

    if formula_h is None or formula_l is None:
        print(f"\n{lang.upper()} P4: formula could not be built — skipped")
        continue

    h_coefs = fit_and_coefs(df_sub, formula_h)
    l_coefs = fit_and_coefs(df_sub, formula_l)

    metrics = concordance_metrics(h_coefs, l_coefs)
    if metrics is None:
        print(f"\n{lang.upper()} P4: fewer than 2 common coefficients — skipped")
        continue

    gd = group_dev_concordance(df_sub, formula_l)

    row = {
        "language":        lang,
        "condition":       CONDITION,
        "n_coefs":         metrics["n_coefs"],
        "sign_agreement":  metrics["sign_agreement"],
        "spearman_rho":    metrics["spearman_rho"],
        "pearson_r":       metrics["pearson_r"],
        "cosine":          metrics["cosine"],
        **gd,
    }
    rows.append(row)

    print(f"\n{lang.upper()} P4 (RAG + probability):")
    print(f"  n_coefs        = {metrics['n_coefs']}")
    print(f"  sign_agreement = {metrics['sign_agreement']*100:.1f}%")
    print(f"  Spearman ρ     = {metrics['spearman_rho']:.3f}")
    print(f"  Pearson r      = {metrics['pearson_r']:.3f}")
    print(f"  cosine         = {metrics['cosine']:.3f}")
    print(f"  group-dev sign = {gd['groupdev_sign']*100:.1f}%  (n={gd['groupdev_n']})")

df_conc = pd.DataFrame(rows)
df_conc.to_csv("results_rag_concordance.csv", index=False)
print(f"\nSaved: results_rag_concordance.csv")

# ── Three-way comparison ──────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("  THREE-WAY COMPARISON: Binary vs Probability vs RAG")
print(f"{'═'*70}")

# Load binary concordance
BINARY_SIGN = {"ar": 0.677, "es": 0.348}   # from Table 4.2 in thesis_main.tex
BINARY_GD   = {"ar": np.nan, "es": np.nan} # not available from saved CSV

# Load probability concordance
try:
    df_prob_conc = pd.read_csv("results_prob_concordance.csv")
    PROB_SIGN = dict(zip(df_prob_conc["language"], df_prob_conc["sign_agreement"]))
    PROB_GD   = dict(zip(df_prob_conc["language"], df_prob_conc["groupdev_sign"]))
except FileNotFoundError:
    PROB_SIGN = {}
    PROB_GD   = {}
    print("WARNING: results_prob_concordance.csv not found — probability column will be empty.")

print(f"\n{'Lang':<5} {'Binary sign':>12} {'Prob sign':>10} {'RAG sign':>10}"
      f"  {'Prob GD':>9} {'RAG GD':>9}")
print("-"*65)

comp_rows = []
for lang in LANGUAGES:
    b_sign = BINARY_SIGN.get(lang, np.nan)
    p_sign = PROB_SIGN.get(lang, np.nan)
    r_row  = df_conc[df_conc["language"] == lang]
    r_sign = float(r_row["sign_agreement"].values[0]) if len(r_row) > 0 else np.nan
    p_gd   = PROB_GD.get(lang, np.nan)
    r_gd   = float(r_row["groupdev_sign"].values[0]) if len(r_row) > 0 else np.nan

    # Δ arrows vs probability baseline
    if not (np.isnan(r_sign) or np.isnan(p_sign)):
        delta = r_sign - p_sign
        arrow = "▲" if delta > 0.02 else ("▼" if delta < -0.02 else "≈")
    else:
        delta, arrow = np.nan, "?"

    print(f"{lang.upper():<5} "
          f"{b_sign*100:>10.1f}%  "
          f"{p_sign*100 if not np.isnan(p_sign) else float('nan'):>8.1f}%  "
          f"{r_sign*100 if not np.isnan(r_sign) else float('nan'):>8.1f}%  "
          f"{arrow} {delta*100:+.1f} pp"
          f"  {p_gd*100 if not np.isnan(p_gd) else float('nan'):>7.1f}%"
          f"  {r_gd*100 if not np.isnan(r_gd) else float('nan'):>7.1f}%")

    comp_rows.append({
        "language":           lang,
        "binary_sign":        b_sign,
        "prob_sign":          p_sign,
        "rag_sign":           r_sign,
        "delta_rag_vs_prob":  delta,
        "prob_groupdev_sign": p_gd,
        "rag_groupdev_sign":  r_gd,
    })

df_comp = pd.DataFrame(comp_rows)
df_comp.to_csv("results_rag_concordance_comparison.csv", index=False)
print(f"\nSaved: results_rag_concordance_comparison.csv")

# ── Interpretation ────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("  INTERPRETATION")
print(f"{'═'*70}")
for _, r in df_comp.iterrows():
    lang  = r["language"].upper()
    delta = r["delta_rag_vs_prob"]
    rag   = r["rag_sign"]
    prob  = r["prob_sign"]
    if np.isnan(delta) or np.isnan(rag):
        print(f"  {lang}: result unavailable")
        continue
    if delta > 0.05:
        verdict = (f"RAG IMPROVES direction (+{delta*100:.1f} pp over probability format). "
                   f"Knowledge-gap hypothesis SUPPORTED for {lang}.")
    elif delta < -0.05:
        verdict = (f"RAG DOES NOT IMPROVE direction ({delta*100:.1f} pp vs probability). "
                   f"Directional failure is NOT explained by a knowledge gap for {lang}.")
    else:
        verdict = (f"RAG has NO MEANINGFUL EFFECT ({delta*100:+.1f} pp). "
                   f"Result is inconclusive for {lang}.")
    if rag < 0.50:
        verdict += " Note: RAG sign agreement is BELOW chance — active anti-alignment."
    print(f"  {lang}: {verdict}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(LANGUAGES))
w = 0.22
bar_sets = [
    ("P2 binary (Ch.4)",    [BINARY_SIGN.get(l, np.nan) for l in LANGUAGES], "#4c9be8", 0.7),
    ("P2 probability (Exp 1)", [PROB_SIGN.get(l, np.nan)  for l in LANGUAGES], "#1a5fa8", 0.7),
    ("P4 RAG (Exp 2)",      [float(df_conc[df_conc["language"]==l]["sign_agreement"].values[0])
                              if len(df_conc[df_conc["language"]==l]) > 0 else np.nan
                              for l in LANGUAGES], "#e85c4c", 0.85),
]
for i, (label, vals, color, alpha) in enumerate(bar_sets):
    ax.bar(x + (i - 1) * w, [v * 100 if v is not None and not np.isnan(v) else 0
                               for v in vals],
           w, label=label, color=color, alpha=alpha)

ax.axhline(50,  color="grey",  linestyle="--", linewidth=1,   label="50% chance")
ax.axhline(60,  color="green", linestyle=":",  linewidth=0.8, label="60% indicative threshold")
ax.set_xticks(x)
ax.set_xticklabels([l.upper() for l in LANGUAGES], fontsize=11)
ax.set_ylabel("Sign agreement (%)", fontsize=11)
ax.set_ylim(0, 105)
ax.set_title(
    "Directional sign agreement: Binary vs Probability vs RAG (P4)\n"
    f"Best condition (P2 formula), MIN_SUPPORT={MIN_SUPPORT}",
    fontsize=10
)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig_rag_concordance_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_rag_concordance_comparison.png")

print(f"\n{'═'*70}")
print("  DONE.  Inspect results_rag_concordance_comparison.csv for the key verdict.")
print(f"{'═'*70}")
