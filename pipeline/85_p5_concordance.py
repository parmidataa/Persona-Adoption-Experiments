"""
85_p5_concordance.py
====================
Directional concordance analysis for P5 (Relative Framing + Chain-of-Thought).

Scientific logic
----------------
P5 asked the model to output a signed DEVIATION from the population baseline.
This means llm_deviation is already measuring what the LMM fixed effects measure:
  "how much does this annotator's demographic group shift irony ratings?"

Two LMM outcomes are compared:

  (A) LMM on llm_deviation  [primary for P5 — most direct alignment]
      formula: llm_deviation ~ demographics + (1|post_id)
      Coefficients directly answer: "how does the model weight demographics
      when asked to explicitly reason about demographic shifts?"

  (B) LMM on llm_prob = baseline + llm_deviation  [for cross-experiment comparison]
      formula: llm_prob ~ demographics + (1|post_id)
      Identical scale to P2-binary, P2-prob, P4-RAG → allows four-way comparison.

The human reference is always:
      formula: human_label ~ demographics + (1|post_id)

Sign agreement:
  #(sign(human_coef) == sign(llm_coef)) / total_common_coefs

Group-deviation cross-check (model-free):
  Per-group mean of llm_prob vs per-group mean of human_label (same as P2/P4).

Both metrics are compared across all four conditions in the final table:
  | Language | P2-binary | P2-prob | P4-RAG | P5-CoT |

Outputs:
    results_p5_concordance.csv            — P5 metrics per language
    results_p5_concordance_comparison.csv — four-way comparison table
    fig_p5_concordance_comparison.png     — four-way bar chart

Usage:
    python 85_p5_concordance.py   (run after 84_p5_r2m.py)
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
CONDITION   = "p5"

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


# ── Helpers (identical to 76_rag_concordance.py) ──────────────────────────────
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


def get_formula_p5(target: str, lang: str, df_check: pd.DataFrame) -> str | None:
    return build_formula(target, M1_BASE[lang] + M4_EXTRAS, df_check)


def fit_and_coefs(df_sub: pd.DataFrame, formula: str) -> dict:
    """Fit LMM; return {term: coef} for terms meeting MIN_SUPPORT."""
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
                coefs[name] = val   # continuous (age_centered) always included
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
        "n_coefs":        len(common),
        "sign_agreement": sign_agr,
        "spearman_rho":   float(rho),
        "pearson_r":      float(r),
        "cosine":         cos,
        "terms":          common,
    }


def group_dev_concordance(df_sub: pd.DataFrame, formula: str,
                          llm_col: str = "llm_prob",
                          min_support: int = MIN_SUPPORT) -> dict:
    """
    Model-free group-mean-deviation cross-check.
    Compares per-group mean deviations in human_label vs llm_col (llm_prob).
    Using llm_prob (= baseline + llm_deviation) keeps the scale identical to
    all previous experiments so the comparison is fair.
    """
    cat_cols  = re.findall(r"C\((\w+)\)", formula)
    overall_h = df_sub["human_label"].mean()
    overall_l = df_sub[llm_col].mean()
    h_devs, l_devs = [], []
    for col in cat_cols:
        if col not in df_sub.columns:
            continue
        for level in df_sub[col].dropna().unique():
            mask = df_sub[col] == level
            if mask.sum() < min_support:
                continue
            h_devs.append(df_sub.loc[mask, "human_label"].mean() - overall_h)
            l_devs.append(df_sub.loc[mask, llm_col].mean() - overall_l)
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


# ── Load P5 predictions ───────────────────────────────────────────────────────
p5_path = Path("p5_predictions_clean.parquet")
if not p5_path.exists():
    print("ERROR: p5_predictions_clean.parquet not found. Run 83_parse_p5_results.py first.")
    exit(1)

df = pd.read_parquet(p5_path)
df["age"] = pd.to_numeric(df["age"], errors="coerce")
lang_age_mean = df.groupby("language")["age"].transform("mean")
df["age_centered"] = df["age"] - lang_age_mean
for col in ["gender", "ethnicity", "nationality", "country_birth",
            "country_residence", "employment_status", "student_status"]:
    if col in df.columns:
        df[col] = df[col].replace("unknown", np.nan)

print(f"Loaded {len(df):,} P5 prediction rows")
print(f"  llm_deviation: mean={df['llm_deviation'].mean():+.4f}  std={df['llm_deviation'].std():.4f}")
print(f"  llm_prob     : mean={df['llm_prob'].mean():.4f}      std={df['llm_prob'].std():.4f}")


# ── Concordance for P5 ────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print(f"  DIRECTIONAL CONCORDANCE — P5 / Relative+CoT (MIN_SUPPORT={MIN_SUPPORT})")
print(f"{'═'*70}")

rows = []
for lang in LANGUAGES:
    mask   = df["language"] == lang
    df_sub = df[mask].copy()

    formula_h    = get_formula_p5("human_label",  lang, df_sub)
    formula_dev  = get_formula_p5("llm_deviation", lang, df_sub)   # Analysis A
    formula_prob = get_formula_p5("llm_prob",      lang, df_sub)   # Analysis B

    if formula_h is None or formula_dev is None:
        print(f"\n{lang.upper()} P5: formula could not be built — skipped")
        continue

    h_coefs    = fit_and_coefs(df_sub, formula_h)
    dev_coefs  = fit_and_coefs(df_sub, formula_dev)
    prob_coefs = fit_and_coefs(df_sub, formula_prob)

    metrics_dev  = concordance_metrics(h_coefs, dev_coefs)
    metrics_prob = concordance_metrics(h_coefs, prob_coefs)
    gd = group_dev_concordance(df_sub, formula_prob, llm_col="llm_prob")

    if metrics_dev is None:
        print(f"\n{lang.upper()} P5: fewer than 2 common coefficients — skipped")
        continue

    row = {
        "language":         lang,
        "condition":        CONDITION,
        # Primary (deviation outcome)
        "n_coefs_dev":      metrics_dev["n_coefs"],
        "sign_agr_dev":     metrics_dev["sign_agreement"],
        "spearman_dev":     metrics_dev["spearman_rho"],
        "pearson_dev":      metrics_dev["pearson_r"],
        "cosine_dev":       metrics_dev["cosine"],
        # Secondary (prob outcome)
        "n_coefs_prob":     metrics_prob["n_coefs"] if metrics_prob else np.nan,
        "sign_agr_prob":    metrics_prob["sign_agreement"] if metrics_prob else np.nan,
        "spearman_prob":    metrics_prob["spearman_rho"] if metrics_prob else np.nan,
        "pearson_prob":     metrics_prob["pearson_r"] if metrics_prob else np.nan,
        "cosine_prob":      metrics_prob["cosine"] if metrics_prob else np.nan,
        # Group-dev (model-free, using llm_prob scale)
        **gd,
    }
    rows.append(row)

    print(f"\n{lang.upper()} P5:")
    print(f"  [A: deviation outcome]")
    print(f"    n_coefs        = {metrics_dev['n_coefs']}")
    print(f"    sign_agreement = {metrics_dev['sign_agreement']*100:.1f}%")
    print(f"    Spearman ρ     = {metrics_dev['spearman_rho']:.3f}")
    print(f"    Pearson r      = {metrics_dev['pearson_r']:.3f}")
    print(f"    cosine         = {metrics_dev['cosine']:.3f}")
    if metrics_prob:
        print(f"  [B: prob outcome (llm_prob = baseline + deviation)]")
        print(f"    n_coefs        = {metrics_prob['n_coefs']}")
        print(f"    sign_agreement = {metrics_prob['sign_agreement']*100:.1f}%")
        print(f"    Spearman ρ     = {metrics_prob['spearman_rho']:.3f}")
        print(f"    Pearson r      = {metrics_prob['pearson_r']:.3f}")
    print(f"  [Group-dev check (model-free, llm_prob scale)]")
    print(f"    sign           = {gd['groupdev_sign']*100:.1f}%  (n={gd['groupdev_n']})")

    # Print top 10 coefficient comparison (human vs P5-dev, signed)
    common_d = sorted(set(h_coefs) & set(dev_coefs))
    if common_d:
        print(f"\n  Top coefficient comparison ({lang.upper()}, A — deviation outcome):")
        print(f"  {'Term':<50} {'Human':>8} {'P5-dev':>8} {'Match':>6}")
        print(f"  {'-'*76}")
        for term in common_d[:10]:
            hv_ = h_coefs[term]
            lv_ = dev_coefs[term]
            match = "✓" if np.sign(hv_) == np.sign(lv_) else "✗"
            print(f"  {term:<50} {hv_:>+8.4f} {lv_:>+8.4f} {match:>6}")

df_conc = pd.DataFrame(rows)
df_conc.to_csv("results_p5_concordance.csv", index=False)
print(f"\nSaved: results_p5_concordance.csv")


# ── Four-way comparison ───────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("  FOUR-WAY COMPARISON: Binary / Prob / RAG / P5-CoT")
print(f"{'═'*70}")

# Hardcoded binary baselines (from Table 4.2 of thesis_main.tex)
BINARY_SIGN = {"ar": 0.677, "es": 0.348}

# Load probability concordance
try:
    df_prob_c = pd.read_csv("results_prob_concordance.csv")
    PROB_SIGN = dict(zip(df_prob_c["language"], df_prob_c["sign_agreement"]))
    PROB_GD   = dict(zip(df_prob_c["language"], df_prob_c.get("groupdev_sign",
                    pd.Series(np.nan, index=df_prob_c.index))))
except FileNotFoundError:
    PROB_SIGN = {}; PROB_GD = {}
    print("WARNING: results_prob_concordance.csv not found.")

# Load RAG concordance
try:
    df_rag_c  = pd.read_csv("results_rag_concordance.csv")
    RAG_SIGN  = dict(zip(df_rag_c["language"], df_rag_c["sign_agreement"]))
    RAG_GD    = dict(zip(df_rag_c["language"], df_rag_c.get("groupdev_sign",
                    pd.Series(np.nan, index=df_rag_c.index))))
except FileNotFoundError:
    RAG_SIGN = {}; RAG_GD = {}
    print("WARNING: results_rag_concordance.csv not found.")

# Print comparison table
print(f"\n{'Lang':<5} {'Binary':>9} {'Prob':>9} {'RAG':>9} {'P5-CoT':>9}  "
      f"{'GD Prob':>8} {'GD RAG':>8} {'GD P5':>7}")
print("-"*75)
comp_rows = []
for lang in LANGUAGES:
    b_sign = BINARY_SIGN.get(lang, np.nan)
    p_sign = PROB_SIGN.get(lang, np.nan)
    r_sign = RAG_SIGN.get(lang, np.nan)
    p5_row = df_conc[df_conc["language"] == lang]
    p5_sign = float(p5_row["sign_agr_dev"].values[0])  if len(p5_row) > 0 else np.nan
    p5_gd   = float(p5_row["groupdev_sign"].values[0]) if len(p5_row) > 0 else np.nan
    p_gd    = PROB_GD.get(lang, np.nan)
    r_gd    = RAG_GD.get(lang, np.nan)

    # Δ vs probability baseline
    delta_vs_prob = (p5_sign - p_sign) if not np.isnan(p5_sign) and not np.isnan(p_sign) else np.nan
    arrow = "▲" if delta_vs_prob > 0.02 else ("▼" if delta_vs_prob < -0.02 else "≈") \
            if not np.isnan(delta_vs_prob) else "?"

    def fmt(v):
        return f"{v*100:>7.1f}%" if not np.isnan(v) else "  N/A  "

    print(f"{lang.upper():<5} {fmt(b_sign)} {fmt(p_sign)} {fmt(r_sign)} "
          f"{fmt(p5_sign)} {arrow}({delta_vs_prob*100:+.1f}pp)  "
          f"{fmt(p_gd)} {fmt(r_gd)} {fmt(p5_gd)}")

    comp_rows.append({
        "language":          lang,
        "binary_sign":       b_sign,
        "prob_sign":         p_sign,
        "rag_sign":          r_sign,
        "p5_sign":           p5_sign,
        "delta_p5_vs_prob":  delta_vs_prob,
        "prob_groupdev":     p_gd,
        "rag_groupdev":      r_gd,
        "p5_groupdev":       p5_gd,
    })

df_comp = pd.DataFrame(comp_rows)
df_comp.to_csv("results_p5_concordance_comparison.csv", index=False)
print(f"\nSaved: results_p5_concordance_comparison.csv")


# ── Final interpretation ──────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("  SCIENTIFIC INTERPRETATION")
print(f"{'═'*70}")
for _, r in df_comp.iterrows():
    lang      = r["language"].upper()
    p5        = r["p5_sign"]
    prob      = r["prob_sign"]
    binary    = r["binary_sign"]
    delta     = r["delta_p5_vs_prob"]

    if np.isnan(p5):
        print(f"  {lang}: result unavailable"); continue

    print(f"\n  {lang}:")
    print(f"    P2-binary={binary*100:.1f}%  "
          f"P2-prob={prob*100:.1f}%  "
          f"P4-RAG={r['rag_sign']*100:.1f}%  "
          f"P5-CoT={p5*100:.1f}%")

    # Verdict
    if p5 >= 0.70:
        verdict = (f"P5 SUCCEEDS (≥70% sign agreement). "
                   f"Relative framing + CoT recovers demographic directionality. "
                   f"Structural hypothesis partially refuted for {lang}.")
    elif p5 >= 0.60:
        if not np.isnan(delta) and delta > 0.02:
            verdict = (f"P5 IMPROVES over probability (+{delta*100:.1f} pp, "
                       f"={p5*100:.1f}%). Relative framing helps but does not "
                       f"fully recover alignment for {lang}.")
        else:
            verdict = (f"P5 is at probability-format level ({p5*100:.1f}%). "
                       f"Relative framing and CoT add no meaningful improvement for {lang}.")
    elif p5 >= 0.50:
        verdict = (f"P5 is marginally above chance ({p5*100:.1f}%). "
                   f"Demographic directionality remains poor despite relative framing. "
                   f"RLHF alignment is the structural cause for {lang}.")
    else:
        verdict = (f"P5 FAILS ({p5*100:.1f}% — below chance). "
                   f"Relative framing amplifies anti-alignment. "
                   f"The failure is deeply structural for {lang}.")

    print(f"    → {verdict}")

print(f"\n  OVERALL CONCLUSION:")
print(f"  If BOTH languages are below 70% across all conditions (binary, prob, RAG, P5-CoT),")
print(f"  the series of elimination experiments establishes that demographic directional")
print(f"  concordance is structurally absent from GPT-4.1-mini's irony annotations —")
print(f"  not caused by output format, knowledge gaps, or prompt framing.")


# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
x = np.arange(4)   # 4 conditions
labels = ["P2\nbinary", "P2\nprob", "P4\nRAG", "P5\nCoT"]
colors = ["#a8d5e2", "#5ba4cf", "#e85c4c", "#2d7f3c"]

for lang_i, lang in enumerate(LANGUAGES):
    ax = axes[lang_i]
    r  = df_comp[df_comp["language"] == lang].iloc[0]
    vals = [
        r["binary_sign"] * 100 if not np.isnan(r["binary_sign"]) else 0,
        r["prob_sign"]   * 100 if not np.isnan(r["prob_sign"])   else 0,
        r["rag_sign"]    * 100 if not np.isnan(r["rag_sign"])    else 0,
        r["p5_sign"]     * 100 if not np.isnan(r["p5_sign"])     else 0,
    ]
    bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.55)
    ax.axhline(50,  color="grey",  linestyle="--", linewidth=1,   label="50% chance")
    ax.axhline(60,  color="green", linestyle=":",  linewidth=0.9, label="60% aligned threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f"{lang.upper()}", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    # Annotate bar tops
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

axes[0].set_ylabel("Sign agreement (%)", fontsize=11)
axes[0].legend(fontsize=8, loc="upper left")

fig.suptitle(
    "Directional Concordance Across Prompt Conditions\n"
    f"P2-binary → P2-prob → P4-RAG → P5-Relative+CoT  (MIN_SUPPORT={MIN_SUPPORT})",
    fontsize=11
)
plt.tight_layout()
plt.savefig("fig_p5_concordance_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: fig_p5_concordance_comparison.png")

print(f"\n{'═'*70}")
print("  DONE.  Check results_p5_concordance_comparison.csv and the figure.")
print(f"{'═'*70}")
