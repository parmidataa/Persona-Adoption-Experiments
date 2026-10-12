"""
Step 22: Should I have used the extra demographic variables?

Answers the reviewer question: "MultiPICo also has employment status, student
status, country of birth, country of residence, and generation. Would R² be
higher with those?"

Strategy:
  1. Load existing M1-M5 results from results_extended_fixed_effects.csv
     (already computed by 18_extended_fixed_effects.py) to show practical gains.
  2. Re-fit M1 and M2 per language using ML (not REML) to run proper
     Likelihood Ratio Tests between nested models where possible.
  3. Produce:
     - fig25_extra_vars_incremental_gain.png  (stacked bar chart)
     - fig26_extra_vars_lrt_significance.png  (dot plot with LRT p-values)
     - results_extra_demographics_lrt.csv
  4. Print a clean, self-contained summary table for the professor.
"""

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("STEP 22: EXTRA DEMOGRAPHIC VARIABLES TEST")
print("=" * 70)

# ── 1. Load existing incremental results ─────────────────────────────────────
fe = pd.read_csv("results_extended_fixed_effects.csv")

LANGS = sorted(fe["language"].unique())
MODELS = ["M1", "M2", "M3", "M4", "M5"]

pivot = fe.pivot(index="language", columns="model", values="r2_marginal") * 100
for m in MODELS:
    if m not in pivot.columns:
        pivot[m] = np.nan
pivot = pivot[MODELS]

# Incremental gains per variable group (pp)
delta = pd.DataFrame(index=pivot.index)
delta["M1 baseline\n(age+gender+eth+nat)"]     = pivot["M1"]
delta["+ country_birth\n+ country_residence"]  = (pivot["M2"] - pivot["M1"]).clip(lower=0)
delta["+ student_status\n+ employment_status"] = (pivot["M3"] - pivot["M2"]).clip(lower=0)
delta["+ generation"]                          = (pivot["M4"] - pivot["M3"]).clip(lower=0)
delta["+ base_rate_loo\n(attitudinal proxy*)"] = (pivot["M5"] - pivot["M4"]).clip(lower=0)

print("\nIncremental R²_m gains (pp) per variable group:\n")
print(delta.round(3).to_string())

# ── 2. Likelihood Ratio Tests (ML, not REML) ─────────────────────────────────
# Compare M1 vs M2 (does adding country_birth + country_residence help?)
# Compare M2 vs M3 (does adding employment + student help?)
# We can only do LRT where both models are LMM-based (not OLS fallback).

method_pivot = fe.pivot(index="language", columns="model", values="method")

def fit_ml(df, formula, groups="text_id"):
    """Fit a mixed LMM with ML (not REML) for LRT."""
    try:
        m = smf.mixedlm(formula, df, groups=df[groups])
        r = m.fit(reml=False, method="lbfgs", maxiter=200)
        return r
    except Exception:
        return None

def lrt(r_reduced, r_full):
    """Likelihood ratio test between two nested ML models."""
    if r_reduced is None or r_full is None:
        return np.nan, np.nan
    delta_ll = 2 * (r_full.llf - r_reduced.llf)
    df_diff   = r_full.df_modelwc - r_reduced.df_modelwc
    if df_diff <= 0:
        return np.nan, np.nan
    p = 1 - stats.chi2.cdf(delta_ll, df_diff)
    return float(delta_ll), float(p)

lrt_rows = []

for lang in LANGS:
    print(f"\n[{lang}] Running LRTs …")
    df = pd.read_parquet(f"data_{lang}_v2.parquet")

    # Which terms were actually used for each model?
    def used_terms(model_label):
        row = fe[(fe["language"] == lang) & (fe["model"] == model_label)]
        if row.empty:
            return None
        t = row["terms_used"].iloc[0]
        return t if isinstance(t, str) else None

    def formula_from_terms(terms_str):
        if terms_str is None:
            return None
        return f"label ~ {'+'.join(terms_str.split('+'))}"

    def run_lrt_pair(m_reduced, m_full):
        t_red  = used_terms(m_reduced)
        t_full = used_terms(m_full)
        if t_red is None or t_full is None:
            return np.nan, np.nan
        # Only do LRT when both models were LMM-based
        meth_red  = method_pivot.at[lang, m_reduced]  if lang in method_pivot.index else "lmm"
        meth_full = method_pivot.at[lang, m_full]     if lang in method_pivot.index else "lmm"
        if meth_red != "lmm" or meth_full != "lmm":
            return np.nan, np.nan
        r_red  = fit_ml(df, formula_from_terms(t_red))
        r_full = fit_ml(df, formula_from_terms(t_full))
        return lrt(r_red, r_full)

    chi2_12, p_12 = run_lrt_pair("M1", "M2")
    chi2_23, p_23 = run_lrt_pair("M2", "M3")
    chi2_34, p_34 = run_lrt_pair("M3", "M4")

    r2_m1 = pivot.at[lang, "M1"]
    r2_m2 = pivot.at[lang, "M2"]
    r2_m3 = pivot.at[lang, "M3"]
    r2_m4 = pivot.at[lang, "M4"]
    r2_m5 = pivot.at[lang, "M5"]

    lrt_rows.append({
        "language": lang,
        "r2_M1": r2_m1, "r2_M2": r2_m2, "r2_M3": r2_m3,
        "r2_M4": r2_m4, "r2_M5": r2_m5,
        "delta_M1_M2": r2_m2 - r2_m1,
        "delta_M2_M3": r2_m3 - r2_m2,
        "delta_M3_M4": r2_m4 - r2_m3,
        "delta_M4_M5": r2_m5 - r2_m4,
        "chi2_M1vsM2": chi2_12, "p_M1vsM2": p_12,
        "chi2_M2vsM3": chi2_23, "p_M2vsM3": p_23,
        "chi2_M3vsM4": chi2_34, "p_M3vsM4": p_34,
    })
    print(f"   M1 vs M2: χ²={chi2_12:.2f}, p={p_12:.4f}" if not np.isnan(chi2_12)
          else "   M1 vs M2: LRT not available (OLS fallback in M1 or M2)")
    print(f"   M2 vs M3: χ²={chi2_23:.2f}, p={p_23:.4f}" if not np.isnan(chi2_23)
          else "   M2 vs M3: LRT not available (OLS fallback in M2 or M3)")
    print(f"   M3 vs M4: χ²={chi2_34:.2f}, p={p_34:.4f}" if not np.isnan(chi2_34)
          else "   M3 vs M4: LRT not available (OLS fallback in M3 or M4)")

lrt_df = pd.DataFrame(lrt_rows).set_index("language")
lrt_df.to_csv("results_extra_demographics_lrt.csv")
print("\nSaved results_extra_demographics_lrt.csv")

# ── 3. Figure 25 — Stacked incremental gain chart ────────────────────────────
COLS   = list(delta.columns)
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
langs_sorted = delta.index.tolist()

fig, ax = plt.subplots(figsize=(12, 6))
bottoms = np.zeros(len(langs_sorted))
for col, color in zip(COLS, COLORS):
    vals = delta.loc[langs_sorted, col].fillna(0).values
    bars = ax.bar(langs_sorted, vals, bottom=bottoms, label=col,
                  color=color, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals):
        if v > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"+{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
    bottoms += vals

# Annotate total R²_m at M4 (demographic-only ceiling)
for i, lang in enumerate(langs_sorted):
    r2_m4 = float(pivot.at[lang, "M4"])
    ax.text(i, r2_m4 + 0.15, f"M4={r2_m4:.2f}%",
            ha="center", va="bottom", fontsize=8, color="#444444")

ax.set_ylabel("R²_m (%)", fontsize=11)
ax.set_xlabel("Language", fontsize=11)
ax.set_title(
    "Incremental gain in R²_m from adding extra demographic variables\n"
    "(*M5 adds a leave-one-out attitudinal proxy — not a demographic; shown as an upper bound)",
    fontsize=11)
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(pivot["M5"].max(), 12) * 1.08)
fig.tight_layout()
fig.savefig("fig25_extra_vars_incremental_gain.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig25_extra_vars_incremental_gain.png")

# ── 4. Figure 26 — LRT significance dot plot ─────────────────────────────────
comparisons = [
    ("p_M1vsM2", "country_birth + country_residence\n(vs. M1 baseline)", "#55A868"),
    ("p_M2vsM3", "student_status + employment_status\n(vs. M2)", "#C44E52"),
    ("p_M3vsM4", "generation\n(vs. M3)", "#8172B2"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
ALPHA_BONF = 0.05 / (len(LANGS) * 3)  # Bonferroni over all tests

for ax, (p_col, title, color) in zip(axes, comparisons):
    ps = lrt_df[p_col]
    deltas_col = {"p_M1vsM2": "delta_M1_M2",
                  "p_M2vsM3": "delta_M2_M3",
                  "p_M3vsM4": "delta_M3_M4"}[p_col]
    ds = lrt_df[deltas_col]
    ys = np.arange(len(LANGS))
    langs_plot = lrt_df.index.tolist()

    for i, (lang, p_val, d_val) in enumerate(
            zip(langs_plot, ps.values, ds.values)):
        if np.isnan(p_val):
            marker, mcolor, zorder = "x", "gray", 1
        elif p_val < ALPHA_BONF:
            marker, mcolor, zorder = "D", color, 3
        elif p_val < 0.05:
            marker, mcolor, zorder = "o", color, 2
        else:
            marker, mcolor, zorder = "o", "#cccccc", 1
        ax.scatter(d_val * 100, i, marker=marker, color=mcolor,
                   s=80, zorder=zorder)
        p_str = f"p={p_val:.3f}" if not np.isnan(p_val) else "n/a (OLS)"
        ax.text(d_val * 100 + 0.02, i, f"  {p_str}", va="center", fontsize=8)

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(langs_plot)
    ax.set_xlabel("ΔR²_m (pp)", fontsize=10)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="x", alpha=0.3)

axes[0].set_ylabel("Language", fontsize=10)

legend_elements = [
    plt.scatter([], [], marker="D", color="#55A868", label=f"Bonferroni-sig. (p<{ALPHA_BONF:.4f})"),
    plt.scatter([], [], marker="o", color="#55A868", label="Nominal sig. (p<0.05)"),
    plt.scatter([], [], marker="o", color="#cccccc", label="Not significant"),
    plt.scatter([], [], marker="x", color="gray", label="LRT not available (OLS fallback)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Likelihood Ratio Tests: do extra variable groups significantly improve fit?",
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig("fig26_extra_vars_lrt_significance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig26_extra_vars_lrt_significance.png")

# ── 5. Clean professor-ready summary table ────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY TABLE — Extra demographic variables: do they add anything?")
print("=" * 70)
print()
print("Columns: R²_m at M1 (baseline) | gain from country fields | gain from")
print("         employment+student | gain from generation | M4 total (demog. only)")
print("         M5 total (+ attitudinal upper bound)")
print()

hdr = f"{'Lang':>4} | {'M1':>5} | {'Δcountry':>8} | {'Δemploy':>7} | {'Δgen':>5} | {'M4':>5} | {'M5*':>6}"
print(hdr)
print("-" * len(hdr))

def sig_flag(p_val):
    if np.isnan(p_val):
        return "  n/a"
    if p_val < ALPHA_BONF:
        return "  ***"
    if p_val < 0.01:
        return "   **"
    if p_val < 0.05:
        return "    *"
    return "     "

for lang in LANGS:
    r = lrt_df.loc[lang]
    d12 = r["delta_M1_M2"]   # already in pp (pivot was multiplied by 100)
    d23 = r["delta_M2_M3"]
    d34 = r["delta_M3_M4"]
    sf12 = sig_flag(r["p_M1vsM2"])
    sf23 = sig_flag(r["p_M2vsM3"])
    sf34 = sig_flag(r["p_M3vsM4"])
    print(f"{lang:>4} | {r['r2_M1']:>5.2f} | {d12:>+6.2f}pp{sf12} | {d23:>+5.2f}pp{sf23} | {d34:>+3.2f}pp{sf34} | {r['r2_M4']:>5.2f} | {r['r2_M5']:>5.2f}")

print()
print("* = p<0.05  ** = p<0.01  *** = Bonferroni-significant  n/a = OLS fallback (LRT not computable)")
print("* M5 adds an attitudinal proxy (LOO base-rate), NOT a demographic — reported as upper bound only")

print("\n" + "=" * 70)
print("STEP 22 COMPLETE")
print("=" * 70)
