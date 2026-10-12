"""
Step 24 (Direction 3): Interaction Robustness — do the Phase A interaction
effects survive richer controls (M4 base)?

The Phase A analysis (Step 5) found significant interactions on top of the
M1 base (age, gender, ethnicity, nationality).  A key confound is omitted
variables: the interaction `gender × nationality` may proxy for `gender ×
country_birth` or similar, which M4 controls for directly.

This script:
  1. Loads Phase A interaction results and selects the Bonferroni-significant
     interactions per language (top candidate per language by ΔR²_m).
  2. Re-tests each interaction with the M4 variable set as the base formula
     (same term pruning / rank-check approach as Step 18).
  3. Compares ΔR²_m (interaction on M1 base) vs. ΔR²_m (interaction on M4
     base) for each language, and whether the interaction is still
     statistically significant after richer controls.
  4. Produces:
       fig29_interaction_robustness_comparison.png
         Side-by-side horizontal bar chart: best-interaction ΔR²_m on M1 base
         (blue) vs. M4 base (green) per language, with significance markers.
       fig30_interaction_delta_heatmap.png
         Heat-map of ΔR²_m attenuation: (ΔR²_m_M4base − ΔR²_m_M1base) for
         all tested interactions × languages, showing which effects shrink and
         which persist.
       results_interaction_robustness.csv

Interpretation guide:
  • If ΔR²_m shrinks substantially on M4 base → the Phase A interaction was
    partly proxying for omitted country/employment variables (confounding).
  • If ΔR²_m persists on M4 base → the interaction is genuinely intersectional
    and cannot be reduced to additive demographic controls.

Note on OLS-fallback languages:
  For languages where the M4 LMM hits a boundary (en, es, hi, de), we attempt
  a fresh LMM with only the M4 terms + interaction term.  If that also
  degenerates, we compute R²_m via OLS with the M1 total-variance anchor
  (same strategy as Step 18) and mark the result as OLS-fallback.
"""

import os
import warnings
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("STEP 24 (Direction 3): INTERACTION ROBUSTNESS ON M4 BASE")
print("=" * 70)

LANG_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "hi": "Hindi", "it": "Italian", "nl": "Dutch",
    "pt": "Portuguese",
}

# ── 1. Load Phase A results & select Bonferroni-significant interactions ───────
ia = pd.read_csv("results_interaction_effects.csv")
fe = pd.read_csv("results_extended_fixed_effects.csv")

# Bonferroni threshold from Phase A (same as Step 5 / Figure 5)
n_tests_phase_a = len(ia)
bonf_thresh_a = 0.05 / n_tests_phase_a
print(f"\nPhase A: {n_tests_phase_a} interactions tested, "
      f"Bonferroni threshold = 0.05/{n_tests_phase_a} = {bonf_thresh_a:.6f}")

ia["bonf_sig_phaseA"] = ia["p_value"] < bonf_thresh_a

# Best (Bonferroni-sig) interaction per language, ranked by ΔR²_m
sig_ia = ia[ia["bonf_sig_phaseA"]].copy()
best_per_lang = (
    sig_ia.sort_values("r2_change", ascending=False)
    .groupby("language")
    .first()
    .reset_index()
)

# All significant interactions (for the heatmap)
all_sig = ia[ia["bonf_sig_phaseA"]].copy()

print("\nSelected interactions (best Bonferroni-sig per language):")
print(best_per_lang[["language", "interaction", "r2_change_pct", "p_value"]]
      .to_string(index=False))

# ── 2. Helper functions (same design as Step 18) ──────────────────────────────
def r2_from_lmm(res):
    fixed_pred = np.asarray(res.model.exog) @ res.fe_params
    var_fixed = float(np.var(fixed_pred))
    cov_re = res.cov_re
    var_random = float(cov_re.iloc[0, 0] if hasattr(cov_re, "iloc")
                       else np.asarray(cov_re).flatten()[0])
    var_residual = float(res.scale)
    total = var_fixed + var_random + var_residual
    return {"var_fixed": var_fixed, "var_random": var_random,
            "var_residual": var_residual,
            "r2_marginal": var_fixed / total if total > 0 else 0.0,
            "total": total}

def is_boundary(r2d):
    tot = r2d["total"]
    return (tot <= 0
            or r2d["var_residual"] / tot < 0.01
            or r2d["var_random"] / tot < 1e-6)

def fit_lmm_terms(df, base_terms, extra_term=None, ml=False):
    """Fit LMM with base_terms as fixed effects + optional extra_term."""
    rhs_parts = []
    for t in base_terms:
        if t in ("age_centered", "base_rate_loo"):
            rhs_parts.append(t)
        else:
            rhs_parts.append(f"C({t})")
    formula = "label ~ " + (" + ".join(rhs_parts) if rhs_parts else "1")
    if extra_term:
        formula += f" + {extra_term}"
    try:
        m = smf.mixedlm(formula, df, groups=df["text_id"])
        res = m.fit(reml=not ml, method=["lbfgs", "powell", "nm"], maxiter=300)
        return res, formula
    except Exception:
        return None, formula

def ols_r2m(df, base_terms, extra_term, anchor_total_var):
    """R²_m via OLS using M1 total-variance anchor."""
    rhs_parts = []
    for t in base_terms:
        if t in ("age_centered",):
            rhs_parts.append(t)
        else:
            rhs_parts.append(f"C({t})")
    formula = "label ~ " + (" + ".join(rhs_parts) if rhs_parts else "1")
    if extra_term:
        formula += f" + {extra_term}"
    res = smf.ols(formula, df).fit()
    fixed_pred = np.asarray(res.model.exog) @ res.params
    var_fixed = float(np.var(fixed_pred))
    return var_fixed / anchor_total_var if anchor_total_var > 0 else np.nan

def lrt(res_base, res_full):
    if res_base is None or res_full is None:
        return np.nan, np.nan
    delta_ll = 2 * (res_full.llf - res_base.llf)
    df_diff = res_full.df_modelwc - res_base.df_modelwc
    if df_diff <= 0:
        return np.nan, np.nan
    p = 1 - stats.chi2.cdf(delta_ll, df_diff)
    return float(delta_ll), float(p)

def get_m4_terms(lang):
    """Return the terms_used string for M4 from Step 18 results."""
    row = fe[(fe["language"] == lang) & (fe["model"] == "M4")]
    if row.empty:
        return None, None
    return row["terms_used"].iloc[0].split("+"), row["method"].iloc[0]

def get_m1_total_var(lang):
    """Return M1 total variance (for OLS anchor) from Step 18 results."""
    row = fe[(fe["language"] == lang) & (fe["model"] == "M1")]
    if row.empty:
        return np.nan
    r = row.iloc[0]
    return float(r["var_fixed"]) + float(r["var_random"]) + float(r["var_residual"])

# ── 3. Interaction term string → statsmodels formula fragment ─────────────────
# Phase A terms used C(gender), age_centered, C(ethnicity), C(nationality)
IA_TERM_MAP = {
    "gender × ethnicity":     "C(gender):C(ethnicity)",
    "age × gender":           "age_centered:C(gender)",
    "age × ethnicity":        "age_centered:C(ethnicity)",
    "gender × nationality":   "C(gender):C(nationality)",
    "age × nationality":      "age_centered:C(nationality)",
    "ethnicity × nationality":"C(ethnicity):C(nationality)",
}

# M4 may have dropped nationality and added country_residence / country_birth.
# When nationality is absent from M4 but was the interaction variable, we
# substitute the best available country proxy or skip gracefully.
COUNTRY_PROXY = "country_residence"   # fallback when nationality is absent in M4


def resolve_interaction_term(ia_name, m4_terms):
    """
    Map a Phase-A interaction name to a formula fragment compatible with
    the M4 term set.  Returns (formula_fragment, note).
    """
    base = IA_TERM_MAP.get(ia_name)
    if base is None:
        return None, "unknown interaction name"

    needs_nat = "nationality" in ia_name
    needs_gen = "gender" in ia_name
    needs_age = "age" in ia_name
    needs_eth = "ethnicity" in ia_name

    # Check components are available in M4
    has_nat = "nationality" in m4_terms
    has_eth = "ethnicity" in m4_terms
    has_gen = "gender" in m4_terms
    has_res = COUNTRY_PROXY in m4_terms

    if needs_nat and not has_nat:
        if has_res:
            # Replace nationality with country_residence
            term = base.replace("C(nationality)", f"C({COUNTRY_PROXY})")
            return term, f"nationality absent in M4; substituted {COUNTRY_PROXY}"
        else:
            return None, "nationality absent in M4 and no country proxy available"

    if needs_eth and not has_eth:
        return None, "ethnicity absent in M4"

    return base, "original term"


# ── 4. Run robustness tests ───────────────────────────────────────────────────
# We test:
#   (a) The best Bonferroni-significant interaction per language on M4 base.
#   (b) ALL Bonferroni-significant interactions (for the heatmap).

def run_interaction_on_m4(lang, ia_name, df_lang, m4_terms, m4_method):
    """
    Returns a dict with ΔR²_m on M4 base, p-value, method used.
    """
    m1_total = get_m1_total_var(lang)
    ia_term, note = resolve_interaction_term(ia_name, m4_terms)

    if ia_term is None:
        return {
            "dr2_m4_base": np.nan, "chi2": np.nan, "p_value": np.nan,
            "method": "skipped", "note": note,
        }

    # Try full LMM on M4 base + interaction
    res_base, _ = fit_lmm_terms(df_lang, m4_terms, extra_term=None, ml=True)
    res_full, _ = fit_lmm_terms(df_lang, m4_terms, extra_term=ia_term, ml=True)

    base_ok = (res_base is not None) and (not is_boundary(r2_from_lmm(res_base)))
    full_ok = (res_full is not None) and (not is_boundary(r2_from_lmm(res_full)))

    if base_ok and full_ok:
        chi2_val, p_val = lrt(res_base, res_full)
        r2_base = r2_from_lmm(res_base)["r2_marginal"]
        r2_full = r2_from_lmm(res_full)["r2_marginal"]
        dr2 = r2_full - r2_base
        return {
            "dr2_m4_base": dr2 * 100,
            "chi2": chi2_val, "p_value": p_val,
            "method": "lmm", "note": note,
        }
    else:
        # OLS fallback: compute R²_m with M1 anchor
        if not np.isfinite(m1_total) or m1_total <= 0:
            return {
                "dr2_m4_base": np.nan, "chi2": np.nan, "p_value": np.nan,
                "method": "ols_no_anchor", "note": note,
            }
        r2_base_ols = ols_r2m(df_lang, m4_terms, None, m1_total)
        r2_full_ols = ols_r2m(df_lang, m4_terms, ia_term, m1_total)
        dr2_ols = r2_full_ols - r2_base_ols
        return {
            "dr2_m4_base": dr2_ols * 100,
            "chi2": np.nan, "p_value": np.nan,
            "method": "ols_fallback", "note": note,
        }


print("\n" + "─" * 70)
print("Running best-interaction-per-language on M4 base …")
print("─" * 70)

best_rows = []
for _, ia_row in best_per_lang.iterrows():
    lang = ia_row["language"]
    ia_name = ia_row["interaction"]
    df_lang = pd.read_parquet(f"data_{lang}_v2.parquet")
    m4_terms, m4_method = get_m4_terms(lang)
    if m4_terms is None:
        print(f"  [{lang}] M4 terms not found — skipping")
        continue

    print(f"\n  [{lang}] Testing '{ia_name}' on M4 base ...")
    result = run_interaction_on_m4(lang, ia_name, df_lang, m4_terms, m4_method)
    print(f"    M1-base ΔR²_m = {ia_row['r2_change_pct']:+.3f} pp  "
          f"(Phase A, p={ia_row['p_value']:.2e})")
    print(f"    M4-base ΔR²_m = {result['dr2_m4_base']:+.3f} pp  "
          f"p={result['p_value']:.2e}  [{result['method']}]  {result['note']}")

    best_rows.append({
        "language": lang,
        "interaction": ia_name,
        "dr2_m1_base_pp": ia_row["r2_change_pct"],
        "p_m1_base": ia_row["p_value"],
        "bonf_sig_m1_base": ia_row["bonf_sig_phaseA"],
        "dr2_m4_base_pp": result["dr2_m4_base"],
        "chi2_m4_base": result["chi2"],
        "p_m4_base": result["p_value"],
        "method_m4": result["method"],
        "note_m4": result["note"],
        "ia_term_m4": IA_TERM_MAP.get(ia_name, "?"),
    })

best_df = pd.DataFrame(best_rows)

# ── Compute Bonferroni threshold for the M4-base tests ────────────────────────
n_tests_m4 = len(best_rows)
bonf_thresh_m4 = 0.05 / n_tests_m4 if n_tests_m4 > 0 else 0.05
best_df["bonf_sig_m4_base"] = best_df["p_m4_base"] < bonf_thresh_m4

print("\n" + "─" * 70)
print("Running ALL Bonferroni-sig interactions on M4 base (for heatmap) …")
print("─" * 70)

all_rows = []
for _, ia_row in all_sig.iterrows():
    lang = ia_row["language"]
    ia_name = ia_row["interaction"]
    df_lang = pd.read_parquet(f"data_{lang}_v2.parquet")
    m4_terms, m4_method = get_m4_terms(lang)
    if m4_terms is None:
        continue
    result = run_interaction_on_m4(lang, ia_name, df_lang, m4_terms, m4_method)
    all_rows.append({
        "language": lang, "interaction": ia_name,
        "dr2_m1_base_pp": ia_row["r2_change_pct"],
        "dr2_m4_base_pp": result["dr2_m4_base"],
        "attenuation_pp": (result["dr2_m4_base"] - ia_row["r2_change_pct"]
                           if np.isfinite(result["dr2_m4_base"]) else np.nan),
        "p_m1": ia_row["p_value"],
        "p_m4": result["p_value"],
        "method_m4": result["method"],
    })

all_df = pd.DataFrame(all_rows)

# Save combined results
best_df.to_csv("results_interaction_robustness.csv", index=False)
print("\nSaved results_interaction_robustness.csv")

# ── 5. Summary table ──────────────────────────────────────────────────────────
n_m4_tests = best_df["p_m4_base"].notna().sum()
bonf_thresh_m4_print = 0.05 / n_m4_tests if n_m4_tests > 0 else np.nan

print("\n" + "=" * 70)
print(f"INTERACTION ROBUSTNESS SUMMARY  "
      f"(M4-base Bonferroni threshold: 0.05/{n_m4_tests} = {bonf_thresh_m4_print:.4f})")
print("=" * 70)
hdr = (f"{'Lang':>10} | {'Interaction':^28} | "
       f"{'ΔR² M1-base':>12} | {'sig M1':>6} | "
       f"{'ΔR² M4-base':>12} | {'sig M4':>6} | "
       f"{'Change':>8} | method")
print(hdr)
print("-" * len(hdr))
for _, row in best_df.iterrows():
    sig_m1 = "***" if row["bonf_sig_m1_base"] else "   "
    p_m4   = row["p_m4_base"]
    sig_m4 = ("***" if (np.isfinite(p_m4) and p_m4 < bonf_thresh_m4_print)
              else ("*  " if (np.isfinite(p_m4) and p_m4 < 0.05)
                    else ("n/a" if np.isnan(p_m4) else "   ")))
    dr1    = row["dr2_m1_base_pp"]
    dr4    = row["dr2_m4_base_pp"]
    chg    = (dr4 - dr1) if np.isfinite(dr4) else np.nan
    chg_s  = f"{chg:+.2f}pp" if np.isfinite(chg) else "   n/a"
    print(
        f"{LANG_NAMES.get(row['language'], row['language']):>10} | "
        f"{row['interaction']:^28} | "
        f"{dr1:>+11.3f}% | {sig_m1:>6} | "
        f"{(dr4 if np.isfinite(dr4) else float('nan')):>+11.3f}% | {sig_m4:>6} | "
        f"{chg_s:>8} | {row['method_m4']}"
    )

persists = best_df[best_df["bonf_sig_m4_base"] == True]
attenuated = best_df[
    best_df["dr2_m4_base_pp"].notna() &
    (best_df["dr2_m4_base_pp"] < best_df["dr2_m1_base_pp"] * 0.5)
]
print(f"\nInteractions that persist (Bonferroni-sig on M4 base): "
      f"{len(persists)}/{len(best_df)}")
if len(persists):
    print("  " + ", ".join(f"{LANG_NAMES[r['language']]} {r['interaction']}"
                            for _, r in persists.iterrows()))
print(f"Interactions that attenuate ≥50% on M4 base: "
      f"{len(attenuated)}/{len(best_df)}")
if len(attenuated):
    print("  " + ", ".join(f"{LANG_NAMES[r['language']]} {r['interaction']}"
                            for _, r in attenuated.iterrows()))

# ── 6. Figure 29 — Side-by-side bar chart (best interaction per lang) ─────────
fig, ax = plt.subplots(figsize=(10, 6.5))

y_pos = np.arange(len(best_df))
bar_h = 0.35

C_M1 = "#4C72B0"
C_M4 = "#55A868"

m1_vals = best_df["dr2_m1_base_pp"].values
m4_vals = best_df["dr2_m4_base_pp"].fillna(0).values
m4_available = best_df["dr2_m4_base_pp"].notna().values
lang_labels = [LANG_NAMES.get(l, l) for l in best_df["language"]]
interaction_labels = best_df["interaction"].values

# Sort by M1 ΔR² descending
order = np.argsort(m1_vals)[::-1]
m1_vals_s  = m1_vals[order]
m4_vals_s  = m4_vals[order]
avail_s    = m4_available[order]
lang_s     = [lang_labels[i] for i in order]
inter_s    = [interaction_labels[i] for i in order]
sig_m1_s   = best_df["bonf_sig_m1_base"].values[order]
sig_m4_s   = best_df["bonf_sig_m4_base"].values[order]
method_s   = best_df["method_m4"].values[order]

ax.barh(y_pos + bar_h / 2, m1_vals_s, height=bar_h,
        color=C_M1, label="M1 base (Phase A — Hu & Collier controls)", zorder=2)

for i in range(len(order)):
    if avail_s[i]:
        color = C_M4
        hatch = "//" if method_s[i] == "ols_fallback" else None
        ax.barh(i - bar_h / 2, m4_vals_s[i], height=bar_h,
                color=color, hatch=hatch, label=None, zorder=2)
    else:
        ax.text(0.05, i - bar_h / 2, "skipped", va="center",
                fontsize=8, color="gray", style="italic")

# Dummy patch for legend
m4_patch = mpatches.Patch(color=C_M4, label="M4 base (richer controls)")
m4_ols_patch = mpatches.Patch(facecolor=C_M4, hatch="//",
                               label="M4 base (OLS fallback)")

# Significance markers on M1 bars
for i, (sig1, sig4, v1, v4, av) in enumerate(
        zip(sig_m1_s, sig_m4_s, m1_vals_s, m4_vals_s, avail_s)):
    if sig1:
        ax.text(v1 + 0.03, i + bar_h / 2, "★",
                va="center", fontsize=9, color="#333333")
    if av and np.isfinite(v4) and sig4:
        ax.text(v4 + 0.03, i - bar_h / 2, "★",
                va="center", fontsize=9, color="#1a6e3d")

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{l}\n{ia}" for l, ia in zip(lang_s, inter_s)],
                   fontsize=8.5)
ax.set_xlabel("ΔR²_m from adding the interaction term (pp)", fontsize=11)
ax.set_title(
    "Figure 29 — Interaction robustness: do Phase A effects survive M4 controls?\n"
    "★ = Bonferroni-significant  //  = OLS fallback (LMM boundary hit)",
    fontsize=11)
ax.legend(handles=[
    mpatches.Patch(color=C_M1, label="M1 base (Phase A)"),
    m4_patch, m4_ols_patch
], loc="lower right", fontsize=9)
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(left=-0.1)
fig.tight_layout()
fig.savefig("fig29_interaction_robustness_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved fig29_interaction_robustness_comparison.png")

# ── 7. Figure 30 — Attenuation heatmap ───────────────────────────────────────
# Pivot: rows = language, cols = interaction, values = attenuation (pp)
pivot_att = all_df.pivot(index="language", columns="interaction",
                          values="attenuation_pp")
pivot_m1  = all_df.pivot(index="language", columns="interaction",
                          values="dr2_m1_base_pp")

# Only keep rows/cols with at least one non-NaN attenuation
pivot_att = pivot_att.dropna(how="all", axis=0).dropna(how="all", axis=1)

if pivot_att.empty:
    print("Heatmap skipped — no attenuation data (all tests were OLS fallback).")
else:
    # Reorder rows by LANG_NAMES order
    lang_order = [l for l in LANG_NAMES if l in pivot_att.index]
    pivot_att = pivot_att.reindex(lang_order)
    pivot_att.index = [LANG_NAMES[l] for l in pivot_att.index]

    fig, ax = plt.subplots(figsize=(11, max(4, len(pivot_att) * 0.8 + 1.5)))

    # Diverging colormap centred at 0: negative = shrank, positive = grew
    vmax = max(abs(pivot_att.values[np.isfinite(pivot_att.values)]).max(), 0.5) \
           if np.isfinite(pivot_att.values).any() else 1.0
    cmap = plt.cm.RdYlGn  # red=shrank, green=grew

    im = ax.imshow(pivot_att.values.astype(float),
                   aspect="auto", cmap=cmap,
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot_att.columns)))
    ax.set_xticklabels(pivot_att.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot_att.index)))
    ax.set_yticklabels(pivot_att.index, fontsize=10)

    # Annotate cells
    for r in range(pivot_att.shape[0]):
        for c in range(pivot_att.shape[1]):
            val = pivot_att.iloc[r, c]
            if np.isfinite(val):
                ax.text(c, r, f"{val:+.2f}", ha="center", va="center",
                        fontsize=8,
                        color="black" if abs(val) < vmax * 0.6 else "white")
            else:
                ax.text(c, r, "—", ha="center", va="center",
                        fontsize=9, color="#aaaaaa")

    plt.colorbar(im, ax=ax, label="Attenuation (pp):  M4-base ΔR² − M1-base ΔR²\n"
                                   "negative = interaction shrank  |  positive = grew")
    ax.set_title(
        "Figure 30 — Interaction attenuation when M4 replaces M1 as base\n"
        "(red = effect shrank after richer controls; green = effect persisted/grew)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig("fig30_interaction_delta_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig30_interaction_delta_heatmap.png")

print("\n" + "=" * 70)
print("STEP 24 COMPLETE")
print("=" * 70)
