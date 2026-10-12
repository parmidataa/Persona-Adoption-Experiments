"""
29_rmm_analysis.py  (v3 — condition-matched formulas + bootstrap CIs)
======================================================================
Fix 2A: Each prompt condition is evaluated with the formula that matches
        what the LLM was actually told:
          P0 / P1 → M1 formula  (age, gender, ethnicity, nationality)
          P2      → M4 formula  (+ country_birth, country_residence,
                                   employment_status, student_status)
          P3      → template formula (only the variables in that language's
                                      prose sentence)

Fix 2B: Bootstrap 95% CIs (100 iterations, powell optimizer) for the
        best condition per language to validate near-complete replication
        claims.

Both fixes also apply to the Sample Human R²_m (same formula per condition,
so the "% of achievable signal" comparison is always apples-to-apples).

Outputs:
    results_r2m_llm.csv        — R²_m per lang × cond (condition-matched formula)
    results_steerability.csv   — % of achievable signal, condition-matched
    results_bootstrap_ci.csv   — 95% CI for best condition per language
    results_spearman.csv       — Spearman ρ (both benchmarks)
    fig_r2m_heatmap.png        — three-panel heatmap
    fig_steerability.png       — condition bars + % achievable + CI error bars
    fig_spearman_scatter.png   — two-panel scatter
"""

import warnings, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Direction 1 benchmarks (full dataset) ────────────────────────────────────
BENCHMARKS = {
    "ar": {"m1": 0.019017, "m4": 0.038825, "target": 0.076480},
    "de": {"m1": 0.002024, "m4": 0.013845, "target": 0.061517},
    "en": {"m1": 0.013586, "m4": 0.048531, "target": 0.143136},
    "es": {"m1": 0.010041, "m4": 0.032465, "target": 0.093269},
    "fr": {"m1": 0.015464, "m4": 0.021825, "target": 0.037138},
    "hi": {"m1": 0.019916, "m4": 0.060088, "target": 0.078475},
    "it": {"m1": 0.002709, "m4": 0.044459, "target": 0.122968},
    "nl": {"m1": 0.012025, "m4": 0.023633, "target": 0.114866},
    "pt": {"m1": 0.010891, "m4": 0.043796, "target": 0.121228},
}

CONDITIONS = ["p0", "p1", "p2", "p3"]
LANGUAGES  = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]

# ── Condition-matched formula definitions ─────────────────────────────────────
# P0 / P1 — M1 base variables per language
M1_BASE = {
    "ar": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "de": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "en": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "es": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "fr": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "hi": ["age_centered", "C(gender)", "C(ethnicity)"],
    "it": ["age_centered", "C(gender)", "C(ethnicity)"],
    "nl": ["age_centered", "C(gender)", "C(ethnicity)"],
    "pt": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
}

# P2 — extra M4 variables (added to M1 base if they have ≥2 unique non-null values)
M4_EXTRAS = ["C(country_birth)", "C(country_residence)",
             "C(employment_status)", "C(student_status)"]

# P3 — template-matched variables (only the fields used in each language's prose)
P3_TEMPLATE = {
    "ar": ["age_centered", "C(nationality)"],
    "de": ["C(Generation)", "C(employment_status)"],
    "en": ["C(nationality)", "C(gender)", "C(country_residence)"],
    "es": ["age_centered", "C(country_birth)", "C(country_residence)"],
    "fr": ["age_centered", "C(ethnicity)", "C(nationality)", "C(country_residence)"],
    "hi": ["age_centered", "C(ethnicity)", "C(country_birth)"],
    "it": ["age_centered", "C(employment_status)"],
    "nl": ["C(employment_status)"],
    "pt": ["C(employment_status)", "C(country_birth)"],
}

# All possible predictor columns (for NaN-dropping)
ALL_PRED_COLS = ["age_centered", "gender", "ethnicity", "nationality",
                 "country_birth", "country_residence",
                 "employment_status", "student_status", "Generation"]


def col_from_term(term: str) -> str:
    """Extract raw column name from a formula term like C(col) or age_centered."""
    m = re.match(r"C\((\w+)\)", term)
    return m.group(1) if m else term


def build_formula(target: str, var_list: list[str], df_check: pd.DataFrame) -> str:
    """Build formula string, silently dropping vars that are constant in df_check."""
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


def get_formula(target: str, lang: str, cond: str, df_check: pd.DataFrame) -> str:
    """Return the condition-appropriate formula, with variance-based column pruning."""
    if cond in ("p0", "p1"):
        return build_formula(target, M1_BASE[lang], df_check)
    elif cond == "p2":
        return build_formula(target, M1_BASE[lang] + M4_EXTRAS, df_check)
    else:  # p3
        return build_formula(target, P3_TEMPLATE[lang], df_check)


def needed_cols(formula: str, df: pd.DataFrame, target: str) -> list[str]:
    """Return list of DataFrame columns needed for a formula string."""
    cols = [target, "post_id"]
    for col in ALL_PRED_COLS:
        if col in formula and col in df.columns:
            cols.append(col)
    return list(dict.fromkeys(cols))  # deduplicate, preserve order


# ── Load data ─────────────────────────────────────────────────────────────────
df_all = pd.read_parquet("data_predictions_clean.parquet")

df_all["age"] = pd.to_numeric(df_all["age"], errors="coerce")
lang_age_mean = df_all.groupby("language")["age"].transform("mean")
df_all["age_centered"] = df_all["age"] - lang_age_mean

DEMO_CATS = ["gender", "ethnicity", "nationality",
             "country_birth", "country_residence",
             "employment_status", "student_status", "Generation"]
for col in DEMO_CATS:
    if col in df_all.columns:
        df_all[col] = df_all[col].replace("unknown", np.nan)

print(f"Loaded {len(df_all):,} rows | {sorted(df_all['language'].unique())} | "
      f"{sorted(df_all['condition'].unique())}")


# ── Core R²_m fitter ─────────────────────────────────────────────────────────
def fit_r2m(df_sub: pd.DataFrame, formula: str,
            method: str = "bfgs") -> dict:
    """Fit LMM, return dict with r2m and diagnostics. Returns NaN on failure."""
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
        return {"r2m": r2m, "n": len(df_fit),
                "converged": m.converged, "mean_pred": df_fit[target].mean()}
    except Exception as e:
        return {"r2m": np.nan, "n": 0, "converged": False, "error": str(e)}


# ── Sample Human R²_m — condition-matched formula ───────────────────────────
# For each condition, compute the human R²_m using the SAME formula as the LLM.
# This makes "% of achievable signal" an apples-to-apples comparison.

print(f"\n{'═'*70}")
print("  SAMPLE HUMAN R²_m (condition-matched formula, p0 slice)")
print(f"{'═'*70}")

SAMPLE_HUMAN = {}  # {(lang, cond): r2m}
for lang in LANGUAGES:
    mask_h = (df_all["language"] == lang) & (df_all["condition"] == "p0")
    df_h = df_all[mask_h].copy()
    for cond in CONDITIONS:
        hf = get_formula("human_label", lang, cond, df_h)
        if hf is None:
            SAMPLE_HUMAN[(lang, cond)] = np.nan
            continue
        res = fit_r2m(df_h, hf)
        SAMPLE_HUMAN[(lang, cond)] = res["r2m"]

# Print summary
print(f"{'Lang':<5} {'SH_P0/P1':>10} {'SH_P2':>10} {'SH_P3':>10} {'Dir1_M4':>10} {'Preserved_M1':>14}")
print("-" * 60)
for lang in LANGUAGES:
    sh01 = SAMPLE_HUMAN[(lang,"p0")]
    sh2  = SAMPLE_HUMAN[(lang,"p2")]
    sh3  = SAMPLE_HUMAN[(lang,"p3")]
    m4   = BENCHMARKS[lang]["m4"]
    pres = sh01/m4*100 if (sh01 and not np.isnan(sh01)) else float("nan")
    print(f"{lang.upper():<5} {sh01*100:>9.3f}%  {sh2*100:>9.3f}%  {sh3*100:>9.3f}%"
          f"  {m4*100:>9.3f}%  {pres:>11.0f}%")


# ── Main analysis loop ────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("  R²_m ANALYSIS — condition-matched formulas")
print(f"{'═'*70}")

results = []
for lang in LANGUAGES:
    bench = BENCHMARKS[lang]
    print(f"\n{lang.upper()}  Dir1 M1={bench['m1']*100:.3f}%  M4={bench['m4']*100:.3f}%")
    for cond in CONDITIONS:
        mask = (df_all["language"] == lang) & (df_all["condition"] == cond)
        df_sub = df_all[mask].copy()
        formula = get_formula("llm_label", lang, cond, df_sub)
        if formula is None:
            print(f"  {cond.upper()}: SKIPPED (no valid predictors)")
            results.append({"language": lang, "condition": cond,
                            "r2m_llm": np.nan, "r2m_llm_pct": np.nan,
                            "formula": "None"})
            continue

        res   = fit_r2m(df_sub, formula)
        r2m   = res["r2m"]
        sh    = SAMPLE_HUMAN[(lang, cond)]
        pct_sh = (r2m / sh * 100) if (not np.isnan(r2m) and sh and sh > 0) else np.nan

        # Zone relative to Dir1 benchmarks
        if np.isnan(r2m):          zone = "ERROR"
        elif r2m < bench["m1"]:    zone = "below_M1"
        elif r2m < bench["m4"]:    zone = "M1_to_M4"
        elif r2m < bench["target"]:zone = "M4_to_target"
        else:                      zone = "at_target"

        print(f"  {cond.upper()} [{formula.split('~')[1][:55].strip()!s:55}] "
              f"R²m={r2m*100:.3f}%  SH={sh*100:.3f}%  %SH={pct_sh:.1f}%  [{zone}]")

        results.append({
            "language":            lang,
            "condition":           cond,
            "formula":             formula,
            "r2m_llm":             r2m,
            "r2m_llm_pct":         r2m * 100 if not np.isnan(r2m) else np.nan,
            "r2m_sample_human":    sh,
            "pct_of_sample_human": pct_sh,
            "r2m_m1":              bench["m1"],
            "r2m_m4":              bench["m4"],
            "r2m_target":          bench["target"],
            "zone":                zone,
            "n":                   res["n"],
            "converged":           res.get("converged", False),
            "mean_llm_label":      res.get("mean_pred", np.nan),
        })

df_res = pd.DataFrame(results)


# ── Bootstrap 95% CIs (best condition per language) ──────────────────────────
N_BOOT = 100
print(f"\n{'═'*70}")
print(f"  BOOTSTRAP CIs  (N={N_BOOT}, powell optimizer, best condition per lang)")
print(f"{'═'*70}")

boot_rows = []
for lang in LANGUAGES:
    lang_df  = df_res[df_res["language"] == lang].dropna(subset=["r2m_llm"])
    if lang_df.empty:
        boot_rows.append({"language": lang, "best_condition": "n/a",
                          "r2m_best": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
        continue

    best_cond = lang_df.loc[lang_df["r2m_llm"].idxmax(), "condition"]
    r2m_point = lang_df["r2m_llm"].max()
    sh_best   = SAMPLE_HUMAN[(lang, best_cond)]

    mask = (df_all["language"] == lang) & (df_all["condition"] == best_cond)
    df_sub  = df_all[mask].copy()
    formula = get_formula("llm_label", lang, best_cond, df_sub)
    nc = needed_cols(formula, df_sub, "llm_label")
    df_fit  = df_sub[[c for c in nc if c in df_sub.columns]].dropna()

    boot_r2ms = []
    for i in range(N_BOOT):
        df_b = df_fit.sample(n=len(df_fit), replace=True, random_state=42 + i)
        res_b = fit_r2m(df_b, formula, method="powell")
        if not np.isnan(res_b["r2m"]) and res_b["r2m"] >= 0:
            boot_r2ms.append(res_b["r2m"] * 100)

    if len(boot_r2ms) < 20:
        ci_lo, ci_hi = np.nan, np.nan
        note = f"only {len(boot_r2ms)} successes"
    else:
        ci_lo = np.percentile(boot_r2ms, 2.5)
        ci_hi = np.percentile(boot_r2ms, 97.5)
        note  = f"{len(boot_r2ms)}/{N_BOOT} ok"

    pct_sh = (r2m_point / sh_best * 100) if (sh_best and sh_best > 0) else np.nan
    print(f"  {lang.upper():<3} best={best_cond.upper()}  "
          f"R²m={r2m_point*100:.3f}%  "
          f"95%CI=[{ci_lo:.3f}%, {ci_hi:.3f}%]  "
          f"% of SH={pct_sh:.1f}%  ({note})")

    boot_rows.append({
        "language":       lang,
        "best_condition": best_cond,
        "r2m_best_pct":   r2m_point * 100,
        "ci_lo_pct":      ci_lo,
        "ci_hi_pct":      ci_hi,
        "r2m_sample_human_pct": sh_best * 100 if sh_best else np.nan,
        "pct_of_sample_human":  pct_sh,
        "n_boot_success": len(boot_r2ms),
    })

df_boot = pd.DataFrame(boot_rows)


# ── Steerability (best condition vs P0, relative to Dir1 M1→M4 gap) ──────────
steer_rows = []
for lang in LANGUAGES:
    bench   = BENCHMARKS[lang]
    lang_df = df_res[df_res["language"] == lang]
    r2m_p0  = lang_df[lang_df["condition"]=="p0"]["r2m_llm"].values[0]
    idx_max = lang_df["r2m_llm"].idxmax()
    r2m_best = lang_df.loc[idx_max, "r2m_llm"]
    best_cond = lang_df.loc[idx_max, "condition"]
    sh_best   = SAMPLE_HUMAN[(lang, best_cond)]

    gap   = bench["m4"] - bench["m1"]
    steer = (r2m_best - r2m_p0) / gap if gap > 0 else np.nan
    steer_clipped = np.clip(steer, -2, 5) if not np.isnan(steer) else np.nan
    pct_sh = (r2m_best / sh_best * 100) if (sh_best and sh_best > 0) else np.nan

    steer_rows.append({
        "language":            lang,
        "r2m_p0":              r2m_p0,
        "r2m_best":            r2m_best,
        "best_condition":      best_cond,
        "r2m_sample_human":    sh_best,
        "pct_of_sample_human": pct_sh,
        "r2m_m1":              bench["m1"],
        "r2m_m4":              bench["m4"],
        "steerability":        steer_clipped,
    })

df_steer = pd.DataFrame(steer_rows)


# ── Spearman ρ ────────────────────────────────────────────────────────────────
r2m_p3_raw = df_res[df_res["condition"]=="p3"]["r2m_llm"].values
r2m_m4     = np.array([BENCHMARKS[l]["m4"] for l in LANGUAGES])
valid_mask = ~np.isnan(r2m_p3_raw)
if valid_mask.sum() > 3:
    rho, pval = spearmanr(r2m_p3_raw[valid_mask], r2m_m4[valid_mask])
else:
    rho, pval = np.nan, np.nan
r2m_p3 = r2m_p3_raw  # keep full array for scatter plot

r2m_best_list = [df_res[df_res["language"]==l]["r2m_llm"].max() for l in LANGUAGES]
r2m_sh_list   = [SAMPLE_HUMAN[(l,"p0")] for l in LANGUAGES]
rho_sh, pval_sh = spearmanr(r2m_best_list, r2m_sh_list)


# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("  SUMMARY TABLE  (condition-matched R²_m, %)")
print(f"{'═'*70}")
pivot = df_res.pivot(index="language", columns="condition",
                     values="r2m_llm_pct").round(3)
pivot.columns.name = None
pivot.index.name = None
pivot["[SH_M1]"] = [SAMPLE_HUMAN[(l,"p0")]*100 for l in pivot.index]
pivot["[SH_M4]"] = [SAMPLE_HUMAN[(l,"p2")]*100 for l in pivot.index]
pivot["[Dir1_M4]"] = [BENCHMARKS[l]["m4"]*100 for l in pivot.index]
print(pivot[["p0","p1","p2","p3","[SH_M1]","[SH_M4]","[Dir1_M4]"]].to_string())

print(f"\n{'═'*70}")
print("  STEERABILITY")
print(f"{'═'*70}")
print(df_steer[["language","r2m_p0","r2m_best","best_condition",
                "pct_of_sample_human","steerability"]].to_string(index=False))

print(f"\n{'═'*70}")
print("  SPEARMAN ρ")
print(f"{'═'*70}")
print(f"  LLM P3 vs Dir1 M4    : ρ={rho:.3f}  p={pval:.4f}")
print(f"  LLM best vs SH (M1)  : ρ={rho_sh:.3f}  p={pval_sh:.4f}")
print(f"{'═'*70}")


# ── Save CSVs ─────────────────────────────────────────────────────────────────
df_res.to_csv("results_r2m_llm.csv", index=False)
df_steer.to_csv("results_steerability.csv", index=False)
df_boot.to_csv("results_bootstrap_ci.csv", index=False)
pd.DataFrame([
    {"comparison": "p3_vs_dir1_m4",        "rho": rho,    "p": pval},
    {"comparison": "llm_best_vs_sample_human", "rho": rho_sh, "p": pval_sh},
]).to_csv("results_spearman.csv", index=False)
print("\nSaved: results_r2m_llm.csv, results_steerability.csv, "
      "results_bootstrap_ci.csv, results_spearman.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: three-panel heatmap ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5),
                         gridspec_kw={"width_ratios": [4, 1, 1], "wspace": 0.06})

llm_data = df_res.pivot(index="language", columns="condition",
                        values="r2m_llm_pct")[["p0","p1","p2","p3"]]
llm_data.index = [l.upper() for l in llm_data.index]
llm_data.columns = ["P0\n(no persona)", "P1\n(M1 fields)",
                     "P2\n(M4 fields)", "P3\n(prose)"]

vmax = max(float(llm_data.values.max()), 4)

sns.heatmap(llm_data, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white",
            ax=axes[0], cbar=False, annot_kws={"size": 9}, vmin=0, vmax=vmax)
axes[0].set_title("LLM R²_m (%) — condition-matched formula", fontsize=11, fontweight="bold")
axes[0].set_xlabel("")
axes[0].set_ylabel("Language")

sh_m1_data = pd.DataFrame(
    {"Sample Human\nM1 formula\n(P0/P1 ref)": [SAMPLE_HUMAN[(l,"p0")]*100 for l in LANGUAGES]},
    index=[l.upper() for l in LANGUAGES])
sns.heatmap(sh_m1_data, annot=True, fmt=".3f", cmap="Blues",
            linewidths=0.5, linecolor="white",
            ax=axes[1], cbar=False, annot_kws={"size": 9}, vmin=0, vmax=vmax)
axes[1].set_title("PRIMARY\nbenchmark\n(M1 formula)", fontsize=10,
                   color="#1a5fa8", fontweight="bold")
axes[1].set_xlabel(""); axes[1].set_ylabel(""); axes[1].set_yticklabels([])

m4_data = pd.DataFrame(
    {"Dir1 M4\n(full dataset)": [BENCHMARKS[l]["m4"]*100 for l in LANGUAGES]},
    index=[l.upper() for l in LANGUAGES])
sns.heatmap(m4_data, annot=True, fmt=".3f", cmap="Greys",
            linewidths=0.5, linecolor="white",
            ax=axes[2], cbar=False, annot_kws={"size": 9}, vmin=0, vmax=vmax)
axes[2].set_title("Context\nbenchmark\n(Dir1 M4)", fontsize=10, color="#555555")
axes[2].set_xlabel(""); axes[2].set_ylabel(""); axes[2].set_yticklabels([])

fig.suptitle(
    "LLM Persona Prompting R²_m — condition-matched formulas\n"
    "P0/P1 use M1 formula  |  P2 uses M4 formula  |  P3 uses template formula",
    fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig("fig_r2m_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_r2m_heatmap.png")


# ── Figure 2: steerability + % achievable with CI error bars ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: R²_m by condition + both reference lines
ax = axes[0]
cond_colors = {"p0": "#aec6e8", "p1": "#4c9be8", "p2": "#1a5fa8", "p3": "#e85c4c"}
x = np.arange(len(LANGUAGES))
width = 0.2

for i, cond in enumerate(CONDITIONS):
    vals = [df_res[(df_res["language"]==l) & (df_res["condition"]==cond)]["r2m_llm_pct"].values[0]
            for l in LANGUAGES]
    ax.bar(x + i*width, vals, width, label=cond.upper(), color=cond_colors[cond], alpha=0.9)

sh_p0_list = [SAMPLE_HUMAN[(l,"p0")]*100 for l in LANGUAGES]
ax.plot(x + 1.5*width, sh_p0_list, color="#27ae60", marker="^", markersize=8,
        linestyle="--", linewidth=1.5, label="Sample Human / M1 (PRIMARY)", zorder=6)

m4_list = [BENCHMARKS[l]["m4"]*100 for l in LANGUAGES]
ax.plot(x + 1.5*width, m4_list, color="#555555", marker="D", markersize=5,
        linestyle=":", linewidth=1.2, label="Dir1 M4 (context)",
        zorder=5, markerfacecolor="none")

ax.set_xticks(x + 1.5*width)
ax.set_xticklabels([l.upper() for l in LANGUAGES])
ax.set_ylabel("R²_m (%)")
ax.set_title("LLM R²_m by Condition\n(▲ Sample Human PRIMARY  ◇ Dir1 M4 context)", fontsize=10)
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.grid(axis="y", alpha=0.3)

# Right: % of achievable signal + 95% CI error bars
ax = axes[1]
pct_vals = df_steer["pct_of_sample_human"].values

bar_colors = []
for v in pct_vals:
    if np.isnan(v):       bar_colors.append("#cccccc")
    elif v > 150:         bar_colors.append("#8e44ad")   # purple = overshoot
    elif v >= 90:         bar_colors.append("#27ae60")   # green  = near-complete
    elif v >= 50:         bar_colors.append("#f39c12")   # orange = partial
    else:                 bar_colors.append("#e74c3c")   # red    = failure

bars = ax.bar([l.upper() for l in LANGUAGES], pct_vals,
              color=bar_colors, alpha=0.88, edgecolor="white")

# Add 95% CI error bars from bootstrap
ci_lo_arr = df_boot["ci_lo_pct"].values
ci_hi_arr = df_boot["ci_hi_pct"].values
r2m_arr   = df_boot["r2m_best_pct"].values
sh_arr    = df_boot["r2m_sample_human_pct"].values

for j, lang in enumerate(LANGUAGES):
    sh = sh_arr[j]
    n_ok = df_boot.iloc[j]["n_boot_success"]
    if sh and sh > 0 and not np.isnan(ci_lo_arr[j]) and n_ok >= 20:
        lo_pct = ci_lo_arr[j] / sh * 100
        hi_pct = ci_hi_arr[j] / sh * 100
        pt     = pct_vals[j]
        lo_err = max(0.0, pt - lo_pct)
        hi_err = max(0.0, hi_pct - pt)
        if lo_err + hi_err > 0:
            ax.errorbar(j, pt,
                        yerr=[[lo_err], [hi_err]],
                        fmt="none", color="black", capsize=4, linewidth=1.5, zorder=10)

ax.axhline(100, color="#27ae60", linewidth=1.5, linestyle="--", alpha=0.8,
           label="100% = matches sample human")
ax.axhline(90,  color="#27ae60", linewidth=0.8, linestyle=":", alpha=0.5)
ax.axhline(50,  color="#f39c12", linewidth=0.8, linestyle=":", alpha=0.5)

ax.set_ylabel("LLM best R²_m as % of Sample Human R²_m")
ax.set_title("% of Achievable Signal Replicated (with 95% bootstrap CI)\n"
             "green ≥90%  orange 50–90%  red <50%  purple = overshoot", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

for bar, row in zip(bars, df_steer.itertuples()):
    h = bar.get_height()
    if not np.isnan(h):
        ax.text(bar.get_x() + bar.get_width()/2, h + 3,
                f"{row.best_condition.upper()}\n{h:.0f}%",
                ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig("fig_steerability.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_steerability.png")


# ── Figure 3: two-panel scatter ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

lang_colors = {
    "ar":"#e74c3c","hi":"#e67e22","fr":"#3498db",
    "en":"#2ecc71","es":"#9b59b6","de":"#1abc9c",
    "it":"#f39c12","nl":"#e91e63","pt":"#607d8b",
}

# Left: original (LLM P3 vs Dir1 M4) — skip languages where P3 = NaN
ax = axes[0]
for lang in LANGUAGES:
    r2_p3 = df_res[(df_res["language"]==lang)&(df_res["condition"]=="p3")]["r2m_llm_pct"].values[0]
    m4    = BENCHMARKS[lang]["m4"] * 100
    if np.isnan(r2_p3):
        continue
    ax.scatter(r2_p3, m4, s=130, color=lang_colors[lang], zorder=5)
    ax.annotate(lang.upper(), (r2_p3, m4), textcoords="offset points",
                xytext=(5,3), fontsize=9, color=lang_colors[lang], fontweight="bold")
lim = max(float(df_res["r2m_llm_pct"].max()),
          max(b["m4"]*100 for b in BENCHMARKS.values())) * 1.15
ax.plot([0,lim],[0,lim],"k--",alpha=0.25,label="y = x  (perfect alignment)")
ax.set_xlabel("LLM R²_m — P3 condition (%)", fontsize=10)
ax.set_ylabel("Dir1 M4 R²_m — full dataset (%)", fontsize=10)
ax.set_title(f"Original: LLM P3 vs Dir1 M4\nSpearman ρ={rho:.3f}  p={pval:.3f}", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_xlim(left=-0.1); ax.set_ylim(bottom=-0.1)

# Right: corrected (LLM best vs Sample Human, with CI error bars)
ax = axes[1]
for j, lang in enumerate(LANGUAGES):
    r2_best = float(df_res[df_res["language"]==lang]["r2m_llm_pct"].max())
    sh      = SAMPLE_HUMAN[(lang,"p0")] * 100
    ax.scatter(r2_best, sh, s=130, color=lang_colors[lang], zorder=5)
    ax.annotate(lang.upper(), (r2_best, sh), textcoords="offset points",
                xytext=(5,3), fontsize=9, color=lang_colors[lang], fontweight="bold")
    # CI horizontal error bars on x-axis (LLM best) — skip if < 20 bootstrap successes
    n_ok_j = int(df_boot.iloc[j]["n_boot_success"])
    if not np.isnan(ci_lo_arr[j]) and n_ok_j >= 20:
        lo_err_x = max(0.0, r2_best - ci_lo_arr[j])
        hi_err_x = max(0.0, ci_hi_arr[j] - r2_best)
        ax.errorbar(r2_best, sh,
                    xerr=[[lo_err_x], [hi_err_x]],
                    fmt="none", color=lang_colors[lang], capsize=3,
                    linewidth=1.2, alpha=0.7, zorder=4)

lim2 = max(float(df_res["r2m_llm_pct"].max()),
           max(SAMPLE_HUMAN[(l,"p0")]*100 for l in LANGUAGES)) * 1.2
ax.plot([0,lim2],[0,lim2],"k--",alpha=0.25,label="y = x  (perfect replication)")
ax.set_xlabel("LLM best-condition R²_m (%) [→ CI]", fontsize=10)
ax.set_ylabel("Sample Human R²_m — same 700 rows (%)", fontsize=10)
ax.set_title(f"Corrected: LLM best vs Sample Human\nSpearman ρ={rho_sh:.3f}  p={pval_sh:.3f}", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_xlim(left=-0.1); ax.set_ylim(bottom=-0.1)

# Annotate PT overshoot
pt_best = float(df_res[df_res["language"]=="pt"]["r2m_llm_pct"].max())
pt_sh   = SAMPLE_HUMAN[("pt","p0")] * 100
ax.annotate("⚠ PT\novershoot", (pt_best, pt_sh),
            textcoords="offset points", xytext=(8,-28), fontsize=7,
            color="#607d8b", arrowprops=dict(arrowstyle="->",color="#607d8b",lw=0.8))

fig.suptitle("Does LLM steerability track where demographics matter?",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_spearman_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_spearman_scatter.png")

print(f"\n{'═'*70}")
print("  ALL DONE")
print(f"{'═'*70}")
