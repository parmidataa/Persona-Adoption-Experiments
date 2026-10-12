"""
35_directional_concordance.py  — Approach A
===========================================
Validity check for the R²_m results: does the LLM move labels in the SAME
DIRECTION as humans for each demographic group, or does it merely create
demographic-shaped variance that points the wrong way?

R²_m measures how much variance demographics explain in the LLM's labels.
It is sign-blind: a coefficient of +0.3 (humans) and −0.3 (LLM) both contribute
positively to R²_m, yet they describe OPPOSITE behaviour.

Method (zero new API calls):
  For each language × condition:
    1. Take the condition-matched formula (identical to 29_rmm_analysis.py).
    2. Fit the SAME formula twice on the SAME rows — once with human_label,
       once with llm_label as the response. Because the design matrix is
       identical, the fixed-effect coefficients are directly comparable.
    3. Drop the intercept (base rate, not a demographic effect).
    4. ROBUSTNESS FILTER: keep only coefficients for categorical levels with
       at least MIN_SUPPORT observations. Rare levels (e.g. a nationality with
       3 rows) produce extreme outlier coefficients that corrupt Pearson r
       (spurious ±1.0 values). Filtering them gives an honest comparison.
    5. Compare the human vs LLM coefficient vectors with robust-first metrics:
         - sign agreement   = fraction of coefficients with matching sign  [PRIMARY]
         - Spearman rho      = rank alignment of effects (outlier-resistant) [PRIMARY]
         - Pearson r         = magnitude+direction (OUTLIER-SENSITIVE, secondary)
         - cosine similarity = direction alignment of the whole vector

Interpretation guide:
    high R²_m + high sign agreement  → genuine demographic replication
    high R²_m + low/negative sign agreement → demographic-shaped variance that
        points the WRONG way (R²_m is sign-blind and overstates "success")

Outputs:
    results_directional_concordance.csv   — per lang × cond metrics
    results_concordance_coefficients.csv  — every paired coefficient (+support)
    fig_directional_concordance.png        — summary figure
"""

MIN_SUPPORT = 15   # minimum observations for a categorical level to be compared

import warnings, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CONDITIONS = ["p0", "p1", "p2", "p3"]
LANGUAGES  = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]

# Best condition per language (from Stage 3 formula-matched analysis in findings.md)
BEST_COND = {
    "ar": "p2", "de": "p3", "en": "p2", "es": "p2", "fr": "p2",
    "hi": "p2", "it": "p3", "nl": "p2", "pt": "p1",
}

# ── Condition-matched formula definitions (identical to 29_rmm_analysis.py) ───
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
M4_EXTRAS = ["C(country_birth)", "C(country_residence)",
             "C(employment_status)", "C(student_status)"]
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


def needed_cols(formula, target):
    cols = [target, "post_id"]
    for col in ALL_PRED_COLS:
        if col in formula:
            cols.append(col)
    return list(dict.fromkeys(cols))


def fit_fe_params(df_fit, formula, method="bfgs"):
    """Fit LMM, return fixed-effect coefficients (pandas Series), Intercept dropped."""
    try:
        m = smf.mixedlm(formula, data=df_fit,
                        groups=df_fit["post_id"]).fit(
                            reml=False, method=method, disp=False)
        fe = m.fe_params.copy()
        fe = fe[[i for i in fe.index if i != "Intercept" and "Intercept" not in str(i)]]
        return fe
    except Exception:
        return None


def parse_coef_name(name):
    """Return (column, level) from a statsmodels term name.
    'C(gender)[T.Male]' -> ('gender','Male'); 'age_centered' -> ('age_centered', None)."""
    m = re.match(r"C\((\w+)\)\[T\.(.+)\]", str(name))
    if m:
        return m.group(1), m.group(2)
    return str(name), None


def coef_support(name, df_fit):
    """Number of observations supporting a coefficient (level frequency).
    Continuous terms (age) get full n; categorical dummies get the level count."""
    col, level = parse_coef_name(name)
    if level is None:
        return len(df_fit)
    if col not in df_fit.columns:
        return 0
    return int((df_fit[col].astype(str) == str(level)).sum())


def group_dev_concordance(df_fit, pred_cols, min_support=MIN_SUPPORT):
    """Model-free cross-check: for each well-supported demographic level,
    compare the human group-mean deviation vs the LLM group-mean deviation
    from the overall mean. Robust to LMM parameterisation / reference-level
    choice. Returns (sign_agreement, spearman_rho, n_groups)."""
    hmean = df_fit["human_label"].mean()
    lmean = df_fit["llm_label"].mean()
    hdev, ldev = [], []
    for col in pred_cols:
        if col == "age_centered" or col not in df_fit.columns:
            continue
        for level, g in df_fit.groupby(col):
            if len(g) < min_support:
                continue
            hdev.append(g["human_label"].mean() - hmean)
            ldev.append(g["llm_label"].mean() - lmean)
    if len(hdev) < 2:
        return np.nan, np.nan, len(hdev)
    h = np.array(hdev); l = np.array(ldev)
    sign_agree = float(np.mean(np.sign(h) == np.sign(l)))
    try:
        rho = spearmanr(h, l)[0] if len(h) >= 3 else np.nan
    except Exception:
        rho = np.nan
    return sign_agree, rho, len(hdev)


# ── Load data ─────────────────────────────────────────────────────────────────
df_all = pd.read_parquet("data_predictions_clean.parquet")
df_all["age"] = pd.to_numeric(df_all["age"], errors="coerce")
lang_age_mean = df_all.groupby("language")["age"].transform("mean")
df_all["age_centered"] = df_all["age"] - lang_age_mean

DEMO_CATS = ["gender", "ethnicity", "nationality", "country_birth",
             "country_residence", "employment_status", "student_status", "Generation"]
for col in DEMO_CATS:
    if col in df_all.columns:
        df_all[col] = df_all[col].replace("unknown", np.nan)

print(f"Loaded {len(df_all):,} rows")
print(f"\n{'═'*78}")
print("  DIRECTIONAL CONCORDANCE — human vs LLM fixed-effect coefficients")
print(f"{'═'*78}")

summary_rows = []
coef_rows    = []

for lang in LANGUAGES:
    for cond in CONDITIONS:
        mask   = (df_all["language"] == lang) & (df_all["condition"] == cond)
        df_sub = df_all[mask].copy()

        # Build formula on the LLM target, then mirror it for the human target.
        formula_llm = get_formula("llm_label", lang, cond, df_sub)
        if formula_llm is None:
            continue
        rhs           = formula_llm.split("~", 1)[1].strip()
        formula_human = f"human_label ~ {rhs}"

        # Same rows for both fits → identical design matrix → comparable coefs.
        nc = needed_cols(formula_llm, "llm_label")
        nc = list(dict.fromkeys(nc + ["human_label"]))
        df_fit = df_sub[[c for c in nc if c in df_sub.columns]].dropna()
        if len(df_fit) < 50:
            continue

        fe_llm   = fit_fe_params(df_fit, formula_llm)
        fe_human = fit_fe_params(df_fit, formula_human)
        if fe_llm is None or fe_human is None:
            continue

        # Robustness filter: keep only well-supported categorical levels.
        common_all = [i for i in fe_human.index if i in fe_llm.index]
        common = [c for c in common_all if coef_support(c, df_fit) >= MIN_SUPPORT]
        n_dropped = len(common_all) - len(common)
        if len(common) < 2:
            continue
        h = fe_human[common].values.astype(float)
        l = fe_llm[common].values.astype(float)

        # Metrics — sign agreement and Spearman are the robust primaries.
        sign_agree = float(np.mean(np.sign(h) == np.sign(l)))
        try:
            spear = spearmanr(h, l)[0] if len(common) >= 3 else np.nan
        except Exception:
            spear = np.nan
        try:
            pear = pearsonr(h, l)[0] if len(common) >= 3 else np.nan
        except Exception:
            pear = np.nan
        denom = (np.linalg.norm(h) * np.linalg.norm(l))
        cosine = float(np.dot(h, l) / denom) if denom > 0 else np.nan

        # Model-free cross-check on the same predictors.
        pred_cols = [c for c in needed_cols(formula_llm, "llm_label")
                     if c not in ("llm_label", "post_id")]
        gd_sign, gd_rho, gd_n = group_dev_concordance(df_fit, pred_cols)

        is_best = (BEST_COND.get(lang) == cond)
        summary_rows.append({
            "language": lang, "condition": cond, "is_best_condition": is_best,
            "n_coefs": len(common), "n_coefs_dropped": n_dropped, "n_rows": len(df_fit),
            "sign_agreement": sign_agree,
            "spearman_rho": spear, "pearson_r": pear,
            "cosine_similarity": cosine,
            "groupdev_sign_agreement": gd_sign,
            "groupdev_spearman": gd_rho,
            "groupdev_n_groups": gd_n,
        })

        for name, hv, lv in zip(common, h, l):
            coef_rows.append({
                "language": lang, "condition": cond, "coef": name,
                "support": coef_support(name, df_fit),
                "human": hv, "llm": lv,
                "same_sign": bool(np.sign(hv) == np.sign(lv)),
            })

        flag = "  ◀ best" if is_best else ""
        print(f"  {lang.upper()} {cond.upper()}  "
              f"n_coef={len(common):2d} (−{n_dropped} rare)  "
              f"sign_agree={sign_agree*100:5.1f}%  "
              f"rho={spear if not np.isnan(spear) else float('nan'):+.3f}  "
              f"cos={cosine:+.3f}{flag}")

df_sum  = pd.DataFrame(summary_rows)
df_coef = pd.DataFrame(coef_rows)
df_sum.to_csv("results_directional_concordance.csv", index=False)
df_coef.to_csv("results_concordance_coefficients.csv", index=False)
print(f"\nSaved: results_directional_concordance.csv ({len(df_sum)} rows)")
print(f"Saved: results_concordance_coefficients.csv ({len(df_coef)} paired coefficients)")

# ── Best-condition summary ─────────────────────────────────────────────────────
print(f"\n{'═'*78}")
print("  BEST-CONDITION CONCORDANCE (the conditions reported as results)")
print(f"{'═'*78}")
df_best = df_sum[df_sum["is_best_condition"]].copy().set_index("language").reindex(LANGUAGES).dropna(how="all")
print("  Coefficient method (LMM)        |  Model-free cross-check (group means)")
print(f"  {'Lang':<5}{'Cond':<5}{'sign':>7}{'rho':>8}   |  {'sign':>7}{'rho':>8}{'n_grp':>7}   verdict")
print("-" * 74)
verdicts = {}
for lang in LANGUAGES:
    r = df_sum[(df_sum.language == lang) & (df_sum.is_best_condition)]
    if r.empty:
        continue
    r = r.iloc[0]
    sa, sp   = r["sign_agreement"], r["spearman_rho"]
    gsa, grho = r["groupdev_sign_agreement"], r["groupdev_spearman"]
    # Verdict from agreement of the two robust sign metrics.
    both_sign = [v for v in (sa, gsa) if not np.isnan(v)]
    mean_sign = np.mean(both_sign) if both_sign else np.nan
    if mean_sign >= 0.6:
        verdict = "aligned"
    elif mean_sign <= 0.45:
        verdict = "WRONG direction"
    else:
        verdict = "mixed"
    verdicts[lang] = verdict
    print(f"  {lang.upper():<5}{r['condition'].upper():<5}{sa*100:>6.0f}%"
          f"{sp if not np.isnan(sp) else float('nan'):>8.2f}   |  "
          f"{gsa*100:>6.0f}%{grho if not np.isnan(grho) else float('nan'):>8.2f}"
          f"{int(r['groupdev_n_groups']):>7}   {verdict}")

print("\n  Verdict counts:",
      {v: sum(1 for x in verdicts.values() if x == v)
       for v in ["aligned", "mixed", "WRONG direction"]})

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

# Left: sign agreement + cosine for the best condition per language, sorted
ax = axes[0]
rows = []
for lang in LANGUAGES:
    r = df_sum[(df_sum.language == lang) & (df_sum.is_best_condition)]
    if not r.empty:
        rows.append(r.iloc[0])
db = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)

db = db.sort_values("sign_agreement", ascending=False)
x = np.arange(len(db))
width = 0.38
sa_colors = ["#27ae60" if v >= 0.6 else ("#e74c3c" if v <= 0.45 else "#f39c12")
             for v in db["sign_agreement"]]
bars1 = ax.bar(x - width/2, db["sign_agreement"]*100, width,
               label="Sign agreement (%)", color=sa_colors, alpha=0.95)
bars2 = ax.bar(x + width/2, db["spearman_rho"]*100, width,
               label="Spearman ρ (×100)", color="#34495e", alpha=0.75)
ax.axhline(50, color="gray", linestyle="--", linewidth=1,
           label="50% = chance (sign)")
ax.axhline(0,  color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{r.language.upper()}\n{r.condition.upper()}" for _, r in db.iterrows()],
                   fontsize=9)
ax.set_ylabel("Concordance metric")
ax.set_title("Directional concordance — best condition per language\n"
             "green=aligned (≥60%)  orange=mixed  red=wrong direction (≤45%)",
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# Right: scatter of paired coefficients for the strongest success (ES best)
ax = axes[1]
focus_lang = "es"
focus_cond = BEST_COND[focus_lang]
dc = df_coef[(df_coef.language == focus_lang) & (df_coef.condition == focus_cond)]
if not dc.empty:
    colors = ["#27ae60" if s else "#e74c3c" for s in dc["same_sign"]]
    ax.scatter(dc["human"], dc["llm"], c=colors, s=70, zorder=5, edgecolor="white")
    lim = max(dc["human"].abs().max(), dc["llm"].abs().max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y = x (perfect agreement)")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Human coefficient")
    ax.set_ylabel("LLM coefficient")
    ax.set_title(f"Coefficient agreement — {focus_lang.upper()} {focus_cond.upper()} "
                 f"(strongest success)\n"
                 f"green = same sign, red = opposite sign", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("Approach A: Directional Concordance Validity Check",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_directional_concordance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: fig_directional_concordance.png")
print(f"\n{'═'*78}")
print("  DONE")
print(f"{'═'*78}")
