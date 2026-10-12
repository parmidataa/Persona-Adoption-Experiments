"""
43_analyze_gpt41.py  — Approach B analysis (gpt-4.1 vs gpt-4.1-mini)
====================================================================
Parses gpt41_batch_output.jsonl, builds a clean predictions table, then runs
the IDENTICAL formula-matched R²_m analysis and directional concordance as the
baseline, and produces a side-by-side comparison.

Answers the four insight questions:
  1. Hindi: does scale fix the competence failure?
  2. English: is persona disruption scale-dependent?
  3. Directional concordance: does scale improve human-aligned direction?
  4. ES/AR successes: do they replicate with a larger model?

Outputs:
    data_predictions_gpt41_clean.parquet
    results_gpt41_vs_mini.csv          — R²_m both models, per lang × cond
    results_gpt41_concordance.csv      — directional concordance for gpt-4.1
    fig_gpt41_r2m_comparison.png       — R²_m mini vs gpt-4.1
    fig_gpt41_concordance_comparison.png
"""

import warnings, re, json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = ["p0", "p1", "p2", "p3"]
LANGUAGES  = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]
BEST_COND  = {"ar":"p2","de":"p3","en":"p2","es":"p2","fr":"p2",
              "hi":"p2","it":"p3","nl":"p2","pt":"p1"}

# ── Formula infrastructure (identical to 29_rmm_analysis.py / 35_*) ───────────
M1_BASE = {
    "ar": ["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
    "de": ["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
    "en": ["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
    "es": ["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
    "fr": ["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
    "hi": ["age_centered","C(gender)","C(ethnicity)"],
    "it": ["age_centered","C(gender)","C(ethnicity)"],
    "nl": ["age_centered","C(gender)","C(ethnicity)"],
    "pt": ["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
}
M4_EXTRAS = ["C(country_birth)","C(country_residence)",
             "C(employment_status)","C(student_status)"]
P3_TEMPLATE = {
    "ar": ["age_centered","C(nationality)"],
    "de": ["C(Generation)","C(employment_status)"],
    "en": ["C(nationality)","C(gender)","C(country_residence)"],
    "es": ["age_centered","C(country_birth)","C(country_residence)"],
    "fr": ["age_centered","C(ethnicity)","C(nationality)","C(country_residence)"],
    "hi": ["age_centered","C(ethnicity)","C(country_birth)"],
    "it": ["age_centered","C(employment_status)"],
    "nl": ["C(employment_status)"],
    "pt": ["C(employment_status)","C(country_birth)"],
}
ALL_PRED_COLS = ["age_centered","gender","ethnicity","nationality","country_birth",
                 "country_residence","employment_status","student_status","Generation"]
MIN_SUPPORT = 15


def col_from_term(t):
    m = re.match(r"C\((\w+)\)", t); return m.group(1) if m else t

def build_formula(target, var_list, df_check):
    valid = []
    for term in var_list:
        col = col_from_term(term)
        if col not in df_check.columns: continue
        if col == "age_centered": valid.append(term); continue
        if df_check[col].dropna().nunique() >= 2: valid.append(term)
    return (f"{target} ~ " + " + ".join(valid)) if valid else None

def get_formula(target, lang, cond, df_check):
    if cond in ("p0","p1"): return build_formula(target, M1_BASE[lang], df_check)
    if cond == "p2":        return build_formula(target, M1_BASE[lang]+M4_EXTRAS, df_check)
    return build_formula(target, P3_TEMPLATE[lang], df_check)

def needed_cols(formula, target):
    cols = [target, "post_id"]
    for col in ALL_PRED_COLS:
        if col in formula: cols.append(col)
    return list(dict.fromkeys(cols))

def fit_r2m(df_sub, formula, method="bfgs"):
    target = formula.split("~")[0].strip()
    nc = [c for c in needed_cols(formula, target) if c in df_sub.columns]
    df_fit = df_sub[nc].dropna()
    if len(df_fit) < 50 or df_fit[target].nunique() < 2:
        return np.nan
    try:
        m = smf.mixedlm(formula, data=df_fit, groups=df_fit["post_id"]).fit(
            reml=False, method=method, disp=False)
        fp = np.dot(m.model.exog, m.fe_params)
        vf = np.var(fp); vr = float(m.cov_re.iloc[0,0])
        return vf / (vf + vr + m.scale)
    except Exception:
        return np.nan

def fit_fe(df_fit, formula, method="bfgs"):
    try:
        m = smf.mixedlm(formula, data=df_fit, groups=df_fit["post_id"]).fit(
            reml=False, method=method, disp=False)
        fe = m.fe_params.copy()
        return fe[[i for i in fe.index if "Intercept" not in str(i)]]
    except Exception:
        return None

def parse_coef_name(name):
    m = re.match(r"C\((\w+)\)\[T\.(.+)\]", str(name))
    return (m.group(1), m.group(2)) if m else (str(name), None)

def coef_support(name, df_fit):
    col, level = parse_coef_name(name)
    if level is None: return len(df_fit)
    if col not in df_fit.columns: return 0
    return int((df_fit[col].astype(str) == str(level)).sum())


# ── 1. Parse gpt-4.1 batch output ─────────────────────────────────────────────
def parse_label(raw):
    if raw is None: return None
    for ch in str(raw).strip():
        if ch in ("0","1"): return int(ch)
    return None

print("Parsing gpt41_batch_output.jsonl...")
rows = []
with open("gpt41_batch_output.jsonl") as f:
    for line in f:
        item = json.loads(line)
        cid = item["custom_id"].replace("g41_", "", 1)  # strip prefix → maps to index
        try:
            choices = item["response"]["body"]["choices"]
            raw = choices[0]["message"]["content"] if choices else None
        except (KeyError, IndexError, TypeError):
            raw = None
        rows.append({"custom_id": cid, "llm_label": parse_label(raw)})
df_out = pd.DataFrame(rows)

df_idx = pd.DataFrame([json.loads(l) for l in open("batch_index.jsonl")])
df = df_out.merge(df_idx, on="custom_id", how="left")

n_fail = df["llm_label"].isna().sum()
print(f"  {len(df):,} rows | parse failures: {n_fail} ({n_fail/len(df)*100:.2f}%)")

df = df[df["llm_label"].notna()].copy()
df["llm_label"] = df["llm_label"].astype(int)
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["age_centered"] = df["age"] - df.groupby("language")["age"].transform("mean")
for col in ["gender","ethnicity","nationality","country_birth","country_residence",
            "employment_status","student_status","Generation"]:
    if col in df.columns:
        df[col] = df[col].replace({"unknown": np.nan, "nan": np.nan})
df.to_parquet("data_predictions_gpt41_clean.parquet", index=False)
print(f"  Saved data_predictions_gpt41_clean.parquet ({len(df):,} rows)")

# label means comparison
print("\n  Mean LLM label by condition (gpt-4.1):")
print(df.groupby(["condition"])["llm_label"].mean().round(3).to_string())

# ── 2. R²_m formula-matched for gpt-4.1 + within-sample human baseline ────────
print(f"\n{'═'*70}\n  R²_m (formula-matched) — gpt-4.1\n{'═'*70}")

# Within-sample human baseline (human_label is identical across models)
SH = {}
for lang in LANGUAGES:
    dfh = df[(df.language==lang) & (df.condition=="p0")].copy()
    for cond in CONDITIONS:
        hf = get_formula("human_label", lang, cond, dfh)
        SH[(lang,cond)] = fit_r2m(dfh, hf) if hf else np.nan

rows_r2m = []
for lang in LANGUAGES:
    for cond in CONDITIONS:
        dsub = df[(df.language==lang) & (df.condition==cond)].copy()
        f_llm = get_formula("llm_label", lang, cond, dsub)
        r2m = fit_r2m(dsub, f_llm) if f_llm else np.nan
        rows_r2m.append({"language":lang, "condition":cond,
                         "r2m_gpt41_pct": r2m*100 if not np.isnan(r2m) else np.nan})
df_r2m41 = pd.DataFrame(rows_r2m)

# Load mini baseline
df_mini = pd.read_csv("results_r2m_llm.csv")[["language","condition","r2m_llm_pct"]]
df_mini = df_mini.rename(columns={"r2m_llm_pct":"r2m_mini_pct"})

df_cmp = df_r2m41.merge(df_mini, on=["language","condition"], how="left")
df_cmp["sh_pct"] = [SH[(r.language,r.condition)]*100 if not np.isnan(SH[(r.language,r.condition)]) else np.nan
                    for r in df_cmp.itertuples()]
df_cmp["delta_pp"] = df_cmp["r2m_gpt41_pct"] - df_cmp["r2m_mini_pct"]
df_cmp.to_csv("results_gpt41_vs_mini.csv", index=False)

# Print best-condition comparison
print(f"\n  {'Lang':<5}{'Best':<5}{'mini%':>9}{'gpt4.1%':>9}{'Δpp':>8}{'SH%':>9}{'gpt4.1 %SH':>12}")
print("-"*60)
for lang in LANGUAGES:
    bc = BEST_COND[lang]
    r = df_cmp[(df_cmp.language==lang) & (df_cmp.condition==bc)].iloc[0]
    pct_sh = (r.r2m_gpt41_pct / r.sh_pct * 100) if (r.sh_pct and not np.isnan(r.sh_pct) and r.sh_pct>0) else np.nan
    print(f"  {lang.upper():<5}{bc.upper():<5}{r.r2m_mini_pct:>9.3f}{r.r2m_gpt41_pct:>9.3f}"
          f"{r.delta_pp:>+8.3f}{r.sh_pct:>9.3f}{pct_sh:>11.1f}%")

# ── 3. Directional concordance for gpt-4.1 ────────────────────────────────────
print(f"\n{'═'*70}\n  DIRECTIONAL CONCORDANCE — gpt-4.1 (best condition)\n{'═'*70}")
conc_rows = []
for lang in LANGUAGES:
    cond = BEST_COND[lang]
    dsub = df[(df.language==lang) & (df.condition==cond)].copy()
    f_llm = get_formula("llm_label", lang, cond, dsub)
    if f_llm is None: continue
    rhs = f_llm.split("~",1)[1].strip()
    f_hum = f"human_label ~ {rhs}"
    nc = list(dict.fromkeys(needed_cols(f_llm,"llm_label") + ["human_label"]))
    dfit = dsub[[c for c in nc if c in dsub.columns]].dropna()
    if len(dfit) < 50: continue
    fe_l = fit_fe(dfit, f_llm); fe_h = fit_fe(dfit, f_hum)
    if fe_l is None or fe_h is None: continue
    common = [i for i in fe_h.index if i in fe_l.index and coef_support(i,dfit) >= MIN_SUPPORT]
    if len(common) < 2: continue
    h = fe_h[common].values.astype(float); l = fe_l[common].values.astype(float)
    sign_agree = float(np.mean(np.sign(h)==np.sign(l)))
    rho = spearmanr(h,l)[0] if len(common)>=3 else np.nan
    conc_rows.append({"language":lang, "condition":cond, "n_coefs":len(common),
                      "sign_agreement":sign_agree, "spearman_rho":rho})
    print(f"  {lang.upper()} {cond.upper()}  n={len(common):2d}  "
          f"sign_agree={sign_agree*100:5.1f}%  rho={rho if not np.isnan(rho) else float('nan'):+.3f}")
df_conc41 = pd.DataFrame(conc_rows)
df_conc41.to_csv("results_gpt41_concordance.csv", index=False)

# ── 4. Figures ─────────────────────────────────────────────────────────────────
# Fig A: R²_m best-condition mini vs gpt-4.1, with human baseline
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(LANGUAGES)); w = 0.28
mini_vals = [df_cmp[(df_cmp.language==l)&(df_cmp.condition==BEST_COND[l])]["r2m_mini_pct"].values[0] for l in LANGUAGES]
g41_vals  = [df_cmp[(df_cmp.language==l)&(df_cmp.condition==BEST_COND[l])]["r2m_gpt41_pct"].values[0] for l in LANGUAGES]
sh_vals   = [SH[(l,BEST_COND[l])]*100 if not np.isnan(SH[(l,BEST_COND[l])]) else np.nan for l in LANGUAGES]
ax.bar(x-w, mini_vals, w, label="gpt-4.1-mini (baseline)", color="#aec6e8")
ax.bar(x,   g41_vals,  w, label="gpt-4.1 (full)", color="#1a5fa8")
ax.bar(x+w, sh_vals,   w, label="Within-sample human baseline", color="#27ae60", alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([f"{l.upper()}\n{BEST_COND[l].upper()}" for l in LANGUAGES], fontsize=9)
ax.set_ylabel("R²_m (%)")
ax.set_title("Approach B: R²_m by model — best condition per language\n"
             "Does scaling gpt-4.1-mini → gpt-4.1 change demographic replication?", fontsize=11)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("fig_gpt41_r2m_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
print("\nSaved: fig_gpt41_r2m_comparison.png")

# Fig B: directional concordance mini vs gpt-4.1 (sign agreement)
try:
    df_conc_mini = pd.read_csv("results_directional_concordance.csv")
    df_conc_mini = df_conc_mini[df_conc_mini["is_best_condition"]==True][["language","sign_agreement"]]
    df_conc_mini = df_conc_mini.rename(columns={"sign_agreement":"sign_mini"})
    dcomp = df_conc41.merge(df_conc_mini, on="language", how="left")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(len(dcomp)); w = 0.38
    ax.bar(xs-w/2, dcomp["sign_mini"]*100, w, label="gpt-4.1-mini", color="#aec6e8")
    ax.bar(xs+w/2, dcomp["sign_agreement"]*100, w, label="gpt-4.1", color="#1a5fa8")
    ax.axhline(50, color="gray", ls="--", lw=1, label="50% = chance")
    ax.set_xticks(xs); ax.set_xticklabels([f"{r.language.upper()}\n{r.condition.upper()}" for r in dcomp.itertuples()], fontsize=9)
    ax.set_ylabel("Sign agreement (%)")
    ax.set_title("Approach B: Directional concordance by model\n"
                 "Does scale improve human-aligned demographic direction?", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig("fig_gpt41_concordance_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
    print("Saved: fig_gpt41_concordance_comparison.png")
except Exception as e:
    print(f"(concordance comparison figure skipped: {e})")

print(f"\n{'═'*70}\n  DONE — gpt-4.1 analysis complete\n{'═'*70}")
