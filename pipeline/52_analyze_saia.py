"""
52_analyze_saia.py — cross-model analysis on the SAIA best-condition subset
============================================================================
Compares four models on the IDENTICAL 900-row best-condition subset
(9 languages x best cond x n=100):

    gpt-4.1-mini   (OpenAI, closed, small)   — re-scored on the subset
    gpt-4.1        (OpenAI, closed, large)   — re-scored on the subset
    llama-3.1-8b   (Meta, open, 8B)          — SAIA
    mistral-large-3 (Mistral, open, 675B)    — SAIA

Everything reuses the SAME condition-matched formula infrastructure and the SAME
directional-concordance method as 43_analyze_gpt41.py / 35_directional_concordance.py,
so results are directly comparable to the main experiment.

For each model x language (best condition only):
  - R²_m (formula-matched), and % of the within-sample human baseline (H-baseline)
  - directional concordance: coefficient sign-agreement + Spearman rho vs humans

Because the subset is identical across models, the human baseline (H-baseline) is
computed once on these rows and shared.

Outputs:
    saia_master_r2m.csv            — model x lang R²_m matrix (+ H-baseline, %SH)
    saia_master_concordance.csv    — model x lang sign-agreement / rho
    fig_saia_r2m_matrix.png        — grouped bars, R²_m by model per language
    fig_saia_concordance_matrix.png
    fig_saia_scale_family.png      — small-vs-large / open-vs-closed summary
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

LANGUAGES = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]
BEST_COND = {"ar":"p2","de":"p3","en":"p2","es":"p2","fr":"p2",
             "hi":"p2","it":"p3","nl":"p2","pt":"p1"}

# ── Formula infrastructure (identical to 43/35) ──────────────────────────────
M1_BASE = {
    "ar":["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
    "de":["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
    "en":["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
    "es":["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
    "fr":["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
    "hi":["age_centered","C(gender)","C(ethnicity)"],
    "it":["age_centered","C(gender)","C(ethnicity)"],
    "nl":["age_centered","C(gender)","C(ethnicity)"],
    "pt":["age_centered","C(gender)","C(ethnicity)","C(nationality)"],
}
M4_EXTRAS = ["C(country_birth)","C(country_residence)",
             "C(employment_status)","C(student_status)"]
P3_TEMPLATE = {
    "ar":["age_centered","C(nationality)"],
    "de":["C(Generation)","C(employment_status)"],
    "en":["C(nationality)","C(gender)","C(country_residence)"],
    "es":["age_centered","C(country_birth)","C(country_residence)"],
    "fr":["age_centered","C(ethnicity)","C(nationality)","C(country_residence)"],
    "hi":["age_centered","C(ethnicity)","C(country_birth)"],
    "it":["age_centered","C(employment_status)"],
    "nl":["C(employment_status)"],
    "pt":["C(employment_status)","C(country_birth)"],
}
ALL_PRED_COLS = ["age_centered","gender","ethnicity","nationality","country_birth",
                 "country_residence","employment_status","student_status","Generation"]
MIN_SUPPORT = 12   # slightly lower than main (n=100 subset) but still guards sparsity

def col_from_term(t):
    m = re.match(r"C\((\w+)\)", t); return m.group(1) if m else t

def build_formula(target, var_list, df_check):
    valid=[]
    for term in var_list:
        col=col_from_term(term)
        if col not in df_check.columns: continue
        if col=="age_centered": valid.append(term); continue
        if df_check[col].dropna().nunique()>=2: valid.append(term)
    return (f"{target} ~ "+" + ".join(valid)) if valid else None

# NOTE ON FORMULA CHOICE (n=100 subset):
# The main experiment used rich per-condition formulas on 700 rows. At n=100 the
# full P2/M4 formula needs 40-50+ dummy columns (e.g. AR nationality=15 +
# country_birth=15 + country_residence=13) and has missingness in
# employment/student status -> the design matrix goes singular for most languages
# and even for the human baseline. That is an artefact of sample size, not model
# quality. For a FAIR, STABLE cross-model comparison we therefore use a REDUCED,
# UNIFORM "core demographic" formula (age + gender + ethnicity: low cardinality,
# no missingness) applied IDENTICALLY to all four models AND the human baseline.
# Absolute values are thus not comparable to the 700-row main numbers, but the
# cross-model comparison (the whole point of this sub-study) is valid.
CORE_TERMS = ["age_centered", "C(gender)", "C(ethnicity)"]

def get_formula(target, lang, cond, df_check):
    return build_formula(target, CORE_TERMS, df_check)

def needed_cols(formula, target):
    cols=[target,"post_id"]
    for col in ALL_PRED_COLS:
        if col in formula: cols.append(col)
    return list(dict.fromkeys(cols))

def fit_r2m(df_sub, formula):
    target=formula.split("~")[0].strip()
    nc=[c for c in needed_cols(formula,target) if c in df_sub.columns]
    df_fit=df_sub[nc].dropna()
    if len(df_fit)<40 or df_fit[target].nunique()<2: return np.nan
    for meth in ("lbfgs","bfgs","cg","powell"):
        try:
            m=smf.mixedlm(formula,data=df_fit,groups=df_fit["post_id"]).fit(
                reml=False,method=meth,disp=False)
            fp=np.dot(m.model.exog,m.fe_params)
            vf=np.var(fp); vr=float(m.cov_re.iloc[0,0])
            return vf/(vf+vr+m.scale)
        except Exception:
            continue
    return np.nan

def fit_fe(df_fit, formula):
    for meth in ("lbfgs","bfgs","cg"):
        try:
            m=smf.mixedlm(formula,data=df_fit,groups=df_fit["post_id"]).fit(
                reml=False,method=meth,disp=False)
            fe=m.fe_params.copy()
            return fe[[i for i in fe.index if "Intercept" not in str(i)]]
        except Exception:
            continue
    return None

def parse_coef_name(name):
    m=re.match(r"C\((\w+)\)\[T\.(.+)\]",str(name))
    return (m.group(1),m.group(2)) if m else (str(name),None)

def coef_support(name, df_fit):
    col,level=parse_coef_name(name)
    if level is None: return len(df_fit)
    if col not in df_fit.columns: return 0
    return int((df_fit[col].astype(str)==str(level)).sum())

# ── Load subset ids + index (demographics) ───────────────────────────────────
subset_ids=set(json.load(open("saia_bestcond_ids.json")))
idx=pd.DataFrame([json.loads(l) for l in open("saia_bestcond_index.jsonl")])

def prep(df):
    """attach age_centered + NA cleaning, restricted to subset rows."""
    df=df[df["custom_id"].isin(subset_ids)].copy()
    df["age"]=pd.to_numeric(df["age"],errors="coerce")
    df["age_centered"]=df["age"]-df.groupby("language")["age"].transform("mean")
    for col in ["gender","ethnicity","nationality","country_birth","country_residence",
                "employment_status","student_status","Generation"]:
        if col in df.columns:
            df[col]=df[col].replace({"unknown":np.nan,"nan":np.nan})
    return df

# OpenAI models (re-scored on subset)
mini = prep(pd.read_parquet("data_predictions_clean.parquet"))
g41  = prep(pd.read_parquet("data_predictions_gpt41_clean.parquet"))

# SAIA models: merge label onto index
def load_saia(path):
    out=pd.DataFrame([json.loads(l) for l in open(path)])
    out=out[["custom_id","label"]].rename(columns={"label":"llm_label"})
    df=idx.merge(out,on="custom_id",how="inner")
    df=df[df["llm_label"].notna()].copy()
    df["llm_label"]=df["llm_label"].astype(int)
    return prep(df)

llama   = load_saia("saia_out_meta-llama-3_1-8b-instruct.jsonl")
mistral = load_saia("saia_out_mistral-large-3-675b-instruct-2512.jsonl")

MODELS = {
    "gpt-4.1-mini": mini,
    "gpt-4.1":      g41,
    "llama-3.1-8b": llama,
    "mistral-large-3": mistral,
}
print("Rows per model (subset, best cond only counts in analysis):")
for k,v in MODELS.items():
    print(f"  {k:<18} {len(v):,} rows")

# ── Human baseline (shared; from mini which has identical human_label) ────────
# Use whichever model has the row; human_label is identical across models.
HB={}
for lang in LANGUAGES:
    cond=BEST_COND[lang]
    dfh=mini[(mini.language==lang)&(mini.condition==cond)].copy()
    hf=get_formula("human_label",lang,cond,dfh)
    HB[lang]=fit_r2m(dfh,hf)*100 if hf else np.nan

# ── R²_m per model x language (best cond) ─────────────────────────────────────
r2m_rows=[]
for mname,df in MODELS.items():
    for lang in LANGUAGES:
        cond=BEST_COND[lang]
        dsub=df[(df.language==lang)&(df.condition==cond)].copy()
        f=get_formula("llm_label",lang,cond,dsub)
        r2m=fit_r2m(dsub,f)*100 if f else np.nan
        r2m_rows.append({"model":mname,"language":lang,"condition":cond,
                         "r2m_pct":r2m,"hbaseline_pct":HB[lang],
                         "pct_of_baseline":(r2m/HB[lang]*100) if (HB[lang] and not np.isnan(HB[lang]) and HB[lang]>0 and not np.isnan(r2m)) else np.nan})
df_r2m=pd.DataFrame(r2m_rows)
df_r2m.to_csv("saia_master_r2m.csv",index=False)

print(f"\n{'='*78}\n  R²_m (%) — best condition per language, on identical 900-row subset\n{'='*78}")
piv=df_r2m.pivot(index="language",columns="model",values="r2m_pct")
piv=piv[["gpt-4.1-mini","gpt-4.1","llama-3.1-8b","mistral-large-3"]]
piv["H-baseline"]=[HB[l] for l in piv.index]
print(piv.round(3).to_string())

# ── Directional concordance per model x language ──────────────────────────────
conc_rows=[]
for mname,df in MODELS.items():
    for lang in LANGUAGES:
        cond=BEST_COND[lang]
        dsub=df[(df.language==lang)&(df.condition==cond)].copy()
        f_llm=get_formula("llm_label",lang,cond,dsub)
        if f_llm is None: continue
        rhs=f_llm.split("~",1)[1].strip()
        f_hum=f"human_label ~ {rhs}"
        nc=list(dict.fromkeys(needed_cols(f_llm,"llm_label")+["human_label"]))
        dfit=dsub[[c for c in nc if c in dsub.columns]].dropna()
        if len(dfit)<40: continue
        fe_l=fit_fe(dfit,f_llm); fe_h=fit_fe(dfit,f_hum)
        if fe_l is None or fe_h is None: continue
        common=[i for i in fe_h.index if i in fe_l.index and coef_support(i,dfit)>=MIN_SUPPORT]
        if len(common)<2: continue
        h=fe_h[common].values.astype(float); l=fe_l[common].values.astype(float)
        sign=float(np.mean(np.sign(h)==np.sign(l)))
        rho=spearmanr(h,l)[0] if len(common)>=3 else np.nan
        conc_rows.append({"model":mname,"language":lang,"condition":cond,
                          "n_coefs":len(common),"sign_agreement":sign,"spearman_rho":rho})
df_conc=pd.DataFrame(conc_rows)
df_conc.to_csv("saia_master_concordance.csv",index=False)

print(f"\n{'='*78}\n  Directional concordance — coefficient sign-agreement (%)\n{'='*78}")
pivc=df_conc.pivot(index="language",columns="model",values="sign_agreement")*100
pivc=pivc.reindex(columns=[c for c in ["gpt-4.1-mini","gpt-4.1","llama-3.1-8b","mistral-large-3"] if c in pivc.columns])
print(pivc.round(1).to_string())

# ── Figures ───────────────────────────────────────────────────────────────────
MODEL_ORDER=["gpt-4.1-mini","gpt-4.1","llama-3.1-8b","mistral-large-3"]
COLORS={"gpt-4.1-mini":"#aec6e8","gpt-4.1":"#1a5fa8",
        "llama-3.1-8b":"#e8a598","mistral-large-3":"#c0392b"}

# Fig 1: R²_m matrix
fig,ax=plt.subplots(figsize=(13,5.8))
x=np.arange(len(LANGUAGES)); w=0.18
for i,m in enumerate(MODEL_ORDER):
    vals=[df_r2m[(df_r2m.model==m)&(df_r2m.language==l)]["r2m_pct"].values[0] for l in LANGUAGES]
    ax.bar(x+(i-1.5)*w,vals,w,label=m,color=COLORS[m])
ax.plot(x,[HB[l] for l in LANGUAGES],"D",color="#27ae60",ms=8,label="Within-sample human baseline",zorder=5)
ax.set_xticks(x); ax.set_xticklabels([f"{l.upper()}\n{BEST_COND[l].upper()}" for l in LANGUAGES])
ax.set_ylabel("R²_m (%)")
ax.set_title("Cross-model R²_m on identical 900-row subset (best condition per language)\n"
             "OpenAI (closed) vs open models — does demographic replication depend on model/family?",fontsize=11)
ax.legend(fontsize=9,ncol=2); ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); plt.savefig("fig_saia_r2m_matrix.png",dpi=150,bbox_inches="tight"); plt.close()
print("\nSaved fig_saia_r2m_matrix.png")

# Fig 2: concordance matrix
fig,ax=plt.subplots(figsize=(13,5.8))
langs_c=[l for l in LANGUAGES if l in pivc.index]
x=np.arange(len(langs_c))
present=[m for m in MODEL_ORDER if m in pivc.columns]; w=0.8/len(present)
for i,m in enumerate(present):
    vals=[pivc.loc[l,m] if l in pivc.index else np.nan for l in langs_c]
    ax.bar(x+(i-(len(present)-1)/2)*w,vals,w,label=m,color=COLORS[m])
ax.axhline(50,color="gray",ls="--",lw=1,label="50% = chance")
ax.set_xticks(x); ax.set_xticklabels([f"{l.upper()}\n{BEST_COND[l].upper()}" for l in langs_c])
ax.set_ylabel("Coefficient sign-agreement (%)")
ax.set_title("Cross-model directional concordance (best condition)\n"
             "Do any models reproduce the DIRECTION of human demographic effects?",fontsize=11)
ax.legend(fontsize=9,ncol=2); ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); plt.savefig("fig_saia_concordance_matrix.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved fig_saia_concordance_matrix.png")

# Fig 3: summary — mean raw R²_m and mean sign-agreement per model
# (%-of-baseline is unstable at n=100 because several human baselines are tiny;
#  raw R²_m on the identical formula/rows is the fair magnitude metric.)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
means_r2m={m:np.nanmean(df_r2m[df_r2m.model==m]["r2m_pct"]) for m in MODEL_ORDER}
means_sign={m:np.nanmean(df_conc[df_conc.model==m]["sign_agreement"])*100 for m in MODEL_ORDER}
hb_mean=np.nanmean(list(HB.values()))
ax1.bar(range(len(MODEL_ORDER)),[means_r2m[m] for m in MODEL_ORDER],color=[COLORS[m] for m in MODEL_ORDER])
ax1.axhline(hb_mean,color="#27ae60",ls="--",lw=1.5,label=f"human baseline ({hb_mean:.1f}%)")
ax1.set_xticks(range(len(MODEL_ORDER))); ax1.set_xticklabels(MODEL_ORDER,rotation=20,ha="right",fontsize=9)
ax1.set_ylabel("Mean R²_m (%) across 9 languages"); ax1.set_title("Magnitude: demographic variance",fontsize=10)
ax1.legend(fontsize=9); ax1.grid(axis="y",alpha=0.3)
ax2.bar(range(len(MODEL_ORDER)),[means_sign[m] for m in MODEL_ORDER],color=[COLORS[m] for m in MODEL_ORDER])
ax2.axhline(50,color="gray",ls="--",lw=1,label="chance")
ax2.set_xticks(range(len(MODEL_ORDER))); ax2.set_xticklabels(MODEL_ORDER,rotation=20,ha="right",fontsize=9)
ax2.set_ylabel("Mean sign-agreement (%)"); ax2.set_title("Direction: human-aligned demographics",fontsize=10)
ax2.legend(fontsize=9); ax2.grid(axis="y",alpha=0.3)
plt.suptitle("Cross-model summary — averaged across 9 languages (identical subset)",fontsize=11)
plt.tight_layout(); plt.savefig("fig_saia_scale_family.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved fig_saia_scale_family.png")

print(f"\n{'='*78}\n  SUMMARY\n{'='*78}")
print(f"  Human baseline mean R²_m across 9 langs: {hb_mean:.2f}%")
print("  Mean raw R²_m (%) across 9 langs:")
for m in MODEL_ORDER: print(f"    {m:<18} {means_r2m[m]:5.2f}%")
print("  Mean directional sign-agreement (50% = chance):")
for m in MODEL_ORDER: print(f"    {m:<18} {means_sign[m]:5.1f}%")
print("\nDONE — SAIA cross-model analysis complete")
