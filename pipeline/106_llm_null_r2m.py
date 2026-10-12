"""
106_llm_null_r2m.py
===================
Shuffled-profile null for the LLM R^2_m values (gpt-4.1-mini and Opus 5),
with the human baseline on the same rows for comparison.

For every language x condition: the formula of 29_rmm_analysis.py, ML,
post intercept. Whole demographic profiles are shuffled across annotators
(N_PERM times) and R^2_m is recomputed. The LLM label was generated from the
annotator's own profile (P1-P3), so a real persona effect should beat the
null; P0 has no persona in the prompt and serves as a sanity check.

Writes only: results_llm_null_r2m.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from fastlmm import prune_cols, shuffled_profile_null

N_PERM = 1000
rng = np.random.default_rng(11)
LANGS = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]
CONDS = ["p0", "p1", "p2", "p3"]

defs = {}
exec(open("29_rmm_analysis.py").read().split("# ── Load data")[0], defs)
CATS = ["gender", "ethnicity", "nationality", "country_birth", "country_residence",
        "employment_status", "student_status", "Generation"]


def load(path):
    d = pd.read_parquet(path)
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d["age_centered"] = d["age"] - d.groupby("language")["age"].transform("mean")
    for c in CATS:
        if c in d.columns:
            d[c] = d[c].replace("unknown", np.nan)
    return d


def cell(d, target, lang, cond):
    f = defs["get_formula"](target, lang, cond, d)
    if f is None:
        return None
    need = defs["needed_cols"](f, d, target) + ["annotator_id"]
    d = d[list(dict.fromkeys(need))].dropna().reset_index(drop=True)
    rhs = f.split("~")[1].strip()
    X = prune_cols(patsy.dmatrix(rhs, d, return_type="dataframe").values)
    y = d[target].values.astype(float)
    tc = pd.factorize(d["post_id"])[0]
    m = sm.MixedLM(y, X, groups=tc).fit(reml=False, method="bfgs", disp=False)
    return shuffled_profile_null(X, y, tc, pd.factorize(d["annotator_id"])[0],
                                 float(m.cov_re[0, 0]), m.scale, N_PERM, rng)


rep_mini = pd.read_csv("results_r2m_llm.csv").set_index(["language", "condition"])["r2m_llm"]
rep_opus = (pd.read_csv("results_opus5_vs_mini.csv")
            .set_index(["language", "condition"])["r2m_opus5_pct"] / 100)

rows = []
for model, path, rep in [("mini", "data_predictions_clean.parquet", rep_mini),
                         ("opus5", "data_predictions_opus5_clean.parquet", rep_opus)]:
    df = load(path)
    for lang in LANGS:
        for cond in CONDS:
            d = df[(df.language == lang) & (df.condition == cond)]
            for target, who in [("llm_label", model), ("human_label", "human")]:
                if who == "human" and cond == "p1":
                    continue
                try:
                    out = cell(d, target, lang, cond)
                except Exception as e:
                    print(f"{model} {lang} {cond} {who}: failed ({e})")
                    continue
                if out is None:
                    continue
                r = rep.get((lang, cond), np.nan) if who != "human" else np.nan
                rows.append({"sample": model, "rater": who, "language": lang,
                             "condition": cond, "r2m_reported": r, **out})
                print(f"{model:5s} {lang} {cond} {who:6s} obs={out['r2m']*100:6.2f}% "
                      f"rep={r*100:6.2f}% null={out['null_mean']*100:5.2f}% "
                      f"p95={out['null_p95']*100:5.2f}% p={out['perm_p']:.3f}")

pd.DataFrame(rows).to_csv("results_llm_null_r2m.csv", index=False)
print("Saved results_llm_null_r2m.csv")
