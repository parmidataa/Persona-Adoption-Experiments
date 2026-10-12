"""
104_null_r2m.py
===============
Shuffled-profile null for every reported R^2_m.

Demographics are annotator-level, so with few annotators any set of labels
attached to annotators absorbs some stable between-annotator variance. For
each R^2_m we shuffle whole demographic profiles across annotators
(N_PERM times), recompute R^2_m with the same formula and the same method,
and report the null mean, the null 95th percentile and the permutation p.

Families:
  phaseA : Table 3.2, M1 on data_<lang>.parquet (REML, text intercept).
  phaseD : M1-M4 on data_<lang>_v2.parquet with the terms actually used in
           18_extended_fixed_effects.py; LMM cells use the fitted variance
           components, OLS-fallback cells use the M1 total as denominator
           (as in script 18).
  sh     : Study 2 sample-human baselines, P0 slice of
           data_predictions_clean.parquet, condition formulas of
           29_rmm_analysis.py (ML, post intercept).

Under permutation the variance components are held at their observed values
(the fixed effects are then the exact GLS estimates); a spot check against
full refits is printed at the end.

Writes only: results_null_r2m.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

from fastlmm import TextGLS

N_PERM = 1000
rng = np.random.default_rng(7)
LANGS = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]


def null_cell(d, rhs, ycol, tcol, var_u, var_e, method="lmm", anchor=None):
    X = patsy.dmatrix(rhs, d, return_type="dataframe").values
    y = d[ycol].values.astype(float)
    tc = pd.factorize(d[tcol])[0]
    ac = pd.factorize(d["annotator_id"])[0]
    first = pd.Series(np.arange(len(d))).groupby(ac).first().values
    Aann = X[first]
    n_ann = len(first)
    k = np.linalg.matrix_rank(Aann)

    if method == "lmm":
        g = TextGLS(tc, var_u, var_e)
        yt = g.transform(y).ravel()
        stat = lambda M: (lambda vf: vf / (vf + var_u + var_e))(g.var_fixed(M, yt))
    else:
        def stat(M):
            b, *_ = np.linalg.lstsq(M, y, rcond=None)
            return float(np.var(M @ b)) / anchor

    obs = stat(X)
    null = np.array([stat(Aann[rng.permutation(n_ann)][ac]) for _ in range(N_PERM)])
    return {"n_annotators": n_ann, "params": k, "r2m": obs,
            "null_mean": null.mean(), "null_p95": np.quantile(null, 0.95),
            "perm_p": (np.sum(null >= obs) + 1) / (N_PERM + 1),
            "excess": obs - null.mean()}


rows = []

# ── Phase A, Table 3.2 ────────────────────────────────────────────────────────
r2a = pd.read_csv("results_r2_per_language.csv").set_index("language")
for lang in LANGS:
    f = r2a.loc[lang, "formula"]
    rhs = f.split("~")[1].strip()
    d = pd.read_parquet(f"data_{lang}.parquet")
    need = ["label", "text_id", "annotator_id"] + [c for c in d.columns if c in rhs]
    d = d[need].dropna().reset_index(drop=True)
    m = smf.mixedlm(f, d, groups=d["text_id"]).fit(reml=True, method="lbfgs")
    out = null_cell(d, rhs, "label", "text_id", float(m.cov_re.iloc[0, 0]), m.scale)
    rows.append({"family": "phaseA", "language": lang, "model": "M1",
                 "r2m_reported": r2a.loc[lang, "r2_marginal"], **out})
    print(f"phaseA {lang} M1  obs={out['r2m']*100:.2f}% rep={r2a.loc[lang,'r2_marginal']*100:.2f}% "
          f"null={out['null_mean']*100:.2f}% p95={out['null_p95']*100:.2f}% p={out['perm_p']:.3f}")

# ── Phase D, M1-M4 ───────────────────────────────────────────────────────────
fe = pd.read_csv("results_extended_fixed_effects.csv")
for lang in LANGS:
    d = pd.read_parquet(f"data_{lang}_v2.parquet")
    sub = fe[fe.language == lang].set_index("model")
    anchor = sub.loc["M1", ["var_fixed", "var_random", "var_residual"]].sum()
    for mname in ["M1", "M2", "M3", "M4"]:
        r = sub.loc[mname]
        rhs = " + ".join(t if t == "age_centered" else f"C({t})" for t in r.terms_used.split("+"))
        out = null_cell(d, rhs, "label", "text_id", r.var_random, r.var_residual,
                        method="lmm" if r.method == "lmm" else "ols", anchor=anchor)
        rows.append({"family": "phaseD", "language": lang, "model": mname,
                     "method": r.method, "r2m_reported": r.r2_marginal, **out})
        print(f"phaseD {lang} {mname}  obs={out['r2m']*100:.2f}% rep={r.r2_marginal*100:.2f}% "
              f"null={out['null_mean']*100:.2f}% p95={out['null_p95']*100:.2f}% "
              f"p={out['perm_p']:.3f}  k/n={out['params']}/{out['n_annotators']}")

# ── Study 2 sample-human baselines ──────────────────────────────────────────
src = open("29_rmm_analysis.py").read().split("# ── Load data")[0]
defs = {}
exec(src, defs)

df_all = pd.read_parquet("data_predictions_clean.parquet")
df_all["age"] = pd.to_numeric(df_all["age"], errors="coerce")
df_all["age_centered"] = df_all["age"] - df_all.groupby("language")["age"].transform("mean")
for c in ["gender", "ethnicity", "nationality", "country_birth", "country_residence",
          "employment_status", "student_status", "Generation"]:
    if c in df_all.columns:
        df_all[c] = df_all[c].replace("unknown", np.nan)

r2l = pd.read_csv("results_r2m_llm.csv").set_index(["language", "condition"])
for lang in LANGS:
    dh = df_all[(df_all.language == lang) & (df_all.condition == "p0")].copy()
    for cond in ["p0", "p2", "p3"]:
        f = defs["get_formula"]("human_label", lang, cond, dh)
        if f is None:
            continue
        need = defs["needed_cols"](f, dh, "human_label") + ["annotator_id"]
        d = dh[list(dict.fromkeys(need))].dropna().reset_index(drop=True)
        try:
            m = smf.mixedlm(f, d, groups=d["post_id"]).fit(reml=False, method="bfgs", disp=False)
        except Exception as e:
            print(f"sh {lang} {cond}: fit failed ({e})")
            continue
        rhs = f.split("~")[1].strip()
        out = null_cell(d, rhs, "human_label", "post_id", float(m.cov_re.iloc[0, 0]), m.scale)
        rep = r2l.loc[(lang, cond), "r2m_sample_human"]
        rows.append({"family": "sh", "language": lang,
                     "model": "p0/p1" if cond == "p0" else cond,
                     "r2m_reported": rep, **out})
        print(f"sh {lang} {cond}  obs={out['r2m']*100:.2f}% rep={rep*100:.2f}% "
              f"null={out['null_mean']*100:.2f}% p95={out['null_p95']*100:.2f}% "
              f"p={out['perm_p']:.3f}  k/n={out['params']}/{out['n_annotators']}")

R = pd.DataFrame(rows)
R.to_csv("results_null_r2m.csv", index=False)
print("\nSaved results_null_r2m.csv")
