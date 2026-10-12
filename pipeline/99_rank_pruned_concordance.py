"""
99_rank_pruned_concordance.py
=============================
Re-runs coefficient-based sign agreement with an identifiable (full-rank)
design, for gpt-4.1-mini (all conditions) and Claude Opus 5 (best condition).

Formulas, rows, support filter, and fitting method are identical to
35_directional_concordance.py / 93_parse_opus5_results.py. The only change:
design columns are taken in formula order and a column is kept only if it
increases the rank of the columns kept so far. Dropped columns are linearly
dependent on earlier ones, so fitted values (and R^2_m) are unchanged.

Writes only new files:
    results_concordance_pruned_summary.csv
    results_concordance_pruned_coefficients.csv
"""
import warnings
warnings.filterwarnings("ignore")

from math import comb
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

src = open("35_directional_concordance.py").read().split("# ── Load data")[0]
ns = {}
exec(src, ns)
get_formula, needed_cols = ns["get_formula"], ns["needed_cols"]
coef_support, MIN_SUPPORT = ns["coef_support"], ns["MIN_SUPPORT"]

LANGS = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]
CONDS = ["p0", "p1", "p2", "p3"]
BEST = {"ar": "p2", "de": "p3", "en": "p2", "es": "p2", "fr": "p2",
        "hi": "p2", "it": "p3", "nl": "p2", "pt": "p1"}
DEMO = ["gender", "ethnicity", "nationality", "country_birth",
        "country_residence", "employment_status", "student_status", "Generation"]


def prep(df):
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_centered"] = df["age"] - df.groupby("language")["age"].transform("mean")
    for c in DEMO:
        if c in df.columns:
            df[c] = df[c].replace({"unknown": np.nan, "nan": np.nan})
    return df


def prune(X):
    keep = []
    for j in range(X.shape[1]):
        cand = keep + [j]
        if np.linalg.matrix_rank(X.values[:, cand]) == len(cand):
            keep = cand
    return X.iloc[:, keep], [X.columns[j] for j in range(X.shape[1]) if j not in keep]


def fit_fe(y, X, groups):
    try:
        m = sm.MixedLM(y, X, groups=groups).fit(reml=False, method="bfgs", disp=False)
        return pd.Series(np.asarray(m.fe_params), index=X.columns).drop("Intercept")
    except Exception:
        return None


def binom_p(k, n):
    pk = comb(n, k)
    return sum(comb(n, i) for i in range(n + 1) if comb(n, i) <= pk) / 2 ** n


def run(df, model, langs_conds):
    summ, coefs = [], []
    for lang, cond in langs_conds:
        sub = df[(df.language == lang) & (df.condition == cond)]
        f = get_formula("llm_label", lang, cond, sub)
        if f is None:
            continue
        nc = list(dict.fromkeys(needed_cols(f, "llm_label") + ["human_label"]))
        d = sub[[c for c in nc if c in sub.columns]].dropna()
        if len(d) < 50:
            continue
        X = patsy.dmatrix(f.split("~", 1)[1], d, return_type="dataframe")
        Xp, dropped = prune(X)
        fh = fit_fe(d["human_label"].values, Xp, d["post_id"].values)
        fl = fit_fe(d["llm_label"].values, Xp, d["post_id"].values)
        if fh is None or fl is None:
            continue
        common = [c for c in fh.index if coef_support(c, d) >= MIN_SUPPORT]
        if len(common) < 2:
            continue
        h, l = fh[common].values, fl[common].values
        same = np.sign(h) == np.sign(l)
        summ.append({"model": model, "language": lang, "condition": cond,
                     "is_best": BEST[lang] == cond, "n_rows": len(d),
                     "cols": X.shape[1], "rank": Xp.shape[1], "n_dropped_cols": len(dropped),
                     "n_coefs": len(common), "n_agree": int(same.sum()),
                     "sign_agreement": same.mean(),
                     "max_abs_human": np.abs(h).max(), "max_abs_llm": np.abs(l).max(),
                     "dropped_cols": "; ".join(dropped)})
        for c, hv, lv, s in zip(common, h, l, same):
            coefs.append({"model": model, "language": lang, "condition": cond,
                          "coef": c, "human": hv, "llm": lv, "same_sign": bool(s)})
    return summ, coefs


mini = prep(pd.read_parquet("data_predictions_clean.parquet"))
opus = prep(pd.read_parquet("data_predictions_opus5_clean.parquet"))

s1, c1 = run(mini, "gpt-4.1-mini", [(l, c) for l in LANGS for c in CONDS])
s2, c2 = run(opus, "opus5", [(l, BEST[l]) for l in LANGS])
S = pd.DataFrame(s1 + s2)
pd.DataFrame(c1 + c2).to_csv("results_concordance_pruned_coefficients.csv", index=False)
S.to_csv("results_concordance_pruned_summary.csv", index=False)

old_mini = pd.read_csv("results_directional_concordance.csv").set_index(["language", "condition"])
old_opus = pd.read_csv("results_opus5_concordance.csv").set_index(["language", "condition"])

for model, old in [("gpt-4.1-mini", old_mini), ("opus5", old_opus)]:
    B = S[(S.model == model) & S.is_best]
    print(f"\n=== {model}: best condition per language ===")
    print(f"{'lang':<5}{'cond':<5}{'rank/cols':>10}{'old':>9}{'new':>14}{'max|coef| h/l':>18}")
    for r in B.itertuples():
        o = old.loc[(r.language, r.condition), "sign_agreement"] * 100
        print(f"{r.language:<5}{r.condition:<5}{f'{r.rank}/{r.cols}':>10}{o:>8.1f}%"
              f"{f'{r.n_agree}/{r.n_coefs} = {r.sign_agreement*100:.1f}%':>14}"
              f"{f'{r.max_abs_human:.2f}/{r.max_abs_llm:.2f}':>18}")
    k, n = int(B.n_agree.sum()), int(B.n_coefs.sum())
    print(f"pooled: {k}/{n} = {k/n*100:.1f}%, exact binomial p = {binom_p(k, n):.3f}; "
          f"mean of language percentages = {B.sign_agreement.mean()*100:.1f}%")
    for r in B.itertuples():
        print(f"  {r.language} per-language binomial p = {binom_p(r.n_agree, r.n_coefs):.3f}")

print("\n=== gpt-4.1-mini, Arabic and Spanish, all conditions ===")
for r in S[(S.model == "gpt-4.1-mini") & S.language.isin(["ar", "es"])].itertuples():
    o = old_mini.loc[(r.language, r.condition), "sign_agreement"] * 100
    print(f"{r.language} {r.condition}: old {o:.1f}%  new {r.n_agree}/{r.n_coefs} = {r.sign_agreement*100:.1f}%")
