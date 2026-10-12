"""
100_saturation_overfit_check.py
===============================
How much between-annotator variance would M1-M4 absorb by chance, given that
demographics are annotator-level and some languages have few annotators?

For each language (complete-data sample, data_<lang>_v2.parquet):
  1. Annotator effect = mean over the annotator's rows of (label - text mean).
  2. Weighted OLS of the annotator effect on the M1..M4 terms actually used in
     18_extended_fixed_effects.py (weights = rows per annotator).
  3. Share captured = weighted R^2; adjusted R^2 penalises parameters.
  4. Permutation null: shuffle whole demographic profiles across annotators
     (N_PERM times) and refit; report the null mean and 95th percentile.

Writes only: results_saturation_overfit_check.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

N_PERM = 1000
rng = np.random.default_rng(42)
LANGS = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]
fe = pd.read_csv("results_extended_fixed_effects.csv")


def design(a, terms):
    parts = [np.ones((len(a), 1))]
    for t in terms:
        if t == "age_centered":
            parts.append(a[[t]].astype(float).values)
        else:
            parts.append(pd.get_dummies(a[t].astype(str), prefix=t, drop_first=True).astype(float).values)
    return np.column_stack(parts)


def wr2(X, y, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    fit = X @ beta
    ybar = np.average(y, weights=w)
    return 1 - np.sum(w * (y - fit) ** 2) / np.sum(w * (y - ybar) ** 2)


rows = []
for lang in LANGS:
    d = pd.read_parquet(f"data_{lang}_v2.parquet")
    d["dev"] = d["label"] - d.groupby("text_id")["label"].transform("mean")
    g = d.groupby("annotator_id")
    a = g.first()
    a["eff"] = g["dev"].mean()
    a["w"] = g.size()
    n = len(a)
    for m in ["M1", "M2", "M3", "M4"]:
        terms = fe[(fe.language == lang) & (fe.model == m)].iloc[0].terms_used.split("+")
        X = design(a, terms)
        k = np.linalg.matrix_rank(X)
        r2 = wr2(X, a["eff"].values, a["w"].values)
        adj = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else np.nan
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            Xp = X[rng.permutation(n)]
            null[i] = wr2(Xp, a["eff"].values, a["w"].values)
        rows.append({"language": lang, "model": m, "n_annotators": n, "params": k,
                     "ratio": k / n, "r2_between": r2, "adj_r2_between": adj,
                     "null_mean": null.mean(), "null_p95": np.quantile(null, 0.95),
                     "perm_p": (np.sum(null >= r2) + 1) / (N_PERM + 1),
                     "excess_over_null": r2 - null.mean()})

R = pd.DataFrame(rows)
R.to_csv("results_saturation_overfit_check.csv", index=False)
sat = {}
print(f"{'lang':<5}{'mod':<4}{'n':>4}{'k':>4}{'R2':>8}{'adjR2':>8}{'null':>8}{'p95':>8}{'perm p':>8}")
for r in R.itertuples():
    print(f"{r.language:<5}{r.model:<4}{r.n_annotators:>4}{r.params:>4}{r.r2_between*100:>7.1f}%"
          f"{r.adj_r2_between*100:>7.1f}%{r.null_mean*100:>7.1f}%{r.null_p95*100:>7.1f}%{r.perm_p:>8.3f}")
