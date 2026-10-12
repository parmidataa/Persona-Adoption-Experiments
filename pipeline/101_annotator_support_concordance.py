"""
101_annotator_support_concordance.py
====================================
Sign agreement with an annotator-based support filter, for gpt-4.1-mini and
Claude Opus 5 at each language's best condition.

Changes relative to 35_directional_concordance.py / 93_parse_opus5_results.py:
  - design is rank-pruned in formula order (as in 99_rank_pruned_concordance.py);
  - a coefficient (or a group, for the model-free check) is compared only if
    its categorical level is held by at least MIN_ANN distinct annotators
    (age, a continuous term, is always kept);
  - the model-free group-deviation check uses the same annotator filter.

Writes only: results_concordance_annotator_support.csv
"""
import warnings
warnings.filterwarnings("ignore")

import re
from math import comb
import numpy as np
import pandas as pd
import patsy

exec(open("99_rank_pruned_concordance.py").read().split("mini = prep")[0])

ANN_THRESHOLDS = [1, 3, 5]


def level_annotators(name, d):
    m = re.match(r"C\((\w+)\)\[T\.(.+)\]", str(name))
    if not m:
        return 10 ** 6
    col, lev = m.group(1), m.group(2)
    return d.loc[d[col].astype(str) == lev, "annotator_id"].nunique()


def groupdev(d, cols, min_ann):
    hm, lm = d.human_label.mean(), d.llm_label.mean()
    out = []
    for c in cols:
        for lev, g in d.groupby(c):
            if len(g) >= MIN_SUPPORT and g.annotator_id.nunique() >= min_ann:
                out.append((g.human_label.mean() - hm, g.llm_label.mean() - lm))
    if len(out) < 2:
        return np.nan, 0, 0
    a = np.array(out)
    same = np.sign(a[:, 0]) == np.sign(a[:, 1])
    return same.mean(), int(same.sum()), len(a)


def binom_p(k, n):
    if n == 0:
        return np.nan
    pk = comb(n, k)
    return sum(comb(n, i) for i in range(n + 1) if comb(n, i) <= pk) / 2 ** n


rows = []
for model, path in [("gpt-4.1-mini", "data_predictions_clean.parquet"),
                    ("opus5", "data_predictions_opus5_clean.parquet")]:
    df = prep(pd.read_parquet(path))
    for lang in LANGS:
        cond = BEST[lang]
        sub = df[(df.language == lang) & (df.condition == cond)]
        f = get_formula("llm_label", lang, cond, sub)
        nc = list(dict.fromkeys(needed_cols(f, "llm_label") + ["human_label", "annotator_id"]))
        d = sub[[c for c in nc if c in sub.columns]].dropna()
        X = patsy.dmatrix(f.split("~", 1)[1], d, return_type="dataframe")
        Xp, _ = prune(X)
        fh = fit_fe(d.human_label.values, Xp, d.post_id.values)
        fl = fit_fe(d.llm_label.values, Xp, d.post_id.values)
        cats = [c for c in needed_cols(f, "llm_label")
                if c not in ("llm_label", "post_id", "age_centered")]
        for t in ANN_THRESHOLDS:
            common = [c for c in fh.index
                      if coef_support(c, d) >= MIN_SUPPORT and level_annotators(c, d) >= t]
            same = np.sign(fh[common].values) == np.sign(fl[common].values)
            gs, gk, gn = groupdev(d, cats, t)
            rows.append({"model": model, "language": lang, "condition": cond,
                         "min_annotators": t, "n_annotators": d.annotator_id.nunique(),
                         "coef_k": int(same.sum()), "coef_n": len(common),
                         "coef_sign": same.mean() if len(common) else np.nan,
                         "gd_k": gk, "gd_n": gn, "gd_sign": gs})

R = pd.DataFrame(rows)
R.to_csv("results_concordance_annotator_support.csv", index=False)

for model in ["gpt-4.1-mini", "opus5"]:
    print(f"\n=== {model} (best condition) — coefficient sign | model-free group-deviation ===")
    print(f"{'lang':<5}{'cond':<5}{'ann':>4}" + "".join(f"{f'>= {t} ann':>26}" for t in ANN_THRESHOLDS))
    for lang in LANGS:
        rr = R[(R.model == model) & (R.language == lang)]
        cells = ""
        for t in ANN_THRESHOLDS:
            r = rr[rr.min_annotators == t].iloc[0]
            c = f"{r.coef_k}/{r.coef_n}" if r.coef_n else "-"
            g = f"{r.gd_k}/{r.gd_n}" if r.gd_n else "-"
            cells += f"{c + ' | ' + g:>26}"
        print(f"{lang:<5}{rr.condition.iloc[0]:<5}{rr.n_annotators.iloc[0]:>4}{cells}")
    for t in ANN_THRESHOLDS:
        s = R[(R.model == model) & (R.min_annotators == t)]
        ck, cn, gk, gn = s.coef_k.sum(), s.coef_n.sum(), s.gd_k.sum(), s.gd_n.sum()
        print(f"  pooled >= {t} ann: coef {ck}/{cn} = {ck/cn*100:.1f}% (p={binom_p(ck, cn):.3f}) | "
              f"group-dev {gk}/{gn} = {gk/gn*100:.1f}% (p={binom_p(gk, gn):.3f})")
