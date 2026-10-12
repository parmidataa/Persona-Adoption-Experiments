"""
103_split_half_reliability.py
=============================
Noise ceiling for directional concordance: how often do two disjoint halves
of the human annotators agree on the sign of a demographic coefficient, and
how does LLM-vs-human agreement compare under the same split?

Per language, N_SPLIT random splits of annotators into halves A and B:
  ceiling : sign(H_A) vs sign(H_B)          (human vs human)
  llm     : sign(H_A) vs sign(L_B)          (human vs LLM on B's rows/personas)
Coefficients come from OLS of the text-centred label (label minus the text
mean within the half) on the condition-matched formula, with the design
rank-pruned in formula order. A coefficient is compared only if its level is
held by at least MIN_ANN annotators in each half (age is always compared).

Samples:
  700-row best-condition samples (gpt-4.1-mini and Claude Opus 5 labels)
  full human corpus, M1 formula (data_<lang>.parquet), ceiling only

Writes only: results_split_half_reliability.csv
"""
import json
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import patsy

exec(open("99_rank_pruned_concordance.py").read().split("mini = prep")[0])

N_SPLIT = 200
MIN_ANN = 2
rng = np.random.default_rng(7)


def coefs(d, rhs, ycol, text_col):
    y = d[ycol] - d.groupby(text_col)[ycol].transform("mean")
    X = patsy.dmatrix(rhs, d, return_type="dataframe")
    Xp, _ = prune(X)
    beta, *_ = np.linalg.lstsq(Xp.values, y.values, rcond=None)
    return pd.Series(beta, index=Xp.columns).drop("Intercept", errors="ignore")


def ok_levels(names, d):
    keep = []
    for n in names:
        m = re.match(r"C\((\w+)\)\[T\.(.+)\]", str(n))
        if not m:
            keep.append(n)
        elif d.loc[d[m.group(1)].astype(str) == m.group(2), "annotator_id"].nunique() >= MIN_ANN:
            keep.append(n)
    return keep


def agree(a, b, names):
    if not names:
        return np.nan, 0
    s = np.sign(a[names].values) == np.sign(b[names].values)
    return s.mean(), len(names)


rows = []

for model, path in [("gpt-4.1-mini", "data_predictions_clean.parquet"),
                    ("opus5", "data_predictions_opus5_clean.parquet")]:
    df = prep(pd.read_parquet(path))
    for lang in LANGS:
        cond = BEST[lang]
        sub = df[(df.language == lang) & (df.condition == cond)]
        f = get_formula("llm_label", lang, cond, sub)
        rhs = f.split("~", 1)[1]
        nc = list(dict.fromkeys(needed_cols(f, "llm_label") + ["human_label", "annotator_id"]))
        d = sub[[c for c in nc if c in sub.columns]].dropna()
        anns = d.annotator_id.unique()
        cei, llm, nco = [], [], []
        for _ in range(N_SPLIT):
            perm = rng.permutation(anns)
            A = d[d.annotator_id.isin(perm[: len(anns) // 2])]
            B = d[d.annotator_id.isin(perm[len(anns) // 2:])]
            try:
                hA = coefs(A, rhs, "human_label", "post_id")
                hB = coefs(B, rhs, "human_label", "post_id")
                lB = coefs(B, rhs, "llm_label", "post_id")
            except Exception:
                continue
            common = [n for n in hA.index if n in hB.index and n in lB.index]
            common = [n for n in ok_levels(common, A) if n in ok_levels(common, B)]
            c1, n1 = agree(hA, hB, common)
            c2, _ = agree(hA, lB, common)
            if n1:
                cei.append(c1); llm.append(c2); nco.append(n1)
        cei, llm = np.array(cei), np.array(llm)
        rows.append({"sample": f"700-row best ({model})", "language": lang, "condition": cond,
                     "n_annotators": len(anns), "splits_used": len(cei),
                     "mean_coefs": np.mean(nco) if nco else np.nan,
                     "ceiling_mean": cei.mean(), "ceiling_lo": np.quantile(cei, .025),
                     "ceiling_hi": np.quantile(cei, .975),
                     "llm_mean": llm.mean(), "llm_lo": np.quantile(llm, .025),
                     "llm_hi": np.quantile(llm, .975),
                     "share_llm_below_ceiling": np.mean(llm < cei)})

meta = json.load(open("lang_metadata.json"))
for lang in LANGS:
    d = pd.read_parquet(f"data_{lang}.parquet")
    rhs = " + ".join(meta[lang]["usable_vars"])
    anns = d.annotator_id.unique()
    cei, nco = [], []
    for _ in range(N_SPLIT):
        perm = rng.permutation(anns)
        A = d[d.annotator_id.isin(perm[: len(anns) // 2])]
        B = d[d.annotator_id.isin(perm[len(anns) // 2:])]
        try:
            hA = coefs(A, rhs, "label", "text_id")
            hB = coefs(B, rhs, "label", "text_id")
        except Exception:
            continue
        common = [n for n in hA.index if n in hB.index]
        common = [n for n in ok_levels(common, A) if n in ok_levels(common, B)]
        c, n = agree(hA, hB, common)
        if n:
            cei.append(c); nco.append(n)
    cei = np.array(cei)
    rows.append({"sample": "full corpus (M1)", "language": lang, "condition": "M1",
                 "n_annotators": len(anns), "splits_used": len(cei),
                 "mean_coefs": np.mean(nco), "ceiling_mean": cei.mean(),
                 "ceiling_lo": np.quantile(cei, .025), "ceiling_hi": np.quantile(cei, .975)})

R = pd.DataFrame(rows)
R.to_csv("results_split_half_reliability.csv", index=False)

for s in R["sample"].unique():
    x = R[R["sample"] == s]
    print(f"\n=== {s} ===")
    print(f"{'lang':<5}{'cond':<5}{'ann':>4}{'coefs':>7}{'human-human':>22}{'human-LLM':>22}{'LLM<ceil':>10}")
    for r in x.itertuples():
        hh = f"{r.ceiling_mean*100:.0f}% [{r.ceiling_lo*100:.0f}, {r.ceiling_hi*100:.0f}]"
        hl = (f"{r.llm_mean*100:.0f}% [{r.llm_lo*100:.0f}, {r.llm_hi*100:.0f}]"
              if "llm_mean" in x.columns and not np.isnan(r.llm_mean) else "-")
        sh = (f"{r.share_llm_below_ceiling*100:.0f}%"
              if "share_llm_below_ceiling" in x.columns and not np.isnan(r.share_llm_below_ceiling) else "-")
        print(f"{r.language:<5}{r.condition:<5}{r.n_annotators:>4}{r.mean_coefs:>7.1f}{hh:>22}{hl:>22}{sh:>10}")
    w = x.mean_coefs
    print(f"coef-weighted mean: human-human {np.average(x.ceiling_mean, weights=w)*100:.1f}%"
          + (f" | human-LLM {np.average(x.llm_mean, weights=w)*100:.1f}%"
             if "llm_mean" in x.columns and x.llm_mean.notna().all() else ""))
