"""
102_annotator_level_retest.py
=============================
Re-tests the Study 1 significance claims with annotator clustering taken into
account. Demographics vary only between annotators, so the original tests
(text random intercept only, rows treated as independent) overstate the
evidence.

Families re-tested:
  (a) Phase D steps M1->M2, M2->M3, M3->M4 (complete-data sample,
      data_<lang>_v2.parquet). Nested sets are built cumulatively from the
      requested fields; fields constant within a language are skipped, and
      design columns are kept in formula order only if they add rank, so
      every step is nested by construction.
  (b) Phase A pairwise interactions (data_<lang>.parquet, M1 base), the same
      specifications as 05_interaction_effects.py.

Two methods per test:
  LRT_ann : y = label minus text mean (absorbs text effects); MixedLM with an
            annotator random intercept, ML; likelihood-ratio test on the
            rank difference.
  PERM_ann: annotator means of the text-centred label, weighted (rows) OLS;
            incremental F statistic; Freedman-Lane permutation (N_PERM).

Writes only:
  results_retest_phaseD_steps.csv
  results_retest_interactions.csv
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats

from fastlmm import nested_design

N_PERM = 2000
rng = np.random.default_rng(42)
LANGS = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]

STEP_FIELDS = {
    "M1": ["age_centered", "gender", "ethnicity", "nationality"],
    "M2": ["country_birth", "country_residence"],
    "M3": ["student_status", "employment_status"],
    "M4": ["generation"],
}


def prune_cols(X, n_base):
    keep = []
    for j in range(X.shape[1]):
        cand = keep + [j]
        if np.linalg.matrix_rank(X[:, cand]) == len(cand):
            keep = cand
    return X[:, keep], sum(1 for j in keep if j < n_base)


def lrt_ann(y, X_full, n_base, groups):
    Xf, kb = prune_cols(X_full, n_base)
    Xb = Xf[:, :kb]
    dfd = Xf.shape[1] - kb
    if dfd <= 0:
        return np.nan, 0, np.nan
    rb = sm.MixedLM(y, Xb, groups=groups).fit(reml=False, method="lbfgs")
    rf = sm.MixedLM(y, Xf, groups=groups).fit(reml=False, method="lbfgs")
    lr = max(2 * (rf.llf - rb.llf), 0.0)
    return lr, dfd, stats.chi2.sf(lr, dfd)


def wls_rss(X, y, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    fit = X @ beta
    return np.sum(w * (y - fit) ** 2), fit


def perm_ann(a_y, a_w, A_full, n_base):
    Af, kb = prune_cols(A_full, n_base)
    Ab = Af[:, :kb]
    n, kf = Af.shape
    dfd = kf - kb
    if dfd <= 0 or n - kf <= 0:
        return np.nan, dfd, np.nan, n
    rss_b, fit_b = wls_rss(Ab, a_y, a_w)
    rss_f, _ = wls_rss(Af, a_y, a_w)
    F = ((rss_b - rss_f) / dfd) / (rss_f / (n - kf))
    res_b = a_y - fit_b
    cnt = 0
    for _ in range(N_PERM):
        ys = fit_b + res_b[rng.permutation(n)]
        rb, _ = wls_rss(Ab, ys, a_w)
        rf, _ = wls_rss(Af, ys, a_w)
        if ((rb - rf) / dfd) / (rf / (n - kf)) >= F:
            cnt += 1
    return F, dfd, (cnt + 1) / (N_PERM + 1), n


def term(f):
    return f if f == "age_centered" else f"C({f})"


def prepare(d):
    d = d.copy()
    d["dev"] = d["label"] - d.groupby("text_id")["label"].transform("mean")
    return d


def annotator_table(d, cols):
    g = d.groupby("annotator_id")
    a = g[cols].first()
    a["y"] = g["dev"].mean()
    a["w"] = g.size()
    return a


# ── (a) Phase D steps ─────────────────────────────────────────────────────────
orig = pd.read_csv("results_extra_demographics_lrt.csv").set_index("language")
step_rows = []
for lang in LANGS:
    d = prepare(pd.read_parquet(f"data_{lang}_v2.parquet"))
    a_first = d.drop_duplicates("annotator_id")
    fields = {m: [f for f in fs if f in d.columns and
                  (f == "age_centered" or a_first[f].nunique() > 1)]
              for m, fs in STEP_FIELDS.items()}
    cum = []
    for prev, new in [("M1", "M2"), ("M2", "M3"), ("M3", "M4")]:
        base_f = cum or [f for m in ["M1", "M2", "M3", "M4"][:1] for f in fields[m]]
        if not cum:
            cum = list(fields["M1"])
        base_f = list(cum)
        full_f = base_f + fields[new]
        key = f"p_{prev}vs{new}"
        row = {"language": lang, "step": f"{prev}->{new}",
               "added_fields": "+".join(fields[new]) or "(none)",
               "p_original": orig.loc[lang, key] if key in orig.columns else np.nan}
        if not fields[new]:
            row.update({"lrt_ann_p": np.nan, "perm_ann_p": np.nan, "df": 0})
        else:
            rhs_b = " + ".join(term(f) for f in base_f)
            added = " + ".join(term(f) for f in fields[new])
            Xf, nb = nested_design(rhs_b, added, d)
            lr, dfd, p_lrt = lrt_ann(d["dev"].values, Xf, nb, d["annotator_id"].values)
            a = annotator_table(d, full_f)
            Af, nbA = nested_design(rhs_b, added, a)
            F, dfA, p_perm, n_ann = perm_ann(a["y"].values, a["w"].values, Af, nbA)
            row.update({"df": dfd, "lrt_ann_stat": lr, "lrt_ann_p": p_lrt,
                        "perm_F": F, "perm_df": dfA, "perm_ann_p": p_perm,
                        "n_annotators": n_ann})
        step_rows.append(row)
        cum = full_f
        print(f"[{lang}] {row['step']:7s} added={row['added_fields']:<38s} "
              f"orig p={row['p_original']:.2g}  LRT_ann p={row.get('lrt_ann_p', np.nan):.3g}  "
              f"PERM_ann p={row.get('perm_ann_p', np.nan):.3g}")

S = pd.DataFrame(step_rows)
S.to_csv("results_retest_phaseD_steps.csv", index=False)

# ── (b) Phase A interactions ──────────────────────────────────────────────────
meta = json.load(open("lang_metadata.json"))
specs = [("gender × ethnicity", "C(gender):C(ethnicity)"),
         ("age × gender", "age_centered:C(gender)"),
         ("age × ethnicity", "age_centered:C(ethnicity)")]
specs_nat = [("gender × nationality", "C(gender):C(nationality)"),
             ("age × nationality", "age_centered:C(nationality)"),
             ("ethnicity × nationality", "C(ethnicity):C(nationality)")]
orig_int = pd.read_csv("results_interaction_effects.csv")

int_rows = []
for lang in LANGS:
    d = prepare(pd.read_parquet(f"data_{lang}.parquet"))
    usable = meta[lang]["usable_vars"]
    rhs_b = " + ".join(usable)
    tests = specs + (specs_nat if "C(nationality)" in usable else [])
    cols = [c.replace("C(", "").replace(")", "") for c in usable]
    a = annotator_table(d, cols)
    for name, t in tests:
        o = orig_int[(orig_int.language == lang) & (orig_int.interaction == name)]
        if o.empty:
            continue
        Xf, nb = nested_design(rhs_b, t, d)
        lr, dfd, p_lrt = lrt_ann(d["dev"].values, Xf, nb, d["annotator_id"].values)
        Af, nbA = nested_design(rhs_b, t, a)
        F, dfA, p_perm, n_ann = perm_ann(a["y"].values, a["w"].values, Af, nbA)
        int_rows.append({"language": lang, "interaction": name,
                         "p_original": float(o.p_value.iloc[0]), "df_original": int(o.df.iloc[0]),
                         "df": dfd, "lrt_ann_p": p_lrt, "perm_df": dfA, "perm_ann_p": p_perm,
                         "n_annotators": n_ann})
        print(f"[{lang}] {name:24s} orig p={o.p_value.iloc[0]:.2g}  "
              f"LRT_ann p={p_lrt:.3g} (df {dfd})  PERM_ann p={p_perm:.3g} (df {dfA}, n={n_ann})")

I = pd.DataFrame(int_rows)
I.to_csv("results_retest_interactions.csv", index=False)

bonf33, bonf45 = 0.05 / 33, 0.05 / 45
print("\n=== Summary ===")
print(f"Phase D steps tested: {S.lrt_ann_p.notna().sum()}")
for col in ["p_original", "lrt_ann_p", "perm_ann_p"]:
    print(f"  {col:12s}: p<0.05 in {(S[col] < 0.05).sum()} | p<0.005 in {(S[col] < 0.005).sum()}")
print(f"Interactions tested: {len(I)}")
for col in ["p_original", "lrt_ann_p", "perm_ann_p"]:
    print(f"  {col:12s}: p<0.05 in {(I[col] < 0.05).sum()} | Bonferroni 0.05/33 in "
          f"{(I[col] < bonf33).sum()} | 0.05/45 in {(I[col] < bonf45).sum()}")
