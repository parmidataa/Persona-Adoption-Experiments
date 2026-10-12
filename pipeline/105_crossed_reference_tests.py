"""
105_crossed_reference_tests.py
==============================
Reference significance tests with crossed random intercepts for text and
annotator: label ~ X + (1|text_id) + (1|annotator_id), maximum likelihood,
likelihood-ratio test on the rank difference. Same nested steps and the same
interaction specifications as 102_annotator_level_retest.py.

Also refits the persona-ceiling decomposition (M1 terms, as in
17_crossed_random_effects.py) with the exact ML fitter, because the
statsmodels crossed fits can stop before the optimum.

Writes only:
  results_crossed_lrt_phaseD_steps.csv
  results_crossed_lrt_interactions.csv
  results_crossed_ceiling_check.csv
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from fastlmm import CrossedML, nested_design, prune_cols

LANGS = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]
STEP_FIELDS = {
    "M1": ["age_centered", "gender", "ethnicity", "nationality"],
    "M2": ["country_birth", "country_residence"],
    "M3": ["student_status", "employment_status"],
    "M4": ["generation"],
}


def term(f):
    return f if f == "age_centered" else f"C({f})"


def crossed_lrt(cm, d, rhs_b, added):
    Xf_raw, nb = nested_design(rhs_b, added, d)
    Xf, kb = prune_cols(Xf_raw, nb)
    dfd = Xf.shape[1] - kb
    if dfd <= 0:
        return np.nan, 0, np.nan, None
    rb = cm.fit(Xf[:, :kb])
    rf = cm.fit(Xf, start=tuple(rb["lg"]))
    lr = max(2 * (rf["llf"] - rb["llf"]), 0.0)
    return lr, dfd, stats.chi2.sf(lr, dfd), rf


def model_for(d):
    return CrossedML(d["label"].values, pd.factorize(d["text_id"])[0],
                     pd.factorize(d["annotator_id"])[0])


# ── Phase D steps ────────────────────────────────────────────────────────────
prev = pd.read_csv("results_retest_phaseD_steps.csv").set_index(["language", "step"])
step_rows = []
for lang in LANGS:
    d = pd.read_parquet(f"data_{lang}_v2.parquet")
    cm = model_for(d)
    a_first = d.drop_duplicates("annotator_id")
    fields = {m: [f for f in fs if f in d.columns and
                  (f == "age_centered" or a_first[f].nunique() > 1)]
              for m, fs in STEP_FIELDS.items()}
    cum = list(fields["M1"])
    for p_, n_ in [("M1", "M2"), ("M2", "M3"), ("M3", "M4")]:
        step = f"{p_}->{n_}"
        full = cum + fields[n_]
        row = {"language": lang, "step": step,
               "added_fields": "+".join(fields[n_]) or "(none)",
               "p_original": prev.loc[(lang, step), "p_original"],
               "lrt_ann_p": prev.loc[(lang, step), "lrt_ann_p"],
               "perm_ann_p": prev.loc[(lang, step), "perm_ann_p"]}
        if fields[n_]:
            lr, dfd, p, _ = crossed_lrt(cm, d, " + ".join(map(term, cum)),
                                        " + ".join(map(term, fields[n_])))
            row.update({"df": dfd, "crossed_lr": lr, "crossed_p": p})
        else:
            row.update({"df": 0, "crossed_lr": np.nan, "crossed_p": np.nan})
        step_rows.append(row)
        cum = full
        print(f"[{lang}] {step:7s} {row['added_fields']:<36s} orig={row['p_original']:.2g} "
              f"crossed={row['crossed_p']:.3g} LRT_ann={row['lrt_ann_p']:.3g} PERM_ann={row['perm_ann_p']:.3g}")
pd.DataFrame(step_rows).to_csv("results_crossed_lrt_phaseD_steps.csv", index=False)

# ── Phase A interactions ─────────────────────────────────────────────────────
meta = json.load(open("lang_metadata.json"))
specs = [("gender × ethnicity", "C(gender):C(ethnicity)"),
         ("age × gender", "age_centered:C(gender)"),
         ("age × ethnicity", "age_centered:C(ethnicity)")]
specs_nat = [("gender × nationality", "C(gender):C(nationality)"),
             ("age × nationality", "age_centered:C(nationality)"),
             ("ethnicity × nationality", "C(ethnicity):C(nationality)")]
prev_i = pd.read_csv("results_retest_interactions.csv").set_index(["language", "interaction"])
int_rows = []
for lang in LANGS:
    d = pd.read_parquet(f"data_{lang}.parquet")
    usable = meta[lang]["usable_vars"]
    cols = [c.replace("C(", "").replace(")", "") for c in usable]
    d = d[["label", "text_id", "annotator_id"] + cols].dropna().reset_index(drop=True)
    cm = model_for(d)
    rhs_b = " + ".join(usable)
    tests = specs + (specs_nat if "C(nationality)" in usable else [])
    for name, t in tests:
        if (lang, name) not in prev_i.index:
            continue
        lr, dfd, p, _ = crossed_lrt(cm, d, rhs_b, t)
        pr = prev_i.loc[(lang, name)]
        int_rows.append({"language": lang, "interaction": name,
                         "p_original": pr.p_original, "lrt_ann_p": pr.lrt_ann_p,
                         "perm_ann_p": pr.perm_ann_p, "df": dfd,
                         "crossed_lr": lr, "crossed_p": p})
        print(f"[{lang}] {name:24s} orig={pr.p_original:.2g} crossed={p:.3g} (df {dfd}) "
              f"LRT_ann={pr.lrt_ann_p:.3g} PERM_ann={pr.perm_ann_p:.3g}")
pd.DataFrame(int_rows).to_csv("results_crossed_lrt_interactions.csv", index=False)

# ── Persona ceiling check ────────────────────────────────────────────────────
old = pd.read_csv("results_crossed_r2_per_language.csv").set_index("language")
ceil_rows = []
for lang in LANGS:
    d = pd.read_parquet(f"data_{lang}_v2.parquet")
    rhs = old.loc[lang, "fixed_rhs"]
    X = prune_cols(patsy.dmatrix(rhs, d, return_type="dataframe").values)
    cm = model_for(d)
    row = {"language": lang, "target_r2_old": old.loc[lang, "target_r2"],
           "icc_annotator_old": old.loc[lang, "icc_annotator"]}
    for tag, reml in [("ml", False), ("reml", True)]:
        r = cm.fit(X, reml=reml)
        tot = r["var_fixed"] + r["var_text"] + r["var_ann"] + r["var_e"]
        row[f"target_r2_{tag}"] = r["var_ann"] / (r["var_fixed"] + r["var_ann"] + r["var_e"])
        row[f"icc_annotator_{tag}"] = r["var_ann"] / tot
    ceil_rows.append(row)
    print(f"[{lang}] ceiling old(REML, statsmodels)={row['target_r2_old']*100:.2f}%  "
          f"REML={row['target_r2_reml']*100:.2f}%  ML={row['target_r2_ml']*100:.2f}%")
pd.DataFrame(ceil_rows).to_csv("results_crossed_ceiling_check.csv", index=False)
