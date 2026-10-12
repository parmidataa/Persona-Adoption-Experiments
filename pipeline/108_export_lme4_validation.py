"""
108_export_lme4_validation.py
=============================
Exports the cells used to validate fastlmm.CrossedML against R lme4
(108_lme4_validation.R) and writes the fastlmm values for the same cells.

Writes: /tmp/lme4val/*.csv and results_lme4_validation_fast.csv
"""
import json
import numpy as np
import pandas as pd
import patsy
from fastlmm import CrossedML, nested_design, prune_cols

CELLS = [  # (id, data, lang, base fields, added terms)
    ("es_M1M2", "v2", "es", "age_centered + gender + ethnicity + nationality", "country_birth + country_residence"),
    ("it_M2M3", "v2", "it", "age_centered + gender + ethnicity + country_birth", "student_status + employment_status"),
    ("pt_M2M3", "v2", "pt", "age_centered + gender + ethnicity + nationality + country_birth + country_residence", "student_status + employment_status"),
    ("de_agexnat", "A", "de", None, "age_centered:nationality"),
    ("fr_agexeth", "A", "fr", None, "age_centered:ethnicity"),
    ("en_genxnat", "A", "en", None, "gender:nationality"),
]
CEIL = ["ar", "fr", "hi"]
meta = json.load(open("lang_metadata.json"))
ceil_rhs = pd.read_csv("results_crossed_r2_per_language.csv").set_index("language")["fixed_rhs"]


def py_rhs(r):
    return " + ".join(t if t in ("age_centered",) or ":" in t else f"C({t})" for t in [s.strip() for s in r.split("+")])


def pyterm(t):
    if ":" in t:
        a, b = t.split(":")
        return ":".join(x if x == "age_centered" else f"C({x})" for x in (a, b))
    return t if t == "age_centered" else f"C({t})"


rows = []
for cid, src, lang, base, add in CELLS:
    if src == "v2":
        d = pd.read_parquet(f"data_{lang}_v2.parquet")
    else:
        d = pd.read_parquet(f"data_{lang}.parquet")
        base = " + ".join(c.replace("C(", "").replace(")", "") for c in meta[lang]["usable_vars"])
    fields = [f.strip() for f in base.split("+")] + [x for t in add.split("+") for x in t.strip().split(":")]
    fields = list(dict.fromkeys(fields))
    d = d[["label", "text_id", "annotator_id"] + fields].dropna().reset_index(drop=True)
    d.to_csv(f"/tmp/lme4val/{cid}.csv", index=False)
    bt = [pyterm(t.strip()) for t in base.split("+")]
    at = [pyterm(t.strip()) for t in add.split("+")]
    Xf_raw, nb = nested_design(" + ".join(bt), " + ".join(at), d)
    Xf, kb = prune_cols(Xf_raw, nb)
    cm = CrossedML(d.label.values, pd.factorize(d.text_id)[0], pd.factorize(d.annotator_id)[0])
    rb, rf = cm.fit(Xf[:, :kb]), cm.fit(Xf)
    rows.append({"cell": cid, "base": base, "added": add, "llf_base": rb["llf"], "llf_full": rf["llf"],
                 "df": Xf.shape[1] - kb, "lr": 2 * (rf["llf"] - rb["llf"])})

for lang in CEIL:
    d = pd.read_parquet(f"data_{lang}_v2.parquet")
    rhs = ceil_rhs[lang]
    flds = [t.replace("C(", "").replace(")", "").strip() for t in rhs.split("+")]
    d = d[["label", "text_id", "annotator_id"] + flds]
    d.to_csv(f"/tmp/lme4val/ceil_{lang}.csv", index=False)
    X = prune_cols(patsy.dmatrix(rhs, d, return_type="dataframe").values)
    r = CrossedML(d.label.values, pd.factorize(d.text_id)[0], pd.factorize(d.annotator_id)[0]).fit(X, reml=True)
    rows.append({"cell": f"ceil_{lang}", "base": " + ".join(flds), "var_text": r["var_text"],
                 "var_ann": r["var_ann"], "var_e": r["var_e"]})
pd.DataFrame(rows).to_csv("results_lme4_validation_fast.csv", index=False)
print(pd.DataFrame(rows).to_string())
