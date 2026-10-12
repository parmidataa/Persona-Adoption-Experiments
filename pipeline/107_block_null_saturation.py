"""
107_block_null_saturation.py
============================
(a) Block-level null for each Phase D step. Base fields keep their observed
    values; only the fields added at the step are shuffled across annotators
    (jointly, as one block). The statistic is the R^2_m gain of the step
    (text-intercept LMM, variance components from the observed full fit).
    Same cumulative nested sets as 102 and 105.
(b) Saturation corrected for the shuffled-profile null:
    (R^2_m - null mean) / R^2_persona, from results_null_r2m.csv and
    results_persona_ceiling_per_language.csv.

Writes only:
  results_block_null_phaseD.csv
  results_saturation_null_corrected.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from fastlmm import TextGLS, prune_cols

N_PERM = 2000
rng = np.random.default_rng(5)
LANGS = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]
STEP_FIELDS = {
    "M1": ["age_centered", "gender", "ethnicity", "nationality"],
    "M2": ["country_birth", "country_residence"],
    "M3": ["student_status", "employment_status"],
    "M4": ["generation"],
}


def term(f):
    return f if f == "age_centered" else f"C({f})"


rows = []
for lang in LANGS:
    d = pd.read_parquet(f"data_{lang}_v2.parquet")
    y = d["label"].values.astype(float)
    tc = pd.factorize(d["text_id"])[0]
    ac = pd.factorize(d["annotator_id"])[0]
    first = pd.Series(np.arange(len(d))).groupby(ac).first().values
    n_ann = len(first)
    a_first = d.drop_duplicates("annotator_id")
    fields = {m: [f for f in fs if f in d.columns and
                  (f == "age_centered" or a_first[f].nunique() > 1)]
              for m, fs in STEP_FIELDS.items()}
    cum = list(fields["M1"])
    for p_, n_ in [("M1", "M2"), ("M2", "M3"), ("M3", "M4")]:
        new = fields[n_]
        if not new:
            cum = cum + new
            continue
        Xb = patsy.dmatrix(" + ".join(map(term, cum)), d, return_type="dataframe").values
        Xn = patsy.dmatrix("0 + " + " + ".join(map(term, new)), d, return_type="dataframe").values
        Xf = np.column_stack([Xb, Xn])
        m = sm.MixedLM(y, prune_cols(Xf), groups=tc).fit(reml=True, method="lbfgs")
        g = TextGLS(tc, float(m.cov_re[0, 0]), m.scale)
        yt = g.transform(y).ravel()
        tot = lambda vf: vf / (vf + g.var_u + g.var_e)
        r2_base = tot(g.var_fixed(Xb, yt))
        gain = tot(g.var_fixed(Xf, yt)) - r2_base
        An = Xn[first]
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            Xp = np.column_stack([Xb, An[rng.permutation(n_ann)][ac]])
            null[i] = tot(g.var_fixed(Xp, yt)) - r2_base
        row = {"language": lang, "step": f"{p_}->{n_}", "added_fields": "+".join(new),
               "n_annotators": n_ann, "gain_pp": gain * 100,
               "null_mean_pp": null.mean() * 100, "null_p95_pp": np.quantile(null, 0.95) * 100,
               "block_perm_p": (np.sum(null >= gain) + 1) / (N_PERM + 1),
               "excess_pp": (gain - null.mean()) * 100}
        rows.append(row)
        print(f"[{lang}] {row['step']:7s} {row['added_fields']:<34s} gain={row['gain_pp']:5.2f}pp "
              f"null={row['null_mean_pp']:5.2f}pp p95={row['null_p95_pp']:5.2f}pp p={row['block_perm_p']:.3f}")
        cum = cum + new
pd.DataFrame(rows).to_csv("results_block_null_phaseD.csv", index=False)

# ── (b) null-corrected saturation ────────────────────────────────────────────
nul = pd.read_csv("results_null_r2m.csv")
nul = nul[nul.family == "phaseD"].set_index(["language", "model"])
ceil = pd.read_csv("results_persona_ceiling_per_language.csv").set_index("language")
sat = []
for lang in LANGS:
    c = ceil.loc[lang, "corrected_ceiling"]
    for mname in ["M1", "M2", "M3", "M4"]:
        r = nul.loc[(lang, mname)]
        sat.append({"language": lang, "model": mname, "r2_persona": c,
                    "r2m": r.r2m, "null_mean": r.null_mean,
                    "saturation_raw": r.r2m / c,
                    "saturation_null_corrected": max(r.r2m - r.null_mean, 0) / c})
S = pd.DataFrame(sat)
S.to_csv("results_saturation_null_corrected.csv", index=False)
print("\nSaturation (raw -> null-corrected), %")
print(S.pivot(index="language", columns="model",
              values=["saturation_raw", "saturation_null_corrected"]).mul(100).round(1).to_string())
print("mean M1 raw/corr:", round(S[S.model == 'M1'].saturation_raw.mean() * 100, 1),
      round(S[S.model == 'M1'].saturation_null_corrected.mean() * 100, 1),
      " mean M4 raw/corr:", round(S[S.model == 'M4'].saturation_raw.mean() * 100, 1),
      round(S[S.model == 'M4'].saturation_null_corrected.mean() * 100, 1))
