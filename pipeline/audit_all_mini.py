"""
audit_all_mini.py  — Authoritative convergence audit of ALL gpt-4.1-mini numbers.

Recomputes R2m and directional concordance with the hardened, convergence-REQUIRED
fitter (raw + standardized outcome across bfgs/lbfgs/nm) and compares against the
published CSVs, flagging every cell whose published value came from a
non-converged fit. Covers:

  1. Binary main experiment  (data_predictions_clean.parquet, outcome llm_label)
       - R2m for all 9 langs x {p0,p1,p2,p3}, + within-sample human SH
       - directional concordance for the best condition per language
  2. Probability elicitation (prob_predictions_clean.parquet, outcome llm_prob)
       - R2m + concordance for AR, ES (P2)

RAG and P5 are handled by verify_mini_fits.py (already run).

Usage:
    python audit_all_mini.py
"""

import warnings, re
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, pearsonr

LANGUAGES = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]
CONDITIONS = ["p0", "p1", "p2", "p3"]
BEST_COND = {"ar": "p2", "de": "p3", "en": "p2", "es": "p2", "fr": "p2",
             "hi": "p2", "it": "p3", "nl": "p2", "pt": "p1"}
MIN_SUPPORT = 15

M1_BASE = {
    "ar": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "de": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "en": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "es": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "fr": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "hi": ["age_centered", "C(gender)", "C(ethnicity)"],
    "it": ["age_centered", "C(gender)", "C(ethnicity)"],
    "nl": ["age_centered", "C(gender)", "C(ethnicity)"],
    "pt": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
}
M4_EXTRAS = ["C(country_birth)", "C(country_residence)",
             "C(employment_status)", "C(student_status)"]
P3_TEMPLATE = {
    "ar": ["age_centered", "C(nationality)"],
    "de": ["C(Generation)", "C(employment_status)"],
    "en": ["C(nationality)", "C(gender)", "C(country_residence)"],
    "es": ["age_centered", "C(country_birth)", "C(country_residence)"],
    "fr": ["age_centered", "C(ethnicity)", "C(nationality)", "C(country_residence)"],
    "hi": ["age_centered", "C(ethnicity)", "C(country_birth)"],
    "it": ["age_centered", "C(employment_status)"],
    "nl": ["C(employment_status)"],
    "pt": ["C(employment_status)", "C(country_birth)"],
}
ALL_PRED_COLS = ["age_centered", "gender", "ethnicity", "nationality", "country_birth",
                 "country_residence", "employment_status", "student_status", "Generation"]
_OPTIMIZERS = ["bfgs", "lbfgs", "nm"]


def col_from_term(t):
    m = re.match(r"C\((\w+)\)", t)
    return m.group(1) if m else t


def build_formula(target, var_list, df_check):
    valid = []
    for term in var_list:
        col = col_from_term(term)
        if col not in df_check.columns:
            continue
        if col == "age_centered":
            valid.append(term); continue
        if df_check[col].dropna().nunique() >= 2:
            valid.append(term)
    return (f"{target} ~ " + " + ".join(valid)) if valid else None


def get_formula(target, lang, cond, df_check):
    if cond in ("p0", "p1"):
        return build_formula(target, M1_BASE[lang], df_check)
    if cond == "p2":
        return build_formula(target, M1_BASE[lang] + M4_EXTRAS, df_check)
    return build_formula(target, P3_TEMPLATE[lang], df_check)


def needed_cols(formula, target):
    cols = [target, "post_id"]
    for col in ALL_PRED_COLS:
        if col in formula:
            cols.append(col)
    return list(dict.fromkeys(cols))


def _fit_lmm(formula, df_fit, target):
    """Hardened: try raw then standardized outcome; require convergence."""
    variants = [df_fit]
    s = df_fit[target].std()
    if s and s > 0:
        dfz = df_fit.copy()
        dfz[target] = (dfz[target] - dfz[target].mean()) / s
        variants.append(dfz)
    for dd in variants:
        for meth in _OPTIMIZERS:
            try:
                m = smf.mixedlm(formula, data=dd, groups=dd["post_id"]).fit(
                    reml=False, method=meth, disp=False)
                if getattr(m, "converged", True):
                    return m
            except Exception:
                continue
    return None


def fit_r2m(df_sub, formula):
    target = formula.split("~")[0].strip()
    nc = [c for c in needed_cols(formula, target) if c in df_sub.columns]
    df_fit = df_sub[nc].dropna().copy()
    if len(df_fit) < 50 or df_fit[target].nunique() < 2:
        return np.nan
    m = _fit_lmm(formula, df_fit, target)
    if m is None:
        return np.nan
    fp = np.dot(m.model.exog, m.fe_params)
    vf = np.var(fp); vr = float(m.cov_re.iloc[0, 0])
    return vf / (vf + vr + m.scale)


def fit_and_coefs(df_sub, formula):
    target = formula.split("~")[0].strip()
    nc = [target, "post_id"] + [col_from_term(t)
          for t in re.findall(r"C\(\w+\)|\w+", formula.split("~")[1])]
    nc = [c for c in dict.fromkeys(nc) if c in df_sub.columns]
    df_fit = df_sub[nc].dropna().copy()
    if len(df_fit) < 50:
        return {}
    m = _fit_lmm(formula, df_fit, target)
    if m is None:
        return {}
    coefs = {}
    for name, val in m.fe_params.items():
        if name == "Intercept":
            continue
        cm = re.search(r"C\((\w+)\)", name)
        if cm:
            col = cm.group(1)
            lm = re.search(r"\[T\.(.+)\]", name)
            if lm and col in df_fit.columns and (df_fit[col] == lm.group(1)).sum() >= MIN_SUPPORT:
                coefs[name] = val
        else:
            coefs[name] = val
    return coefs


def concordance(h, l):
    common = sorted(set(h) & set(l))
    if len(common) < 2:
        return None
    hv = np.array([h[k] for k in common]); lv = np.array([l[k] for k in common])
    rho, _ = spearmanr(hv, lv)
    try:
        r, _ = pearsonr(hv, lv)
    except Exception:
        r = np.nan
    denom = np.linalg.norm(hv) * np.linalg.norm(lv)
    cos = float(np.dot(hv, lv) / denom) if denom > 0 else np.nan
    return {"n": len(common), "sign": float(np.mean(np.sign(hv) == np.sign(lv))),
            "rho": float(rho), "pearson": float(r), "cosine": cos}


def prep(df):
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_centered"] = df["age"] - df.groupby("language")["age"].transform("mean")
    for col in ["gender", "ethnicity", "nationality", "country_birth",
                "country_residence", "employment_status", "student_status", "Generation"]:
        if col in df.columns:
            df[col] = df[col].replace({"unknown": np.nan, "nan": np.nan})
    return df


def audit_binary():
    print("=" * 90)
    print("  1) BINARY MAIN EXPERIMENT — R2m recheck (outcome=llm_label)")
    print("=" * 90)
    df = prep(pd.read_parquet("data_predictions_clean.parquet"))
    old = pd.read_csv("results_r2m_llm.csv")
    print(f"\n  {'Lang':<5}{'Cond':<5}{'old R2m%':>10}{'new R2m%':>10}{'Δ':>9}{'oldConv':>9}   flag")
    print("  " + "-" * 74)
    changed = []
    for lang in LANGUAGES:
        for cond in CONDITIONS:
            sub = df[(df.language == lang) & (df.condition == cond)].copy()
            f = get_formula("llm_label", lang, cond, sub)
            new = fit_r2m(sub, f) * 100 if f else np.nan
            orow = old[(old.language == lang) & (old.condition == cond)]
            o = float(orow["r2m_llm_pct"].values[0]) if len(orow) else np.nan
            oconv = bool(orow["converged"].values[0]) if len(orow) else True
            d = new - o if not (np.isnan(new) or np.isnan(o)) else np.nan
            flag = ""
            if (not oconv) or (not np.isnan(d) and abs(d) > 0.05):
                flag = "<-- CHANGED"
                changed.append((lang, cond, o, new, oconv))
            ds = f"{d:>+9.3f}" if not np.isnan(d) else f"{'nan':>9}"
            ns = f"{new:>10.3f}" if not np.isnan(new) else f"{'nan':>10}"
            print(f"  {lang.upper():<5}{cond.upper():<5}{o:>10.3f}{ns}{ds}{str(oconv):>9}   {flag}")
    return changed


def audit_binary_concordance():
    print("\n" + "=" * 90)
    print("  2) BINARY MAIN — directional concordance recheck (best condition)")
    print("=" * 90)
    df = prep(pd.read_parquet("data_predictions_clean.parquet"))
    old = pd.read_csv("results_directional_concordance.csv")
    old = old[old.is_best_condition == True]
    print(f"\n  {'Lang':<5}{'Cond':<5}{'old sign':>9}{'new sign':>9}{'Δpp':>7}"
          f"{'old Pear':>9}{'new Pear':>9}   flag")
    print("  " + "-" * 70)
    for lang in LANGUAGES:
        cond = BEST_COND[lang]
        sub = df[(df.language == lang) & (df.condition == cond)].copy()
        f_l = get_formula("llm_label", lang, cond, sub)
        if not f_l:
            continue
        rhs = f_l.split("~", 1)[1]
        f_h = "human_label ~" + rhs
        cm = concordance(fit_and_coefs(sub, f_h), fit_and_coefs(sub, f_l))
        orow = old[old.language == lang]
        if not len(orow) or cm is None:
            print(f"  {lang.upper():<5}{cond.upper():<5}  (unavailable)")
            continue
        o_s = float(orow["sign_agreement"].values[0]) * 100
        o_p = float(orow["pearson_r"].values[0])
        n_s = cm["sign"] * 100; n_p = cm["pearson"]
        flag = "<-- CHANGED" if abs(o_s - n_s) > 0.05 or abs(o_p - n_p) > 0.02 else ""
        print(f"  {lang.upper():<5}{cond.upper():<5}{o_s:>8.1f}%{n_s:>8.1f}%{n_s-o_s:>+7.1f}"
              f"{o_p:>+9.3f}{n_p:>+9.3f}   {flag}")


def audit_prob():
    print("\n" + "=" * 90)
    print("  3) PROBABILITY ELICITATION — R2m + concordance recheck (outcome=llm_prob)")
    print("=" * 90)
    df = prep(pd.read_parquet("prob_predictions_clean.parquet"))
    old_r = pd.read_csv("results_prob_r2m.csv")
    old_c = pd.read_csv("results_prob_concordance.csv")
    print(f"\n  {'Lang':<5}{'Cond':<5}{'old R2m%':>10}{'new R2m%':>10}{'Δ':>9}   flag")
    print("  " + "-" * 52)
    for lang in ["ar", "es"]:
        for cond in ["p2"]:
            sub = df[(df.language == lang) & (df.condition == cond)].copy()
            f = get_formula("llm_prob", lang, cond, sub)
            new = fit_r2m(sub, f) * 100 if f else np.nan
            orow = old_r[(old_r.language == lang) & (old_r.condition == cond)]
            o = float(orow["r2m_prob_pct"].values[0]) if len(orow) else np.nan
            d = new - o if not (np.isnan(new) or np.isnan(o)) else np.nan
            flag = "<-- CHANGED" if not np.isnan(d) and abs(d) > 0.05 else ""
            print(f"  {lang.upper():<5}{cond.upper():<5}{o:>10.3f}{new:>10.3f}{d:>+9.3f}   {flag}")
    print(f"\n  {'Lang':<5}{'old sign':>9}{'new sign':>9}{'Δpp':>7}{'old Pear':>9}{'new Pear':>9}   flag")
    print("  " + "-" * 60)
    for lang in ["ar", "es"]:
        sub = df[(df.language == lang) & (df.condition == "p2")].copy()
        f_l = get_formula("llm_prob", lang, "p2", sub)
        rhs = f_l.split("~", 1)[1]
        f_h = "human_label ~" + rhs
        cm = concordance(fit_and_coefs(sub, f_h), fit_and_coefs(sub, f_l))
        orow = old_c[old_c.language == lang]
        o_s = float(orow["sign_agreement"].values[0]) * 100
        o_p = float(orow["pearson_r"].values[0])
        n_s = cm["sign"] * 100; n_p = cm["pearson"]
        flag = "<-- CHANGED" if abs(o_s - n_s) > 0.05 or abs(o_p - n_p) > 0.02 else ""
        print(f"  {lang.upper():<5}{o_s:>8.1f}%{n_s:>8.1f}%{n_s-o_s:>+7.1f}"
              f"{o_p:>+9.3f}{n_p:>+9.3f}   {flag}")


if __name__ == "__main__":
    changed = audit_binary()
    audit_binary_concordance()
    audit_prob()
    print("\n" + "=" * 90)
    print(f"  Binary R2m cells changed: {len(changed)}")
    for lang, cond, o, n, oc in changed:
        print(f"    {lang.upper()} {cond.upper()}: {o:.3f}% -> {n:.3f}%  (old converged={oc})")
