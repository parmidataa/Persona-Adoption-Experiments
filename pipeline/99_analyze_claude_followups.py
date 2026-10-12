"""
99_analyze_claude_followups.py  — Analyse Claude RAG (P4) and P5 runs
=====================================================================
Runs the SAME R2m + directional-concordance pipeline used for the gpt-4.1-mini
RAG and P5 experiments (scripts 75/76 and 84/85) on the Claude Opus 5 outputs,
then prints a mini-vs-Opus5 side-by-side and writes claude-suffixed CSVs +
comparison figures.

Reads:
    rag_predictions_claude_clean.parquet
    p5_predictions_claude_clean.parquet
    results_rag_r2m.csv / results_rag_concordance.csv          (mini baselines)
    results_p5_r2m.csv  / results_p5_concordance.csv           (mini baselines)
    results_r2m_llm.csv                                        (binary SH denom)

Writes:
    results_rag_claude_r2m.csv,  results_rag_claude_concordance.csv
    results_p5_claude_r2m.csv,   results_p5_claude_concordance.csv
    fig_rag_claude_comparison.png,  fig_p5_claude_comparison.png

Usage:
    python 99_analyze_claude_followups.py
    (Run AFTER 98_parse_claude_followups.py)
"""

import warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

LANGUAGES   = ["ar", "es"]
MIN_SUPPORT = 15

M1_BASE = ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"]
M4_EXTRAS = ["C(country_birth)", "C(country_residence)",
             "C(employment_status)", "C(student_status)"]
ALL_PRED_COLS = ["age_centered", "gender", "ethnicity", "nationality",
                 "country_birth", "country_residence",
                 "employment_status", "student_status", "Generation"]


# ── Shared helpers (identical logic to 75/76/84/85) ──────────────────────────
def col_from_term(t):
    m = re.match(r"C\((\w+)\)", t)
    return m.group(1) if m else t


def build_formula(target, df_check):
    valid = []
    for term in M1_BASE + M4_EXTRAS:
        col = col_from_term(term)
        if col not in df_check.columns:
            continue
        if col == "age_centered":
            valid.append(term); continue
        if df_check[col].dropna().nunique() >= 2:
            valid.append(term)
    return (f"{target} ~ " + " + ".join(valid)) if valid else None


def needed_cols(formula, df, target):
    cols = [target, "post_id"]
    for col in ALL_PRED_COLS:
        if col in formula and col in df.columns:
            cols.append(col)
    return list(dict.fromkeys(cols))


# Robust fit for near-singular designs (Arabic: nationality/country dummies on
# ~500 rows). We try each optimizer on the RAW outcome first (matches the
# gpt-4.1-mini runs so those numbers are unchanged), then on a STANDARDIZED
# outcome. R2m and every direction metric (sign agreement, Spearman, Pearson,
# cosine) are invariant to affine transforms of the outcome, so a z-scored fit
# is numerically stabilizing but statistically equivalent. Different outcomes
# (binary human vs continuous LLM) happen to converge under different scalings,
# so trying both rescues every fit without biasing any estimate.
_OPTIMIZERS = ["bfgs", "lbfgs", "nm"]


def _fit_lmm(formula, df_fit, target):
    """Fit MixedLM robustly; return fitted model or None."""
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


def fit_r2m(df_sub, formula, method=None):
    target = formula.split("~")[0].strip()
    nc = [c for c in needed_cols(formula, df_sub, target) if c in df_sub.columns]
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


def concordance(h_coefs, l_coefs):
    common = sorted(set(h_coefs) & set(l_coefs))
    if len(common) < 2:
        return None
    hv = np.array([h_coefs[k] for k in common])
    lv = np.array([l_coefs[k] for k in common])
    sign_agr = float(np.mean(np.sign(hv) == np.sign(lv)))
    rho, _ = spearmanr(hv, lv)
    try:
        r, _ = pearsonr(hv, lv)
    except Exception:
        r = np.nan
    denom = np.linalg.norm(hv) * np.linalg.norm(lv)
    cos = float(np.dot(hv, lv) / denom) if denom > 0 else np.nan
    return {"n_coefs": len(common), "sign_agreement": sign_agr,
            "spearman_rho": float(rho), "pearson_r": float(r), "cosine": cos}


def group_dev(df_sub, formula, llm_col):
    cats = re.findall(r"C\((\w+)\)", formula)
    oh = df_sub["human_label"].mean(); ol = df_sub[llm_col].mean()
    hd, ld = [], []
    for col in cats:
        if col not in df_sub.columns:
            continue
        for level in df_sub[col].dropna().unique():
            mask = df_sub[col] == level
            if mask.sum() < MIN_SUPPORT:
                continue
            hd.append(df_sub.loc[mask, "human_label"].mean() - oh)
            ld.append(df_sub.loc[mask, llm_col].mean() - ol)
    if len(hd) < 2:
        return {"groupdev_sign": np.nan, "groupdev_rho": np.nan, "groupdev_n": 0}
    hv = np.array(hd); lv = np.array(ld)
    rho, _ = spearmanr(hv, lv)
    return {"groupdev_sign": float(np.mean(np.sign(hv) == np.sign(lv))),
            "groupdev_rho": float(rho), "groupdev_n": len(hd)}


def prep(df):
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_centered"] = df["age"] - df.groupby("language")["age"].transform("mean")
    for col in ["gender", "ethnicity", "nationality", "country_birth",
                "country_residence", "employment_status", "student_status"]:
        if col in df.columns:
            df[col] = df[col].replace("unknown", np.nan)
    return df


def load_binary_sh():
    """P2 binary sample-human baseline denominator (same convention as 75)."""
    sh = {}
    try:
        d = pd.read_csv("results_r2m_llm.csv")
        for lang in LANGUAGES:
            row = d[(d.language == lang) & (d.condition == "p2")]
            if len(row) and "sample_human_r2m" in row.columns and not pd.isna(row["sample_human_r2m"].values[0]):
                sh[lang] = float(row["sample_human_r2m"].values[0])
    except FileNotFoundError:
        pass
    # Fallback Study-1 M1 benchmarks if CSV missing
    sh.setdefault("ar", 0.019017)
    sh.setdefault("es", 0.010041)
    return sh


BINARY_SH = load_binary_sh()


# ── RAG analysis ──────────────────────────────────────────────────────────────
def analyse_rag():
    p = Path("rag_predictions_claude_clean.parquet")
    if not p.exists():
        print("SKIP RAG: rag_predictions_claude_clean.parquet not found.")
        return
    print("\n" + "=" * 70)
    print("  RAG / P4  —  Claude Opus 5  (R2m + directional concordance)")
    print("=" * 70)
    df = prep(pd.read_parquet(p))

    r2m_rows, conc_rows = [], []
    for lang in LANGUAGES:
        sub = df[(df.language == lang) & (df.condition == "p4")].copy()
        f_llm = build_formula("llm_prob", sub)
        f_hum = build_formula("human_label", sub)
        if f_llm is None:
            continue
        r2m = fit_r2m(sub, f_llm)
        sh = BINARY_SH.get(lang, np.nan)
        pct_sh = (r2m / sh * 100) if (not np.isnan(r2m) and sh) else np.nan
        r2m_rows.append({"language": lang, "r2m_claude": r2m,
                         "r2m_claude_pct": r2m * 100 if not np.isnan(r2m) else np.nan,
                         "pct_sh_claude": pct_sh})

        h = fit_and_coefs(sub, f_hum); l = fit_and_coefs(sub, f_llm)
        cm = concordance(h, l)
        gd = group_dev(sub, f_llm, "llm_prob")
        if cm:
            conc_rows.append({"language": lang, "condition": "p4", **cm, **gd})

    df_r2m = pd.DataFrame(r2m_rows); df_conc = pd.DataFrame(conc_rows)
    df_r2m.to_csv("results_rag_claude_r2m.csv", index=False)
    df_conc.to_csv("results_rag_claude_concordance.csv", index=False)

    # mini baselines
    mini_r2m = pd.read_csv("results_rag_r2m.csv") if Path("results_rag_r2m.csv").exists() else pd.DataFrame()
    mini_conc = pd.read_csv("results_rag_concordance.csv") if Path("results_rag_concordance.csv").exists() else pd.DataFrame()

    print(f"\n  {'Lang':<5}{'mini R2m%':>11}{'Opus5 R2m%':>12}{'Δpp':>8}"
          f"{'mini sign':>11}{'Opus5 sign':>12}")
    print("  " + "-" * 58)
    for lang in LANGUAGES:
        cr = df_r2m[df_r2m.language == lang]
        cc = df_conc[df_conc.language == lang]
        mr = mini_r2m[mini_r2m.language == lang] if len(mini_r2m) else pd.DataFrame()
        mc = mini_conc[mini_conc.language == lang] if len(mini_conc) else pd.DataFrame()
        mr_v = float(mr["r2m_rag"].values[0] * 100) if len(mr) and "r2m_rag" in mr else np.nan
        cr_v = float(cr["r2m_claude_pct"].values[0]) if len(cr) else np.nan
        mc_v = float(mc["sign_agreement"].values[0]) if len(mc) else np.nan
        cc_v = float(cc["sign_agreement"].values[0]) if len(cc) else np.nan
        d = cr_v - mr_v if not (np.isnan(cr_v) or np.isnan(mr_v)) else np.nan
        print(f"  {lang.upper():<5}{mr_v:>11.3f}{cr_v:>12.3f}{d:>+8.3f}"
              f"{mc_v*100:>10.1f}%{cc_v*100:>11.1f}%")
    _fig_two_model(df_r2m, df_conc, mini_r2m, mini_conc,
                   r2m_mini_col="r2m_rag", title="RAG / P4",
                   fname="fig_rag_claude_comparison.png")
    print("  Saved: results_rag_claude_r2m.csv, results_rag_claude_concordance.csv, "
          "fig_rag_claude_comparison.png")


# ── P5 analysis ───────────────────────────────────────────────────────────────
def analyse_p5():
    p = Path("p5_predictions_claude_clean.parquet")
    if not p.exists():
        print("SKIP P5: p5_predictions_claude_clean.parquet not found.")
        return
    print("\n" + "=" * 70)
    print("  P5 / Relative+CoT  —  Claude Opus 5  (R2m + directional concordance)")
    print("=" * 70)
    df = prep(pd.read_parquet(p))

    r2m_rows, conc_rows = [], []
    for lang in LANGUAGES:
        sub = df[df.language == lang].copy()
        f_dev  = build_formula("llm_deviation", sub)   # primary outcome
        f_prob = build_formula("llm_prob", sub)
        f_hum  = build_formula("human_label", sub)
        if f_dev is None:
            continue
        r2m_dev = fit_r2m(sub, f_dev)
        r2m_rows.append({"language": lang,
                         "r2m_dev_claude": r2m_dev,
                         "r2m_dev_claude_pct": r2m_dev * 100 if not np.isnan(r2m_dev) else np.nan})

        h = fit_and_coefs(sub, f_hum)
        ld = fit_and_coefs(sub, f_dev)
        lp = fit_and_coefs(sub, f_prob)
        cm_dev  = concordance(h, ld)
        cm_prob = concordance(h, lp)
        gd = group_dev(sub, f_prob, "llm_prob")
        if cm_dev:
            conc_rows.append({
                "language": lang, "condition": "p5",
                "n_coefs_dev": cm_dev["n_coefs"],
                "sign_agr_dev": cm_dev["sign_agreement"],
                "spearman_dev": cm_dev["spearman_rho"],
                "pearson_dev": cm_dev["pearson_r"],
                "cosine_dev": cm_dev["cosine"],
                "sign_agr_prob": cm_prob["sign_agreement"] if cm_prob else np.nan,
                **gd})

    df_r2m = pd.DataFrame(r2m_rows); df_conc = pd.DataFrame(conc_rows)
    df_r2m.to_csv("results_p5_claude_r2m.csv", index=False)
    df_conc.to_csv("results_p5_claude_concordance.csv", index=False)

    mini_r2m = pd.read_csv("results_p5_r2m.csv") if Path("results_p5_r2m.csv").exists() else pd.DataFrame()
    mini_conc = pd.read_csv("results_p5_concordance.csv") if Path("results_p5_concordance.csv").exists() else pd.DataFrame()

    print(f"\n  {'Lang':<5}{'mini R2m%':>11}{'Opus5 R2m%':>12}{'Δpp':>8}"
          f"{'mini sign':>11}{'Opus5 sign':>12}  (deviation outcome)")
    print("  " + "-" * 62)
    for lang in LANGUAGES:
        cr = df_r2m[df_r2m.language == lang]
        cc = df_conc[df_conc.language == lang]
        mr = mini_r2m[mini_r2m.language == lang] if len(mini_r2m) else pd.DataFrame()
        mc = mini_conc[mini_conc.language == lang] if len(mini_conc) else pd.DataFrame()
        mr_v = float(mr["r2m_dev_pct"].values[0]) if len(mr) and "r2m_dev_pct" in mr else np.nan
        cr_v = float(cr["r2m_dev_claude_pct"].values[0]) if len(cr) else np.nan
        mc_v = float(mc["sign_agr_dev"].values[0]) if len(mc) and "sign_agr_dev" in mc else np.nan
        cc_v = float(cc["sign_agr_dev"].values[0]) if len(cc) else np.nan
        d = cr_v - mr_v if not (np.isnan(cr_v) or np.isnan(mr_v)) else np.nan
        print(f"  {lang.upper():<5}{mr_v:>11.3f}{cr_v:>12.3f}{d:>+8.3f}"
              f"{mc_v*100:>10.1f}%{cc_v*100:>11.1f}%")

    # Figure: sign agreement mini vs Opus5 (deviation outcome)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(LANGUAGES)); w = 0.38
    mini_s = [float(mini_conc[mini_conc.language == l]["sign_agr_dev"].values[0]) * 100
              if len(mini_conc) and l in set(mini_conc.language) else np.nan for l in LANGUAGES]
    op_s = [float(df_conc[df_conc.language == l]["sign_agr_dev"].values[0]) * 100
            if l in set(df_conc.language) else np.nan for l in LANGUAGES]
    ax.bar(x - w/2, mini_s, w, label="gpt-4.1-mini", color="#aec6e8")
    ax.bar(x + w/2, op_s, w, label="Claude Opus 5", color="#c0562e")
    ax.axhline(50, color="gray", ls="--", lw=1, label="50% = chance")
    ax.set_xticks(x); ax.set_xticklabels([l.upper() for l in LANGUAGES])
    ax.set_ylabel("Sign agreement (%) — deviation outcome")
    ax.set_title("P5 (Relative+CoT): directional concordance\ngpt-4.1-mini vs Claude Opus 5")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig("fig_p5_claude_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results_p5_claude_r2m.csv, results_p5_claude_concordance.csv, "
          "fig_p5_claude_comparison.png")


def _fig_two_model(df_r2m, df_conc, mini_r2m, mini_conc, r2m_mini_col, title, fname):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(LANGUAGES)); w = 0.38
    mini_r = [float(mini_r2m[mini_r2m.language == l][r2m_mini_col].values[0]) * 100
              if len(mini_r2m) and l in set(mini_r2m.language) else np.nan for l in LANGUAGES]
    op_r = [float(df_r2m[df_r2m.language == l]["r2m_claude_pct"].values[0])
            if l in set(df_r2m.language) else np.nan for l in LANGUAGES]
    axL.bar(x - w/2, mini_r, w, label="gpt-4.1-mini", color="#aec6e8")
    axL.bar(x + w/2, op_r, w, label="Claude Opus 5", color="#c0562e")
    axL.set_xticks(x); axL.set_xticklabels([l.upper() for l in LANGUAGES])
    axL.set_ylabel("R²_m (%)"); axL.set_title(f"{title}: magnitude (R²_m)")
    axL.legend(); axL.grid(axis="y", alpha=0.3)

    mini_s = [float(mini_conc[mini_conc.language == l]["sign_agreement"].values[0]) * 100
              if len(mini_conc) and l in set(mini_conc.language) else np.nan for l in LANGUAGES]
    op_s = [float(df_conc[df_conc.language == l]["sign_agreement"].values[0]) * 100
            if l in set(df_conc.language) else np.nan for l in LANGUAGES]
    axR.bar(x - w/2, mini_s, w, label="gpt-4.1-mini", color="#aec6e8")
    axR.bar(x + w/2, op_s, w, label="Claude Opus 5", color="#c0562e")
    axR.axhline(50, color="gray", ls="--", lw=1, label="50% = chance")
    axR.set_xticks(x); axR.set_xticklabels([l.upper() for l in LANGUAGES])
    axR.set_ylabel("Sign agreement (%)"); axR.set_title(f"{title}: direction")
    axR.legend(); axR.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()


def main():
    analyse_rag()
    analyse_p5()
    print("\nDONE — Claude follow-up analysis complete.")


if __name__ == "__main__":
    main()
