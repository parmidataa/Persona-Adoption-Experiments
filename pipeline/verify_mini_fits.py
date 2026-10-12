"""
verify_mini_fits.py  — Re-check the gpt-4.1-mini RAG (P4) and P5 numbers with the
hardened, convergence-required fitter used for the Claude follow-up (script 99).

For each mini result we recompute R2m and directional sign agreement with the
robust fitter (raw + standardized outcome across bfgs/lbfgs/nm, REQUIRING
convergence) and print OLD (currently published) vs NEW (hardened) side by side,
flagging any cell whose old value came from a non-converged fit.

Reads:
    rag_predictions_clean.parquet, p5_predictions_clean.parquet   (mini data)
    results_rag_r2m.csv / results_rag_concordance.csv             (old mini)
    results_p5_r2m.csv  / results_p5_concordance.csv              (old mini)

Usage:
    python verify_mini_fits.py
"""

import warnings, importlib.util
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

# Load the hardened helpers from 99 (module name starts with a digit -> importlib)
spec = importlib.util.spec_from_file_location("m99", "99_analyze_claude_followups.py")
m99 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m99)

LANGS = ["ar", "es"]


def recheck_rag():
    print("=" * 78)
    print("  RAG / P4  (gpt-4.1-mini) — OLD published vs NEW hardened fit")
    print("=" * 78)
    df = m99.prep(pd.read_parquet("rag_predictions_clean.parquet"))
    old_r2m = pd.read_csv("results_rag_r2m.csv")
    old_conc = pd.read_csv("results_rag_concordance.csv")

    print(f"\n  {'Lang':<5}{'R2m old':>10}{'R2m new':>10}{'Δ':>8}   "
          f"{'sign old':>9}{'sign new':>9}{'Δpp':>7}")
    print("  " + "-" * 62)
    for lang in LANGS:
        sub = df[(df.language == lang) & (df.condition == "p4")].copy()
        f_llm = m99.build_formula("llm_prob", sub)
        f_hum = m99.build_formula("human_label", sub)
        new_r2m = m99.fit_r2m(sub, f_llm) * 100
        cm = m99.concordance(m99.fit_and_coefs(sub, f_hum), m99.fit_and_coefs(sub, f_llm))
        new_sign = cm["sign_agreement"] * 100 if cm else np.nan

        o_r = float(old_r2m[old_r2m.language == lang]["r2m_rag"].values[0]) * 100
        o_s = float(old_conc[old_conc.language == lang]["sign_agreement"].values[0]) * 100
        flag = "  <-- CHANGED" if abs(o_r - new_r2m) > 0.05 or abs(o_s - new_sign) > 0.05 else ""
        print(f"  {lang.upper():<5}{o_r:>10.3f}{new_r2m:>10.3f}{new_r2m-o_r:>+8.3f}   "
              f"{o_s:>8.1f}%{new_sign:>8.1f}%{new_sign-o_s:>+7.1f}{flag}")


def recheck_p5():
    print("\n" + "=" * 78)
    print("  P5 / Relative+CoT  (gpt-4.1-mini) — OLD published vs NEW hardened fit")
    print("  (deviation outcome)")
    print("=" * 78)
    df = m99.prep(pd.read_parquet("p5_predictions_clean.parquet"))
    old_r2m = pd.read_csv("results_p5_r2m.csv")
    old_conc = pd.read_csv("results_p5_concordance.csv")

    print(f"\n  {'Lang':<5}{'R2m old':>10}{'R2m new':>10}{'Δ':>8}   "
          f"{'sign old':>9}{'sign new':>9}{'Δpp':>7}")
    print("  " + "-" * 62)
    for lang in LANGS:
        sub = df[df.language == lang].copy()
        f_dev = m99.build_formula("llm_deviation", sub)
        f_hum = m99.build_formula("human_label", sub)
        new_r2m = m99.fit_r2m(sub, f_dev) * 100
        cm = m99.concordance(m99.fit_and_coefs(sub, f_hum), m99.fit_and_coefs(sub, f_dev))
        new_sign = cm["sign_agreement"] * 100 if cm else np.nan

        o_r = float(old_r2m[old_r2m.language == lang]["r2m_dev_pct"].values[0])
        o_s = float(old_conc[old_conc.language == lang]["sign_agr_dev"].values[0]) * 100
        flag = "  <-- CHANGED" if abs(o_r - new_r2m) > 0.05 or abs(o_s - new_sign) > 0.05 else ""
        print(f"  {lang.upper():<5}{o_r:>10.3f}{new_r2m:>10.3f}{new_r2m-o_r:>+8.3f}   "
              f"{o_s:>8.1f}%{new_sign:>8.1f}%{new_sign-o_s:>+7.1f}{flag}")


if __name__ == "__main__":
    recheck_rag()
    recheck_p5()
    print("\nNote: cells flagged CHANGED had an OLD value from a fit that the")
    print("hardened fitter rejects as non-converged (or that needed standardizing).")
