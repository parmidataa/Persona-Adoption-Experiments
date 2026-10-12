"""
75_rag_r2m.py
=============
R²m analysis for the RAG experiment (P4 condition).

Computes marginal R² for the P4 (RAG-augmented, probability-format) condition
and produces a three-way comparison table:

  | Condition          | AR R²m  | ES R²m  | AR %SH  | ES %SH  |
  |--------------------|---------|---------|---------|---------|
  | P2 binary  (Ch.4)  |         |         |         |         |
  | P2 prob.   (Exp 1) |         |         |         |         |
  | P4 RAG+prob (Exp2) |         |         |         |         |

%SH for all three rows uses the binary sample-human baseline from
results_r2m_llm.csv — the human_prop approach used in 64_prob_r2m.py
was found to degenerate for most conditions (see §4.8.4 footnote).
This gives a consistent denominator across all three experiments.

P4 uses the same LMM formula as P2 (M1_BASE + M4_EXTRAS), because P4
is designed as an enhanced version of P2 with the same demographic fields
but with retrieved annotation examples replacing the abstract field list.

Outputs:
    results_rag_r2m.csv           — R²m for P4 per language
    results_rag_comparison.csv    — three-way comparison table
    fig_rag_r2m_comparison.png    — bar chart: binary vs prob vs RAG

Usage:
    python 75_rag_r2m.py
    (Run AFTER 74_parse_rag_results.py)
"""

import warnings, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

LANGUAGES  = ["ar", "es"]
CONDITION  = "p4"          # P4 uses the P2-style formula

# ── Full-dataset Study 1 benchmarks (from 29_rmm_analysis.py) ────────────────
# Used as context only — %SH uses binary sample-human baseline below.
BENCHMARKS = {
    "ar": {"m1": 0.019017, "m4": 0.038825, "target": 0.076480},
    "es": {"m1": 0.010041, "m4": 0.032465, "target": 0.093269},
}

# ── P4 uses the P2 (M1_BASE + M4_EXTRAS) formula ─────────────────────────────
M1_BASE = {
    "ar": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "es": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
}
M4_EXTRAS = [
    "C(country_birth)", "C(country_residence)",
    "C(employment_status)", "C(student_status)",
]
ALL_PRED_COLS = [
    "age_centered", "gender", "ethnicity", "nationality",
    "country_birth", "country_residence",
    "employment_status", "student_status", "Generation",
]


def col_from_term(term: str) -> str:
    m = re.match(r"C\((\w+)\)", term)
    return m.group(1) if m else term


def build_formula(target: str, var_list: list, df_check: pd.DataFrame) -> str | None:
    valid = []
    for term in var_list:
        col = col_from_term(term)
        if col not in df_check.columns:
            continue
        if col == "age_centered":
            valid.append(term); continue
        if df_check[col].dropna().nunique() >= 2:
            valid.append(term)
    if not valid:
        return None
    return f"{target} ~ " + " + ".join(valid)


def get_formula_p4(target: str, lang: str, df_check: pd.DataFrame) -> str | None:
    """P4 always uses the full P2 (M1_BASE + M4_EXTRAS) formula."""
    return build_formula(target, M1_BASE[lang] + M4_EXTRAS, df_check)


def needed_cols(formula: str, df: pd.DataFrame, target: str) -> list:
    cols = [target, "post_id"]
    for col in ALL_PRED_COLS:
        if col in formula and col in df.columns:
            cols.append(col)
    return list(dict.fromkeys(cols))


def fit_r2m(df_sub: pd.DataFrame, formula: str, method: str = "bfgs") -> dict:
    target = formula.split("~")[0].strip()
    nc     = needed_cols(formula, df_sub, target)
    df_fit = df_sub[[c for c in nc if c in df_sub.columns]].dropna()

    if len(df_fit) < 50 or df_fit[target].nunique() < 2:
        return {"r2m": np.nan, "n": len(df_fit), "converged": False}

    try:
        m   = smf.mixedlm(formula, data=df_fit,
                          groups=df_fit["post_id"]).fit(
                              reml=False, method=method, disp=False)
        fp  = np.dot(m.model.exog, m.fe_params)
        vf  = np.var(fp)
        vr  = float(m.cov_re.iloc[0, 0])
        r2m = vf / (vf + vr + m.scale)
        return {"r2m": r2m, "n": len(df_fit), "converged": m.converged}
    except Exception as e:
        return {"r2m": np.nan, "n": 0, "converged": False, "error": str(e)}


# ── Load RAG predictions ───────────────────────────────────────────────────────
rag_path = Path("rag_predictions_clean.parquet")
if not rag_path.exists():
    print("ERROR: rag_predictions_clean.parquet not found. Run 74_parse_rag_results.py first.")
    exit(1)

df_rag = pd.read_parquet(rag_path)
df_rag["age"] = pd.to_numeric(df_rag["age"], errors="coerce")
lang_age_mean = df_rag.groupby("language")["age"].transform("mean")
df_rag["age_centered"] = df_rag["age"] - lang_age_mean
for col in ["gender", "ethnicity", "nationality", "country_birth",
            "country_residence", "employment_status", "student_status"]:
    if col in df_rag.columns:
        df_rag[col] = df_rag[col].replace("unknown", np.nan)

print(f"Loaded {len(df_rag):,} RAG prediction rows")
print(f"Languages  : {sorted(df_rag['language'].unique())}")
print(f"Condition  : {sorted(df_rag['condition'].unique())}")

# ── Load binary sample-human baselines for %SH denominator ───────────────────
# We use the binary SH as the consistent denominator for all three experiments.
# (See §4.8.4 footnote for why human_prop-based SH degenerates.)
try:
    df_bin_r2m = pd.read_csv("results_r2m_llm.csv")
    BINARY_SH = {}
    for lang in LANGUAGES:
        # P2 condition SH = the condition-matched human baseline for P2
        row_sh = df_bin_r2m[
            (df_bin_r2m["language"] == lang) &
            (df_bin_r2m["condition"] == "p2")
        ]
        if len(row_sh) > 0 and "sample_human_r2m" in row_sh.columns:
            BINARY_SH[lang] = float(row_sh["sample_human_r2m"].values[0])
        else:
            # Fallback: derive from pct_of_sample_human × r2m_llm_pct
            # if sample_human_r2m column is not present, use the Study 1 M1 benchmark
            BINARY_SH[lang] = BENCHMARKS[lang]["m1"]
    print(f"\nBinary SH baselines (from results_r2m_llm.csv, P2 condition):")
    for lang, sh in BINARY_SH.items():
        print(f"  {lang.upper()}: {sh*100:.3f}%")
except FileNotFoundError:
    print("WARNING: results_r2m_llm.csv not found — using Study 1 M1 benchmark as SH.")
    BINARY_SH = {lang: BENCHMARKS[lang]["m1"] for lang in LANGUAGES}

# ── Compute P4 R²m ────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("  R²m  —  P4 (RAG + probability format)")
print(f"{'═'*65}")

rag_results = []
for lang in LANGUAGES:
    mask   = (df_rag["language"] == lang) & (df_rag["condition"] == CONDITION)
    df_sub = df_rag[mask].copy()
    formula = get_formula_p4("llm_prob", lang, df_sub)
    if formula is None:
        print(f"  {lang.upper()} P4: formula could not be built — skipped")
        rag_results.append({"language": lang, "r2m_rag": np.nan, "pct_sh_rag": np.nan})
        continue

    res  = fit_r2m(df_sub, formula)
    r2m  = res["r2m"]
    sh   = BINARY_SH.get(lang, np.nan)
    pct  = (r2m / sh * 100) if (not np.isnan(r2m) and sh and sh > 0) else np.nan

    bench = BENCHMARKS[lang]
    print(f"  {lang.upper()} P4: R²m={r2m*100:.3f}%   SH(binary P2)={sh*100:.3f}%"
          f"   %SH={pct:.1f}%   n={res['n']}")
    rag_results.append({"language": lang, "r2m_rag": r2m, "pct_sh_rag": pct})

df_rag_res = pd.DataFrame(rag_results)
df_rag_res.to_csv("results_rag_r2m.csv", index=False)
print(f"\nSaved: results_rag_r2m.csv")

# ── Three-way comparison ──────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("  THREE-WAY COMPARISON: Binary vs Probability vs RAG")
print(f"{'═'*65}")

try:
    df_bin = pd.read_csv("results_r2m_llm.csv")
    df_bin_p2 = df_bin[
        (df_bin["language"].isin(LANGUAGES)) &
        (df_bin["condition"] == "p2")
    ][["language", "r2m_llm_pct", "pct_of_sample_human"]].rename(columns={
        "r2m_llm_pct":         "r2m_binary_pct",
        "pct_of_sample_human": "pct_sh_binary",
    })
except FileNotFoundError:
    df_bin_p2 = pd.DataFrame({"language": LANGUAGES,
                               "r2m_binary_pct": [np.nan]*2,
                               "pct_sh_binary":  [np.nan]*2})

try:
    df_prob = pd.read_csv("results_prob_r2m.csv")
    df_prob_p2 = df_prob[
        (df_prob["language"].isin(LANGUAGES)) &
        (df_prob["condition"] == "p2")
    ][["language", "r2m_prob_pct"]].rename(columns={
        "r2m_prob_pct": "r2m_prob_pct",
    })
    # Recompute %SH(prob) using binary SH for consistency
    df_prob_p2 = df_prob_p2.copy()
    df_prob_p2["pct_sh_prob"] = df_prob_p2.apply(
        lambda r: r["r2m_prob_pct"] / (BINARY_SH.get(r["language"], np.nan) * 100) * 100
        if BINARY_SH.get(r["language"]) else np.nan, axis=1
    )
except FileNotFoundError:
    df_prob_p2 = pd.DataFrame({"language": LANGUAGES,
                                "r2m_prob_pct": [np.nan]*2,
                                "pct_sh_prob":  [np.nan]*2})

df_rag_res["r2m_rag_pct"] = df_rag_res["r2m_rag"] * 100

comp = (df_bin_p2
        .merge(df_prob_p2, on="language", how="outer")
        .merge(df_rag_res[["language", "r2m_rag_pct", "pct_sh_rag"]], on="language", how="outer"))
comp = comp[comp["language"].isin(LANGUAGES)].reset_index(drop=True)
comp.to_csv("results_rag_comparison.csv", index=False)

print(f"\n{'Lang':<5} {'P2-binary':>12} {'P2-prob':>10} {'P4-RAG':>10}")
print(f"{'':5} {'R²m  %SH':>12} {'R²m  %SH':>10} {'R²m  %SH':>10}")
print("-"*55)
for _, r in comp.iterrows():
    print(f"{r['language'].upper():<5} "
          f"{r.get('r2m_binary_pct',np.nan):>5.2f}% "
          f"{r.get('pct_sh_binary',np.nan):>5.1f}%    "
          f"{r.get('r2m_prob_pct',np.nan):>5.2f}% "
          f"{r.get('pct_sh_prob',np.nan):>5.1f}%   "
          f"{r.get('r2m_rag_pct',np.nan):>5.2f}% "
          f"{r.get('pct_sh_rag',np.nan):>5.1f}%")

print(f"\nSaved: results_rag_comparison.csv")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x   = np.arange(len(LANGUAGES))
w   = 0.22
bar_data = [
    ("P2 binary (Ch.4)",    comp["r2m_binary_pct"].values, "#4c9be8", 0.7),
    ("P2 probability (Exp 1)", comp["r2m_prob_pct"].values, "#1a5fa8", 0.7),
    ("P4 RAG (Exp 2)",      comp["r2m_rag_pct"].values,   "#e85c4c", 0.85),
]
for i, (label, vals, color, alpha) in enumerate(bar_data):
    ax.bar(x + (i - 1) * w, vals, w, label=label, color=color, alpha=alpha)

ax.set_xticks(x)
ax.set_xticklabels([l.upper() for l in LANGUAGES], fontsize=11)
ax.set_ylabel("$R^2_m$ (%)", fontsize=11)
ax.set_title("$R^2_m$ comparison: Binary vs Probability vs RAG (P4)\n"
             "Arabic and Spanish, P2-equivalent formula", fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for lang_i, lang in enumerate(LANGUAGES):
    sh = BINARY_SH.get(lang, np.nan)
    if not np.isnan(sh):
        ax.axhline(sh * 100, color="grey", linestyle="--", linewidth=0.8,
                   label=f"{lang.upper()} SH" if lang_i == 0 else None)

plt.tight_layout()
plt.savefig("fig_rag_r2m_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_rag_r2m_comparison.png")

print(f"\n{'═'*65}")
print("  DONE — next step: python 76_rag_concordance.py")
print(f"{'═'*65}")
