"""
84_p5_r2m.py
============
R²m analysis for the P5 (Relative Framing + Chain-of-Thought) condition.

Two parallel analyses on the P5 predictions:

  Analysis A — Deviation outcome (llm_deviation ∈ [−1, 1])
    formula: llm_deviation ~ demographics + (1|post_id)
    This is the most natural outcome for P5: R²m directly measures how much
    demographic variance the model encodes in its deviation estimates.
    Since deviation is centred by design (~0 mean), the fixed effects are
    directly comparable to human LMM coefficients.

  Analysis B — Reconstructed probability (llm_prob = baseline + llm_deviation)
    formula: llm_prob ~ demographics + (1|post_id)
    This allows direct comparison with P2-binary, P2-prob, and P4-RAG R²m
    values using the same scale (all three report LMM on an absolute prob/label).

Both use the P2 (M1_BASE + M4_EXTRAS) formula — P5 is designed as an
enhanced P2.

%SH denominator: binary sample-human baseline (from P2 rows in
results_r2m_llm.csv), consistent with the P4 analysis.

Outputs:
    results_p5_r2m.csv         — both analyses per language
    results_p5_comparison.csv  — four-way comparison table
    fig_p5_r2m_comparison.png  — bar chart

Usage:
    python 84_p5_r2m.py   (run after 83_parse_p5_results.py)
"""

import warnings
warnings.filterwarnings("ignore")

import re
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

LANGUAGES  = ["ar", "es"]
CONDITION  = "p5"

# ── Full-dataset Study 1 benchmarks ──────────────────────────────────────────
BENCHMARKS = {
    "ar": {"m1": 0.019017, "m4": 0.038825, "target": 0.076480},
    "es": {"m1": 0.010041, "m4": 0.032465, "target": 0.093269},
}

# ── P5 uses P2 formula (M1_BASE + M4_EXTRAS) ─────────────────────────────────
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


def get_formula(target: str, lang: str, df_check: pd.DataFrame) -> str | None:
    return build_formula(target, M1_BASE[lang] + M4_EXTRAS, df_check)


def needed_cols(formula: str, df: pd.DataFrame, target: str) -> list:
    cols = [target, "post_id"]
    for col in ALL_PRED_COLS:
        if col in formula and col in df.columns:
            cols.append(col)
    return list(dict.fromkeys(cols))


def fit_r2m(df_sub: pd.DataFrame, formula: str, method: str = "bfgs") -> dict:
    target  = formula.split("~")[0].strip()
    nc      = needed_cols(formula, df_sub, target)
    df_fit  = df_sub[[c for c in nc if c in df_sub.columns]].dropna()

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
        try:
            return fit_r2m.__wrapped__(df_sub, formula, method="lbfgs")
        except Exception:
            return {"r2m": np.nan, "n": 0, "converged": False, "error": str(e)}


# ── Load data ─────────────────────────────────────────────────────────────────
p5_path = Path("p5_predictions_clean.parquet")
if not p5_path.exists():
    print("ERROR: p5_predictions_clean.parquet not found. Run 83_parse_p5_results.py first.")
    exit(1)

df = pd.read_parquet(p5_path)
df["age"] = pd.to_numeric(df["age"], errors="coerce")
lang_age_mean = df.groupby("language")["age"].transform("mean")
df["age_centered"] = df["age"] - lang_age_mean

for col in ["gender", "ethnicity", "nationality", "country_birth",
            "country_residence", "employment_status", "student_status"]:
    if col in df.columns:
        df[col] = df[col].replace("unknown", np.nan)

print(f"Loaded {len(df):,} P5 prediction rows")
print(f"  llm_deviation: mean={df['llm_deviation'].mean():+.3f}  "
      f"std={df['llm_deviation'].std():.3f}")
print(f"  llm_prob     : mean={df['llm_prob'].mean():.3f}  "
      f"std={df['llm_prob'].std():.3f}")

# ── Load binary SH baselines ──────────────────────────────────────────────────
BINARY_SH = {}
try:
    df_bin = pd.read_csv("results_r2m_llm.csv")
    for lang in LANGUAGES:
        row_sh = df_bin[
            (df_bin["language"] == lang) & (df_bin["condition"] == "p2")
        ]
        if len(row_sh) > 0 and "sample_human_r2m" in row_sh.columns:
            BINARY_SH[lang] = float(row_sh["sample_human_r2m"].values[0])
        else:
            BINARY_SH[lang] = BENCHMARKS[lang]["m1"]
    print(f"\nBinary SH baselines (P2 from results_r2m_llm.csv):")
    for lang, sh in BINARY_SH.items():
        print(f"  {lang.upper()}: {sh*100:.3f}%")
except FileNotFoundError:
    print("WARNING: results_r2m_llm.csv not found — using Study 1 M1 benchmark as SH.")
    BINARY_SH = {lang: BENCHMARKS[lang]["m1"] for lang in LANGUAGES}


# ── Run both analyses ─────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("  P5 R²m RESULTS")
print(f"{'═'*65}")

p5_rows = []
for lang in LANGUAGES:
    mask   = df["language"] == lang
    df_sub = df[mask].copy()

    sh = BINARY_SH.get(lang, np.nan)

    # Analysis A: deviation outcome
    formula_dev = get_formula("llm_deviation", lang, df_sub)
    res_dev = fit_r2m(df_sub, formula_dev) if formula_dev else {"r2m": np.nan, "n": 0}

    # Analysis B: reconstructed probability outcome
    formula_prob = get_formula("llm_prob", lang, df_sub)
    res_prob = fit_r2m(df_sub, formula_prob) if formula_prob else {"r2m": np.nan, "n": 0}

    r2m_dev  = res_dev["r2m"]
    r2m_prob = res_prob["r2m"]
    pct_dev  = (r2m_dev  / sh * 100) if (not np.isnan(r2m_dev)  and sh > 0) else np.nan
    pct_prob = (r2m_prob / sh * 100) if (not np.isnan(r2m_prob) and sh > 0) else np.nan

    print(f"\n  {lang.upper()} P5 (deviation outcome):    "
          f"R²m={r2m_dev*100:.3f}%   %SH={pct_dev:.1f}%   n={res_dev['n']}")
    print(f"  {lang.upper()} P5 (prob. outcome):       "
          f"R²m={r2m_prob*100:.3f}%  %SH={pct_prob:.1f}%   n={res_prob['n']}")
    print(f"  {lang.upper()} Human SH ceiling (binary): {sh*100:.3f}%")
    print(f"  {lang.upper()} Study 1 Full-data target:  {BENCHMARKS[lang]['target']*100:.3f}%")

    p5_rows.append({
        "language":       lang,
        "condition":      CONDITION,
        "r2m_dev":        r2m_dev,
        "r2m_dev_pct":    r2m_dev * 100 if not np.isnan(r2m_dev) else np.nan,
        "pct_sh_dev":     pct_dev,
        "r2m_prob":       r2m_prob,
        "r2m_prob_pct":   r2m_prob * 100 if not np.isnan(r2m_prob) else np.nan,
        "pct_sh_prob":    pct_prob,
        "n":              res_dev["n"],
        "binary_sh":      sh,
    })

df_p5 = pd.DataFrame(p5_rows)
df_p5.to_csv("results_p5_r2m.csv", index=False)
print(f"\nSaved: results_p5_r2m.csv")


# ── Four-way comparison ───────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("  FOUR-WAY COMPARISON: Binary / Prob / RAG / P5")
print(f"{'═'*65}")

# Load previous results
try:
    df_bin_raw = pd.read_csv("results_r2m_llm.csv")
    df_bin_p2  = df_bin_raw[
        (df_bin_raw["language"].isin(LANGUAGES)) & (df_bin_raw["condition"] == "p2")
    ][["language", "r2m_llm_pct", "pct_of_sample_human"]].rename(columns={
        "r2m_llm_pct":         "r2m_binary_pct",
        "pct_of_sample_human": "pct_sh_binary",
    })
except FileNotFoundError:
    df_bin_p2  = pd.DataFrame({"language": LANGUAGES,
                                "r2m_binary_pct": [np.nan]*2,
                                "pct_sh_binary":  [np.nan]*2})

try:
    df_prob_raw = pd.read_csv("results_prob_r2m.csv")
    df_prob_p2  = df_prob_raw[
        (df_prob_raw["language"].isin(LANGUAGES)) & (df_prob_raw["condition"] == "p2")
    ][["language", "r2m_prob_pct"]].rename(columns={"r2m_prob_pct": "r2m_prob_pct"})
    df_prob_p2["pct_sh_prob"] = df_prob_p2.apply(
        lambda r: r["r2m_prob_pct"] / (BINARY_SH.get(r["language"], np.nan) * 100) * 100
        if BINARY_SH.get(r["language"]) else np.nan, axis=1
    )
except FileNotFoundError:
    df_prob_p2 = pd.DataFrame({"language": LANGUAGES,
                                "r2m_prob_pct": [np.nan]*2,
                                "pct_sh_prob":  [np.nan]*2})

try:
    df_rag_raw = pd.read_csv("results_rag_r2m.csv")
    df_rag_p4  = df_rag_raw[
        df_rag_raw["language"].isin(LANGUAGES)
    ][["language", "r2m_rag", "pct_sh_rag"]].copy()
    df_rag_p4["r2m_rag_pct"] = df_rag_p4["r2m_rag"] * 100
except FileNotFoundError:
    df_rag_p4  = pd.DataFrame({"language": LANGUAGES,
                                "r2m_rag_pct": [np.nan]*2,
                                "pct_sh_rag":  [np.nan]*2})

comp = (df_bin_p2
        .merge(df_prob_p2, on="language", how="outer")
        .merge(df_rag_p4[["language", "r2m_rag_pct", "pct_sh_rag"]], on="language", how="outer")
        .merge(df_p5[["language", "r2m_dev_pct", "pct_sh_dev"]].rename(
                columns={"r2m_dev_pct": "r2m_p5_pct", "pct_sh_dev": "pct_sh_p5"}
               ), on="language", how="outer"))
comp = comp[comp["language"].isin(LANGUAGES)].reset_index(drop=True)
comp.to_csv("results_p5_comparison.csv", index=False)

header = f"{'Lang':<5} {'P2-binary':>12} {'P2-prob':>10} {'P4-RAG':>10} {'P5-CoT':>10}"
print(f"\n{header}")
print(f"{'':5} {'R²m  %SH':>12} {'R²m  %SH':>10} {'R²m  %SH':>10} {'R²m  %SH':>10}")
print("-"*65)
for _, r in comp.iterrows():
    print(f"{r['language'].upper():<5} "
          f"{r.get('r2m_binary_pct', np.nan):>5.2f}% "
          f"{r.get('pct_sh_binary',  np.nan):>5.1f}%    "
          f"{r.get('r2m_prob_pct',   np.nan):>5.2f}% "
          f"{r.get('pct_sh_prob',    np.nan):>5.1f}%   "
          f"{r.get('r2m_rag_pct',    np.nan):>5.2f}% "
          f"{r.get('pct_sh_rag',     np.nan):>5.1f}%   "
          f"{r.get('r2m_p5_pct',     np.nan):>5.2f}% "
          f"{r.get('pct_sh_p5',      np.nan):>5.1f}%")
print(f"\nSaved: results_p5_comparison.csv")


# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x   = np.arange(len(LANGUAGES))
w   = 0.17

bar_data = [
    ("P2 binary",       comp["r2m_binary_pct"].values, "#a8d5e2", 0.85),
    ("P2 probability",  comp["r2m_prob_pct"].values,   "#5ba4cf", 0.85),
    ("P4 RAG",          comp["r2m_rag_pct"].values,    "#e85c4c", 0.85),
    ("P5 relative+CoT", comp["r2m_p5_pct"].values,     "#2d7f3c", 0.85),
]
for i, (label, vals, color, alpha) in enumerate(bar_data):
    offsets = x + (i - 1.5) * w
    ax.bar(offsets, np.nan_to_num(vals), w, label=label, color=color, alpha=alpha)
    for xi, v in zip(offsets, vals):
        if not np.isnan(v):
            ax.text(xi, v + 0.05, f"{v:.2f}%", ha="center", va="bottom",
                    fontsize=7, rotation=0)

ax.set_xticks(x)
ax.set_xticklabels([l.upper() for l in LANGUAGES], fontsize=11)
ax.set_ylabel("$R^2_m$ (%)", fontsize=11)
ax.set_title("$R^2_m$ by Prompt Condition: Binary vs Probability vs RAG vs Relative+CoT\n"
             "Arabic and Spanish, P2-equivalent LMM formula", fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for lang_i, lang in enumerate(LANGUAGES):
    sh = BINARY_SH.get(lang, np.nan)
    if not np.isnan(sh):
        ax.axhline(sh * 100, color="grey", linestyle="--", linewidth=0.8,
                   xmin=lang_i * 0.5 + 0.02, xmax=lang_i * 0.5 + 0.48,
                   label=f"{lang.upper()} human ceiling" if lang_i == 0 else None)

plt.tight_layout()
plt.savefig("fig_p5_r2m_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_p5_r2m_comparison.png")

print(f"\n{'═'*65}")
print("  DONE — next step: python 85_p5_concordance.py")
print(f"{'═'*65}")
