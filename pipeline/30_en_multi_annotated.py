"""
30_en_multi_annotated.py  — Option 2
=====================================
Re-analyse English using only the subset of texts that have >= 2 annotators
in the 700-row sample.  No new API calls needed.

Hypothesis: P0's inflated R²_m (1.307%) is an artifact of the 74% solo-
annotated structure in the EN sample.  If so, the solo-annotated texts carry
a spurious demographic signal (which texts certain demographics happen to see),
not genuine annotator-level demographic variance.

Two thresholds tested:
  • >=2 annotators/post  (315 rows, 134 posts)
  • >=3 annotators/post  (105 rows, 30 posts) — note: very small

Outputs:
    results_en_multi_annotated.csv   — R²_m per condition on each subset
    fig_en_multi_annotated.png       — comparison bar chart
"""

import warnings, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Reuse formula definitions from 29_rmm_analysis.py ──────────────────────
M1_BASE = {
    "en": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
}
M4_EXTRAS = ["C(country_birth)", "C(country_residence)",
             "C(employment_status)", "C(student_status)"]
P3_TEMPLATE = {
    "en": ["C(nationality)", "C(gender)", "C(country_residence)"],
}
ALL_PRED_COLS = ["age_centered", "gender", "ethnicity", "nationality",
                 "country_birth", "country_residence",
                 "employment_status", "student_status", "Generation"]

CONDITIONS = ["p0", "p1", "p2", "p3"]
FULL_R2M   = {"p0": 1.307, "p1": 1.222, "p2": 3.566, "p3": 0.914}  # from Stage 3
FULL_SH_M1 = 2.740   # H-baseline(M1) on full 700-row EN sample
FULL_SH_M4 = 8.546   # H-baseline(M4) on full 700-row EN sample


def col_from_term(term):
    m = re.match(r"C\((\w+)\)", term)
    return m.group(1) if m else term


def build_formula(target, var_list, df_check):
    valid = []
    for term in var_list:
        col = col_from_term(term)
        if col not in df_check.columns:
            continue
        if col == "age_centered":
            valid.append(term)
            continue
        if df_check[col].dropna().nunique() >= 2:
            valid.append(term)
    if not valid:
        return None
    return f"{target} ~ " + " + ".join(valid)


def get_formula(target, cond, df_check):
    if cond in ("p0", "p1"):
        return build_formula(target, M1_BASE["en"], df_check)
    elif cond == "p2":
        return build_formula(target, M1_BASE["en"] + M4_EXTRAS, df_check)
    else:
        return build_formula(target, P3_TEMPLATE["en"], df_check)


def needed_cols(formula, target):
    cols = [target, "post_id"]
    for col in ALL_PRED_COLS:
        if col in formula:
            cols.append(col)
    return list(dict.fromkeys(cols))


def fit_r2m(df_sub, formula):
    target = formula.split("~")[0].strip()
    nc = [c for c in needed_cols(formula, target) if c in df_sub.columns]
    df_fit = df_sub[nc].dropna()
    if len(df_fit) < 30 or df_fit[target].nunique() < 2:
        return {"r2m": np.nan, "n": len(df_fit)}
    try:
        m = smf.mixedlm(formula, data=df_fit,
                        groups=df_fit["post_id"]).fit(
                            reml=False, method="bfgs", disp=False)
        fp = np.dot(m.model.exog, m.fe_params)
        vf = np.var(fp)
        vr = float(m.cov_re.iloc[0, 0])
        r2m = vf / (vf + vr + m.scale)
        return {"r2m": r2m, "n": len(df_fit), "converged": m.converged}
    except Exception as e:
        return {"r2m": np.nan, "n": 0, "error": str(e)}


# ── Load data ─────────────────────────────────────────────────────────────────
df_all = pd.read_parquet("data_predictions_clean.parquet")
df_all["age"] = pd.to_numeric(df_all["age"], errors="coerce")
lang_age_mean = df_all.groupby("language")["age"].transform("mean")
df_all["age_centered"] = df_all["age"] - lang_age_mean

DEMO_CATS = ["gender", "ethnicity", "nationality", "country_birth",
             "country_residence", "employment_status", "student_status", "Generation"]
for col in DEMO_CATS:
    if col in df_all.columns:
        df_all[col] = df_all[col].replace("unknown", np.nan)

df_en = df_all[df_all["language"] == "en"].copy()

# Identify multi-annotated posts (based on p0 slice which has one row per annotator)
ann_per_post = (df_en[df_en["condition"] == "p0"]
                .groupby("post_id")["annotator_id"].nunique())

posts_ge2 = ann_per_post[ann_per_post >= 2].index
posts_ge3 = ann_per_post[ann_per_post >= 3].index

print(f"EN sample:  total posts={len(ann_per_post)}, >=2ann posts={len(posts_ge2)}, >=3ann posts={len(posts_ge3)}")

# ── Compute human R²_m on each subset ─────────────────────────────────────────
print(f"\n{'═'*70}")
print("  SAMPLE HUMAN R²_m on each subset")
print(f"{'═'*70}")

subsets = {
    "full_700":  df_en,
    "multi_ge2": df_en[df_en["post_id"].isin(posts_ge2)],
    "multi_ge3": df_en[df_en["post_id"].isin(posts_ge3)],
}

human_r2m = {}
for subset_name, df_sub in subsets.items():
    df_h = df_sub[df_sub["condition"] == "p0"].copy()
    sh = {}
    for cond in CONDITIONS:
        hf = get_formula("human_label", cond, df_h)
        if hf is None:
            sh[cond] = np.nan
            continue
        res = fit_r2m(df_h, hf)
        sh[cond] = res["r2m"]
    human_r2m[subset_name] = sh
    print(f"  {subset_name:15} (n={len(df_h):4})  SH(M1)={sh['p0']*100:.3f}%  SH(M4)={sh['p2']*100 if sh['p2'] else float('nan'):.3f}%  SH(P3)={sh['p3']*100 if sh['p3'] else float('nan'):.3f}%")

# ── Main R²_m loop ─────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("  LLM R²_m per condition × subset")
print(f"{'═'*70}")

records = []
for subset_name, df_sub_all in subsets.items():
    for cond in CONDITIONS:
        df_cond = df_sub_all[df_sub_all["condition"] == cond].copy()
        formula = get_formula("llm_label", cond, df_cond)
        if formula is None:
            r2m_val = np.nan
        else:
            res = fit_r2m(df_cond, formula)
            r2m_val = res["r2m"]

        sh = human_r2m[subset_name].get(cond, np.nan)
        pct_sh = (r2m_val / sh * 100) if (r2m_val and sh and not np.isnan(r2m_val) and sh > 0) else np.nan
        n_posts = df_cond["post_id"].nunique()

        print(f"  {subset_name:15} {cond.upper()} "
              f"R²m={r2m_val*100 if r2m_val else float('nan'):.3f}%  "
              f"SH={sh*100 if sh else float('nan'):.3f}%  "
              f"%SH={pct_sh:.1f}%  (posts={n_posts})")

        records.append({
            "subset":     subset_name,
            "condition":  cond,
            "r2m_llm_pct": r2m_val * 100 if not np.isnan(r2m_val) else np.nan,
            "r2m_sh_pct":  sh * 100 if sh and not np.isnan(sh) else np.nan,
            "pct_of_sh":   pct_sh,
            "n_posts":     n_posts,
        })

df_out = pd.DataFrame(records)
df_out.to_csv("results_en_multi_annotated.csv", index=False)
print("\nSaved: results_en_multi_annotated.csv")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

cond_labels = {"p0": "P0 (no persona)", "p1": "P1 (M1 fields)",
               "p2": "P2 (M4 fields)", "p3": "P3 (prose)"}
cond_colors = {"p0": "#aec6e8", "p1": "#4c9be8", "p2": "#1a5fa8", "p3": "#e85c4c"}
subset_labels = {"full_700": "Full 700 rows", "multi_ge2": "≥2 ann/post\n(315 rows)",
                 "multi_ge3": "≥3 ann/post\n(105 rows)"}

# Left: R²_m by condition for each subset
ax = axes[0]
subset_keys = ["full_700", "multi_ge2", "multi_ge3"]
x = np.arange(len(subset_keys))
width = 0.2

for i, cond in enumerate(CONDITIONS):
    vals = [df_out[(df_out["subset"]==s) & (df_out["condition"]==cond)]["r2m_llm_pct"].values[0]
            for s in subset_keys]
    ax.bar(x + i*width, vals, width, label=cond_labels[cond],
           color=cond_colors[cond], alpha=0.9)

# Add human baseline line
sh_m1_vals = [human_r2m[s]["p0"] * 100 for s in subset_keys]
ax.plot(x + 1.5*width, sh_m1_vals, "g^--", markersize=8, linewidth=1.5,
        label="H-baseline(M1)", zorder=6)

ax.set_xticks(x + 1.5*width)
ax.set_xticklabels([subset_labels[s] for s in subset_keys], fontsize=9)
ax.set_ylabel("R²_m (%)")
ax.set_title("EN LLM R²_m by condition\nper annotator-density subset", fontsize=10)
ax.legend(fontsize=7.5)
ax.grid(axis="y", alpha=0.3)

# Right: P0 R²_m comparison (the key finding)
ax = axes[1]
p0_vals = [df_out[(df_out["subset"]==s) & (df_out["condition"]=="p0")]["r2m_llm_pct"].values[0]
           for s in subset_keys]
sh_vals  = [human_r2m[s]["p0"] * 100 for s in subset_keys]
bar_colors = ["#aec6e8", "#4c9be8", "#1a5fa8"]

x2 = np.arange(len(subset_keys))
bars = ax.bar(x2, p0_vals, color=bar_colors, alpha=0.9, width=0.5, label="P0 R²_m")
ax.scatter(x2, sh_vals, color="#27ae60", s=100, zorder=6, marker="^",
           label="H-baseline(M1)", clip_on=False)

for bar, v in zip(bars, p0_vals):
    if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.03,
                f"{v:.3f}%", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x2)
ax.set_xticklabels([subset_labels[s] for s in subset_keys], fontsize=9)
ax.set_ylabel("R²_m (%)")
ax.set_title("EN P0 (no persona) R²_m\nacross subsets — is it an artifact?", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("English: Multi-Annotated Subset Reanalysis (Option 2)\n"
             "Does solo-annotation inflate P0 R²_m?", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_en_multi_annotated.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_en_multi_annotated.png")
print(f"\n{'═'*70}")
print("  DONE")
print(f"{'═'*70}")
