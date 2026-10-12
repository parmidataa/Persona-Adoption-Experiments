"""
34_parse_extended_results.py
============================
Parses extended_batch_output.jsonl, merges with extended_batch_index.csv,
and produces a clean predictions parquet: extended_predictions.parquet

Runs all three analyses automatically after parsing:
  - Option 1: EN variable ablation R²_m analysis
  - Option 3: HI scaled sample R²_m analysis
  - Option 4: IT/PT P4 employment-targeted analysis
"""

import warnings, re
warnings.filterwarnings("ignore")

import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Parse batch output ────────────────────────────────────────────────────────
print("Parsing extended_batch_output.jsonl...")
outputs = {}
errors  = []
with open("extended_batch_output.jsonl") as f:
    for line in f:
        rec = json.loads(line.strip())
        cid = rec["custom_id"]
        if rec.get("error"):
            errors.append(cid)
            continue
        try:
            content = rec["response"]["body"]["choices"][0]["message"]["content"].strip()
            label   = int(content) if content in ("0","1") else -1
        except Exception:
            label = -1
        outputs[cid] = label

df_index = pd.read_csv("extended_batch_index.csv")
df_index["llm_label"] = df_index["custom_id"].map(outputs)
parse_ok   = (df_index["llm_label"] != -1) & df_index["llm_label"].notna()
parse_fail = (~parse_ok).sum()
print(f"  Parsed: {parse_ok.sum()} ok, {parse_fail} failures, {len(errors)} API errors")

# Drop failures
df_preds = df_index[parse_ok].copy()
df_preds["age"] = pd.to_numeric(df_preds["age"], errors="coerce")
lang_age_mean = df_preds.groupby("language")["age"].transform("mean")
df_preds["age_centered"] = df_preds["age"] - lang_age_mean

DEMO_CATS = ["gender","ethnicity","nationality","country_birth","country_residence",
             "employment_status","student_status"]
for col in DEMO_CATS:
    if col in df_preds.columns:
        df_preds[col] = df_preds[col].replace({"unknown": np.nan, "nan": np.nan})

df_preds.to_csv("extended_predictions.csv", index=False)
print(f"  Saved extended_predictions.csv ({len(df_preds)} rows)")


# ── Shared helpers ─────────────────────────────────────────────────────────────
M1_BASE = {
    "en": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
    "hi": ["age_centered", "C(gender)", "C(ethnicity)"],
    "it": ["age_centered", "C(gender)", "C(ethnicity)"],
    "pt": ["age_centered", "C(gender)", "C(ethnicity)", "C(nationality)"],
}
M4_EXTRAS = ["C(country_birth)", "C(country_residence)",
             "C(employment_status)", "C(student_status)"]
P3_TEMPLATE = {
    "en": ["C(nationality)", "C(gender)", "C(country_residence)"],
    "hi": ["age_centered", "C(ethnicity)", "C(country_birth)"],
    "it": ["age_centered", "C(employment_status)"],
    "pt": ["C(employment_status)", "C(country_birth)"],
}
ABLATION_VARS = {
    "p_age":       ["age_centered"],
    "p_gender":    ["C(gender)"],
    "p_ethnicity": ["C(ethnicity)"],
    "p_nat":       ["C(nationality)"],
    "p_country":   ["C(country_residence)"],
}
BENCHMARKS = {
    "en": {"m1": 0.013586, "m4": 0.048531},
    "hi": {"m1": 0.019916, "m4": 0.060088},
    "it": {"m1": 0.002709, "m4": 0.044459},
    "pt": {"m1": 0.010891, "m4": 0.043796},
}
# From Stage 3 analysis
STAGE3 = {
    "en": {"p0": 1.307, "p1": 1.222, "p2": 3.566, "p3": 0.914,
           "sh_m1": 2.740, "sh_m4": 8.546},
    "hi": {"p0": 0.044, "p1": 0.377, "p2": 1.305, "p3": 0.049,
           "sh_m1": 1.150, "sh_m4": 7.190},
    "it": {"p0": 0.341, "p1": 0.086, "p2": 0.811, "p3": 0.869,
           "sh_m1": 0.606},
    "pt": {"p0": 0.023, "p1": 1.497, "p2": np.nan, "p3": 0.668,
           "sh_m1": 0.519},
}
ALL_PRED_COLS = ["age_centered","gender","ethnicity","nationality",
                 "country_birth","country_residence","employment_status",
                 "student_status","Generation"]


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


def needed_cols(formula, target):
    cols = [target, "post_id"]
    for col in ALL_PRED_COLS:
        if col in formula:
            cols.append(col)
    return list(dict.fromkeys(cols))


def fit_r2m(df_sub, formula, method="bfgs"):
    target = formula.split("~")[0].strip()
    nc = [c for c in needed_cols(formula, target) if c in df_sub.columns]
    df_fit = df_sub[nc].dropna()
    if len(df_fit) < 30 or df_fit[target].nunique() < 2:
        return {"r2m": np.nan, "n": len(df_fit)}
    try:
        m = smf.mixedlm(formula, data=df_fit,
                        groups=df_fit["post_id"]).fit(reml=False, method=method, disp=False)
        fp = np.dot(m.model.exog, m.fe_params)
        vf = np.var(fp)
        vr = float(m.cov_re.iloc[0, 0])
        r2m = vf / (vf + vr + m.scale)
        return {"r2m": r2m, "n": len(df_fit), "converged": m.converged}
    except Exception as e:
        return {"r2m": np.nan, "n": 0, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  OPTION 1: EN variable ablation
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print("  OPTION 1: EN variable ablation")
print(f"{'═'*70}")

df_en_abl = df_preds[df_preds["language"] == "en"].copy()
ablation_conds = ["p_age", "p_gender", "p_ethnicity", "p_nat", "p_country"]

records_abl = []
for cond in ablation_conds:
    df_c = df_en_abl[df_en_abl["condition"] == cond].copy()
    if df_c.empty:
        print(f"  {cond}: NO DATA")
        continue
    var_list = ABLATION_VARS[cond]
    formula  = build_formula("llm_label", var_list, df_c)
    if formula is None:
        r2m_val = np.nan
        print(f"  {cond}: formula failed (no valid predictors)")
    else:
        res = fit_r2m(df_c, formula)
        r2m_val = res["r2m"]
        # Compare to P0 (1.307%) — does this variable alone cause disruption?
        ref_p0 = STAGE3["en"]["p0"] / 100
        direction = "↑ above P0" if (r2m_val and r2m_val > ref_p0) else "↓ below P0"
        print(f"  {cond}: R²m={r2m_val*100:.3f}%  ({direction}, P0=1.307%)")

    records_abl.append({
        "condition":    cond,
        "language":     "en",
        "r2m_pct":      r2m_val * 100 if r2m_val and not np.isnan(r2m_val) else np.nan,
        "n":            len(df_c),
    })

df_abl_out = pd.DataFrame(records_abl)
df_abl_out.to_csv("results_en_variable_ablation.csv", index=False)
print("Saved: results_en_variable_ablation.csv")

# Figure: EN variable ablation
fig, ax = plt.subplots(figsize=(9, 5))
cond_labels_abl = {
    "p_age": "P_age\n(age only)",
    "p_gender": "P_gender\n(gender only)",
    "p_ethnicity": "P_ethnicity\n(ethnicity only)",
    "p_nat": "P_nat\n(nationality only)",
    "p_country": "P_country\n(country_res only)",
}
vals = [df_abl_out[df_abl_out["condition"]==c]["r2m_pct"].values[0]
        if not df_abl_out[df_abl_out["condition"]==c].empty else np.nan
        for c in ablation_conds]
bar_colors = ["#27ae60" if (v and v > STAGE3["en"]["p0"]) else "#e74c3c" for v in vals]
bars = ax.bar([cond_labels_abl[c] for c in ablation_conds], vals,
              color=bar_colors, alpha=0.88, edgecolor="white")

# Reference lines
ax.axhline(STAGE3["en"]["p0"], color="black", linestyle="--", linewidth=1.5,
           label=f"P0 (no persona) = {STAGE3['en']['p0']:.3f}%")
ax.axhline(STAGE3["en"]["p1"], color="#4c9be8", linestyle=":", linewidth=1.5,
           label=f"P1 (all M1 vars) = {STAGE3['en']['p1']:.3f}%")

for bar, v in zip(bars, vals):
    if v and not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}%",
                ha="center", va="bottom", fontsize=9)

ax.set_ylabel("R²_m (%)")
ax.set_title("EN Variable Ablation (Option 1)\n"
             "Which single demographic variable drives or disrupts demographic signal?\n"
             "Green = above P0 (helpful), Red = below P0 (disruptive)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig_en_variable_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_en_variable_ablation.png")


# ══════════════════════════════════════════════════════════════════════════════
#  OPTION 3: HI sample scaling
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print("  OPTION 3: HI sample scaling (1400 rows)")
print(f"{'═'*70}")

df_hi_new = df_preds[df_preds["language"] == "hi"].copy()
hi_conds  = ["p0", "p1", "p2", "p3"]

# Compute sample human baseline on 1400 rows
df_hi_h = df_hi_new[df_hi_new["condition"] == "p0"].copy()
sh_hi = {}
for cond in hi_conds:
    if cond in ("p0","p1"):
        formula = build_formula("human_label", M1_BASE["hi"], df_hi_h)
    elif cond == "p2":
        formula = build_formula("human_label", M1_BASE["hi"] + M4_EXTRAS, df_hi_h)
    else:
        formula = build_formula("human_label", P3_TEMPLATE["hi"], df_hi_h)
    res = fit_r2m(df_hi_h, formula) if formula else {"r2m": np.nan}
    sh_hi[cond] = res["r2m"]

print(f"  HI 1400-row within-sample human baseline:")
print(f"    M1  (P0/P1): {sh_hi['p0']*100:.3f}%  (700-row was: {STAGE3['hi']['sh_m1']:.3f}%)")
print(f"    M4  (P2)   : {sh_hi['p2']*100:.3f}%  (700-row was: {STAGE3['hi']['sh_m4']:.3f}%)")
print(f"    Full-dataset M4 : {BENCHMARKS['hi']['m4']*100:.3f}%")

records_hi = []
for cond in hi_conds:
    df_c = df_hi_new[df_hi_new["condition"] == cond].copy()
    if df_c.empty:
        print(f"  HI {cond}: NO DATA"); continue
    if cond in ("p0","p1"):
        formula = build_formula("llm_label", M1_BASE["hi"], df_c)
    elif cond == "p2":
        formula = build_formula("llm_label", M1_BASE["hi"] + M4_EXTRAS, df_c)
    else:
        formula = build_formula("llm_label", P3_TEMPLATE["hi"], df_c)
    res = fit_r2m(df_c, formula) if formula else {"r2m": np.nan}
    r2m_val = res["r2m"]
    sh = sh_hi.get(cond, np.nan)
    pct_sh = (r2m_val / sh * 100) if (r2m_val and sh and not np.isnan(r2m_val) and sh > 0) else np.nan
    old_r2m = STAGE3["hi"].get(cond, np.nan)
    print(f"  HI {cond.upper()}: R²m={r2m_val*100:.3f}%  (was {old_r2m:.3f}%)  SH={sh*100:.3f}%  %SH={pct_sh:.1f}%")
    records_hi.append({
        "condition": cond, "language": "hi",
        "r2m_pct_1400": r2m_val * 100 if not np.isnan(r2m_val) else np.nan,
        "r2m_pct_700": old_r2m,
        "sh_pct_1400": sh * 100 if sh and not np.isnan(sh) else np.nan,
        "pct_of_sh": pct_sh,
    })

df_hi_out = pd.DataFrame(records_hi)
df_hi_out.to_csv("results_hi_scaled.csv", index=False)
print("Saved: results_hi_scaled.csv")

# Figure: HI scaling comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
ax = axes[0]
x = np.arange(len(hi_conds))
width = 0.35
r700 = [STAGE3["hi"].get(c, np.nan) for c in hi_conds]
r1400= [df_hi_out[df_hi_out["condition"]==c]["r2m_pct_1400"].values[0]
        if not df_hi_out[df_hi_out["condition"]==c].empty else np.nan
        for c in hi_conds]
ax.bar(x - width/2, r700,  width, label="700 rows (Stage 3)", color="#aec6e8", alpha=0.9)
ax.bar(x + width/2, r1400, width, label="1400 rows (Stage 3)", color="#1a5fa8", alpha=0.9)
ax.axhline(STAGE3["hi"]["sh_m1"], color="#27ae60", linestyle="--", linewidth=1.5,
           label=f"H-baseline M1 (700r) = {STAGE3['hi']['sh_m1']:.3f}%")
sh_1400_m1 = sh_hi.get("p0", np.nan)
if sh_1400_m1 and not np.isnan(sh_1400_m1):
    ax.axhline(sh_1400_m1*100, color="#1a7840", linestyle="-.", linewidth=1.5,
               label=f"H-baseline M1 (1400r) = {sh_1400_m1*100:.3f}%")
ax.set_xticks(x)
ax.set_xticklabels([c.upper() for c in hi_conds])
ax.set_ylabel("R²_m (%)")
ax.set_title("HI: 700-row vs 1400-row sample\nDoes more data help?", fontsize=10)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

ax = axes[1]
sh_m1_new = sh_hi.get("p0", np.nan)
pcts = [df_hi_out[df_hi_out["condition"]==c]["pct_of_sh"].values[0]
        if not df_hi_out[df_hi_out["condition"]==c].empty else np.nan
        for c in hi_conds]
old_pct = [(STAGE3["hi"].get(c, np.nan) / STAGE3["hi"]["sh_m1"] * 100)
           if not np.isnan(STAGE3["hi"].get(c, np.nan)) else np.nan
           for c in hi_conds]
ax.bar(x - width/2, old_pct,  width, label="700 rows", color="#aec6e8", alpha=0.9)
ax.bar(x + width/2, pcts,     width, label="1400 rows", color="#1a5fa8", alpha=0.9)
ax.axhline(100, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([c.upper() for c in hi_conds])
ax.set_ylabel("% of achievable H-baseline(M1)")
ax.set_title("HI: % of Achievable Signal\n(relative to condition-matched H-baseline)", fontsize=10)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

fig.suptitle("Hindi Sample Scaling: Option 3\nSampling power vs LLM competence disentanglement",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_hi_scaled.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_hi_scaled.png")


# ══════════════════════════════════════════════════════════════════════════════
#  OPTION 4: IT and PT targeted P4
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print("  OPTION 4: IT and PT targeted P4 (employment-focused)")
print(f"{'═'*70}")

records_p4 = []
for lang in ["it", "pt"]:
    df_p4 = df_preds[(df_preds["language"] == lang) & (df_preds["condition"] == "p4_emp")].copy()
    if df_p4.empty:
        print(f"  {lang.upper()} P4: NO DATA"); continue

    # P4 formula: only employment_status
    formula = build_formula("llm_label", ["C(employment_status)"], df_p4)
    if formula is None:
        r2m_p4 = np.nan
        print(f"  {lang.upper()} P4: formula failed")
    else:
        res = fit_r2m(df_p4, formula)
        r2m_p4 = res["r2m"]

    # Compare to P0, P3 (the best Stage 3 condition)
    s3 = STAGE3[lang]
    p0_r2m = s3["p0"]
    best_stage3 = max(v for k,v in s3.items() if k.startswith("p") and not np.isnan(v))
    direction = "↑ above P3/best" if (r2m_p4 and r2m_p4 * 100 > best_stage3) else "↓ below best"
    print(f"  {lang.upper()} P4: R²m={r2m_p4*100:.3f}%  P0={p0_r2m:.3f}%  best_stage3={best_stage3:.3f}%  ({direction})")

    records_p4.append({
        "language": lang, "condition": "p4_emp",
        "r2m_pct": r2m_p4 * 100 if r2m_p4 and not np.isnan(r2m_p4) else np.nan,
        "p0_stage3": p0_r2m,
        "best_stage3": best_stage3,
        "sh_m1_stage3": s3.get("sh_m1", np.nan),
    })

df_p4_out = pd.DataFrame(records_p4)
df_p4_out.to_csv("results_it_pt_p4.csv", index=False)
print("Saved: results_it_pt_p4.csv")

# Figure: IT and PT P4 vs existing conditions
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
for ax_idx, lang in enumerate(["it", "pt"]):
    ax = axes[ax_idx]
    s3 = STAGE3[lang]
    conds_plot = ["p0", "p1", "p2", "p3", "p4_emp"]
    vals_plot  = [s3.get(c, np.nan) for c in ["p0","p1","p2","p3"]]
    p4_val = (df_p4_out[df_p4_out["language"]==lang]["r2m_pct"].values[0]
              if not df_p4_out[df_p4_out["language"]==lang].empty else np.nan)
    vals_plot.append(p4_val)
    colors_plot = ["#aec6e8","#4c9be8","#1a5fa8","#e85c4c","#27ae60"]
    labels_plot = ["P0\n(no persona)","P1\n(M1 fields)","P2\n(M4 fields)","P3\n(prose)","P4\n(emp only)"]
    bars = ax.bar(labels_plot, vals_plot, color=colors_plot, alpha=0.88, edgecolor="white")
    sh_val = s3.get("sh_m1", np.nan)
    if not np.isnan(sh_val):
        ax.axhline(sh_val, color="#27ae60", linestyle="--", linewidth=1.5,
                   label=f"H-baseline M1 = {sh_val:.3f}%")
    for bar, v in zip(bars, vals_plot):
        if v and not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("R²_m (%)")
    ax.set_title(f"{lang.upper()}: Employment-Focused P4 vs Stage 3 Conditions", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

fig.suptitle("IT and PT: Targeted P4 Prompt (Option 4)\n"
             "Does focusing on employment status help?", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_it_pt_p4.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_it_pt_p4.png")

print(f"\n{'═'*70}")
print("  ALL DONE — extended analysis complete")
print(f"{'═'*70}")
