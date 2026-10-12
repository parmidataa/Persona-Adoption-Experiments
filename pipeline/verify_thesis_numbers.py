"""Recompute headline values from the result files and compare them with the thesis.

Usage:  python3 verify_thesis_numbers.py [results_dir]

Every value is recomputed from the raw per-language / per-cell rows, not read from
summary columns. A row is flagged when the recomputed value, rounded to the precision
stated in the thesis, differs from the stated value. Standard library only.
"""
import csv
import math
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
LANGS = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]
BEST = {"ar": "p2", "de": "p3", "en": "p2", "es": "p2", "fr": "p2",
        "hi": "p2", "it": "p3", "nl": "p2", "pt": "p1"}


def load(name):
    with open(os.path.join(D, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def lt(x, t):
    """p-value below threshold; a missing p-value counts as not significant."""
    v = num(x)
    return v is not None and v < t


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs)


n_flag = 0


def check(label, computed, stated, decimals=None, unit=""):
    """decimals=None means an exact (count) comparison."""
    global n_flag
    if decimals is None:
        ok = computed == stated
        c, s = f"{computed}", f"{stated}"
    else:
        ok = abs(round(computed, decimals) - stated) < 0.5 * 10 ** -decimals + 1e-12
        c, s = f"{computed:.{decimals + 2}f}{unit}", f"{stated:.{decimals}f}{unit}"
    n_flag += not ok
    print(f"  {'OK  ' if ok else 'FLAG'}  {label:<58s} recomputed {c:>12s}   thesis {s:>9s}")


def section(title):
    print(f"\n== {title}")


# ── Study 1: shuffled-profile null ─────────────────────────────────────────────
null = load("results_null_r2m.csv")
pa = {r["language"]: r for r in null if r["family"] == "phaseA" and r["model"] == "M1"}
pd_ = {(r["language"], r["model"]): r for r in null if r["family"] == "phaseD"}

section("Study 1, Phase A M1 against the shuffled-profile null (results_null_r2m.csv)")
check("real M1 mean R2m (%)", mean(num(pa[l]["r2m"]) for l in LANGS) * 100, 1.27, 2, "%")
check("null M1 mean R2m (%)", mean(num(pa[l]["null_mean"]) for l in LANGS) * 100, 1.04, 2, "%")
check("French permutation p", num(pa["fr"]["perm_p"]), 0.014, 3)
check("languages with perm p < 0.05", sum(num(pa[l]["perm_p"]) < 0.05 for l in LANGS), 1)
check("French p above Bonferroni 0.05/9 (1 = yes)", int(num(pa["fr"]["perm_p"]) > 0.05 / 9), 1)

section("Study 1, Phase D M1-M4 null means (complete-data sample)")
stated_null = {"M1": 1.16, "M4": 3.36}
stated_obs = {"M1": 1.17, "M3": 3.54, "M4": 3.64}
for m in ["M1", "M2", "M3", "M4"]:
    o = mean(num(pd_[(l, m)]["r2m"]) for l in LANGS) * 100
    n = mean(num(pd_[(l, m)]["null_mean"]) for l in LANGS) * 100
    if m in stated_null:
        check(f"{m} null mean (%)", n, stated_null[m], 2, "%")
    else:
        print(f"  info  {m} null mean (%) {n:.4f}%   (not stated in the thesis)")
    if m in stated_obs:
        check(f"{m} observed mean (%)", o, stated_obs[m], 2, "%")

# ── Study 1: Phase D steps ─────────────────────────────────────────────────────
section("Study 1, Phase D steps (results_crossed_lrt_phaseD_steps.csv)")
steps = load("results_crossed_lrt_phaseD_steps.csv")
testable = [s for s in steps if num(s["crossed_p"]) is not None]
row_run = [s for s in steps if num(s["p_original"]) is not None]
check("steps with a row-level test", len(row_run), 17)
check("steps testable under the corrected tests", len(testable), 26)
check("steps significant at row level (p < 0.05)",
      sum(lt(s["p_original"], 0.05) for s in row_run), 13)
check("row-level-significant steps also sig. in crossed model",
      sum(lt(s["p_original"], 0.05) and lt(s["crossed_p"], 0.05) for s in row_run), 4)
for col, stated in [("crossed_p", 6), ("lrt_ann_p", 6), ("perm_ann_p", 2)]:
    check(f"steps p < 0.05 under {col}", sum(lt(s[col], 0.05) for s in testable), stated)
robust = [s for s in testable
          if all(lt(s[c], 0.05) for c in ("crossed_p", "lrt_ann_p", "perm_ann_p"))]
check("steps significant under all three corrected tests", len(robust), 2)
print("        robust steps:", ", ".join(f"{s['language']} {s['step']}" for s in robust))

# ── Study 1: interactions ──────────────────────────────────────────────────────
section("Study 1, interactions (results_crossed_lrt_interactions.csv)")
inter = load("results_crossed_lrt_interactions.csv")
b33 = 0.05 / 33
check("interactions tested", len(inter), 33)
check("Bonferroni-significant at row level (p < 0.05/33)",
      sum(lt(r["p_original"], b33) for r in inter), 19)
check("Bonferroni-significant under any corrected test",
      sum(any(lt(r[c], b33) for c in ("crossed_p", "lrt_ann_p", "perm_ann_p")) for r in inter), 0)
check("p < 0.05 under all three corrected tests",
      sum(all(lt(r[c], 0.05) for c in ("crossed_p", "lrt_ann_p", "perm_ann_p")) for r in inter), 2)

# ── Study 1: null-corrected saturation ─────────────────────────────────────────
section("Study 1, saturation = R2m / R2_persona (results_null_r2m + persona ceiling)")
ceil = {r["language"]: num(r["corrected_ceiling"]) for r in load("results_persona_ceiling_per_language.csv")}
for m, raw_s, cor_s in [("M1", 15.7, 3.8), ("M4", 45.0, 6.8)]:
    raw = mean(num(pd_[(l, m)]["r2m"]) / ceil[l] for l in LANGS) * 100
    cor = mean(max(num(pd_[(l, m)]["r2m"]) - num(pd_[(l, m)]["null_mean"]), 0) / ceil[l]
               for l in LANGS) * 100
    check(f"{m} mean raw saturation (%)", raw, raw_s, 1, "%")
    check(f"{m} mean null-corrected saturation (%)", cor, cor_s, 1, "%")

# ── Study 2: cells above the null ──────────────────────────────────────────────
section("Study 2, cells above the shuffled-profile null (results_llm_null_r2m.csv)")
llm = load("results_llm_null_r2m.csv")
cell = {(r["sample"], r["rater"], r["language"], r["condition"]): r for r in llm}


def above(r):
    return num(r["perm_p"]) < 0.05


for sample, model in [("mini", "mini"), ("opus5", "opus5")]:
    hum = [r for r in llm if r["sample"] == sample and r["rater"] == "human"]
    check(f"human baselines above null, {sample} sample (of {len(hum)})", sum(map(above, hum)), 0)
    p0 = [cell[(sample, model, l, "p0")] for l in LANGS]
    check(f"{model} P0 cells above null (of 9)", sum(map(above, p0)), 0)
    if model == "mini":
        check("max P0 excess over null, either model (pp)",
              max(num(cell[(s, s, l, "p0")]["excess"]) for s in ("mini", "opus5") for l in LANGS) * 100,
              0.39, 2)
    n = 0
    for l in LANGS:
        p0x = num(cell[(sample, model, l, "p0")]["excess"])
        for c in ("p1", "p2", "p3"):
            r = cell.get((sample, model, l, c))
            n += r is not None and above(r) and num(r["excess"]) > p0x
    check(f"{model} persona cells above null and above P0 excess (of 27)", n, 4 if model == "mini" else 8)

# ── Study 2: split-half agreement ──────────────────────────────────────────────
section("Study 2, human split-half sign agreement (results_split_half_reliability.csv)")
sh = load("results_split_half_reliability.csv")
for sample, stated in [("700-row best (gpt-4.1-mini)", 46.6), ("700-row best (opus5)", 47.2),
                       ("full corpus (M1)", 53.4)]:
    x = [r for r in sh if r["sample"] == sample]
    w = [num(r["mean_coefs"]) for r in x]
    v = sum(num(r["ceiling_mean"]) * wi for r, wi in zip(x, w)) / sum(w) * 100
    check(f"{sample}, coefficient-weighted (%)", v, stated, 1, "%")

# ── Study 2: best-condition magnitude ──────────────────────────────────────────
section("Study 2, best-condition mean R2m (Table 4.3 fits and null-file refits)")
tab_mini = {(r["language"], r["condition"]): num(r["r2m_llm_pct"]) for r in load("results_r2m_llm.csv")}
tab_opus = {(r["language"], r["condition"]): num(r["r2m_opus5_pct"]) for r in load("results_opus5_vs_mini.csv")}
stated = {"mini": (2.44, 2.05, 0.36), "opus5": (4.04, 3.60, 0.43)}
for model, tab in [("mini", tab_mini), ("opus5", tab_opus)]:
    rows = [cell[(model, model, l, BEST[l])] for l in LANGS]
    raw_table = mean(tab[(l, BEST[l])] for l in LANGS)
    raw_refit = mean(num(r["r2m"]) for r in rows) * 100
    nul = mean(num(r["null_mean"]) for r in rows) * 100
    exc = mean(num(r["excess"]) for r in rows) * 100
    s_raw, s_nul, s_exc = stated[model]
    check(f"{model}: raw best-condition mean, table fits (%)", raw_table, s_raw, 2, "%")
    check(f"{model}: null mean for the same conditions (%)", nul, s_nul, 2, "%")
    check(f"{model}: mean excess over null (pp)", exc, s_exc, 2)
    print(f"  info  {model}: raw mean of the null-file refits        {raw_refit:.4f}%")
    print(f"  info  {model}: table raw - null mean                    {raw_table - nul:.4f} pp")
    print(f"  info  {model}: refit raw - null mean (= mean excess)    {raw_refit - nul:.4f} pp")
    for l, r in zip(LANGS, rows):
        d = num(r["r2m"]) * 100 - tab[(l, BEST[l])]
        if abs(d) >= 0.005:
            print(f"        {l} {BEST[l]}: table {tab[(l, BEST[l])]:.3f}%  refit {num(r['r2m']) * 100:.3f}%"
                  f"  ({d:+.3f} pp)")

print(f"\n{n_flag} value(s) flagged.")
print("""
Why 2.44 - 2.05 != 0.36:
  The 2.44% is the mean of the Table 4.3 fits. The null and the excess come from
  106_llm_null_r2m.py, which refits every best-condition model (ML, design matrix
  pruned to full rank) before shuffling the profiles, and computes
  excess = refit R2m - null mean per cell. The refit matches the table in every
  language except Spanish P2: the table value (6.787%) is the convergence-audit refit
  of the full, near-singular design (thesis_corrections_log.md), the null script's
  rank-pruned refit gives 6.524%. That -0.263 pp in one language lowers the
  nine-language mean by 0.03 pp (2.44% -> 2.41%), so
      mean excess = 2.41 - 2.05 = 0.36,   whereas   2.44 - 2.05 = 0.39.
  For Opus 5 the same cell differs by -0.058 pp (4.04% -> 4.03%), giving 0.43.""")
