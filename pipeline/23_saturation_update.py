"""
Step 23 (Direction 1): Saturation Update — How much of the annotator ceiling
does M4 (richest demographics) recover vs. M1 (Hu & Collier baseline)?

Reads the pre-computed ceiling file from Step 19 and produces two new figures:

  fig27_saturation_update_bars.png
    Grouped horizontal bar chart: saturation at M1, M4, and M5 per language,
    sorted by M4 saturation. Shows how much headroom remains after adding all
    available demographic variables.

  fig28_saturation_gap_scatter.png
    Two-panel scatter:
      Left  — observed R²_m (M1, M4, M5) vs target_R², same y=x diagonal as
               Figure 21 but now annotated with the saturation % at M4 to make
               the per-language gap concrete.
      Right — "gap to ceiling" bar chart:  target_R² − R²_m at M1 and M4 per
               language, showing how much persona-reachable variance is still
               uncaptured after each model.

Both figures feed directly into the persona-prompting discussion:
languages with high M4 saturation have little room left from demographics;
languages with low saturation are the priority targets for richer prompts.

Outputs:
  - fig27_saturation_update_bars.png
  - fig28_saturation_gap_scatter.png
  - results_saturation_update.csv      (machine-readable supplement)
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("STEP 23 (Direction 1): SATURATION UPDATE")
print("=" * 70)

# ── 1. Load data ──────────────────────────────────────────────────────────────
ceil = pd.read_csv("results_persona_ceiling_per_language.csv")

LANG_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "hi": "Hindi", "it": "Italian", "nl": "Dutch",
    "pt": "Portuguese",
}

# Convert fractions → percentages for display
pct_cols = ["target_r2", "corrected_ceiling", "r2_m_M1", "r2_m_M2", "r2_m_M3",
            "r2_m_M4", "r2_m_M5",
            "saturation_M1", "saturation_M4", "saturation_M5"]
disp = ceil.copy()
for c in pct_cols:
    if c in disp.columns:
        disp[c] = disp[c] * 100

disp["language_label"] = disp["language"].map(LANG_NAMES)

# Derived columns
disp["gap_M1"] = disp["corrected_ceiling"] - disp["r2_m_M1"]   # unreached after M1
disp["gap_M4"] = disp["corrected_ceiling"] - disp["r2_m_M4"]   # unreached after M4
disp["sat_gain_M1_M4"] = disp["saturation_M4"] - disp["saturation_M1"]

# Sort by M4 saturation descending
disp = disp.sort_values("saturation_M4", ascending=True).reset_index(drop=True)

disp.to_csv("results_saturation_update.csv", index=False)
print("Saved results_saturation_update.csv")

# ── 2. Summary table ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SATURATION TABLE (all values in %)")
print("=" * 70)
hdr = (f"{'Lang':>10} | {'Target R²':>9} | "
       f"{'R²_m M1':>8} | {'Sat M1':>7} | "
       f"{'R²_m M4':>8} | {'Sat M4':>7} | "
       f"{'R²_m M5':>8} | {'Sat M5':>7} | "
       f"{'Sat gain':>8}")
print(hdr)
print("-" * len(hdr))
for _, row in disp.sort_values("saturation_M4", ascending=False).iterrows():
    print(
        f"{row['language_label']:>10} | "
        f"{row['target_r2']:>9.2f} | "
        f"{row['r2_m_M1']:>8.2f} | {row['saturation_M1']:>7.1f} | "
        f"{row['r2_m_M4']:>8.2f} | {row['saturation_M4']:>7.1f} | "
        f"{row['r2_m_M5']:>8.2f} | {row['saturation_M5']:>7.1f} | "
        f"{row['sat_gain_M1_M4']:>+8.1f}"
    )

mean_sat_M1 = disp["saturation_M1"].mean()
mean_sat_M4 = disp["saturation_M4"].mean()
mean_sat_M5 = disp["saturation_M5"].mean()
print(f"\nMean saturation:  M1={mean_sat_M1:.1f}%  M4={mean_sat_M4:.1f}%  M5={mean_sat_M5:.1f}%")
print(f"Mean sat gain M1→M4: {(mean_sat_M4 - mean_sat_M1):+.1f} pp")

# ── 3. Figure 27 — Grouped horizontal bar chart ───────────────────────────────
# Languages sorted by M4 saturation (ascending, so highest is at top of chart)
fig, ax = plt.subplots(figsize=(10, 6.5))

y_pos = np.arange(len(disp))
bar_h = 0.25

C_M1 = "#4C72B0"   # blue
C_M4 = "#55A868"   # green
C_M5 = "#C44E52"   # red (upper bound)
C_ceil = "#BBBBBB"  # light gray

for i, row in disp.iterrows():
    # Target R² background bar (full ceiling)
    ax.barh(i, row["target_r2"], height=0.78, color=C_ceil, alpha=0.25,
            zorder=1, left=0)

ax.barh(y_pos - bar_h, disp["saturation_M1"], height=bar_h,
        color=C_M1, label="M1 saturation (Hu & Collier baseline)", zorder=2)
ax.barh(y_pos,         disp["saturation_M4"], height=bar_h,
        color=C_M4, label="M4 saturation (richest demographics)", zorder=2)
ax.barh(y_pos + bar_h, disp["saturation_M5"], height=bar_h,
        color=C_M5, label="M5 saturation (LOO proxy — upper bound)", zorder=2,
        alpha=0.75, hatch="//")

# Annotate M4 saturation value
for i, row in disp.iterrows():
    ax.text(row["saturation_M4"] + 0.8, i,
            f"{row['saturation_M4']:.0f}%", va="center",
            fontsize=8.5, color=C_M4, fontweight="bold")

ax.axvline(100, color="gray", linestyle="--", linewidth=1, label="100% = ceiling")
ax.set_yticks(y_pos)
ax.set_yticklabels(disp["language_label"], fontsize=11)
ax.set_xlabel("Saturation (% of target R² reached)", fontsize=11)
ax.set_title(
    "Figure 27 — Saturation update: how much of the persona ceiling\n"
    "does each model recover?   (gray background = 100% of target R²)",
    fontsize=11)
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 115)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("fig27_saturation_update_bars.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved fig27_saturation_update_bars.png")

# ── 4. Figure 28 — Two-panel: ceiling scatter + gap chart ─────────────────────
fig, (ax_scatter, ax_gap) = plt.subplots(1, 2, figsize=(14, 7))

# ── Left panel: ceiling scatter with M4 saturation annotations ────────────────
xs = disp["corrected_ceiling"].values

lim = max(xs.max(), disp["r2_m_M5"].max()) * 1.10
ax_scatter.plot([0, lim], [0, lim], color="gray", linestyle="--",
                linewidth=1.0, label="y = x  (perfect saturation)", zorder=0)

# M1
ax_scatter.scatter(xs, disp["r2_m_M1"], marker="o", s=80,
                   color=C_M1, label="M1  (Hu & Collier baseline)", zorder=3)
# M4
ax_scatter.scatter(xs, disp["r2_m_M4"], marker="s", s=80,
                   color=C_M4, label="M4  (richest demographics)", zorder=3)
# M5
ax_scatter.scatter(xs, disp["r2_m_M5"], marker="^", s=80,
                   color=C_M5, label="M5  (LOO proxy — upper bound)", zorder=3,
                   alpha=0.75)

# Connect M1→M4→M5 per language with thin line
for _, row in disp.iterrows():
    x = row["corrected_ceiling"]
    ax_scatter.plot([x, x, x],
                    [row["r2_m_M1"], row["r2_m_M4"], row["r2_m_M5"]],
                    color="lightgray", linewidth=0.7, zorder=1)
    # Annotate language label + M4 saturation
    ax_scatter.annotate(
        f"{row['language_label']}\n({row['saturation_M4']:.0f}%)",
        (x, row["r2_m_M5"]),
        textcoords="offset points", xytext=(5, 3),
        fontsize=7.5, color="black"
    )

ax_scatter.set_xlabel(
    "Corrected ceiling  =  (σ²_f + σ²_ann) / total variance   (%)",
    fontsize=10)
ax_scatter.set_ylabel("Observed R²_m (%)", fontsize=10)
ax_scatter.set_title("Left: Ceiling scatter — M4 labels show\nM4 saturation %",
                     fontsize=10)
ax_scatter.legend(loc="upper left", fontsize=9)
ax_scatter.grid(alpha=0.3)
ax_scatter.set_xlim(0, lim)
ax_scatter.set_ylim(0, lim)

# ── Right panel: gap to ceiling bar chart ─────────────────────────────────────
gap_sort = disp.sort_values("gap_M4", ascending=True).reset_index(drop=True)
yp = np.arange(len(gap_sort))
bh = 0.35

# Gap after M1 (total unreached)
ax_gap.barh(yp + bh / 2, gap_sort["gap_M1"], height=bh,
            color=C_M1, alpha=0.7, label="Gap remaining after M1")
# Gap after M4 (still unreached)
ax_gap.barh(yp - bh / 2, gap_sort["gap_M4"], height=bh,
            color=C_M4, alpha=0.85, label="Gap remaining after M4")

# Annotate gap_M4 value
for i, row in gap_sort.iterrows():
    ax_gap.text(row["gap_M4"] + 0.1, i - bh / 2,
                f"{row['gap_M4']:.1f} pp", va="center",
                fontsize=8.5, color=C_M4)

ax_gap.set_yticks(yp)
ax_gap.set_yticklabels(gap_sort["language_label"], fontsize=11)
ax_gap.set_xlabel("Unreached persona-reachable variance (pp)", fontsize=10)
ax_gap.set_title("Right: Gap to ceiling  (target R² − R²_m)\n"
                 "after M1 vs M4 — smaller = better", fontsize=10)
ax_gap.legend(fontsize=9)
ax_gap.grid(axis="x", alpha=0.3)
ax_gap.set_xlim(0, gap_sort["gap_M1"].max() * 1.25)

fig.suptitle(
    "Figure 28 — Saturation update: ceiling scatter (left) and unreached"
    " persona-reachable gap (right)\n"
    "M4 closes the gap meaningfully in most languages but leaves 24–79%"
    " of the annotator ceiling unreached",
    fontsize=11, y=1.01
)
fig.tight_layout()
fig.savefig("fig28_saturation_gap_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig28_saturation_gap_scatter.png")

print("\n" + "=" * 70)
print("KEY FINDINGS — Saturation Update")
print("=" * 70)
print(f"\nM4 (richest demographics) raises mean saturation from "
      f"{mean_sat_M1:.1f}% (M1) to {mean_sat_M4:.1f}% — a gain of "
      f"{mean_sat_M4 - mean_sat_M1:+.1f} pp.")
print(f"M5 (attitudinal proxy) further raises mean saturation to "
      f"{mean_sat_M5:.1f}% ({mean_sat_M5 - mean_sat_M4:+.1f} pp vs M4).")
print()
below50 = disp[disp["saturation_M4"] < 50]["language_label"].tolist()
above50 = disp[disp["saturation_M4"] >= 50]["language_label"].tolist()
print(f"Languages with M4 saturation < 50% (largest unrealized headroom):")
print(f"  {', '.join(sorted(below50))}")
print(f"Languages with M4 saturation >= 50% (relatively well covered):")
print(f"  {', '.join(sorted(above50))}")

hi_row = disp[disp["language"] == "hi"].iloc[0]
en_row = disp[disp["language"] == "en"].iloc[0]
nl_row = disp[disp["language"] == "nl"].iloc[0]
print(f"\nHindi:   target_R²={hi_row['target_r2']:.2f}%, M4 R²_m={hi_row['r2_m_M4']:.2f}%,"
      f" M4 saturation={hi_row['saturation_M4']:.1f}%  (near-ceiling demographic coverage)")
print(f"English: target_R²={en_row['target_r2']:.2f}%, M4 R²_m={en_row['r2_m_M4']:.2f}%,"
      f" M4 saturation={en_row['saturation_M4']:.1f}%  (large unrealized headroom)")
print(f"Dutch:   target_R²={nl_row['target_r2']:.2f}%, M4 R²_m={nl_row['r2_m_M4']:.2f}%,"
      f" M4 saturation={nl_row['saturation_M4']:.1f}%  (large unrealized headroom)")

print("\n" + "=" * 70)
print("STEP 23 COMPLETE")
print("=" * 70)
