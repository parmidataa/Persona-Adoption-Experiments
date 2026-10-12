"""
Step 19 (Task 3A1): Cross-lingual persona ceiling calibration.

Combines the Target_R^2 from Task 2A (`results_crossed_r2_per_language.csv`)
with the M1, M4, M5 R^2_m values from Task 2B
(`results_extended_fixed_effects.csv`) into the multilingual analogue of
Hu & Collier (2024) Figure 1.

For each language we compute saturation = R^2_m / Target_R^2 at three model
configurations (M1 = current full model, M4 = richest *demographic* model,
M5 = informational upper bound including the LOO base-rate proxy).

Outputs:
  - results_persona_ceiling_per_language.csv
  - fig21_persona_ceiling_scatter.png
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 70)
print("STEP 19 (Task 3A1): PERSONA CEILING CALIBRATION")
print("=" * 70)

crossed = pd.read_csv("results_crossed_r2_per_language.csv")
fe = pd.read_csv("results_extended_fixed_effects.csv")

m_pivot = fe.pivot(index="language", columns="model", values="r2_marginal")
m_pivot.columns.name = None
m_pivot = m_pivot.reset_index()

# ── corrected ceiling ──────────────────────────────────────────────────────────
# The investigation note (persona_ceiling_investigation.md) identifies two issues
# with the original R²_target formula:
#   1. Numerator: only σ²_ann is used, but σ²_f is also stable annotator-level
#      variance already captured by demographics. The full stable annotator signal
#      is σ²_f + σ²_ann.
#   2. Denominator mismatch: R²_m uses total_var; R²_target uses non-text variance.
#
# Corrected saturation = σ²_f / (σ²_f + σ²_ann)
#                      = R²_m / [(var_fixed + var_annotator) / total_var]
#
# All required components (var_fixed, var_annotator, total_var) are available
# in results_crossed_r2_per_language.csv. No model re-runs required.
crossed["corrected_ceiling"] = (
    (crossed["var_fixed"] + crossed["var_annotator"]) / crossed["total_var"]
)

merged = crossed[["language", "target_r2", "corrected_ceiling"]].merge(
    m_pivot, on="language"
)
merged = merged.rename(columns={
    "M1": "r2_m_M1",
    "M2": "r2_m_M2",
    "M3": "r2_m_M3",
    "M4": "r2_m_M4",
    "M5": "r2_m_M5",
})

for col, target_col in [
    ("r2_m_M1", "saturation_M1"),
    ("r2_m_M4", "saturation_M4"),
    ("r2_m_M5", "saturation_M5"),
]:
    merged[target_col] = merged[col] / merged["corrected_ceiling"]

merged.to_csv("results_persona_ceiling_per_language.csv", index=False)

print("\n" + "=" * 70)
print("CEILING TABLE (percentages)")
print("=" * 70)
disp = merged.copy()
for c in ["target_r2", "r2_m_M1", "r2_m_M4", "r2_m_M5",
          "saturation_M1", "saturation_M4", "saturation_M5"]:
    disp[c] = disp[c] * 100
print(disp[[
    "language", "target_r2",
    "r2_m_M1", "saturation_M1",
    "r2_m_M4", "saturation_M4",
    "r2_m_M5", "saturation_M5",
]].to_string(index=False, float_format="%.3f"))


# --- Figure 21: y = R^2_m, x = corrected ceiling ---
# x-axis: corrected ceiling = (σ²_f + σ²_ann) / total_var
# (replaces old target_r2 = σ²_ann / (σ²_f + σ²_ann + σ²_resid))
fig, ax = plt.subplots(figsize=(9, 8))

xs = merged["corrected_ceiling"].values * 100

# Diagonal y = x
lim = max(xs.max(), (merged["r2_m_M5"].max() * 100)) * 1.05
ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1.0,
        label="y = x (perfect saturation)")

# M1
ax.scatter(xs, merged["r2_m_M1"] * 100, marker="o", s=70,
           color="#4C72B0", label="M1 (current full model)")
# M4
ax.scatter(xs, merged["r2_m_M4"] * 100, marker="s", s=70,
           color="#55A868", label="M4 (richest demographics)")
# M5 (upper bound)
ax.scatter(xs, merged["r2_m_M5"] * 100, marker="^", s=70,
           color="#C44E52", label="M5 (LOO base-rate proxy)")

# Connect M1->M4->M5 per language with thin lines
for _, row in merged.iterrows():
    x = row["corrected_ceiling"] * 100
    ax.plot([x, x, x],
            [row["r2_m_M1"] * 100, row["r2_m_M4"] * 100,
             row["r2_m_M5"] * 100],
            color="lightgray", linewidth=0.6, zorder=0)
    ax.annotate(row["language"], (x, row["r2_m_M5"] * 100),
                textcoords="offset points", xytext=(6, 2),
                fontsize=10, fontweight="bold")

ax.set_xlabel(
    "Corrected ceiling  =  ($\\sigma^2_f + \\sigma^2_{\\mathrm{ann}}$)"
    " / total variance   (%)"
)
ax.set_ylabel("Observed R$^2_m$ (%)")
ax.set_title("Task 3A1: Persona ceiling across 9 languages (corrected)\n"
             "(below diagonal = persona variables under-saturate the ceiling)")
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
fig.tight_layout()
fig.savefig("fig21_persona_ceiling_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved fig21_persona_ceiling_scatter.png")

# --- Figure 19 (regenerated with corrected saturation): M1 saturation bar ---
# Regenerate fig19 here so it uses the corrected formula without re-running
# the slow model fits in 17_crossed_random_effects.py.
LANG_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "hi": "Hindi", "it": "Italian", "nl": "Dutch",
    "pt": "Portuguese",
}
sat_data = merged[["language", "saturation_M1", "r2_m_M1", "corrected_ceiling"]].copy()
sat_data["language_label"] = sat_data["language"].map(LANG_NAMES)
sat_data["sat_pct"] = sat_data["saturation_M1"] * 100
sat_data["ceil_pct"] = sat_data["corrected_ceiling"] * 100
sat_data["r2m_pct"] = sat_data["r2_m_M1"] * 100
sat_data = sat_data.sort_values("sat_pct", ascending=True).reset_index(drop=True)

fig19, ax19 = plt.subplots(figsize=(10, 5.5))
y_pos = np.arange(len(sat_data))

# Background: full corrected ceiling (normalised to 100%)
ax19.barh(y_pos, [100] * len(sat_data), height=0.6,
          color="#DDDDDD", alpha=0.4, zorder=1, label="Corrected ceiling (100%)")
# Saturation bar
ax19.barh(y_pos, sat_data["sat_pct"], height=0.6,
          color="#4C72B0", zorder=2, label="M1 saturation")
# Annotations
for i, row in sat_data.iterrows():
    ax19.text(row["sat_pct"] + 0.5, i,
              f"R²_m={row['r2m_pct']:.2f}% / Ceiling={row['ceil_pct']:.2f}%",
              va="center", fontsize=8)
ax19.set_xlabel(
    "Saturation = $\\sigma^2_f$ / ($\\sigma^2_f + \\sigma^2_{\\mathrm{ann}}$)  (%)"
)
ax19.set_yticks(y_pos)
ax19.set_yticklabels(sat_data["language_label"], fontsize=11)
ax19.set_title(
    "Task 2A: How much of the stable annotator-level signal do current\n"
    "persona variables capture? (corrected saturation formula)"
)
ax19.set_xlim(0, 110)
ax19.grid(axis="x", alpha=0.3)
ax19.legend(fontsize=9)
fig19.tight_layout()
fig19.savefig("fig19_saturation_per_language.png", dpi=150, bbox_inches="tight")
plt.close(fig19)
print("Saved fig19_saturation_per_language.png (corrected)")

print("\n" + "=" * 70)
print("STEP 19 COMPLETE")
print("=" * 70)
