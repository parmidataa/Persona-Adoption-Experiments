"""
80_build_p5_batch.py
====================
Build the P5 (Relative Framing + Chain-of-Thought) batch for Arabic and Spanish.

Scientific motivation
---------------------
P4 (RAG) showed that providing raw annotation examples amplifies demographic
variance (high R²m) but worsens directional concordance.  The model performs
label copying instead of generalising the demographic group's annotation style.

P5 addresses this by changing WHAT we ask the model for.  The R²m and sign-
agreement metrics measure signed DEVIATIONS from the grand mean, not absolute
probabilities.  P5 therefore:
  (a) Tells the model the population-level baseline irony rate for the language,
  (b) Asks the model to reason about how its specific demographic background
      would shift that rate (chain-of-thought), and
  (c) Requests a signed decimal deviation rather than an absolute probability.

This perfectly aligns the model's output format with what the LMM fixed effects
actually measure.  If the model cannot produce correct demographic deviations
even when asked for them directly with explicit reasoning, the directional failure
is structural — residing in the model's encoded demographic associations rather
than in any prompt-engineering choice.

Output format (per response)
----------------------------
  Reasoning: [one sentence describing demographic influence on irony perception]
  Deviation: [signed decimal, e.g. +0.12 or -0.08]

  Deviation is in [−1, 1] where:
    +0.12  = model rates this 12 pp MORE ironic than the 34 % average
    −0.08  = model rates this 8 pp LESS ironic than the 34 % average

Scope: AR + ES, one condition (P5), 700 rows each = 1,400 requests total.

Outputs:
    p5_batch_input.jsonl   — batch API requests
    p5_batch_index.jsonl   — custom_id → row metadata
"""

import json
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL      = "gpt-4.1-mini"
LANGUAGES  = ["ar", "es"]
CONDITION  = "p5"

# Baseline irony rates (mean of human_label in data_selected per language)
# Computed from: data_selected.parquet — AR: 700 rows, ES: 700 rows
BASELINE_RATE = {
    "ar": 0.340,   # 34.0 %
    "es": 0.383,   # 38.3 %
}

INPUT_PATH  = Path("data_selected.parquet")
OUTPUT_PATH = Path("p5_batch_input.jsonl")
INDEX_PATH  = Path("p5_batch_index.jsonl")


# ── Sanitiser (identical to all previous batch scripts) ───────────────────────
def sanitise(row: dict) -> dict:
    clean = {}
    for k, v in row.items():
        s = str(v).strip()
        if s in ("DATA_EXPIRED", "nan", "None", ""):
            clean[k] = "unknown"
        elif k == "age":
            try:
                clean[k] = str(int(float(s)))
            except ValueError:
                clean[k] = s
        else:
            clean[k] = s
    return clean


# ── Prompt builders ───────────────────────────────────────────────────────────
def build_system(row: dict) -> str:
    """P2 full demographic list — identical to previous experiments."""
    return (
        "You are annotating text for irony.\n"
        f"Persona: age={row['age']}, gender={row['gender']}, "
        f"ethnicity={row['ethnicity']}, nationality={row['nationality']}, "
        f"country_birth={row['country_birth']}, "
        f"country_residence={row['country_residence']}, "
        f"employment={row['employment_status']}, "
        f"student={row['student_status']}, "
        f"generation={row['Generation']}"
    )


def build_user(row: dict, baseline: float) -> str:
    """
    Relative framing + chain-of-thought instruction.

    Key design choices:
    1. Baseline rate is stated explicitly so the model knows what 'average' means.
    2. Step 1 forces the model to articulate its demographic influence BEFORE scoring.
    3. Step 2 asks for the signed deviation — exactly what the LMM fixed effect measures.
    4. Format is strict so the parser can reliably extract the deviation.
    5. The valid range [-1, 1] is stated to prevent out-of-bound outputs.
    """
    baseline_pct = f"{baseline * 100:.0f}%"
    return (
        f"Across all annotators in this dataset, texts like this one are rated "
        f"as ironic approximately {baseline_pct} of the time on average.\n\n"
        "Think step by step:\n"
        "Step 1 — How does your specific demographic background (nationality, "
        "gender, ethnicity) tend to make you perceive irony differently from "
        "the average annotator?\n"
        "Step 2 — Given your reasoning, how much would your irony rating for "
        f"the text below deviate from the {baseline_pct} average?\n\n"
        "Reply in exactly this format:\n"
        "Reasoning: [one sentence]\n"
        "Deviation: [signed decimal between -1 and +1, "
        "e.g. +0.15 or -0.08 or 0.00]\n\n"
        f"Text: {row['reply']}"
    )


def build_request(custom_id: str, row: dict, baseline: float) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": build_system(row)},
                {"role": "user",   "content": build_user(row, baseline)},
            ],
            "temperature": 0,
            # CoT reasoning (~30 words) + "Deviation: +0.12" → ~150 tokens is safe
            "max_completion_tokens": 150,
        },
    }


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data_selected.parquet ...")
df_full = pd.read_parquet(INPUT_PATH)
df_base = df_full[df_full["language"].isin(LANGUAGES)].copy()
print(f"  {len(df_base):,} rows (AR + ES)")

for lang in LANGUAGES:
    n   = (df_base["language"] == lang).sum()
    b   = BASELINE_RATE[lang]
    print(f"  {lang.upper()}: {n:,} rows, baseline irony rate = {b*100:.1f}%")

# ── Build batch ───────────────────────────────────────────────────────────────
print(f"\nBuilding P5 batch ...")
total_written = 0

with open(OUTPUT_PATH, "w", encoding="utf-8") as batch_f, \
     open(INDEX_PATH,  "w", encoding="utf-8") as idx_f:

    for _, row in df_base.iterrows():
        row_clean  = sanitise(row.to_dict())
        lang       = row_clean["language"]
        baseline   = BASELINE_RATE[lang]
        custom_id  = f"p5_{lang}_{row.name}"

        req = build_request(custom_id, row_clean, baseline)
        batch_f.write(json.dumps(req, ensure_ascii=False) + "\n")

        idx_entry = {
            "custom_id":         custom_id,
            "condition":         CONDITION,
            "orig_row_idx":      int(row.name),
            "language":          lang,
            "annotator_id":      row_clean["annotator_id"],
            "post_id":           row_clean["post_id"],
            "human_label":       int(float(row_clean["label"])),
            "text_entropy":      float(row.get("text_entropy", 0)),
            "baseline_rate":     baseline,
            "age":               row_clean["age"],
            "age_bin":           row_clean["age_bin"],
            "gender":            row_clean["gender"],
            "ethnicity":         row_clean["ethnicity"],
            "nationality":       row_clean["nationality"],
            "country_birth":     row_clean["country_birth"],
            "country_residence": row_clean["country_residence"],
            "employment_status": row_clean["employment_status"],
            "student_status":    row_clean["student_status"],
            "Generation":        row_clean["Generation"],
        }
        idx_f.write(json.dumps(idx_entry, ensure_ascii=False) + "\n")
        total_written += 1

# ── Cost estimate ─────────────────────────────────────────────────────────────
# Input: system (~110 tok) + user baseline+CoT (~120 tok) + text (~40 tok) = ~270 tok
# Output: reasoning (~30 tok) + deviation line (~8 tok) = ~38 tok
est_in  = total_written * 270
est_out = total_written * 38
cost_in  = est_in  * 0.40 / 1_000_000 * 0.5
cost_out = est_out * 1.60 / 1_000_000 * 0.5

print(f"\n  Written  : {total_written:,} requests → {OUTPUT_PATH}")
print(f"  Est. cost: ~${cost_in + cost_out:.3f}  (Batch API, 50% off)")
print(f"\nNext step: python 81_submit_p5_batch.py")
