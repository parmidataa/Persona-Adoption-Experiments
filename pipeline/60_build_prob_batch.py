"""
60_build_prob_batch.py
======================
Probability-elicitation experiment (Variant A).

Scope: Arabic (ar) and Spanish (es) only, conditions P0-P3.
Format change: instead of "Reply with 0 or 1 only", the model is asked for
P(ironic) as a decimal in [0, 1].  System prompts are identical to the
original experiment so that any change in results is attributable solely
to the output format.

Total requests: 2 languages × 4 conditions × 700 rows = 5,600.

Outputs:
    prob_batch_input.jsonl   — batch API requests
    prob_batch_index.jsonl   — custom_id → row metadata (for merging after parse)
"""

import json
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL      = "gpt-4.1-mini"
LANGUAGES  = ["ar", "es"]          # Arabic and Spanish only
CONDITIONS = ["p0", "p1", "p2", "p3"]
INPUT_PATH  = Path("data_selected.parquet")   # same source as 26_build_batch.py (has reply text)
OUTPUT_PATH = Path("prob_batch_input.jsonl")
INDEX_PATH  = Path("prob_batch_index.jsonl")

# ── P3 prose templates (identical to 26_build_batch.py) ──────────────────────
P3_TEMPLATES = {
    "en": "You are a {nationality} {gender} living in {country_residence}.",
    "ar": "You are a {age}-year-old person of {nationality} nationality.",
    "hi": "You are a {age}-year-old {ethnicity} person originally from {country_birth}.",
    "fr": "You are a {age}-year-old {ethnicity} person with {nationality} nationality, living in {country_residence}.",
    "it": "You are a {age}-year-old {employment_status}.",
    "nl": "You are a {employment_status} annotator.",
    "pt": "You are a {employment_status} person originally from {country_birth}.",
    "es": "You are a {age}-year-old person born in {country_birth}, living in {country_residence}.",
    "de": "You are a {Generation} {employment_status} person.",
}

# ── Sanitise helpers (identical to 26_build_batch.py) ────────────────────────
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


def build_system(row: dict, condition: str) -> str:
    """Identical system prompts to the original experiment."""
    if condition == "p0":
        return "You are a text annotator."

    elif condition == "p1":
        return (
            "You are annotating text for irony.\n"
            f"Persona: age={row['age']}, gender={row['gender']}, "
            f"ethnicity={row['ethnicity']}, nationality={row['nationality']}"
        )

    elif condition == "p2":
        return (
            "You are annotating text for irony.\n"
            f"Persona: age={row['age']}, gender={row['gender']}, "
            f"ethnicity={row['ethnicity']}, nationality={row['nationality']}, "
            f"country_birth={row['country_birth']}, country_residence={row['country_residence']}, "
            f"employment={row['employment_status']}, student={row['student_status']}, "
            f"generation={row['Generation']}"
        )

    elif condition == "p3":
        template = P3_TEMPLATES.get(row["language"], "You are a text annotator.")
        try:
            prose = template.format(**row)
        except KeyError as e:
            prose = f"You are a text annotator. (template field missing: {e})"
        return "You are annotating text for irony. " + prose

    raise ValueError(f"Unknown condition: {condition}")


def build_request(custom_id: str, row: dict, condition: str) -> dict:
    system_msg = build_system(row, condition)
    # KEY CHANGE: probability elicitation instead of forced binary
    user_msg = (
        "Rate the probability that the following text is ironic. "
        "Reply with a decimal number between 0 and 1, and nothing else.\n"
        f"Text: {row['reply']}"
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0,
            "max_completion_tokens": 10,   # "0.73" fits in 3 tokens; 10 for safety
        },
    }


# ── Load data — filter to AR and ES (data_selected has 700 rows per language) ──
# data_selected.parquet is the pre-sampled file with the 'reply' text column.
# It has one row per (language, annotator-text pair) — 6,300 rows total (9 langs × 700).
df_full = pd.read_parquet(INPUT_PATH)
df_base = df_full[df_full["language"].isin(LANGUAGES)].copy()
print(f"Base rows (AR+ES): {len(df_base):,}")

total_written = 0
cond_counts   = {c: 0 for c in CONDITIONS}

with open(OUTPUT_PATH, "w", encoding="utf-8") as batch_f, \
     open(INDEX_PATH,  "w", encoding="utf-8") as idx_f:

    for condition in CONDITIONS:
        for _, row in df_base.iterrows():
            # Build a unique custom_id that won't clash with the original experiment
            custom_id = f"prob_{condition}_{row['language']}_{row.name}"
            row_clean = sanitise(row.to_dict())

            req = build_request(custom_id, row_clean, condition)
            batch_f.write(json.dumps(req, ensure_ascii=False) + "\n")

            idx_entry = {
                "custom_id":           custom_id,
                "condition":           condition,
                "orig_row_idx":        int(row.name),
                "language":            row_clean["language"],
                "annotator_id":        row_clean["annotator_id"],
                "post_id":             row_clean["post_id"],
                "human_label":         int(float(row_clean["label"])),
                "text_entropy":        float(row.get("text_entropy", 0)),
                "age":                 row_clean["age"],
                "age_bin":             row_clean["age_bin"],
                "gender":              row_clean["gender"],
                "ethnicity":           row_clean["ethnicity"],
                "nationality":         row_clean["nationality"],
                "country_birth":       row_clean["country_birth"],
                "country_residence":   row_clean["country_residence"],
                "employment_status":   row_clean["employment_status"],
                "student_status":      row_clean["student_status"],
                "Generation":          row_clean["Generation"],
            }
            idx_f.write(json.dumps(idx_entry, ensure_ascii=False) + "\n")
            cond_counts[condition] += 1
            total_written += 1

print(f"\nWrote {total_written:,} requests to {OUTPUT_PATH}")
for c, n in cond_counts.items():
    print(f"  {c.upper()}: {n:,} requests")

# Cost estimate (input ~160 tok, output ~5 tok, Batch API 50% discount)
est_input_tok  = total_written * 160
est_output_tok = total_written * 5
cost_input  = est_input_tok  * 0.40 / 1_000_000  # gpt-4.1-mini standard price
cost_output = est_output_tok * 1.60 / 1_000_000
total_cost  = (cost_input + cost_output) * 0.5   # 50% batch discount
print(f"\nEstimated cost (Batch API, 50% off):")
print(f"  Input tokens : ~{est_input_tok:,}  → ${cost_input*0.5:.4f}")
print(f"  Output tokens: ~{est_output_tok:,}  → ${cost_output*0.5:.4f}")
print(f"  Total         : ~${total_cost:.3f}")
print(f"\nNext step: python 61_submit_prob_batch.py")
