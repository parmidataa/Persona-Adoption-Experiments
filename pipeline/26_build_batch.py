"""
26_build_batch.py
=================
Step 2: Convert data_selected.parquet into a single batch_input.jsonl file
with all API requests for conditions P0, P1, P2, P3.

Output: batch_input.jsonl  (~25,200 lines)

Each line is one OpenAI Batch API request.
custom_id encodes: {condition}_{row_index}  — unique, short, parseable.
A companion index file (batch_index.jsonl) maps each custom_id back to the
original annotator_id, post_id, language, and human_label for downstream analysis.

Column notes on data_selected.parquet:
  - age            : float (e.g. 30.0) → rounded to int in prompts
  - Generation     : GenZ/GenY/GenX/Boomer  (used in DE P3 template)
  - employment_status / student_status: may contain "DATA_EXPIRED" → sanitised
"""

import json
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL      = "gpt-4.1-mini"
CONDITIONS = ["p0", "p1", "p2", "p3"]
INPUT_PATH  = Path("data_selected.parquet")
OUTPUT_PATH = Path("batch_input.jsonl")
INDEX_PATH  = Path("batch_index.jsonl")   # maps custom_id → row metadata

# ── P3 prose templates (one per language, from Phase E findings) ───────────────
# Fields in {braces} come from the sanitised row dict.
# - DE uses {Generation} (GenZ/GenY/GenX/Boomer) instead of GenerationAggregated
# - All fields sanitised: DATA_EXPIRED → "unknown", float age → int
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def sanitise(row: dict) -> dict:
    """
    Return a cleaned copy of the row dict suitable for .format(**row):
      - age float → int string (e.g. 30.0 → "30")
      - DATA_EXPIRED / nan / empty → "unknown"
      - all values → str
    """
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
            # Fallback if a field is missing in this row
            prose = f"You are a text annotator. (template field missing: {e})"
        return "You are annotating text for irony. " + prose

    raise ValueError(f"Unknown condition: {condition}")


def build_request(custom_id: str, row: dict, condition: str) -> dict:
    system_msg = build_system(row, condition)
    user_msg   = f"Ironic or not? Reply with 0 or 1 only.\nText: {row['reply']}"
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
            "max_completion_tokens": 3,
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded: {len(df):,} rows from {INPUT_PATH}")

total_written = 0
condition_counts = {c: 0 for c in CONDITIONS}

with open(OUTPUT_PATH, "w", encoding="utf-8") as batch_f, \
     open(INDEX_PATH,  "w", encoding="utf-8") as idx_f:

    for condition in CONDITIONS:
        for row_idx, row in df.iterrows():
            custom_id  = f"{condition}_{row_idx}"
            row_clean  = sanitise(row.to_dict())

            # Build and write the API request
            req = build_request(custom_id, row_clean, condition)
            batch_f.write(json.dumps(req, ensure_ascii=False) + "\n")

            # Write the index entry (maps custom_id → all metadata needed for analysis)
            idx_entry = {
                "custom_id":    custom_id,
                "condition":    condition,
                "row_idx":      int(row_idx),
                "language":     row_clean["language"],
                "annotator_id": row_clean["annotator_id"],
                "post_id":      row_clean["post_id"],
                "human_label":  int(float(row_clean["label"])),
                "text_entropy": float(row.get("text_entropy", 0)),
                # demographic fields for the analysis model
                "age":          row_clean["age"],
                "age_bin":      row_clean["age_bin"],
                "gender":       row_clean["gender"],
                "ethnicity":    row_clean["ethnicity"],
                "nationality":  row_clean["nationality"],
                "country_birth":       row_clean["country_birth"],
                "country_residence":   row_clean["country_residence"],
                "employment_status":   row_clean["employment_status"],
                "student_status":      row_clean["student_status"],
                "Generation":          row_clean["Generation"],
            }
            idx_f.write(json.dumps(idx_entry, ensure_ascii=False) + "\n")

            total_written += 1
            condition_counts[condition] += 1

print(f"\nBatch file written: {total_written:,} requests → {OUTPUT_PATH}")
print(f"Index file written: {total_written:,} entries → {INDEX_PATH}")
print()
print("Per-condition breakdown:")
for cond, count in condition_counts.items():
    print(f"  {cond.upper()}: {count:,} requests")

# ── Cost estimate ──────────────────────────────────────────────────────────────
# gpt-5.4-mini Batch API: $0.075/1M input tokens, $0.30/1M output tokens (50% discount)
# Rough estimate: ~118 tokens input per request, ~2 tokens output
estimated_input_tokens  = total_written * 118
estimated_output_tokens = total_written * 2
cost_input  = estimated_input_tokens  / 1_000_000 * 0.075
cost_output = estimated_output_tokens / 1_000_000 * 0.30
print(f"\nEstimated cost (batch API 50% discount):")
print(f"  Input:  ~{estimated_input_tokens:,} tokens → ${cost_input:.3f}")
print(f"  Output: ~{estimated_output_tokens:,} tokens → ${cost_output:.3f}")
print(f"  Total:  ~${cost_input + cost_output:.3f}")

# File size sanity check
import os
size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
print(f"\nFile size: {size_mb:.1f} MB  (OpenAI limit: 200 MB — {'OK' if size_mb < 200 else 'EXCEEDS LIMIT'})")
print(f"Request count: {total_written:,}  (OpenAI limit: 50,000 — {'OK' if total_written <= 50000 else 'EXCEEDS LIMIT'})")
