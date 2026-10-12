"""
31_build_extended_batch.py
===========================
Builds a single combined batch JSONL covering three new experiments:

  Option 1 — EN variable ablation (5 single-variable prompts × 700 EN rows)
    p_age, p_gender, p_ethnicity, p_nat, p_country
    Question: Which individual demographic variable causes persona disruption?

  Option 3 — HI sample scaling (1400 rows × 4 conditions)
    Re-samples Hindi at 1400 rows using the same pipeline as 25_prepare_data.py
    Question: Is HI failure a sampling-power problem or an LLM-competence problem?

  Option 4 — IT and PT targeted P4 prompt (1 new condition × 700 rows each)
    "You are a {employment_status}." for both IT and PT
    Question: Does focusing on the single strongest demographic predictor help?

Output: extended_batch_input.jsonl (total calls ≈ 10,500)
"""

import json, re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy as scipy_entropy

MODEL  = "gpt-4.1-mini"
MAX_TOKENS = 3
TEMPERATURE = 0

# ── Load original EN/IT/PT predictions data (for existing 700-row samples) ────
df_all = pd.read_parquet("data_predictions_clean.parquet")

# ── Build row_idx → text lookup from original batch_input.jsonl ──────────────
print("Building text lookup from batch_input.jsonl...")
text_lookup = {}  # row_idx (int) → text string  (only for p0 condition)
with open("batch_input.jsonl") as _f:
    for _line in _f:
        _row = json.loads(_line)
        _cid = _row["custom_id"]
        if not _cid.startswith("p0_"):
            continue
        _user_msg = _row["body"]["messages"][1]["content"]
        _text = _user_msg.replace("Ironic or not? Reply with 0 or 1 only.\nText: ", "")
        try:
            _ridx = int(_cid.split("_")[1])
            text_lookup[_ridx] = _text
        except (IndexError, ValueError):
            pass
print(f"  Loaded {len(text_lookup)} texts")

# ── Load raw per-language parquets (for HI 1400-row resample) ─────────────────
ENTROPY_LOW  = 0.25
ENTROPY_HIGH = 0.92
MIN_DEPTH    = 5
RANDOM_STATE = 42

def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    from scipy.stats import entropy as sp_ent
    return sp_ent([p, 1 - p], base=2)


def safe_load(lang):
    df = pd.read_parquet(f"data_{lang}.parquet")
    for col in df.columns:
        arr = df[col].to_numpy()
        if col == "annotator_id":
            df[col] = arr.astype(str)
        elif arr.dtype.kind in ("U", "O") or "str" in str(df[col].dtype).lower() or "arrow" in str(df[col].dtype).lower():
            df[col] = [str(v) for v in arr]
    for col in ["label", "age", "age_centered"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "age_bin" not in df.columns:
        bins   = [0, 26, 36, 46, 200]
        labels = ["18-25", "26-35", "36-45", "46+"]
        df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=False).astype(str)
    return df


def sample_language(df_raw, target_n, rng):
    """Replicate the 4-step pipeline from 25_prepare_data.py at a given target_n."""
    DEMO_COLS = ["age_bin", "gender", "ethnicity", "nationality"]

    # Step A: entropy filter
    text_stats = df_raw.groupby("post_id")["label"].agg(["mean","count"])
    text_stats["entropy"] = text_stats["mean"].apply(binary_entropy)
    valid_posts = text_stats[(text_stats["entropy"] >= ENTROPY_LOW) &
                              (text_stats["entropy"] <= ENTROPY_HIGH)].index
    df_filt = df_raw[df_raw["post_id"].isin(valid_posts)].copy()

    # Step B: annotator depth filter
    ann_depth = df_filt.groupby("annotator_id")["post_id"].nunique()
    deep_anns = ann_depth[ann_depth >= MIN_DEPTH].index
    df_filt = df_filt[df_filt["annotator_id"].isin(deep_anns)].copy()

    # Step C: greedy diversity (use all annotators)
    # simplified: just use all annotators that passed Step B
    selected_anns = list(deep_anns)

    df_sel = df_filt[df_filt["annotator_id"].isin(selected_anns)].copy()

    # Step D: stratified sample
    ann_list = sorted(df_sel["annotator_id"].unique())
    n_ann = len(ann_list)
    base  = target_n // n_ann
    extra = target_n - base * n_ann

    parts = []
    for i, ann in enumerate(ann_list):
        n_take = base + (1 if i < extra else 0)
        df_ann = df_sel[df_sel["annotator_id"] == ann]
        take   = min(n_take, len(df_ann))
        parts.append(df_ann.sample(n=take, random_state=rng))

    df_out = pd.concat(parts, ignore_index=True)
    # Top up or trim
    if len(df_out) < target_n:
        shortfall = target_n - len(df_out)
        remaining = df_sel[~df_sel.index.isin(df_out.index)]
        if len(remaining) >= shortfall:
            df_out = pd.concat([df_out, remaining.sample(n=shortfall, random_state=rng)], ignore_index=True)
    df_out = df_out.iloc[:target_n].copy()
    return df_out


def make_request(custom_id, system_msg, user_msg):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_completion_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
        },
    }


requests = []
meta_rows = []  # for parsing later: custom_id → metadata

# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION 1: EN variable ablation (5 single-variable prompts × 700 EN rows)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Option 1: EN variable ablation...")

df_en = df_all[(df_all["language"] == "en") & (df_all["condition"] == "p0")].copy()

ablation_conditions = {
    "p_age":      lambda r: f"You are annotating text for irony. Persona: age={int(r['age']) if pd.notna(r['age']) else 'unknown'}",
    "p_gender":   lambda r: f"You are annotating text for irony. Persona: gender={r['gender']}",
    "p_ethnicity":lambda r: f"You are annotating text for irony. Persona: ethnicity={r['ethnicity']}",
    "p_nat":      lambda r: f"You are annotating text for irony. Persona: nationality={r['nationality']}",
    "p_country":  lambda r: f"You are annotating text for irony. Persona: country_residence={r['country_residence']}",
}

for cond_name, sys_fn in ablation_conditions.items():
    for idx, row in df_en.iterrows():
        sys_msg  = sys_fn(row)
        row_text = text_lookup.get(int(row["row_idx"]), "")
        user_msg = f"Ironic or not? Reply with 0 or 1 only.\nText: {row_text}"
        custom_id = f"ext_{cond_name}_en_{row['row_idx']}"
        requests.append(make_request(custom_id, sys_msg, user_msg))
        meta_rows.append({
            "custom_id":    custom_id,
            "condition":    cond_name,
            "language":     "en",
            "row_idx":      row["row_idx"],
            "annotator_id": row["annotator_id"],
            "post_id":      row["post_id"],
            "human_label":  row["human_label"],
            "age": row["age"], "gender": row["gender"],
            "ethnicity": row["ethnicity"], "nationality": row["nationality"],
            "country_birth": row["country_birth"],
            "country_residence": row["country_residence"],
            "employment_status": row["employment_status"],
            "student_status": row["student_status"],
        })

print(f"  EN ablation: {len(ablation_conditions)} conditions × {len(df_en)} rows = {len(ablation_conditions)*len(df_en)} requests")

# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION 3: HI sample scaling (1400 rows × 4 conditions)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Option 3: HI sample scaling to 1400 rows...")

rng = np.random.RandomState(RANDOM_STATE)
df_hi_raw = safe_load("hi")
df_hi_1400 = sample_language(df_hi_raw, 1400, rng)
print(f"  HI 1400-row sample: {len(df_hi_1400)} rows, {df_hi_1400['annotator_id'].nunique()} annotators")

# Find which 1400-row rows overlap with original 700 (for continuity reporting)
df_orig_hi = df_all[(df_all["language"] == "hi") & (df_all["condition"] == "p0")][["row_idx","post_id","annotator_id"]]

hi_texts_raw = df_hi_raw.set_index(["annotator_id", "post_id"])["text"] if "text" in df_hi_raw.columns else None

# Check text column name
print(f"  HI raw columns: {df_hi_raw.columns.tolist()[:10]}")

# P0-P3 system messages for HI (from 26_build_batch.py logic)
def hi_system_msg(cond, row):
    if cond == "p0":
        return "You are a text annotator."
    elif cond == "p1":
        age_val = int(row['age']) if pd.notna(row.get('age')) else 'unknown'
        gender  = row.get('gender','unknown')
        ethnicity = row.get('ethnicity','unknown')
        return (f"You are annotating text for irony. "
                f"Persona: age={age_val}, gender={gender}, ethnicity={ethnicity}")
    elif cond == "p2":
        age_val = int(row['age']) if pd.notna(row.get('age')) else 'unknown'
        gender  = row.get('gender','unknown')
        ethnicity = row.get('ethnicity','unknown')
        country_birth = row.get('country_birth','unknown')
        country_res   = row.get('country_residence','unknown')
        emp   = row.get('employment_status','unknown')
        stud  = row.get('student_status','unknown')
        return (f"You are annotating text for irony. "
                f"Persona: age={age_val}, gender={gender}, ethnicity={ethnicity}, "
                f"country_birth={country_birth}, country_residence={country_res}, "
                f"employment={emp}, student={stud}")
    else:  # p3 for HI: age, ethnicity, country_birth
        age_val = int(row['age']) if pd.notna(row.get('age')) else 'unknown'
        ethnicity = row.get('ethnicity','unknown')
        country_birth = row.get('country_birth','unknown')
        return (f"You are annotating text for irony. "
                f"You are a {age_val}-year-old {ethnicity} person originally from {country_birth}.")

text_col = "text" if "text" in df_hi_raw.columns else "reply"
print(f"  HI text column: {text_col}")

hi_conditions = ["p0", "p1", "p2", "p3"]
for cond in hi_conditions:
    for i, (_, row) in enumerate(df_hi_1400.iterrows()):
        sys_msg  = hi_system_msg(cond, row)
        user_msg = f"Ironic or not? Reply with 0 or 1 only.\nText: {row.get(text_col, row.get('reply',''))}"
        ann_id   = str(row['annotator_id'])
        post_id  = str(row['post_id'])
        custom_id = f"ext_hi1400_{cond}_{ann_id}_{post_id}_{i}"
        requests.append(make_request(custom_id, sys_msg, user_msg))
        meta_rows.append({
            "custom_id":    custom_id,
            "condition":    cond,
            "language":     "hi",
            "row_idx":      i,
            "annotator_id": ann_id,
            "post_id":      post_id,
            "human_label":  float(row.get('label', np.nan)),
            "age": row.get('age'), "gender": row.get('gender',''),
            "ethnicity": row.get('ethnicity',''), "nationality": row.get('nationality',''),
            "country_birth": row.get('country_birth',''),
            "country_residence": row.get('country_residence',''),
            "employment_status": row.get('employment_status',''),
            "student_status": row.get('student_status',''),
        })

print(f"  HI 1400 rows × 4 conditions = {len(hi_conditions) * len(df_hi_1400)} requests")

# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION 4: IT and PT targeted P4 prompt (employment_status only)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Option 4: IT and PT targeted P4 prompts...")

for lang in ["it", "pt"]:
    df_lang = df_all[(df_all["language"] == lang) & (df_all["condition"] == "p0")].copy()
    for _, row in df_lang.iterrows():
        emp = row.get('employment_status', 'unknown')
        if lang == "it":
            sys_msg = f"You are annotating text for irony. You are a {emp}."
        else:  # pt
            country_birth = row.get('country_birth', 'unknown')
            sys_msg = f"You are annotating text for irony. You are a {emp} person originally from {country_birth}."

        row_text  = text_lookup.get(int(row["row_idx"]), "")
        user_msg  = f"Ironic or not? Reply with 0 or 1 only.\nText: {row_text}"
        custom_id = f"ext_p4_{lang}_{row['row_idx']}"
        requests.append(make_request(custom_id, sys_msg, user_msg))
        meta_rows.append({
            "custom_id":    custom_id,
            "condition":    "p4_emp",
            "language":     lang,
            "row_idx":      row["row_idx"],
            "annotator_id": row["annotator_id"],
            "post_id":      row["post_id"],
            "human_label":  row["human_label"],
            "age": row["age"], "gender": row["gender"],
            "ethnicity": row["ethnicity"], "nationality": row["nationality"],
            "country_birth": row["country_birth"],
            "country_residence": row["country_residence"],
            "employment_status": row["employment_status"],
            "student_status": row["student_status"],
        })

print(f"  IT + PT × 700 rows × 1 condition = {2 * 700} requests")

# ═══════════════════════════════════════════════════════════════════════════════
#  Write outputs
# ═══════════════════════════════════════════════════════════════════════════════
with open("extended_batch_input.jsonl", "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")

df_meta = pd.DataFrame(meta_rows)
# Ensure all columns have consistent types for parquet serialisation
for col in ["age"]:
    df_meta[col] = pd.to_numeric(df_meta[col], errors="coerce")
for col in ["gender","ethnicity","nationality","country_birth","country_residence","employment_status","student_status"]:
    df_meta[col] = df_meta[col].astype(str)
df_meta.to_csv("extended_batch_index.csv", index=False)

print(f"\n{'═'*70}")
print(f"  Total requests: {len(requests)}")
print(f"    Option 1 (EN ablation):    {5 * 700:>6}")
print(f"    Option 3 (HI 1400):        {4 * len(df_hi_1400):>6}")
print(f"    Option 4 (IT+PT P4):       {2 * 700:>6}")
print(f"  Estimated cost: ~${len(requests) * 0.000001 * 3 * 0.40:.4f}  (rough upper bound)")
print(f"  Saved: extended_batch_input.jsonl ({len(requests)} lines)")
print(f"         extended_batch_index.parquet")
print(f"{'═'*70}")
