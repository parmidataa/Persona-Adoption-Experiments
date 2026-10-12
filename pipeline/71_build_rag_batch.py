"""
71_build_rag_batch.py
=====================
Build the P4 (RAG-augmented) batch input for the RAG experiment.

For each of the 1,400 rows in data_selected (AR + ES, 700 per language):
  1. Retrieve the query text's embedding from rag_text_index.parquet.
  2. Compute cosine similarity against all texts in the same language,
     excluding the query text's own post_id (prevents data leakage).
  3. Take the TOP_N_CANDIDATES (=50) most similar candidate texts.
  4. For each candidate text, find the annotation from the most
     demographically similar annotator in rag_annotations.parquet.
  5. Rank candidates by (demographic_score DESC, text_similarity DESC).
  6. Select the top K=3 as few-shot examples.
  7. Build a P4 prompt: P2-style demographic list (system) + K examples
     + probability question (user).  Output format: float in [0, 1].

Demographic scoring (per example annotation vs query annotator):
  +3 pts : same nationality
  +2 pts : same gender
  +1 pt  : same ethnicity
  "unknown" values are never credited (score += 0 for that field).
  Maximum score per example = 6.

No API calls are made in this script — only local computation.

Outputs:
    rag_batch_input.jsonl   — batch API requests (1,400 rows)
    rag_batch_index.jsonl   — custom_id → row metadata + retrieval diagnostics

Usage:
    python 71_build_rag_batch.py
    (Run AFTER 70_build_rag_index.py)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL             = "gpt-4.1-mini"
LANGUAGES         = ["ar", "es"]
K                 = 3           # retrieved examples per prompt
TOP_N_CANDIDATES  = 50          # similar texts to consider before demographic ranking
MAX_EXAMPLE_CHARS = 200         # truncate example texts in the prompt

INPUT_PATH        = Path("data_selected.parquet")
TEXT_INDEX_PATH   = Path("rag_text_index.parquet")
ANNOTATIONS_PATH  = Path("rag_annotations.parquet")
OUTPUT_PATH       = Path("rag_batch_input.jsonl")
INDEX_PATH        = Path("rag_batch_index.jsonl")

# ── Sanitiser (identical to 60_build_prob_batch.py) ───────────────────────────
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


# ── Demographic scoring ────────────────────────────────────────────────────────
def demo_score(ann: dict, query: dict) -> int:
    """
    Score how closely an annotation's author matches the query annotator.
    Fields and weights chosen to match the variable-importance ranking
    established in Chapter 3 (nationality > gender > ethnicity for most languages).

    nationality : +3  (strongest predictor in Chapter 3)
    gender      : +2
    ethnicity   : +1
    "unknown"   :  0  (missing data never credited as a match)
    Maximum     :  6
    """
    score = 0
    for field, pts in [("nationality", 3), ("gender", 2), ("ethnicity", 1)]:
        q_val = str(query.get(field, "unknown")).strip()
        a_val = str(ann.get(field, "unknown")).strip()
        if q_val != "unknown" and a_val != "unknown" and q_val == a_val:
            score += pts
    return score


# ── Prompt builders ───────────────────────────────────────────────────────────
def build_p2_system(row: dict) -> str:
    """
    P2 full demographic list — identical to 60_build_prob_batch.py.
    Anchors the model's persona before the retrieved examples are shown.
    """
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


def format_example(text: str, label: int, idx: int) -> str:
    """
    Format one retrieved example for the few-shot block.
    Text is truncated to MAX_EXAMPLE_CHARS to keep prompts compact.
    """
    truncated = text.strip()[:MAX_EXAMPLE_CHARS]
    if len(text.strip()) > MAX_EXAMPLE_CHARS:
        truncated += "..."
    label_str = "ironic (1)" if label == 1 else "not ironic (0)"
    return f'  Example {idx}: "{truncated}"  \u2192  {label_str}'


def build_rag_request(custom_id: str, row: dict, examples: list) -> dict:
    """
    Build a single OpenAI Batch API request for the P4 condition.
    System  = P2 demographic list (persona anchor).
    User    = retrieved examples + probability elicitation question.
    """
    system_msg = build_p2_system(row)

    if examples:
        example_block = "\n".join(
            format_example(ex["reply"], ex["label"], i + 1)
            for i, ex in enumerate(examples)
        )
        example_intro = (
            "Here are examples of how annotators with a similar background "
            "have labeled texts like this one:\n\n"
            f"{example_block}\n\n"
        )
    else:
        # Degenerate fallback: no examples found (should not occur in practice)
        example_intro = ""

    user_msg = (
        f"{example_intro}"
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
            "max_completion_tokens": 10,   # "0.73" fits in ≤3 tokens; 10 for safety
        },
    }


# ── Load inputs ────────────────────────────────────────────────────────────────
print("Loading data_selected.parquet ...")
df_sel = pd.read_parquet(INPUT_PATH)
df_sel = df_sel[df_sel["language"].isin(LANGUAGES)].copy()
print(f"  {len(df_sel):,} query rows (AR + ES)")

print("Loading rag_text_index.parquet ...")
if not TEXT_INDEX_PATH.exists():
    print("ERROR: rag_text_index.parquet not found. Run 70_build_rag_index.py first.")
    exit(1)
index_df = pd.read_parquet(TEXT_INDEX_PATH)
print(f"  {len(index_df):,} unique texts with embeddings")

print("Loading rag_annotations.parquet ...")
if not ANNOTATIONS_PATH.exists():
    print("ERROR: rag_annotations.parquet not found. Run 70_build_rag_index.py first.")
    exit(1)
ann_df = pd.read_parquet(ANNOTATIONS_PATH)
print(f"  {len(ann_df):,} annotation rows in retrieval pool")

# ── Pre-build fast lookup structures ──────────────────────────────────────────
print("\nBuilding lookup structures ...")

# post_id → numpy embedding (float32)
emb_by_postid: dict[str, np.ndarray] = {}
for _, row in index_df.iterrows():
    emb_by_postid[str(row["post_id"])] = np.array(row["embedding"], dtype=np.float32)

# post_id → list of annotation dicts (for demographic lookup)
ann_by_postid: dict[str, list] = {}
for post_id, grp in ann_df.groupby("post_id"):
    ann_by_postid[str(post_id)] = grp.to_dict("records")

# Per-language: pre-normalised embedding matrix + post_id array
# Normalised once here so cosine similarity is just a dot product.
lang_data: dict[str, dict] = {}
for lang in LANGUAGES:
    lang_idx  = index_df[index_df["language"] == lang].reset_index(drop=True)
    post_ids  = lang_idx["post_id"].astype(str).values
    embs      = np.stack(
        [np.array(e, dtype=np.float32) for e in lang_idx["embedding"].values]
    )  # shape (N_lang, 1536)
    norms     = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / np.maximum(norms, 1e-10)
    lang_data[lang] = {"post_ids": post_ids, "embs_norm": embs_norm}
    print(f"  {lang.upper()} retrieval pool: {len(post_ids):,} texts")

# ── Main retrieval + prompt-building loop ──────────────────────────────────────
print(f"\nBuilding P4 batch  (K={K} examples, TOP_N={TOP_N_CANDIDATES}) ...")

total_written = 0
n_fallback    = 0          # rows where fewer than K examples were found
score_hist    = {}         # demographic score → count (across all selected examples)

with open(OUTPUT_PATH, "w", encoding="utf-8") as batch_f, \
     open(INDEX_PATH,  "w", encoding="utf-8") as idx_f:

    for _, row in df_sel.iterrows():
        row_clean    = sanitise(row.to_dict())
        lang         = row_clean["language"]
        query_postid = str(row_clean["post_id"])
        custom_id    = f"rag_p4_{lang}_{row.name}"

        # 1. Retrieve query embedding ──────────────────────────────────────────
        if query_postid not in emb_by_postid:
            # Query post not found in index (unexpected if data_selected ⊆ data_{lang})
            print(f"  WARNING: post_id {query_postid} not in embedding index — "
                  f"using zero vector (row {row.name})")
            q_emb_norm = np.zeros(
                next(iter(emb_by_postid.values())).shape, dtype=np.float32
            )
        else:
            q_emb   = emb_by_postid[query_postid]
            q_norm  = np.linalg.norm(q_emb)
            q_emb_norm = q_emb / max(float(q_norm), 1e-10)

        # 2. Cosine similarity with all same-language texts ───────────────────
        ld       = lang_data[lang]
        cos_sims = ld["embs_norm"] @ q_emb_norm       # shape (N_lang,)
        post_ids = ld["post_ids"]

        # Exclude the query text itself (data-leakage prevention)
        exclude_mask          = post_ids == query_postid
        cos_sims[exclude_mask] = -2.0   # guaranteed below any real cosine value

        # Top-N candidate indices (descending similarity)
        n_pool = len(cos_sims)
        top_n  = min(TOP_N_CANDIDATES, n_pool - int(exclude_mask.sum()))
        if top_n <= 0:
            candidates = []
        else:
            # np.argpartition is O(N) vs O(N log N) for full sort — faster for large pools
            part_idx  = np.argpartition(cos_sims, -top_n)[-top_n:]
            top_idx   = part_idx[np.argsort(cos_sims[part_idx])[::-1]]

            # 3. For each candidate, pick best-matching annotation ─────────────
            candidates = []
            for idx in top_idx:
                cand_pid = post_ids[idx]
                sim      = float(cos_sims[idx])
                if sim < -1.5:       # masked; skip
                    continue

                cand_anns = ann_by_postid.get(str(cand_pid), [])
                if not cand_anns:
                    continue

                best_ann   = None
                best_score = -1
                for ann in cand_anns:
                    sc = demo_score(ann, row_clean)
                    if sc > best_score:
                        best_score = sc
                        best_ann   = ann

                if best_ann is None:
                    continue

                candidates.append({
                    "post_id":           str(cand_pid),
                    "reply":             str(best_ann.get("reply", "")),
                    "label":             int(float(best_ann.get("label", 0))),
                    "demographic_score": best_score,
                    "text_similarity":   sim,
                })

        # 4. Rank by demographic match first, text similarity second ──────────
        candidates.sort(
            key=lambda x: (x["demographic_score"], x["text_similarity"]),
            reverse=True,
        )
        examples = candidates[:K]

        if len(examples) < K:
            n_fallback += 1

        # Track demographic score distribution
        for ex in examples:
            sc = ex["demographic_score"]
            score_hist[sc] = score_hist.get(sc, 0) + 1

        # 5. Build and write batch request ────────────────────────────────────
        req = build_rag_request(custom_id, row_clean, examples)
        batch_f.write(json.dumps(req, ensure_ascii=False) + "\n")

        # 6. Write index entry (metadata + retrieval diagnostics) ─────────────
        idx_entry = {
            "custom_id":         custom_id,
            "condition":         "p4",
            "orig_row_idx":      int(row.name),
            "language":          lang,
            "annotator_id":      row_clean["annotator_id"],
            "post_id":           query_postid,
            "human_label":       int(float(row_clean["label"])),
            "text_entropy":      float(row.get("text_entropy", 0)),
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
            # Retrieval diagnostics
            "n_examples_retrieved": len(examples),
            "best_demo_score":      examples[0]["demographic_score"] if examples else 0,
            "mean_text_sim":        float(np.mean([e["text_similarity"]
                                                   for e in examples])) if examples else 0.0,
        }
        idx_f.write(json.dumps(idx_entry, ensure_ascii=False) + "\n")
        total_written += 1

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Written : {total_written:,} requests → {OUTPUT_PATH}")
print(f"  Fallback rows (< {K} examples found): {n_fallback}")

print(f"\n  Demographic match score distribution")
print(f"  (across all {total_written * K} example slots):")
print(f"  {'Score':>6}  {'Description':<30}  {'Count':>6}  {'%':>6}")
print(f"  {'-'*55}")
SCORE_LABELS = {
    6: "nationality + gender + ethnicity",
    5: "nationality + gender",
    4: "nationality + ethnicity",
    3: "nationality only",
    2: "gender only",
    1: "ethnicity only",
    0: "no match  (any annotator)",
}
total_ex = sum(score_hist.values())
for sc in sorted(score_hist.keys(), reverse=True):
    n   = score_hist[sc]
    pct = n / total_ex * 100 if total_ex > 0 else 0
    lbl = SCORE_LABELS.get(sc, "")
    print(f"  {sc:>6}  {lbl:<30}  {n:>6}  {pct:>5.1f}%")

# Cost estimate
avg_input_tok  = 320   # system (~110) + examples (~160) + question (~50)
est_input_tok  = total_written * avg_input_tok
est_output_tok = total_written * 5
cost_in  = est_input_tok  * 0.40 / 1_000_000 * 0.5   # gpt-4.1-mini, 50% batch discount
cost_out = est_output_tok * 1.60 / 1_000_000 * 0.5
print(f"\n  Estimated cost (Batch API, 50% off): ~${cost_in + cost_out:.3f}")
print(f"{'='*60}")
print(f"\nNext step: python 72_submit_rag_batch.py")
