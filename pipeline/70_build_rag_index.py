"""
70_build_rag_index.py
=====================
Build the retrieval index for the RAG experiment (P4 condition).

For every unique text in the full Arabic (data_ar.parquet) and Spanish
(data_es.parquet) corpora, compute a text embedding using
text-embedding-3-small (1536 dims).  The resulting index is used by
71_build_rag_batch.py to retrieve semantically similar texts annotated
by demographically similar annotators.

Outputs:
    rag_text_index.parquet    — unique texts + embeddings
                                columns: post_id, language, reply, embedding (list[float])
    rag_annotations.parquet   — full annotation pool (all rows, AR + ES)
                                columns: all original columns from data_{lang}.parquet

Usage:
    export OPENAI_API_KEY=sk-...
    python 70_build_rag_index.py
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
LANGUAGES       = ["ar", "es"]
EMBED_MODEL     = "text-embedding-3-small"
BATCH_SIZE      = 500        # texts per embedding API call (well under 2048 limit)
MAX_CHARS       = 2000       # safety truncation — Arabic/Spanish tweets are short;
                             # this is a guard against any unusually long posts

TEXT_INDEX_OUT  = Path("rag_text_index.parquet")
ANNOTATIONS_OUT = Path("rag_annotations.parquet")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Load full per-language corpora ─────────────────────────────────────────────
frames = []
for lang in LANGUAGES:
    path = Path(f"data_{lang}.parquet")
    if not path.exists():
        print(f"ERROR: {path} not found. Expected full-corpus file for {lang.upper()}.")
        exit(1)
    df = pd.read_parquet(path)
    frames.append(df)
    print(f"Loaded {path}: {len(df):,} annotation rows, "
          f"{df['post_id'].nunique():,} unique texts")

df_full = pd.concat(frames, ignore_index=True)
print(f"\nTotal annotation rows (AR + ES): {len(df_full):,}")

# ── Save full annotation pool ──────────────────────────────────────────────────
# This file is the lookup table that 71_build_rag_batch.py queries for
# demographic information and irony labels.
df_full.to_parquet(ANNOTATIONS_OUT, index=False)
print(f"Saved annotation pool : {ANNOTATIONS_OUT}  ({len(df_full):,} rows)")

# ── Deduplicate to unique texts ────────────────────────────────────────────────
# Each unique text (post_id) is embedded once, regardless of how many
# annotators rated it.  The annotation pool retains all rows for retrieval.
unique_texts = (
    df_full[["post_id", "language", "reply"]]
    .drop_duplicates(subset="post_id")
    .reset_index(drop=True)
)
n_unique = len(unique_texts)
print(f"\nUnique texts to embed: {n_unique:,}")
for lang in LANGUAGES:
    n = (unique_texts["language"] == lang).sum()
    print(f"  {lang.upper()}: {n:,}")

# Truncate for the embedding call only
texts_to_embed = unique_texts["reply"].str[:MAX_CHARS].tolist()

# ── Embed in batches ───────────────────────────────────────────────────────────
all_embeddings = []
total_tokens   = 0
n_batches      = (n_unique + BATCH_SIZE - 1) // BATCH_SIZE

print(f"\nEmbedding {n_unique:,} texts in {n_batches} batches of ≤{BATCH_SIZE}...")

for i in range(0, n_unique, BATCH_SIZE):
    batch_texts = texts_to_embed[i : i + BATCH_SIZE]
    batch_num   = i // BATCH_SIZE + 1

    print(f"  Batch {batch_num:>3}/{n_batches}  ({len(batch_texts)} texts) ... ", end="", flush=True)

    try:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch_texts,
        )
    except Exception as e:
        print(f"FAILED — {e}")
        exit(1)

    batch_embs = [item.embedding for item in resp.data]
    assert len(batch_embs) == len(batch_texts), "Embedding count mismatch in batch"

    all_embeddings.extend(batch_embs)
    batch_tokens = resp.usage.total_tokens
    total_tokens += batch_tokens
    batch_cost   = batch_tokens * 0.02 / 1_000_000
    print(f"OK  ({batch_tokens:,} tokens, ~${batch_cost:.5f})")

    # Small pause to stay well within rate limits
    if i + BATCH_SIZE < n_unique:
        time.sleep(0.3)

assert len(all_embeddings) == n_unique, \
    f"Final embedding count {len(all_embeddings)} != expected {n_unique}"

# ── Verify embedding dimension ─────────────────────────────────────────────────
emb_dim = len(all_embeddings[0])
print(f"\nEmbedding dimension  : {emb_dim}")
print(f"Total tokens used    : {total_tokens:,}")
total_cost = total_tokens * 0.02 / 1_000_000
print(f"Total embedding cost : ~${total_cost:.5f}")

# ── Save text index ────────────────────────────────────────────────────────────
# Stored as a list[float] column; pyarrow handles this as a nested type.
# When loading, convert with: np.array(row["embedding"], dtype=np.float32)
unique_texts["embedding"] = all_embeddings
unique_texts.to_parquet(TEXT_INDEX_OUT, index=False)
print(f"\nSaved text index     : {TEXT_INDEX_OUT}  ({len(unique_texts):,} rows)")

# ── Sanity check: round-trip a single embedding ────────────────────────────────
check_df  = pd.read_parquet(TEXT_INDEX_OUT)
check_emb = np.array(check_df["embedding"].iloc[0], dtype=np.float32)
assert check_emb.shape == (emb_dim,), "Round-trip embedding shape mismatch"
print(f"Round-trip check     : OK  (shape {check_emb.shape})")

print(f"\n{'='*60}")
print("  DONE")
print(f"{'='*60}")
print(f"  rag_text_index.parquet   : {n_unique:,} unique texts with {emb_dim}-dim embeddings")
print(f"  rag_annotations.parquet  : {len(df_full):,} annotation rows for retrieval")
print(f"\nNext step: python 71_build_rag_batch.py")
