"""
50_build_saia_sample.py — build the budget-fitted SAIA subset
=============================================================
SAIA has a hard ~3,000 requests/month key limit, so we run only the BEST
condition per language (already identified from the OpenAI experiment) on a
small n per language. This is a strict subset of the original 700-row sample,
so every SAIA model stays directly comparable to gpt-4.1-mini / gpt-4.1 on the
same rows (we re-score OpenAI on this subset in the analysis step).

Design A: 9 languages x best condition x n=100  = 900 requests per model.

Outputs:
    saia_bestcond_ids.json    — 900 custom_ids to run (same set for every model)
    saia_bestcond_index.jsonl — batch_index rows restricted to the subset
"""

import json, random
from collections import defaultdict

SEED = 42
N_PER_LANG = 100
BEST_COND = {"ar": "p2", "de": "p3", "en": "p2", "es": "p2", "fr": "p2",
             "hi": "p2", "it": "p3", "nl": "p2", "pt": "p1"}

idx = [json.loads(l) for l in open("batch_index.jsonl")]

rows_by_lang = defaultdict(list)
for r in idx:
    if r["condition"] == "p0":
        rows_by_lang[r["language"]].append(r["row_idx"])

rng = random.Random(SEED)
subset_ids = []
sel_rowidx_by_lang = {}
for lang in sorted(rows_by_lang):
    ris = sorted(rows_by_lang[lang])
    chosen = rng.sample(ris, N_PER_LANG)
    sel_rowidx_by_lang[lang] = set(chosen)
    cond = BEST_COND[lang]
    subset_ids.extend(f"{cond}_{ri}" for ri in chosen)
    print(f"  {lang}: {N_PER_LANG} rows, best cond = {cond.upper()}")

subset_ids = sorted(subset_ids)
json.dump(subset_ids, open("saia_bestcond_ids.json", "w"))
print(f"\nTotal requests per model: {len(subset_ids):,}  "
      f"(x3 models = {len(subset_ids)*3:,} of ~2,990 monthly budget)")

subset_set = set(subset_ids)
with open("saia_bestcond_index.jsonl", "w") as f:
    for r in idx:
        if r["custom_id"] in subset_set:
            f.write(json.dumps(r) + "\n")
print("Wrote saia_bestcond_ids.json and saia_bestcond_index.jsonl")
