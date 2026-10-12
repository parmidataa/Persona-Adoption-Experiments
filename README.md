# Persona Effects in Multilingual Irony Annotation

Code and results for the Master's thesis *A Multi-Dimensional Evaluation of Persona Adoption by Large Language Models in Cross-Lingual Irony Detection* (Parmida Talebi, University of Marburg, 2026).

The thesis has two studies on the [MultiPICo](https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/MultiPICo) corpus (nine languages, per-annotator irony labels with demographic metadata):

- **Study 1** measures how much of the variance in human irony labels annotator demographics explain, using linear mixed models with marginal $R^2$ (Nakagawa and Schielzeth), a crossed-random-effects persona ceiling, tests with the annotator as the unit of analysis, and shuffled-profile null baselines.
- **Study 2** gives LLMs (`gpt-4.1-mini`, `gpt-4.1`, Claude Opus 5, Llama-3.1-8B, Mistral-Large-3) the same demographic profiles as persona prompts and compares the demographic structure in their labels with the human one, in magnitude and direction, against null and split-half reliability baselines.

## Repository layout

```
pipeline/   all scripts, shared modules, result tables, and LLM predictions
figures/    the figures used in the thesis
```

All scripts read and write files in their own working directory, so run them from inside `pipeline/`.

## Setup

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The `lme4` cross-check additionally needs R with the `lme4` package.

## What you can reproduce

**Without any API access (everything except new LLM calls).**

1. Download and preprocess the corpus (the texts are not redistributed here):

   ```bash
   cd pipeline
   python 01_data_exploration.py        # downloads MultiPICo from Hugging Face -> data_raw.parquet
   python 02_preprocessing.py           # Phase A sample -> data_<lang>.parquet
   python 07_undersampled_preprocessing.py
   python 15_preprocessing_extended.py  # complete-data sample -> data_<lang>_v2.parquet
   python 25_prepare_data.py            # Study 2 700-row samples (seed 42) -> data_selected.parquet
   ```

2. Run any Study 1 script, and any Study 2 analysis script. The LLM labels used in the thesis are included (see below), so the Study 2 analyses run on the same predictions without new API calls.

**With API access.** The `build`, `submit`/`run`, `retrieve`, and `parse` scripts send new requests. They read keys from the environment:

```bash
export OPENAI_API_KEY=...      # gpt-4.1-mini, gpt-4.1, RAG embeddings
export ANTHROPIC_API_KEY=...   # Claude Opus 5
export SAIA_API_KEY=...        # Llama / Mistral through the SAIA gateway
```

New runs of the commercial models may not reproduce the exact labels, since API behaviour can change over time and the Claude Opus 5 runs used the API's default sampling.

## Included data

| File | Content |
|---|---|
| `results_*.csv`, `saia_master_*.csv` | Numerical results behind the thesis tables |
| `data_predictions_clean.parquet` | `gpt-4.1-mini`, P0--P3, 9 languages x 700 rows |
| `data_predictions_gpt41_clean.parquet` | `gpt-4.1`, same design |
| `data_predictions_opus5_clean.parquet` | Claude Opus 5, same design |
| `prob_*`, `rag_*`, `p5_*_clean.parquet` | Follow-up experiments (probability output, retrieval augmentation, relative deviation), `gpt-4.1-mini` and Claude |
| `extended_predictions.csv` | English single-field prompts and Hindi 1,400-row sample |
| `saia_out_*.jsonl` | Llama and Mistral outputs on the 900-row subset |
| `*_index.jsonl`, `*_ids.json`, `extended_batch_index.csv` | Which rows and personas each request used |

The prediction files contain IDs, labels, and annotator demographics, but no post or reply text. Texts are re-attached from the downloaded corpus through `post_id`.

## Scripts by thesis section

Shared modules: `utils.py` (mixed models and Nakagawa--Schielzeth $R^2$), `fastlmm.py` (maximum-likelihood crossed model, text-intercept GLS, and nested designs used by the corrected tests).

### Study 1 (Chapter 3)

| Section | Scripts |
|---|---|
| Data and preprocessing (§3.2) | `01`, `02`, `15` |
| Phase A: $R^2_m$, variable importance, decomposition, interactions (§3.4) | `03`, `04`, `05`, `06` |
| Phase B: undersampling, bootstrap, 50 seeds (§3.5) | `07`--`11`, `12`, `13`, `14` |
| Phase C: commonality, persona ceiling, M1--M5, saturation (§3.6) | `16a`, `16b`, `17`, `18`, `19` |
| Phase D: incremental steps, saturation update, interactions on M4 base (§3.7, Appendix) | `22`, `23`, `24` |
| Annotator-level tests (§3.3.5) | `102`, `105` |
| Shuffled-profile nulls (§3.3.6) | `100`, `104`, `107` |
| `lme4` validation (Appendix) | `108_export_lme4_validation.py`, then `Rscript 108_lme4_validation.R` |

### Study 2 (Chapter 4)

| Section | Scripts |
|---|---|
| Sample selection and prompts (§4.2) | `25`, `26` |
| `gpt-4.1-mini` main run | `27`, `27b`, `28` |
| Magnitude, steerability, calibration (§4.4, Appendix) | `29` |
| Direction (§4.5) | `35`, `99_rank_pruned_concordance.py`, `101`, `103` |
| Shuffled-profile null for LLM and human labels (§4.4) | `106` |
| English and Hindi checks (§4.9.1) | `30`, `31`--`34` |
| `gpt-4.1` (§4.6) | `40`, `44` (chunked submission used for the run), `42`, `43` |
| Llama and Mistral (§4.7) | `50`, `51`, `52` |
| Claude Opus 5 (§4.8) | `90`--`94` |
| Probability elicitation (§4.9.2) | `60`--`65` |
| Retrieval augmentation, P4 (§4.9.2) | `70`--`76` |
| Relative deviation, P5 (§4.9.2) | `80`--`85` |
| Claude replications of P4 and P5 | `95`--`98`, `99_analyze_claude_followups.py` |
| Convergence audits | `audit_all_mini.py`, `verify_mini_fits.py` |

`41_run_gpt41_batch.py` is the original single-batch submission, which exceeded the token limit; `44` replaced it. Script `31` also builds an Italian/Portuguese employment-only prompt that is not reported in the thesis.

## Regenerating the figures

Several figures were updated after the null baselines were added, so the order matters:

1. Run the analysis scripts.
2. Run `fix_figures.py` (recomputes the Spanish P2 cells with a fitter that requires convergence and redraws the affected figures).
3. Run `109_null_figures.py` **last**. Scripts `06`, `23`, `94`, and `fix_figures.py` also write some of the same files and would overwrite the final versions.

| Figure | Final script |
|---|---|
| `fig1`, `fig5`, `fig20`, `fig25`, `fig28`, `fig_3model_comparison`, `fig_directional_concordance` | `109_null_figures.py` |
| `fig_gpt41_concordance_comparison`, `fig_prob_`, `fig_rag_`, `fig_p5_concordance_comparison`, `fig_steerability` | `fix_figures.py` |
| `fig_spearman_scatter` | `29` |
| `fig2`, `fig3`, `fig4` | `06` |
| `fig16` | `14` |
| `fig17` | `16b` |
| `fig18` | `17` |
| `fig21` | `19` |
| `fig29`, `fig30` | `24` |

## Notes

- Random seed 42 is used for all primary analyses; Phase B uses seeds 100--149.
- `108_export_lme4_validation.py` writes its exchange files to `/tmp/lme4val/`.
- The crossed-model routine in `fastlmm.py` agreed with `lme4` to at least five significant digits on the validated cells (`results_lme4_validation_*.csv`).

## Citation

If you use this code, please cite the thesis:

> Talebi, P. (2026). *A Multi-Dimensional Evaluation of Persona Adoption by Large Language Models in Cross-Lingual Irony Detection*. Master's thesis, University of Marburg.

MultiPICo: Casola et al. (2024). Please follow the corpus licence when using the data.
