# Protein Origami: Energetics-Informed Generative Modeling for Stable Protein Mutants

This repository contains the code and experiments for the CHE1148 Team Protein Origami project.
The central objective is to model protein folding stability from sequence and use that signal in a
generative workflow for proposing stabilizing mutants.


## Project Scope

Protein structure predictors can be highly confident while still missing thermodynamic stability under
experimental conditions. This project focuses on experimentally measured folding free energy ($\Delta G$)
and frames stability design as both a prediction and generation problem.

The modeling strategy in the draft centers on a semi-supervised VAE (ss-VAE), with two baselines:
- Predictive baseline: MLP on ESM2-650M embeddings
- Generative baseline: random single-residue mutation followed by feasibility mapping to observed data

## Data and Curation

Source dataset:
- Tsuboyama et al. mega-scale stability dataset (raw scale: about 776k measurements)

Curated pipeline (implemented in src/process_tsuboyama_phase1.py):
- WT reconstruction and cluster-aware organization
- replicate handling with confidence interval fields
- single-substitution filtering for phase-1 modeling
- cluster-aware train/validation construction
- forced OOD validation clusters: 71 and 213

Primary processed artifacts:
- data/processed/tsuboyama_processed_train_full.csv
- data/processed/tsuboyama_processed_train_sampled.csv
- data/processed/tsuboyama_processed_val_full.csv

Representation used for predictive/generative modeling:
- ESM2-650M pooled embeddings (1280-dimensional per sequence)

## Repository Map

```text
configs/                      baseline configuration
data/
  processed/                  curated CSV and embedding payloads
  interim/                    intermediate experiment files and generated datasets
notebooks/
  VAE.ipynb                   ss-VAE prototype, ablations, and analysis plots
results/                      metrics, logs, generated figures, evaluation outputs
scripts/                      SLURM launchers and utility render scripts
src/
  process_tsuboyama_phase1.py
  run_baseline_mlp.py
  train_graphnet_generative.py
  train_esm2_regressor.py
  train_esm_graph_models.py
  generate_random_single_mutant_mapping.py
  embed_esm2_from_pt.py
```

## Methods Summary

### 1) Predictive baseline (MLP)

Baseline MLP consumes 1280-D ESM2 embeddings and predicts $\Delta G$.
The draft reports strong ranking performance and stable seed-to-seed behavior.

Entry point:
- src/run_baseline_mlp.py

### 2) Generative baseline (random single mutation)

For each WT sequence, one amino-acid position is sampled uniformly, and one alternative residue is
sampled from the remaining 19 amino acids. The generated variant is mapped back to curated full data as
a feasibility proxy.

Entry point:
- src/generate_random_single_mutant_mapping.py

### 3) ss-VAE prototype

The VAE prototype in notebooks/VAE.ipynb uses ESM2 embeddings and a multi-objective loss:
- reconstruction term
- KL regularization term
- property prediction term for stability

It also supports latent-space optimization for stability-guided candidate search.

Notebook-embedded implementation details:
- Input representation:
  - 1280-D ESM2 pooled embeddings from .pt payloads
  - paired amino-acid sequences for sequence-side processing
- Default model/training constants in the notebook:
  - INPUT_DIM=1280, LATENT_DIM=256, HIDDEN_DIM=512
  - NUM_EPOCHS=100, BATCH_SIZE=128, PATIENCE=7
  - LEARNING_RATE=8e-5, DROPOUT_RATE=0.2, WEIGHT_DECAY=2e-6
  - OPTIM_STEPS=25, OPTIM_LR=0.05 for latent-space optimization
- VAE architecture in notebook:
  - Encoder: Linear -> BatchNorm -> ReLU -> Dropout -> Linear(2*latent)
  - Decoder: latent -> hidden/2 -> hidden -> reconstructed 1280-D embedding
  - Sequence head: projection to per-position logits (21-token output space)
  - Regressor head: latent -> scalar stability prediction
- Current training objective used in train_and_validate:
  - reconstruction MSE + 0.1 * KL + prediction MSE
  - validation monitoring uses prediction MSE with early stopping
  - epoch logging includes RMSE, R2, and Kendall Tau on inverse-scaled predictions
- Additional notebook experiment block:
  - latent dimension sweep over [64, 128, 256, 512]
  - 5 random initiations per latent dimension
  - reports mean and standard deviation for Kendall Tau and RMSE

Implementation note:
- The notebook defines RL_WEIGHT/KL_WEIGHT/PL_WEIGHT/SEQ_WEIGHT constants for configurable
  multi-term objectives, but the currently active train_and_validate loop uses a fixed weighted
  objective (recon + 0.1*KL + pred) and does not currently include sequence cross-entropy in loss.

## Reported Results (From Draft)

### Baseline MLP (validation, 5 seeds)
- RMSE: 1.355 ± 0.032
- R2: 0.517 ± 0.023
- Kendall Tau: 0.5483 ± 0.0166
- Spearman: 0.719 ± 0.021

### VAE regressor head (validation)
- MAE: 0.8007
- RMSE: 1.1447
- R2: 0.6556
- Spearman: 0.7874

### Generative baseline
- Random-mutation feasibility mapping remains consistently above zero across repeated runs,
  providing a baseline for comparing learned generative methods.

## Reproducing Core Workflows

### Data processing

```bash
python src/process_tsuboyama_phase1.py
```

### Baseline MLP

```bash
python -m src.run_baseline_mlp --config configs/baseline_mlp.yaml
```

### ESM2 embedding + regressor

```bash
python src/train_esm2_regressor.py \
  --mode train_eval \
  --train_csv data/processed/tsuboyama_processed_train_sampled.csv \
  --val_csv data/processed/tsuboyama_processed_val_full.csv \
  --out_dir results/esm2_650m \
  --prepare_embeddings
```

### Generative feasibility mapping (random single mutants)

```bash
python src/generate_random_single_mutant_mapping.py \
  --wt_source_csv data/processed/tsuboyama_processed_val_full.csv \
  --lookup_csv data/processed/tsuboyama_processed_train_full.csv \
  --output_csv results/generative_eval/random_single_mutants_val_wt_mapped_to_train_full.csv \
  --summary_json results/generative_eval/random_single_mutants_val_wt_mapped_to_train_full.summary.json \
  --seed 42
```

### Optional bootstrap over multiple seeds

```bash
for s in $(seq 0 9); do
  python src/generate_random_single_mutant_mapping.py \
    --seed "$s" \
    --output_csv "results/generative_eval/bootstrap_runs/random_single_mutants_seed_${s}.csv" \
    --summary_json "results/generative_eval/bootstrap_runs/random_single_mutants_seed_${s}.summary.json"
done
```

## Current Limitations

- Current curated modeling scope is single substitutions.
- The present VAE implementation is embedding-space and not yet a full sequence decoder.
- Generative evaluation currently uses feasibility mapping against observed variants, not full wet-lab
  validation.


