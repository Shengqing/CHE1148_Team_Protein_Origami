# Protein Origami: Energetics-Informed Modeling for Protein Stability

This repository contains the code used for the CHE1148 Protein Origami project on learning protein stability landscapes from sequence, with an emphasis on **experimentally measured folding free energy** ($\Delta G$). The workflow combines data curation, exploratory analysis, discriminative baselines, ESM2 embedding pipelines, graph-based models, and generative evaluation utilities.

The accompanying project report is available as `Interim_Final_ProteinOrigami.pdf` in the repo root (for reading context only; no need to include it in code commits).

## 1) Project Summary

### Problem
Protein structure predictors can produce plausible structures in silico while still failing thermodynamic stability in vitro. This project focuses on learning a direct mapping from sequence to measured stability and using that signal to guide mutation proposals.

### Dataset
- Source: `Tsuboyama2023_DS2and3_20230416_ColFiltered.csv` (raw)
- Scale: ~776k experimental records in the raw table
- Curated scope used here:
	- replicate collapsing and WT mapping by cluster
	- single-substitution filtering for initial modeling
	- cluster-aware splitting with deliberate OOD clusters (`71`, `213`) forced into validation

### Core modeling tracks in this repo
- **Baseline MLP** on tokenized amino-acid sequences (`src/run_baseline_mlp.py`)
- **GraphNet + generative evaluation** (`src/train_graphnet_generative.py`)
- **ESM2-650M pooled embeddings + MLP regressor** (`src/train_esm2_regressor.py`)
- **ESM2-650M residue embeddings + linear/attention graph models** (`src/train_esm_graph_models.py`)
- **VAE prototype (notebook)**: `notebooks/VAE.ipynb`

## 2) Repository Layout

```text
configs/                     # YAML configs (baseline MLP)
data/
	raw/                       # Raw Tsuboyama CSV
	processed/                 # Curated train/val CSV + embedding .pt payloads
notebooks/
	VAE.ipynb                  # VAE prototype and latent optimization experiments
results/                     # Metrics, checkpoints, figures, logs
scripts/                     # Batch job launcher scripts
src/
	process_tsuboyama_phase1.py
	eda.py
	data.py
	model.py
	train.py
	run_baseline_mlp.py
	train_graphnet_generative.py
	train_esm2_regressor.py
	train_esm_graph_models.py
```

## 3) Data Processing Pipeline

Primary curation logic lives in `src/process_tsuboyama_phase1.py`.

### What the processor does
1. Builds WT candidates per cluster and maps each variant to a parent WT.
2. Collapses replicate measurements using inverse-variance weighting from reported 95% CI.
3. Keeps resolved **single substitutions** for phase-1 modeling.
4. Splits by `WT_cluster` with cluster-aware stratification.
5. Forces clusters `71` and `213` into validation for OOD-like testing.
6. Produces:
	 - `data/processed/tsuboyama_processed_train_full.csv`
	 - `data/processed/tsuboyama_processed_val_full.csv`
	 - `data/processed/tsuboyama_processed_train_sampled.csv` (60k stratified sample)

### Run
```bash
python src/process_tsuboyama_phase1.py
```

## 4) Exploratory Data Analysis (EDA)

EDA utilities are implemented in `src/eda.py` and called by the baseline pipeline.

Generated diagnostics include:
- train/val $\Delta G$ histograms
- sequence-length histograms
- CI-width histograms (if CI columns are present)
- top WT-cluster frequency plots
- split sanity report (cluster overlap counts)

## 5) Models and Training Scripts

### A. Baseline sequence MLP
Files: `src/data.py`, `src/model.py`, `src/train.py`, `src/run_baseline_mlp.py`

Pipeline:
- amino-acid tokenization and padding/truncation
- learned residue embedding
- masked mean pooling
- MLP regression head

Run:
```bash
python -m src.run_baseline_mlp --config configs/baseline_mlp.yaml
```

### B. GraphNet + generative retrieval evaluation
File: `src/train_graphnet_generative.py`

Model:
- 1D residue graph (adjacent-position edges)
- learned amino-acid + positional embeddings
- stacked message-passing GraphNet layers
- masked mean pooling and scalar $\Delta G$ prediction

Includes:
- regression evaluation
- mutation proposal/generative retrieval diagnostics against observed validation variants
- top-k match analysis and summary plots

Run:
```bash
python src/train_graphnet_generative.py \
	--train_csv data/processed/tsuboyama_processed_train_sampled.csv \
	--val_csv data/processed/tsuboyama_processed_val_full.csv \
	--out_dir results/graphnet
```

### C. ESM2-650M pooled embeddings + MLP regressor
File: `src/train_esm2_regressor.py`

Highlights:
- uses `facebook/esm2_t33_650M_UR50D`
- optional embedding precomputation to `.pt`
- train/dev split by cluster inside train set
- final evaluation on full validation set
- optional generative/top-k analysis

Run (sampled train):
```bash
python src/train_esm2_regressor.py \
	--mode train_eval \
	--train_csv data/processed/tsuboyama_processed_train_sampled.csv \
	--val_csv data/processed/tsuboyama_processed_val_full.csv \
	--out_dir results/esm2_650m \
	--prepare_embeddings
```

Run (full train):
```bash
python src/train_esm2_regressor.py \
	--mode train_eval \
	--train_csv data/processed/tsuboyama_processed_train_full.csv \
	--val_csv data/processed/tsuboyama_processed_val_full.csv \
	--out_dir results/esm2_650m_train_full \
	--train_embed_path data/processed/train_full_esm2_650m.pt \
	--val_embed_path data/processed/val_full_esm2_650m.pt \
	--prepare_embeddings
```

### D. ESM residue embeddings + graph heads (linear and attention)
File: `src/train_esm_graph_models.py`

Variants trained in one run:
- `esm_linear_graphnet`: linear message passing on adjacent residues
- `esm_attention_graph`: transformer-style attention blocks over residue embeddings

Run:
```bash
python src/train_esm_graph_models.py \
	--train_csv data/processed/tsuboyama_processed_train_sampled.csv \
	--val_csv data/processed/tsuboyama_processed_val_full.csv \
	--out_dir results/esm_graph_60k \
	--train_embed_path data/processed/train_sampled_esm2_650m_seq.pt \
	--val_embed_path data/processed/val_full_esm2_650m_seq.pt \
	--prepare_embeddings
```

### E. VAE prototype
Prototype code is in `notebooks/VAE.ipynb`.

Current notebook implementation:
- consumes ESM embedding `.pt` payloads
- multi-task VAE with reconstruction + KL + property regression
- latent optimization routine for stability-guided search

Note: this prototype is notebook-first and not yet packaged as a production CLI under `src/`.

## 6) Representative Validation Results

From the current result artifacts:

- **GraphNet baseline** (`results/graphnet/metrics.json`):
	- MAE 1.0419, RMSE 1.5477, $R^2$ 0.3704, Pearson 0.6799, Spearman 0.6685
- **ESM2 pooled + MLP (60k train)** (`results/esm2_650m/metrics.json`):
	- MAE 0.9055, RMSE 1.3087, $R^2$ 0.5499, Pearson 0.7804, Spearman 0.7581
- **ESM2 pooled + MLP (full train)** (`results/esm2_650m_train_full/metrics.json`):
	- MAE 0.7596, RMSE 1.1113, $R^2$ 0.6754, Pearson 0.8375, Spearman 0.8050
- **ESM linear GraphNet (60k)** (`results/esm_graph_60k/metrics.json`, val metrics):
	- MAE 0.8729, RMSE 1.3002, $R^2$ 0.5557, Pearson 0.7860, Spearman 0.7367

These trends support the report-level conclusion that ESM-based representations significantly improve predictive stability modeling over sequence-only baselines.

## 7) Reproducibility Notes

- Most training scripts fix random seed (`42`) by default.
- Train/dev split is cluster-stratified; validation is pre-generated in processed CSV files.
- Metrics are logged as JSON under each experiment folder in `results/`.
- Lint/format is configured through Ruff in `pyproject.toml`.

## 8) Limitations and Active Directions

Aligned with the final report and current code status:
- Current primary training scope is single-point mutation curation.
- OOD behavior is partially probed via held-out clusters; broader generalization remains open.
- ESM-attention graph variant currently underperforms the linear graph variant in the present setup.
- VAE work is currently notebook-based and not yet fully integrated into the `src/` training stack.
- Generative evaluation is retrieval-oriented; sequence-native generation remains an active target.

## 9) Citation

If you use this repository, please cite the project report and the Tsuboyama et al. dataset paper.

---

For manuscript context and experimental framing, see `Interim_Final_ProteinOrigami.pdf`.
