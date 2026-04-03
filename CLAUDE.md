# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InsilicoCell is a deep learning framework for predicting cellular responses to drugs and genetic perturbations. It uses pre-trained PyTorch JIT models to predict gene expression changes, drug responses, binding affinities, and genetic effects.

## Environment Setup

```bash
conda env create -f InsilicoCell/InsilicoCell_env.yaml
conda activate InsilicoCell
```

Key dependencies: Python 3.9, PyTorch 2.5.1, RDKit, Hugging Face Transformers (T5 protein encoder), pandas, numpy.

Model checkpoints must be downloaded separately from S3 — see `InsilicoCell/model_checkpoint/checkpoint.md` for links.

## Running Prediction Tasks

All 7 tasks follow the same CLI pattern:

```bash
cd InsilicoCell
python demo_<task>.py --File1 <sample_info.csv> --File2 <transcriptome.csv> --out <output_name> --BatchSize <int> --device <GPU|CPU>
```

Tasks: `drug-induced_gene_expression`, `drug-protein_binding_affinity`, `TF-gene_association`, `drug_sensitivity`, `gene_effect_score`, `gene_mutation`, `CNV`.

The drug-protein binding and TF-gene tasks do not require `--File2` (transcriptome data).

## Architecture

Each demo script is self-contained and follows this pipeline:
1. Parse CLI args, load input CSVs (sample metadata + optional transcriptome)
2. Generate embeddings — Morgan fingerprints for drugs (RDKit), T5 encoder for proteins, pre-computed CSV for genes, JIT model for cell lines
3. Validate inputs (SMILES validity, gene name existence in `gene_list.csv`)
4. Run batch prediction through a task-specific PyTorch JIT model
5. Write predictions to `InsilicoCell/prediction/`

There is no shared library — embedding generation, data preprocessing (duplicate gene averaging, missing gene imputation), and batch prediction logic are duplicated across scripts.

## Data Format

- **File1** (sample metadata): CSV with task-dependent columns (`SMILES`, `cell_iname`, `gene_name`, `target_sequence`, `time_h`, `dose_uM`)
- **File2** (transcriptome): log2(TPM+1) gene expression matrix, rows=cell lines, columns=genes (~19,144 genes)
- **Output**: Input CSV with appended `prediction` column

Demo data is in `InsilicoCell/demo_data/`.

## No Tests or CI

There is no test suite, no CI/CD, no Makefile, and no package distribution setup (no setup.py/pyproject.toml).
