# InsilicoCell


## Supported Prediction Tasks

| Task | Input | Output | Interpretation |
|------|-------|--------|----------------|
| **Drug-induced gene expression** | SMILES, cell line, gene, time, dose | z-score (−10 to 10) | Positive = up-regulation, negative = down-regulation |
| **Drug-protein binding affinity** | SMILES, protein sequence | log₁₀(IC50) in nM | Lower = stronger binding |
| **TF-gene association** | Gene name, TF protein sequence | Binary (0/1) | 1 = association exists |
| **Drug response** | SMILES, cell line | AUC (0–30) | Lower = stronger growth inhibition |
| **Gene effect score** | Gene name, cell line | Dependency score | >0 = reduces viability, <0 = improves growth |
| **Gene mutation status** | Gene name, cell line | Binary (0/1) | 1 = mutation present |
| **Copy number variation (CNV)** | Gene name, cell line | log₂(copy number + 1) | Gene-level copy number |

Tasks that require cell transcriptome data (File 2): drug-induced gene expression, drug response, gene effect score, gene mutation status, and CNV. The drug-protein binding and TF-gene association tasks do **not** require transcriptome input.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Bin-Chen-Lab/insilicoCell
cd insilicoCell/InsilicoCell
```

### 2. Create the conda environment

```bash
conda env create -f InsilicoCell_env.yaml
conda activate InsilicoCell
```

**Requirements:** Python 3.9, PyTorch 2.5.1, RDKit, Hugging Face Transformers, pandas, numpy.

### 3. Download pretrained model 

Download [the following files](https://chenlab-data-public.s3.amazonaws.com/InsilicoCell/model_checkpoint/InsilicoCell_pretrained.zip) and place them under the directory `InsilicoCell/model_checkpoint/`:

## Preparing Input Data

### File 1: Sample metadata (required for all tasks)

A CSV file where each row is one prediction sample. Columns vary by task:

| Column | Description | Used by |
|--------|-------------|---------|
| `SMILES` | Drug structure in SMILES notation | Drug expression, binding, drug response |
| `cell_iname` | Cell line name (must match row names in File 2) | Drug expression, drug response, gene effect, mutation, CNV |
| `gene_name` | Gene name from the [supported gene list](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/model_checkpoint/gene_list.csv) (~20,000 genes) | Drug expression, TF-gene, gene effect, mutation, CNV |
| `target_sequence` | Protein amino acid sequence | Binding, TF-gene |
| `time_h` | Treatment duration in hours | Drug expression |
| `dose_uM` | Treatment concentration in μM | Drug expression |

See example files in [`demo_data/`](InsilicoCell/demo_data/):
[drug expression](InsilicoCell/demo_data/input_sample_info.csv) ·
[binding](InsilicoCell/demo_data/input_sample_info_drug-protein_binding.csv) ·
[TF-gene](InsilicoCell/demo_data/input_sample_info_TF-gene_association.csv) ·
[drug response](InsilicoCell/demo_data/input_sample_info_drug_response.csv) ·
[gene effect](InsilicoCell/demo_data/input_sample_info_gene_effect_score.csv) ·
[mutation](InsilicoCell/demo_data/input_sample_info_gene_mutation.csv) ·
[CNV](InsilicoCell/demo_data/input_sample_info_CNV.csv)

### File 2: Cell transcriptome (required for 5 of 7 tasks)

A CSV file of untreated transcriptomic profiles, formatted as log₂(TPM + 1):
- **Rows:** cell line names (matching the `cell_iname` column in File 1)
- **Columns:** gene names

This file is used for cell representation. InsilicoCell can handle both seen and unseen cells. If your cell is in the [pretraining dataset](https://chenlab-data-public.s3.amazonaws.com/InsilicoCell/cell_line_expression_matrix.csv), you can extract its expression profile directly from there. If your cell is unseen, you should obtain its transcriptomic profile by yourself.

Duplicate gene names are automatically averaged. Missing genes are imputed with the row mean.

## Running Predictions

All commands are run from the `InsilicoCell/` directory:

```bash
cd InsilicoCell
```

### Drug-induced gene expression

```bash
python demo_drug-induced_gene_expression.py \
  --File1 ./demo_data/input_sample_info.csv \
  --File2 ./demo_data/cell_transcriptomes_log2TPM.csv \
  --out pred_drug_expression --BatchSize 512 --device GPU
```

### Drug-protein binding affinity

```bash
python demo_drug-protein_binding_affinity.py \
  --File1 ./demo_data/input_sample_info_drug-protein_binding.csv \
  --out pred_binding --BatchSize 512 --device GPU
```

> Note: The first run downloads the T5 protein encoder model (~3 GB) from Hugging Face, which may take extra time.

### TF-gene association

```bash
python demo_TF-gene_association.py \
  --File1 ./demo_data/input_sample_info_TF-gene_association.csv \
  --out pred_TF_gene --BatchSize 512 --device GPU
```
> Note: The first run downloads the T5 protein encoder model (~3 GB) from Hugging Face, which may take extra time.

### Drug response

```bash
python demo_drug_sensitivity.py \
  --File1 ./demo_data/input_sample_info_drug_response.csv \
  --File2 ./demo_data/task_drug_response_cell_transcriptomes_log2TPM.csv \
  --out pred_drug_response --BatchSize 512 --device GPU
```

### Gene effect score

```bash
python demo_gene_effect_score.py \
  --File1 ./demo_data/input_sample_info_gene_effect_score.csv \
  --File2 ./demo_data/task_gene_effect_score_cell_transcriptomes_log2TPM.csv \
  --out pred_gene_effect --BatchSize 512 --device GPU
```

### Gene mutation status

```bash
python demo_gene_mutation.py \
  --File1 ./demo_data/input_sample_info_gene_mutation.csv \
  --File2 ./demo_data/task_gene_mutation_cell_transcriptomes_log2TPM.csv \
  --out pred_mutation --BatchSize 512 --device GPU
```

### Copy number variation (CNV)

```bash
python demo_CNV.py \
  --File1 ./demo_data/input_sample_info_CNV.csv \
  --File2 ./demo_data/task_CNV_cell_transcriptomes_log2TPM.csv \
  --out pred_CNV --BatchSize 512 --device GPU
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--File1` | Path to sample metadata CSV |
| `--File2` | Path to cell transcriptome CSV (not needed for binding and TF-gene tasks) |
| `--out` | Output filename (no extension) — saved to `./prediction/` |
| `--BatchSize` | Samples per batch. Larger = faster but uses more memory. Does not affect results. |
| `--device` | `GPU` or `CPU` |

## Output

Predictions are saved as CSV files in `InsilicoCell/prediction/`. Each output file contains all columns from your input File 1 with an appended `prediction` column.





