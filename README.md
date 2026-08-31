# InsilicoCell

InsilicoCell is a virtual platform for cellular functional profile prediction and multi-objective drug screening. Watch the introduction below:


https://github.com/user-attachments/assets/0b18bfd1-b983-4bdd-8c18-46fcdc0c1aa7



You can also read our preprint here: [A pretrained unified model enables cellular functional profile prediction and multi-objective virtual drug screening](https://doi.org/10.64898/2026.08.25.746866).

InsilicoCell can be used in two ways:

1. **Claude/Codex (ChatGPT) desktop interface (recommended for non-programmers):** ask biological questions to InsilicoCell in ordinary language through the chatting interface, which connects to our background remote server to run InsilicoCell. A list of current APIs can be viewed [here](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/API.md).

2. **Terminal interface:** install and run the original Python scripts on your own laptop, workstation, or server. This is especially recommended if you need to do large-scale inference, as there can be fewer limitations on runtime and data size on your own device.

Internally, InsilicoCell is built on [seven pretrained tasks](https://github.com/Bin-Chen-Lab/insilicoCell#21-supported-prediction-tasks).

# 1. Use InsilicoCell from Claude/Codex (ChatGPT) desktop

## 1.1. Connect once

The hosted Streamable HTTP MCP endpoint is:

```text
https://apps.octad.org/insilicocell/mcp
```

### Claude desktop:
open **Settings → Connectors → Add → Add custom connector**, name it `InsilicoCell`, paste the URL above, click on **continue**, and restart the app. 

![Claude interface](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/figures/Screenshot%202026-08-19%20at%204.45.25%20PM.png)

Make sure that InsilicoCell is connected in the Claude interface by clicking on the **+ → Connectors** and see if InsilicoCell is enabled there:

![Claude interface](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/figures/Screenshot%202026-08-20%20at%2011.38.45%20AM.png)

### Codex (ChatGPT) desktop:
open **Settings → Plugins → Add → Add MCP server**, name it `InsilicoCell`, choose **Streamable HTTP**, paste the URL above, click on **save**, and restart the app. 

![chatGPT interface](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/figures/Screenshot%202026-08-19%20at%204.41.43%20PM.png)

Make sure that InsilicoCell is connected in the **Codex** interface by typing **"/mcp"** in the chatbot, click on **MCP** and see if InsilicoCell is enabled there:

![chatGPT interface](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/figures/Screenshot%202026-08-20%20at%2011.35.10%20AM.png)

## 1.2. Ask questions naturally

Example questions and requests:

- “Using InsilicoCell, predict the drug sensitivity of erlotinib in HepG2.” 

- “Can you use InsilicoCell to infer whether or not this compound, O=C(NOCC(O)CO)c1cnc2ncnn2c1Nc1ccc(I)cc1F, affects MYC expression in liver cancer cells?”

- “Use InsilicoCell to identify top compound candidates that kill lung cancer cells. You can screen on the built-in 5 million compounds and take a commonly used lung cancer cell line as an example.”

- “Predict compounds which strongly bind to PD-L1 protein via InsilicoCell. You can screen on the built-in compound library and search for PD-L1 sequence if needed.”

- "Use InsilicoCell to rank the importance of the MYC gene in various types of cancer. In which type of cancer is it the most important?"

<Br> 

Example answers from the Claude interface:

<p align="center">
  <img src="https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/figures/Screenshot%202026-08-19%20at%208.47.15%20PM.png" width="40%" />
  <span>&nbsp;&nbsp;</span>
  <img src="https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/figures/Screenshot%202026-08-19%20at%209.01.10%20PM.png" width="55%" />
</p>

<Br> 

You can chat with InsilicoCell for multiple rounds in the same conversation. Depending on your question, the interface may need to discuss with you for more details for a better answer. For example, if you ask questions about perturbing a cellular context that is not in the built-in cell library of InsilicoCell, the assistant will first ask you to either upload the originally untreated cell transcriptome data ([File 2](https://github.com/Bin-Chen-Lab/insilicoCell#file-2-cell-transcriptome-required-for-5-of-7-tasks)), or it will search on web for the originally untreated cell transcriptome data. After that, it will predict the perturbed state of that cellular context. An example conversation with InsilicoCell about uploading files of your own interested cell type which is not originally in the built-in cell library can be viewed [here](https://claude.ai/share/63905f82-4bf1-406a-9a1d-92bc7dc26462). We also showed an example in the introduction video with user file uploading, you can reproduce the conversation in the video using the demo file of untreated cell transcriptome data [here](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/demo_data/M229_transposed.csv).

For drug screening questions, the assistant will offer two choices:

1. Use InsilicoCell's built-in compound library, which contains ~4.7 million Enamine compounds. 
2. Upload your own CSV containing unique `compound_id` and `SMILES` columns (`compound_name` is optional).

Large screens run in the background. The actual job running time depends on the real-time workload on the remote server. We recommend that you run a small-scale screening job to test running speed before trying a large-scale screening job with millions of compounds. You may close the conversation and later ask for job status using the returned job ID. You should actively ask the assistant to check on the job status, when a job completes, the assistant will provide you with a result-file download link. **You have 8 hours for downloading after job completion, before the result file expires.**

An example involving a longer conversation with InsilicoCell on the topic of PD-L1 inhibitor screening can be viewed [here](https://claude.ai/share/aaea143e-17dc-4e3d-9f10-1be214f6b782). 

## 1.3. Hosted-service limits and data retention

| Item | Hosted-service policy |
|---|---|
| Maximum run time | 10 hours per background screening job. For jobs running beyond 10 hours, we recommend you to use the [terminal interface](https://github.com/Bin-Chen-Lab/insilicoCell#2-use-insilicocell-from-terminal-interface)|
| Inline prediction request | Up to 5,000 rows; larger libraries use the background workflow |
| User upload file size | Up to 1 GB per uncompressed file |
| Accepted user compound library | CSV with `compound_id` and `SMILES` |
| Completed result retention | 8 hours after the model run completes, before the result file expires for downloading |

# 2. Use InsilicoCell from terminal interface

## 2.1. Supported Prediction Tasks 

| Task | Input | Output | Interpretation |
|------|-------|--------|----------------|
| **Drug-induced gene expression change** | SMILES, cell line, gene, time, dose | z-score (−10~10) | More positive = more up-regulation, more negative = more down-regulation |
| **Drug-protein binding affinity** | SMILES, protein sequence | log₁₀(IC50) in nM | Lower = stronger binding |
| **TF-gene association** | Gene name, TF protein sequence | Probability (0~1) | >0.5 represents association, <=0.5 represents no association |
| **Drug sensitivity** | SMILES, cell line | AUC (0~30) | Lower = stronger cell growth inhibition |
| **Gene effect score** | Gene name, cell line | Dependency score | More negative = stronger gene dependency |
| **Gene mutation status** | Gene name, cell line | Probability (0~1) | >0.5 represents mutation, <=0.5 represents no mutation |
| **Copy number variation (CNV)** | Gene name, cell line | log₂(copy number + 1) | Gene-level copy number |

Tasks that require cell transcriptome data ([File 2](https://github.com/Bin-Chen-Lab/insilicoCell#file-2-cell-transcriptome-required-for-5-of-7-tasks)): drug-induced gene expression change, drug sensitivity, gene effect score, gene mutation status, and CNV. The drug-protein binding and TF-gene association tasks do **not** require transcriptome input.

## 2.2. Installation

### 2.2.1. Clone the repository

```bash
git clone https://github.com/Bin-Chen-Lab/insilicoCell
cd insilicoCell/InsilicoCell
```

### 2.2.2. Create the conda environment

```bash
conda env create -f InsilicoCell_env.yaml
conda activate InsilicoCell
```

**Requirements:** Python 3.9, PyTorch 2.5.1, RDKit, Hugging Face Transformers, pandas, numpy.

### 2.2.3. Download pretrained model 

Download [the following folder](https://chenlab-data-public.s3.amazonaws.com/InsilicoCell/model_checkpoint/model_checkpoint.zip), unzip it, which is named "model_checkpoint", and place it under the directory `insilicoCell/InsilicoCell/`.

## 2.3. Preparing Input Data

### File 1: Sample metadata (required for all tasks)

A CSV file where each row is one prediction sample. Columns vary by task:

| Column | Description | Used by |
|--------|-------------|---------|
| `SMILES` | Drug structure in SMILES notation | Drug-induced expression, drug-protein binding, drug sensitivity |
| `cell_iname` | Cell line name (must match row names in File 2) | Drug-induced expression, drug sensitivity, gene effect, mutation, CNV |
| `gene_name` | Gene name from the [supported gene list](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/demo_data/gene_list.csv) (~20k genes) | Drug-induced expression, TF-gene, gene effect, mutation, CNV |
| `target_sequence` | Protein amino acid sequence* | Drug-protein binding, TF-gene |
| `time_h` | Treatment duration in hours | Drug-induced expression |
| `dose_uM` | Treatment concentration in μM | Drug-induced expression |

See example files in [`demo_data/`](InsilicoCell/demo_data/):
[drug-induced expression](InsilicoCell/demo_data/input_sample_info.csv) ·
[drug-protein binding](InsilicoCell/demo_data/input_sample_info_drug-protein_binding.csv) ·
[TF-gene](InsilicoCell/demo_data/input_sample_info_TF-gene_association.csv) ·
[drug sensitivity](InsilicoCell/demo_data/input_sample_info_drug_response.csv) ·
[gene effect](InsilicoCell/demo_data/input_sample_info_gene_effect_score.csv) ·
[mutation](InsilicoCell/demo_data/input_sample_info_gene_mutation.csv) ·
[CNV](InsilicoCell/demo_data/input_sample_info_CNV.csv)

*For protein amino acid sequence sequences, if your protein is among the [Uniprot list](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/demo_data/uniprot_human_proteome_sp.csv), we recommend you to directly copy its sequence from here, which will largely reduce model running time. You can also provide other protein sequences beyond the list, but will cost longer model running time.

### File 2: Cell transcriptome (required for 5 of 7 tasks)

A CSV file of untreated transcriptomic profiles, formatted as log₂(TPM + 1):
- **Rows:** cell line names (matching the `cell_iname` column in File 1)
- **Columns:** gene names

This file is used for cell representation. InsilicoCell can handle both seen and unseen cells. If your cell is in the [pretraining dataset](https://chenlab-data-public.s3.amazonaws.com/InsilicoCell/cell_line_expression_matrix.csv), you can extract its expression profile directly from there. If your cell is unseen, you should obtain its transcriptomic profile by yourself.

Duplicate gene names are automatically averaged. Missing genes are imputed with the row mean.

## 2.4. Running Predictions

All commands are run from the `InsilicoCell/` directory:

```bash
cd InsilicoCell
```
### Parameters

| Parameter | Description |
|-----------|-------------|
| `--File1` | Path to sample metadata CSV |
| `--File2` | Path to cell transcriptome CSV (not needed for binding and TF-gene tasks) |
| `--out` | Output filename (no extension) — saved to `./prediction/` |
| `--BatchSize` | Samples per batch. Larger = faster but uses more memory. Does not affect results. |
| `--device` | `GPU` or `CPU` |

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

### Drug sensitivity

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

## 2.5. Output

Predictions are saved as CSV files in `InsilicoCell/prediction/`. Each output file contains all columns from your input File 1 with an appended `prediction` column.

# 3. Benchmarking evaluation

We have released the [benchmarking evaluation datasets](https://huggingface.co/datasets/binchenlab/InsilicoCell) for zero-shot prediction of the model. Check the link for data file descriptions. We will release the development version, with training set and model fine-tuning code that reproduce the results in the paper after paper acceptance.

For downloading, prediction and evaluation procedures on the benchmarking datasets, refer to [this tutorial](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/benchmarking.md).


# Cumulative usage statistics

<!-- USAGE_METRICS_START -->
Total unique InsilicoCell installations: 19  
Total completed predictions: 69  
Total GitHub release downloads: 0  
Tracking began: 2026-08-19
<!-- USAGE_METRICS_END -->

# Cite InsilicoCell

Please cite our preprint:<Br>
Ruoqiao Chen, Li Huang, Yu Qiao, Saurabh Mandal, Linqing Mo, LingXiao Li, Dmitry Leshchiner, Xiaodan Zhang, Jing Pu, Yuying Xie, Reda Girgis, Edmund Ellsworth, Ling Huang, Xin Chen, Xiaopeng Li, Jiayu Zhou, Bin Chen
bioRxiv 2026.08.25.746866; doi: https://doi.org/10.64898/2026.08.25.746866







