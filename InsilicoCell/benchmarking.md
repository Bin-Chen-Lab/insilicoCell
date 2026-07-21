## Benchmarking evaluation

We have released the [benchmarking evaluation datasets](https://huggingface.co/datasets/binchenlab/InsilicoCell) for zero-shot prediction of the model. Check the link and read data file descriptions. We will further release the training set and model fine-tuning code after the acceptance of the paper.

### Downloading benchmaarking datasets:

All commands are run from the `InsilicoCell/` directory:

```bash
cd InsilicoCell
```
Use shell commands to conveniently download all data files or specified data files: 

Downloading all data files:
```bash
huggingface-cli download binchenlab/InsilicoCell --repo-type=dataset --local-dir ./demo_data
```

Downloading specified data files, e.g., "drug_sensitivity_sample-level_holdout_test_set.csv":
```bash
huggingface-cli download binchenlab/InsilicoCell drug_sensitivity_sample-level_holdout_test_set.csv --repo-type=dataset --local-dir ./demo_data
```

The benchmarking datasets use [cell_line_expression_matrix.csv](https://chenlab-data-public.s3.amazonaws.com/InsilicoCell/cell_line_expression_matrix.csv) as [File2](https://github.com/Bin-Chen-Lab/insilicoCell#file-2-cell-transcriptome-required-for-5-of-7-tasks). You need to download it and save to `./demo_data/`.

### Running Predictions:
This is the same as [previously described](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/README.md#running-predictions). According to your device capacity, you can revise on the `--BatchSize` and `--device` parameters, but keep the other parameters unchanged.

```bash
python demo_drug-induced_gene_expression.py \
  --File1 ./demo_data/drug-induced_gene_expression_change_sample-level_holdout_test_set.csv \
  --File2 ./demo_data/cell_line_expression_matrix.csv \
  --out pred_drug-induced_gene_expression_change_sample-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_drug-induced_gene_expression.py \
  --File1 ./demo_data/drug-induced_gene_expression_change_entity-level_holdout_test_set.csv \
  --File2 ./demo_data/cell_line_expression_matrix.csv \
  --out pred_drug-induced_gene_expression_change_entity-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_drug-protein_binding_affinity.py \
  --File1 ./demo_data/drug-protein_binding_sample-level_holdout_test_set.csv \
  --out pred_drug-protein_binding_sample-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_drug-protein_binding_affinity.py \
  --File1 ./demo_data/drug-protein_binding_entity-level_holdout_test_set.csv \
  --out pred_drug-protein_binding_entity-level_holdout_test_set --BatchSize 512 --device GPU

```bash
python demo_TF-gene_association.py \
  --File1 ./demo_data/TF-gene_association_sample-level_holdout_test_set.csv \
  --out pred_TF-gene_association_sample-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_TF-gene_association.py \
  --File1 ./demo_data/TF-gene_association_entity-level_holdout_test_set.csv \
  --out pred_TF-gene_association_entity-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_drug_sensitivity.py \
  --File1 ./demo_data/drug_sensitivity_sample-level_holdout_test_set.csv \
  --File2 ./demo_data/cell_line_expression_matrix.csv \
  --out pred_drug_sensitivity_sample-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_drug_sensitivity.py \
  --File1 ./demo_data/drug_sensitivity_entity-level_holdout_test_set.csv \
  --File2 ./demo_data/cell_line_expression_matrix.csv \
  --out pred_drug_sensitivity_entity-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_gene_effect_score.py \
  --File1 ./demo_data/gene_effect_score_sample-level_holdout_test_set.csv \
  --File2 ./demo_data/cell_line_expression_matrix.csv \
  --out pred_gene_effect_score_sample-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_gene_effect_score.py \
  --File1 ./demo_data/gene_effect_score_entity-level_holdout_test_set.csv \
  --File2 ./demo_data/cell_line_expression_matrix.csv \
  --out pred_gene_effect_score_entity-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_gene_mutation.py \
  --File1 ./demo_data/gene_mutation_sample-level_holdout_test_set.csv \
  --File2 ./demo_data/cell_line_expression_matrix.csv \
  --out pred_gene_mutation_sample-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_gene_mutation.py \
  --File1 ./demo_data/gene_mutation_entity-level_holdout_test_set.csv \
  --File2 ./demo_data/cell_line_expression_matrix.csv \
  --out pred_gene_mutation_entity-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_CNV.py \
  --File1 ./demo_data/CNV_sample-level_holdout_test_set.csv \
  --File2 ./demo_data/cell_line_expression_matrix.csv \
  --out pred_CNV_sample-level_holdout_test_set --BatchSize 512 --device GPU
```

```bash
python demo_CNV.py \
  --File1 ./demo_data/CNV_entity-level_holdout_test_set.csv \
  --File2 ./demo_data/cell_line_expression_matrix.csv \
  --out pred_CNV_entity-level_holdout_test_set --BatchSize 512 --device GPU
```


