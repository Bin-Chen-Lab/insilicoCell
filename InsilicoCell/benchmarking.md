### Downloading benchmaarking datasets:

All commands are run from the `InsilicoCell/` directory:

```bash
cd InsilicoCell
```
Use shell commands to conveniently download all data files: 

Downloading all data files:
```bash
huggingface-cli download binchenlab/InsilicoCell --repo-type=dataset --local-dir ./demo_data
```

The benchmarking datasets use [cell_line_expression_matrix.csv](https://chenlab-data-public.s3.amazonaws.com/InsilicoCell/cell_line_expression_matrix.csv) as [File2](https://github.com/Bin-Chen-Lab/insilicoCell#file-2-cell-transcriptome-required-for-5-of-7-tasks). You need to download it and save to `./demo_data/`.

An example for benchmarking evaluation on the drug sensitivity task with sample-level holdout:
```bash
python demo_drug_sensitivity.py \
  --File1 ./your_local_dir_name/drug_sensitivity_sample-level_holdout_test_set.csv \
  --File2 ./your_local_dir_name/cell_line_expression_matrix.csv \
  --out pred_drug_sensitivity_sample-level_holdout_test_set --BatchSize 512 --device GPU
```

