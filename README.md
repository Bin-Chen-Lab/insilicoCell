# InsilicoCell

# Dependencies
pandas <br />
numpy <br />
torch <br />
rdkit <br />
argparse <br />

# Step 1: Download model and checkpoints
View [this file](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/model_checkpoint/checkpoint.md) for downloading checkpoints.

# Step 2: Prepare your data
To run InsilicoCell for prediction, you need to prepare two files:
File 1: A meta info file of your input samples. The format of this file can be viewd [here](https://github.com/Bin-Chen-Lab/insilicoCell/tree/main/InsilicoCell/demo_data/input_sample_info.csv). <br />
File 2: log2TPM data of transcriptomic profiles in cells. Row names are cell names (should match the cell names in the "cell_iname" column in your File 1), column names must use the same gene names in orders as shown in the [demo data](https://github.com/Bin-Chen-Lab/insilicoCell/tree/main/InsilicoCell/demo_data/cell_transcriptomes_log2TPM.csv). If there are missing genes' expression in your data, you may use the average expression of the other genes in the cell to impute.


# Step 3: Run InsilicoCell for prediction
Predicting drug-induced gene expression change: <br />
Under the "./InsilicoCell" directory, run the following shell command for prediction: <br />
```
python demo_drug-induced_gene_expression.py --File1 ./demo_data/input_sample_info.csv --File2 ./demo_data/cell_transcriptomes_log2TPM.csv --out pred_demo --device GPU
```
File1: The path to your File 1 as in Step 2. <br />
File2: The path to your File 2 as in Step 2. <br />
out: Set a file name for the output prediction file (no extension). <br />
device: GPU or CPU. <br />




