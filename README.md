# InsilicoCell

# Dependencies
Clone the repo to your device, using the following shell command:
```
git clone https://github.com/Bin-Chen-Lab/insilicoCell
```
Under the "./InsilicoCell" directory, run the following shell command to create the required conda environment: <br />
```
conda env create -f InsilicoCell_env.yaml
```
Activate the environment after creation: <br />
```
conda activate InsilicoCell
```

# Step 1: Download model and checkpoints
View [this file](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/model_checkpoint/checkpoint.md) for downloading checkpoints.

# Step 2: Prepare your data
To run InsilicoCell for prediction, you need to prepare 1~2 input files. Note that File 1 is required for all tasks, while File 2 is only required for the following tasks: (1) drug-induced gene expression change (2) drug response (3) gene effect score
(4) gene mutation status (5) CNV. <br />

File 1: A meta info file of your input samples. The format of this file for each task can be viewd below: <br />
(1) [Drug-induced gene expression change](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/demo_data/input_sample_info.csv). <br />
(2) [Drug-protein binding affinity](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/demo_data/input_sample_info_drug-protein_binding.csv) <br />
(3) [TF-gene association](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/demo_data/input_sample_info_TF-gene_association.csv) <br />
(4) [Drug response](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/demo_data/input_sample_info_drug_response.csv) <br />
(5) [Gene effect score](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/demo_data/input_sample_info_gene_effect_score.csv) <br />
(6) [Gene mutation status](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/demo_data/input_sample_info_gene_mutation.csv) <br />
(7) [CNV](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/demo_data/input_sample_info_CNV.csv) <br />

Clarification on the columns of File 1:  <br />
Each row represent one input sample, which is a combination of the information you fill in all columns. For each task, different combinations of information are required from the following: <br />
• "SMILES": SMILES string of a drug. <br />
• "cell_iname": Cell name (you can define cell names by yourself if you use your own new cells, but make sure they match the cell names in your File 2 as described below). <br />
• "gene_name": Gene name. You can either use all ~20000 gene names as shown [here](https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/model_checkpoint/gene_list.csv) or select a subset of it. <br />
• "target_sequence": Amino sequence of a protein. <br />
• "time_h": A scalar value of treatment time, in the unit of hour. <br />
• "dose_uM": A scalar value of treatment concentration, in the unit of uM. <br />

File 2: log2(TPM+1) (using pseudocount of 1) data of untreated transcriptomic profiles in cells. Row names are cell names (they should match the cell names in the "cell_iname" column in your File 1), column names are gene names. If your cell line appears [here](https://chenlab-data-public.s3.amazonaws.com/InsilicoCell/cell_line_expression_matrix.csv) in our pretraining data, you can directly extract its gene expression profile from there.

# Step 3: Run InsilicoCell for prediction
Under the "./InsilicoCell" directory, run the following shell command for prediction. <br />

clarification on the input parameters: <br />
• File1: The path to your File 1 as in Step 2. <br />
• File2: The path to your File 2 as in Step 2. <br />
• out: Set a file name for the output prediction file (no extension). <br />
• BatchSize: An integer indicating the number of rows in File1 to be processed by your device in every batch. This parameter does not affect prediction results. The larger the batch size, the faster the prediction, but also requires more memory on your device. <br />
• device: GPU or CPU. <br />

Predicting drug-induced gene expression change: <br />
```
python 'demo_drug-induced_gene_expression.py' --File1 './demo_data/input_sample_info.csv' --File2 './demo_data/cell_transcriptomes_log2TPM.csv' --out 'pred_demo_drug-induced_gene_exp_change' --BatchSize 512 --device GPU
```
Predicting drug-protein binding affinity (this task may take a little longer if it's the first time for you to use T5 model): <br />
```
python 'demo_drug-protein_binding_affinity.py' --File1 './demo_data/input_sample_info_drug-protein_binding.csv' --out 'pred_demo_drug-protein_binding' --BatchSize 512 --device GPU
```
Predicting TF-gene association (this task may take a little longer if it's the first time for you to use T5 model): <br />
```
python 'demo_TF-gene_association.py' --File1 './demo_data/input_sample_info_TF-gene_association.csv' --out 'pred_demo_TF-gene_association' --BatchSize 512 --device GPU
```
Predicting drug response: <br />
```
python 'demo_drug_sensitivity.py' --File1 './demo_data/input_sample_info_drug_response.csv' --File2 './demo_data/task_drug_response_cell_transcriptomes_log2TPM.csv' --out 'pred_demo_drug_response' --BatchSize 512 --device GPU
```
Predicting gene effect score: <br />
```
python 'demo_gene_effect_score.py' --File1 './demo_data/input_sample_info_gene_effect_score.csv' --File2 './demo_data/task_gene_effect_score_cell_transcriptomes_log2TPM.csv' --out 'pred_demo_gene_effect_score' --BatchSize 512 --device GPU
```
Predicting gene mutation status: <br />
```
python 'demo_gene_mutation.py' --File1 './demo_data/input_sample_info_gene_mutation.csv' --File2 './demo_data/task_gene_mutation_cell_transcriptomes_log2TPM.csv' --out 'pred_demo_gene_mutation' --BatchSize 512 --device GPU
```
Predicting CNV: <br />
```
python 'demo_CNV.py' --File1 './demo_data/input_sample_info_CNV.csv' --File2 './demo_data/task_CNV_cell_transcriptomes_log2TPM.csv' --out 'pred_demo_CNV' --BatchSize 512 --device GPU
```

For each task, InsilicoCell will returns the predicted values in the "prediction" column in your output prediction file, which is saved to the "./prediction" directory. <br /> 
Clarification on the predicted values: <br /> 
(1) Drug-induced gene expression change: z-scores representing up/down-regulation, ranging from -10 to 10. Larger values represent more up-regulation, vice versa. <br /> 
(2) Drug-protein binding affinity: log10(IC50) (in the unit of nM). <br /> 
(3) TF-gene association: Binary values where 1 represents association, 0 represents no association. <br /> 
(4) Drug response: AUC values from the dose response curve, ranging from 0 to 30. Smaller values represent stronger drug response (growth inhibition effect) in cell. <br />
(5) Gene effect score: Dependency scores as in Depmap. >0 represents that gene knockout reduces cell viability, <0 represents that gene knockout improves growth. <br /> 
(6) Gene mutation status: Binary values where 1 represents mutation, 0 represents no mutation. <br /> 
(7) CNV: Gene level copy number values as in Depmap, log2-transformed with a pseudo count of 1. <br /> 




