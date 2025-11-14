import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import sys

parser = argparse.ArgumentParser(description="Drug-induced gene expression prediction")
parser.add_argument("--File1", required=True, help="Path to input sample info CSV file.")
parser.add_argument("--File2", required=True, help="Path to cell transcriptome log2TPM CSV file.")
parser.add_argument("--out", required=True, help="Output prediction file name (no extension).")
parser.add_argument("--device", required=True, help="GPU or CPU?")

args = parser.parse_args()

File_1_path = args.File1
File_2_path = args.File2
save_prediction_file_name = args.out
my_device = args.device
if my_device=="GPU":
    if not torch.cuda.is_available():
        print("No GPU on your device")
        sys.exit()
    else:
        device = torch.device( f'cuda:{0}')
else:
    device = 'cpu'

#------------------------------------------------------------------------------------
#Users need to input: 
#File 1: Input sample info file (the column "cell_iname" should match the cell names in the row names of file 2). 
#File 2: log2TPM data of transcriptomic profiles in cells. Row names are cell names, column names must use the same gene names in orders as shown in the demo data. For missing genes' expression, you may use the average expression of the other genes in the cell to impute.
File_1 = pd.read_csv(File_1_path, index_col = 'Unnamed: 0')
File_2 = pd.read_csv(File_2_path, index_col = 'Unnamed: 0')
#------------------------------------------------------------------------------------
#Load gene embedding file:
gene_embeddings = pd.read_csv('./model_checkpoint/GO_terms_all_19614_human_genes_pecanpy_embeddings_20241212.csv', index_col = '0')

#------------------------------------------------------------------------------------
#Generate drug embeddings:
all_smiles = File_1['SMILES'].unique().tolist()
fingerprints = []

for smi in all_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        arr = np.zeros((1,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)

fingerprint_matrix = np.array(fingerprints)
drug_embeddings = pd.DataFrame(fingerprint_matrix, index = all_smiles)

#------------------------------------------------------------------------------------
#Generate cell line embeddings:
model2 = torch.jit.load("/egr/research-aidd/chenruo4/self-built_transformer/upload_no_source_version/model_checkpoint/cell_embedding.pt", map_location=device)
model2.eval()
cellline_embeddings = model2(torch.tensor(File_2.values, dtype=torch.float32).to(device))[0].to('cpu')
cellline_embeddings = pd.DataFrame(cellline_embeddings.detach().numpy(), index = File_2.index)

#------------------------------------------------------------------------------------
#convert embeddings to torch tensor format:
tmp_drug = drug_embeddings.loc[File_1['SMILES'], :] 
tmp_drug = torch.tensor(tmp_drug.values, dtype=torch.float32) 

tmp_cellline = cellline_embeddings.loc[File_1['cell_iname'], :]
tmp_cellline = torch.tensor(tmp_cellline.values, dtype=torch.float32) 

tmp_gene = gene_embeddings.loc[File_1['gene_name'], :]
tmp_gene = torch.tensor(tmp_gene.values, dtype=torch.float32) 

tmp_time = pd.DataFrame(File_1['time_h'])
tmp_time = torch.tensor(tmp_time.values, dtype=torch.float32) 

tmp_dose = pd.DataFrame(File_1['dose_uM'])
tmp_dose = torch.tensor(tmp_dose.values, dtype=torch.float32)  

tmp_input_all = torch.cat((tmp_drug, tmp_cellline, tmp_gene, tmp_time, tmp_dose), dim = 1) 

def predict_in_batches(model, inputs, batch_size=512, device="cpu"):
    """
    Run prediction in batches to avoid GPU/CPU memory overflow.
    """
    model = model.to(device)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            end = start + batch_size
            batch = inputs[start:end].to(device)
            y_batch = model(batch)[0]    
            outputs.append(y_batch.cpu())
    return torch.cat(outputs, dim=0)

if device == 'cpu':
    model3 = torch.jit.load("./model_checkpoint/drug-induced_gene_expression_CPU.pt", map_location=device)
else:
    model3 = torch.jit.load("./model_checkpoint/drug-induced_gene_expression_GPU.pt", map_location=device)

model3.eval()
y = predict_in_batches(model3, tmp_input_all, device=device)
File_1["prediction"] = y.numpy()
#y = model3.to(device)(tmp_input_all.to(device))[0]
#File_1['prediction'] = y.to('cpu').detach().numpy()
File_1.to_csv('./prediction/' + save_prediction_file_name + '.csv')

#-----------------------------------------------------------------------------------------
#python demo_drug-induced_gene_expression.py --File1 /egr/research-aidd/chenruo4/self-built_transformer/upload_no_source_version/demo_data/input_sample_info.csv --File2 /egr/research-aidd/chenruo4/self-built_transformer/upload_no_source_version/demo_data/cell_transcriptomes_log2TPM.csv --out pred_demo --device GPU
