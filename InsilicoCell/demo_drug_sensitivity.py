import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import sys
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser(description="Drug response prediction")
parser.add_argument("--File1", required=True, help="Path to input sample info CSV file.")
parser.add_argument("--File2", required=True, help="Path to cell transcriptome log2TPM CSV file.")
parser.add_argument("--out", required=True, help="Output prediction file name (no extension).")
parser.add_argument("--BatchSize", required=True, help="How many samples to predict per batch")
parser.add_argument("--device", required=True, help="GPU or CPU?")

args = parser.parse_args()

File_1_path = args.File1
File_2_path = args.File2
save_prediction_file_name = args.out
batch_size = args.BatchSize
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
File_1 = pd.read_csv(File_1_path)
File_2 = pd.read_csv(File_2_path, index_col = 'Unnamed: 0')

#------------------------------------------------------------------------------------
#Double-check of column names:
demo_file = pd.read_csv('./demo_data/cell_transcriptomes_log2TPM.csv', index_col = 'Unnamed: 0')
File_2 = File_2.groupby(File_2.columns, axis=1).mean() #handle duplicated gene names
File_2 = File_2.reindex(columns=demo_file.columns)
File_2 = File_2.apply(lambda row: row.fillna(row.mean()), axis=1)


#------------------------------------------------------------------------------------
#Generate drug embeddings:
all_smiles = File_1['SMILES'].unique().tolist()
fingerprints = []
invalid_smiles = []

for smi in all_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        arr = np.zeros((1,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
    else:
        invalid_smiles.append(smi)

fingerprint_matrix = np.array(fingerprints)

if fingerprint_matrix.shape[0] != len(all_smiles):
    print("Your File1 contains invalid SMILES strings, please remove them from your File1 before proceeding:", invalid_smiles)
    sys.exit()

drug_embeddings = pd.DataFrame(fingerprint_matrix, index = all_smiles)

#------------------------------------------------------------------------------------
#Generate cell line embeddings:
model2 = torch.jit.load("./model_checkpoint/cell_embedding.pt", map_location=device)
model2.eval()
cellline_embeddings = model2(torch.tensor(File_2.values, dtype=torch.float32).to(device))[0].to('cpu')
cellline_embeddings = pd.DataFrame(cellline_embeddings.detach().numpy(), index = File_2.index)

#------------------------------------------------------------------------------------
#convert embeddings to torch tensor format:

def predict_in_batches(model, batch_size, device="cpu"):
    """
    Run prediction in batches to avoid GPU/CPU memory overflow.
    """
    model = model.to(device)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(File_1), batch_size):
            end = start + batch_size
            tmp_drug = drug_embeddings.loc[File_1.iloc[start:end, :]['SMILES'], :] 
            tmp_drug = torch.tensor(tmp_drug.values, dtype=torch.float32) 
            tmp_cellline = cellline_embeddings.loc[File_1.iloc[start:end, :]['cell_iname'], :] 
            tmp_cellline = torch.tensor(tmp_cellline.values, dtype=torch.float32) 
            batch = torch.cat((tmp_drug, tmp_cellline), dim = 1).to(device)
            y_batch = model(batch)[0]    
            outputs.append(y_batch.cpu())
    return torch.cat(outputs, dim=0)

model3 = torch.jit.load("./model_checkpoint/drug_response.pt", map_location=device)

model3.eval()
y = predict_in_batches(model3, int(float(batch_size)), device=device) #predicted values are ln(AUC+1) 
File_1['prediction'] = np.exp(y.numpy()) - 1 #covert back to the original AUC scale

File_1.to_csv('./prediction/' + save_prediction_file_name + '.csv')

