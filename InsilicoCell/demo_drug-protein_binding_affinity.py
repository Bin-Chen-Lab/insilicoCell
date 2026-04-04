import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import sys
from rdkit import RDLogger
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import re

RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser(description="Drug-protein binding affinity prediction")
parser.add_argument("--File1", required=True, help="Path to input sample info CSV file.")
parser.add_argument("--File2", required=False, help="Path to cell transcriptome log2TPM CSV file.")
parser.add_argument("--out", required=True, help="Output prediction file name (no extension).")
parser.add_argument("--BatchSize", required=True, help="How many samples to predict per batch")
parser.add_argument("--device", required=True, help="GPU or CPU?")

args = parser.parse_args()

File_1_path = args.File1
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

#------------------------------------------------------------------------------------
#Generate protein embeddings:
#(The first time for downloading the model will take longer time)
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
sequences = File_1["target_sequence"].unique().tolist()
embeddings = []
MAX_LENGTH = 1200
for sequence in tqdm(sequences, desc="Generating embeddings"):
    processed_sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    processed_sequence = processed_sequence[:MAX_LENGTH]
    original_length = min(len(sequence), MAX_LENGTH)
    ids = tokenizer(processed_sequence, add_special_tokens=True, padding="longest", return_tensors="pt")
    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    emb_per_residue = embedding_repr.last_hidden_state[0, :original_length]  # (length, hidden_dim)
    emb_per_protein = emb_per_residue.mean(dim=0) 
    embeddings.append(emb_per_protein.cpu())
    del input_ids, attention_mask, embedding_repr, emb_per_residue
    torch.cuda.empty_cache()


protein_embeddings = pd.DataFrame([t.numpy() for t in embeddings], index = sequences)

#torch.save(embeddings, "./protein_embeddings.pt")


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
            tmp_protein = protein_embeddings.loc[File_1.iloc[start:end, :]['target_sequence'], :] 
            tmp_protein = torch.tensor(tmp_protein.values, dtype=torch.float32) 
            batch = torch.cat((tmp_drug, tmp_protein), dim = 1).to(device)
            y_batch = model(batch)[0]    
            outputs.append(y_batch.cpu())
    return torch.cat(outputs, dim=0)

model3 = torch.jit.load("./model_checkpoint/drug-protein_binding_affinity.pt", map_location=device)

model3.eval()
y = predict_in_batches(model3, int(float(batch_size)), device=device) #predicted values are log10(IC50) (IC50 is in the unit of nM) 
File_1["prediction"] = y.numpy()
File_1.to_csv('./prediction/' + save_prediction_file_name + '.csv')

