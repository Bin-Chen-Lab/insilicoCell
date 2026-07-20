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

parser = argparse.ArgumentParser(description="TF-gene association prediction")
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
# Generate protein embeddings:
# First try to load embeddings from the cache. Only activate T5 for sequences missing from the cache.
cache = torch.load('./model_checkpoint/uniprot_human_proteome_sp_cache.pt', map_location="cpu", weights_only=True)
cache_embeddings = cache["embeddings"]
cache_sequence_to_index = cache["sequence_to_index"]

sequences = File_1["target_sequence"].unique().tolist()
embeddings = []
MAX_LENGTH = 1200

missing_sequences = [sequence for sequence in sequences if sequence not in cache_sequence_to_index]
missing_embedding_dict = {}

if len(missing_sequences) > 0:
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
    model.eval()

    for sequence in tqdm(missing_sequences, desc="Generating missing embeddings"):
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
        missing_embedding_dict[sequence] = emb_per_protein.cpu().numpy()
        del input_ids, attention_mask, embedding_repr, emb_per_residue
        torch.cuda.empty_cache()

for sequence in sequences:
    if sequence in cache_sequence_to_index:
        idx = cache_sequence_to_index[sequence]
        embeddings.append(cache_embeddings[idx].cpu().numpy())
    else:
        embeddings.append(missing_embedding_dict[sequence])

protein_embeddings = pd.DataFrame(embeddings, index=sequences)

#------------------------------------------------------------------------------------
#Load gene embedding file:
gene_embeddings = pd.read_csv('./model_checkpoint/gene_embedding.csv', index_col = '0')
nonexist_genes = list(set(File_1['gene_name']) - set(gene_embeddings.index))
if len(nonexist_genes) != 0:
    print("There are invalid gene names in your File1: " + str(nonexist_genes) + ", please use genes from the following gene names: https://github.com/Bin-Chen-Lab/insilicoCell/blob/main/InsilicoCell/demo_data/gene_list.csv")
    sys.exit()

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
            tmp_gene = gene_embeddings.loc[File_1.iloc[start:end, :]['gene_name'], :] 
            tmp_gene = torch.tensor(tmp_gene.values, dtype=torch.float32) 
            tmp_protein = protein_embeddings.loc[File_1.iloc[start:end, :]['target_sequence'], :] 
            tmp_protein = torch.tensor(tmp_protein.values, dtype=torch.float32) 
            batch = torch.cat((tmp_gene, tmp_protein), dim = 1).to(device)
            y_batch = model(batch)[0]    
            outputs.append(y_batch.cpu())
    return torch.cat(outputs, dim=0)

model3 = torch.jit.load("./model_checkpoint/TF-gene_association.pt", map_location=device)

model3.eval()
y = predict_in_batches(model3, int(float(batch_size)), device=device) #predicted values are probability
File_1["prediction"] = y.numpy()
#File_1.loc[File_1['prediction'] > 0.5, 'prediction'] = 1
#File_1.loc[File_1['prediction'] <= 0.5, 'prediction'] = 0

output_file = './prediction/' + save_prediction_file_name + '.csv'
File_1.to_csv(output_file, index=False)

print(f"Prediction completed successfully. Output saved to: {output_file}")