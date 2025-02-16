import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import random
import os
import time
import collections
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, f1_score, accuracy_score


device = torch.device( f'cuda:{1}' if torch.cuda.is_available() else 'cpu')

#----------------------------------------------------------------------------------------------------------------
#set seed:
def seed_everything(seed):
    torch.cuda.manual_seed_all(seed)
    #if args.cuda:
    #    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


#----------------------------------------------------------------------------------------------------------------
# Model:
class Model(nn.Module):
    def __init__(self, num_attention_heads=16, hidden_size = 768):
        super(Model, self).__init__()
        self.hidden_size = hidden_size  # Default hidden size 
        self.embedding_drug1 = nn.Linear(600, 384)
        self.embedding_drug2 = nn.Linear(384, hidden_size)
        self.embedding_cellline1 = nn.Linear(128, 384)
        self.embedding_cellline2 = nn.Linear(384, hidden_size)
        self.embedding_gene1 = nn.Linear(128, 384)
        self.embedding_gene2 = nn.Linear(384, hidden_size)
        self.embedding_time1 = nn.Linear(1, hidden_size)
        self.embedding_dose1 = nn.Linear(1, hidden_size)
        self.embedding_protein1 = nn.Linear(1024, 512)
        self.embedding_protein2 = nn.Linear(512, hidden_size)
        self.LayerNorm_drug = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout_drug = nn.Dropout(0.1)
        self.LayerNorm_cellline = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout_cellline = nn.Dropout(0.1)
        self.LayerNorm_gene = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout_gene = nn.Dropout(0.1)
        self.LayerNorm_time = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout_time = nn.Dropout(0.1)
        self.LayerNorm_dose = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout_dose = nn.Dropout(0.1)
        self.LayerNorm_protein = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout_protein = nn.Dropout(0.1)
        # Encoder layers (12 layers)
        self.encoder_layers = nn.ModuleList([ModelLayer(hidden_size, num_attention_heads) for _ in range(12)])
        # Pooler (output the representation of the [CLS] token)
        self.pooler = ModelPooler(hidden_size)
        # Regression head for each task:
        self.regressor_1 = nn.Linear(hidden_size, 1)
        self.regressor_2 = nn.Linear(hidden_size, 1)
        self.regressor_3 = nn.Linear(hidden_size, 1)
        self.regressor_4 = nn.Linear(hidden_size, 1)
        self.regressor_5 = nn.Linear(hidden_size, 1)
        self.regressor_6 = nn.Linear(hidden_size, 1)
        self.regressor_7 = nn.Linear(hidden_size, 1)
    #
    def forward(self, inputs_embeds, input_task_id = None, CLS_token = None):
        # Gather each modality's data from all tasks together:
        #task_id == 'drug-induced_gene_exp', #order of modality: drug (600), cellline (128), gene (128), time (1), dose (1)
        embeddings = inputs_embeds[0]
        task_id = input_task_id[0]
        sample_size = embeddings.shape[0]
        drug, cellline, gene, time, dose = embeddings[:, :600], embeddings[:, 600:728], embeddings[:, 728:856], embeddings[:, 856], embeddings[:, 857]
        all_drug = drug
        all_cellline = cellline
        all_gene = gene
        all_time = time
        all_dose = dose
        #
        #task_id == 'drug-protein_binding': #order of modality: drug (600), protein (1024)
        embeddings = inputs_embeds[1]
        task_id = input_task_id[1]
        sample_size = embeddings.shape[0]
        drug, protein = embeddings[:, :600], embeddings[:, 600:]
        all_drug = torch.cat((all_drug, drug), dim = 0)
        all_protein = protein
        #
        #task_id == 'TF-gene_regulation': #order of modality: gene (128), protein (1024)
        embeddings = inputs_embeds[2]
        task_id = input_task_id[2]
        sample_size = embeddings.shape[0]
        gene, protein = embeddings[:, :128], embeddings[:, 128:]
        all_gene = torch.cat((all_gene, gene), dim = 0)
        all_protein = torch.cat((all_protein, protein), dim = 0)
        #
        #task_id == 'drug_sensitivity': #order of modality: drug (600), cellline (128)
        embeddings = inputs_embeds[3]
        task_id = input_task_id[3]
        sample_size = embeddings.shape[0]
        drug, cellline = embeddings[:, :600], embeddings[:, 600:]
        all_drug = torch.cat((all_drug, drug), dim = 0)
        all_cellline = torch.cat((all_cellline, cellline), dim = 0)
        #
        #task_id == 'gene_effect_score': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[4]
        task_id = input_task_id[4]
        sample_size = embeddings.shape[0]
        cellline, gene = embeddings[:, :128], embeddings[:, 128:]
        all_cellline = torch.cat((all_cellline, cellline), dim = 0)
        all_gene = torch.cat((all_gene, gene), dim = 0)
        #
        #task_id == 'gene_mutation': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[5]
        task_id = input_task_id[5]
        sample_size = embeddings.shape[0]
        cellline, gene = embeddings[:, :128], embeddings[:, 128:]
        all_cellline = torch.cat((all_cellline, cellline), dim = 0)
        all_gene = torch.cat((all_gene, gene), dim = 0)
        #
        #task_id == 'gene_CNV': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[6]
        task_id = input_task_id[6]
        sample_size = embeddings.shape[0]
        cellline, gene = embeddings[:, :128], embeddings[:, 128:]
        all_cellline = torch.cat((all_cellline, cellline), dim = 0)
        all_gene = torch.cat((all_gene, gene), dim = 0)
        #
        all_drug = self.embedding_drug2(F.gelu(self.embedding_drug1(all_drug)))
        all_cellline = self.embedding_cellline2(F.gelu(self.embedding_cellline1(all_cellline)))
        all_gene = self.embedding_gene2(F.gelu(self.embedding_gene1(all_gene)))
        all_time = self.embedding_time1((all_time/48).unsqueeze(1))
        all_dose = self.embedding_dose1((all_dose/20).unsqueeze(1))
        all_protein = self.embedding_protein2(F.gelu(self.embedding_protein1(all_protein)))
        #
        # Normalize + dropout:
        all_drug = self.LayerNorm_drug(all_drug)
        all_drug = self.dropout_drug(all_drug)
        all_gene = self.LayerNorm_gene(all_gene)
        all_gene = self.dropout_gene(all_gene)
        all_cellline = self.LayerNorm_cellline(all_cellline)
        all_cellline = self.dropout_cellline(all_cellline)
        all_time = self.LayerNorm_time(all_time)
        all_time = self.dropout_time(all_time)
        all_dose = self.LayerNorm_dose(all_dose)
        all_dose = self.dropout_dose(all_dose)
        all_protein = self.LayerNorm_protein(all_protein)
        all_protein = self.dropout_protein(all_protein)
        #
        # Split the data back into each task:
        # Add CLS token & padding tokens, concatenate all tokens together for each task:
        #
        #task_id == 'drug-induced_gene_exp', #order of modality: drug (600), cellline (128), gene (128), time (1), dose (1)
        embeddings = inputs_embeds[0]
        task_id = input_task_id[0]
        sample_size_1 = embeddings.shape[0]
        embeddings_1 = torch.cat((CLS_token.repeat(sample_size_1, 1), all_drug[:sample_size_1, :], all_gene[:sample_size_1, :], all_cellline[:sample_size_1, :], all_time[:sample_size_1, :], all_dose[:sample_size_1, :]), dim=1)
        #No padding for this task.
        embeddings_1 = embeddings_1.reshape(sample_size_1, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_1 = (torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)).repeat(sample_size_1, 1).to(device)
        #
        #task_id == 'drug-protein_binding': #order of modality: drug (600), protein (1024)
        embeddings = inputs_embeds[1]
        task_id = input_task_id[1]
        sample_size_2 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_2, self.hidden_size)).repeat(1, 3).to(device)
        embeddings_2 = torch.cat((CLS_token.repeat(sample_size_2, 1), all_drug[(sample_size_1):(sample_size_1+sample_size_2), :], all_protein[:sample_size_2, :], padding), dim=1)
        embeddings_2 = embeddings_2.reshape(sample_size_2, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_2 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_2, 1).to(device)
        #
        #task_id == 'TF-gene_regulation': #order of modality: gene (128), protein (1024)
        embeddings = inputs_embeds[2]
        task_id = input_task_id[2]
        sample_size_3 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_3, self.hidden_size)).repeat(1, 3).to(device)
        embeddings_3 = torch.cat((CLS_token.repeat(sample_size_3, 1), all_gene[(sample_size_1):(sample_size_1+sample_size_3), :], all_protein[(sample_size_2):(sample_size_2+sample_size_3), :], padding), dim=1)
        embeddings_3 = embeddings_3.reshape(sample_size_3, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_3 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_3, 1).to(device)
        #
        #task_id == 'drug_sensitivity': #order of modality: drug (600), cellline (128)
        embeddings = inputs_embeds[3]
        task_id = input_task_id[3]
        sample_size_4 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_4, self.hidden_size)).repeat(1, 3).to(device)
        embeddings_4 = torch.cat((CLS_token.repeat(sample_size_4, 1), all_drug[(sample_size_1 + sample_size_2):(sample_size_1 + sample_size_2 + sample_size_4), :], all_cellline[(sample_size_1):(sample_size_1+sample_size_4), :], padding), dim=1)
        embeddings_4 = embeddings_4.reshape(sample_size_4, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_4 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_4, 1).to(device)
        #
        #task_id == 'gene_effect_score': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[4]
        task_id = input_task_id[4]
        sample_size_5 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_5, self.hidden_size)).repeat(1, 3).to(device)
        embeddings_5 = torch.cat((CLS_token.repeat(sample_size_5, 1), all_cellline[(sample_size_1 + sample_size_4):(sample_size_1 + sample_size_4 + sample_size_5), :], all_gene[(sample_size_1 + sample_size_3):(sample_size_1 + sample_size_3 + sample_size_5), :], padding), dim=1)
        embeddings_5 = embeddings_5.reshape(sample_size_5, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_5 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_5, 1).to(device)
        #
        #task_id == 'gene_mutation': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[5]
        task_id = input_task_id[5]
        sample_size_6 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_6, self.hidden_size)).repeat(1, 3).to(device)
        embeddings_6 = torch.cat((CLS_token.repeat(sample_size_6, 1), all_cellline[(sample_size_1 + sample_size_4 + sample_size_5):(sample_size_1 + sample_size_4 + sample_size_5 + sample_size_6), :], all_gene[(sample_size_1 + sample_size_3 + sample_size_5):(sample_size_1 + sample_size_3 + sample_size_5 + sample_size_6), :], padding), dim=1)
        embeddings_6 = embeddings_6.reshape(sample_size_6, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_6 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_6, 1).to(device)
        #
        #task_id == 'gene_CNV': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[6]
        task_id = input_task_id[6]
        sample_size_7 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_7, self.hidden_size)).repeat(1, 3).to(device)
        embeddings_7 = torch.cat((CLS_token.repeat(sample_size_7, 1), all_cellline[(sample_size_1 + sample_size_4 + sample_size_5 + sample_size_6):(sample_size_1 + sample_size_4 + sample_size_5 + sample_size_6 + sample_size_7), :], all_gene[(sample_size_1 + sample_size_3 + sample_size_5 + sample_size_6):(sample_size_1 + sample_size_3 + sample_size_5 + sample_size_6 + sample_size_7), :], padding), dim=1)
        embeddings_7 = embeddings_7.reshape(sample_size_7, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_7 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_7, 1).to(device)
        #
        #Gather all tasks' data again for transfomer encoder:
        embeddings = torch.cat((embeddings_1, embeddings_2, embeddings_3, embeddings_4, embeddings_5, embeddings_6, embeddings_7), dim = 0)
        attention_mask = torch.cat((attention_mask_1, attention_mask_2, attention_mask_3, attention_mask_4, attention_mask_5, attention_mask_6, attention_mask_7), dim = 0)
        #
        # Pass through the encoder layers: 
        for layer in self.encoder_layers:
            embeddings = layer(embeddings, attention_mask)
        # Pool the output of the [CLS] token
        pooled_output = self.pooler(embeddings) #for classification tasks, consider using Sigmoid activation
        #
        #Split the data by task again:
        pooled_output_1 = pooled_output[:sample_size_1, :]
        pooled_output_2 = pooled_output[(sample_size_1):(sample_size_1 + sample_size_2), :]
        pooled_output_3 = pooled_output[(sample_size_1 + sample_size_2):(sample_size_1 + sample_size_2 + sample_size_3), :]
        pooled_output_4 = pooled_output[(sample_size_1 + sample_size_2 + sample_size_3):(sample_size_1 + sample_size_2 + sample_size_3 + sample_size_4), :]
        pooled_output_5 = pooled_output[(sample_size_1 + sample_size_2 + sample_size_3 + sample_size_4):(sample_size_1 + sample_size_2 + sample_size_3 + sample_size_4 + sample_size_5), :]
        pooled_output_6 = pooled_output[(sample_size_1 + sample_size_2 + sample_size_3 + sample_size_4 + sample_size_5):(sample_size_1 + sample_size_2 + sample_size_3 + sample_size_4 + sample_size_5 + sample_size_6), :]
        pooled_output_7 = pooled_output[(sample_size_1 + sample_size_2 + sample_size_3 + sample_size_4 + sample_size_5 + sample_size_6):(sample_size_1 + sample_size_2 + sample_size_3 + sample_size_4 + sample_size_5 + sample_size_6 + sample_size_7), :]
        # Output layer for regression and classification tasks:
        out_1 = self.regressor_1(pooled_output_1)
        out_2 = self.regressor_2(pooled_output_2)
        out_3 = F.sigmoid(self.regressor_3(pooled_output_3))
        out_4 = self.regressor_4(pooled_output_4)
        out_5 = self.regressor_5(pooled_output_5)
        out_6 = F.sigmoid(self.regressor_6(pooled_output_6))
        out_7 = self.regressor_7(pooled_output_7)
        return out_1, out_2, out_3, out_4, out_5, out_6, out_7

#----------------------------------------------------------------------------------------------------------------
# Model layer (attention + feed-forward sub-layers):
class ModelLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=16):
        super(ModelLayer, self).__init__()
        self.attention = ModelAttention(hidden_size, num_attention_heads)
        self.intermediate = ModelIntermediate(hidden_size)
        self.output = ModelOutput(hidden_size)
    #
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

#----------------------------------------------------------------------------------------------------------------
# Attention sub-layer (scaled dot product attention):
class ModelAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=16):
        super(ModelAttention, self).__init__()
        self.self_attention = ModelSdpaSelfAttention(hidden_size, num_attention_heads)
        self.output = ModelSelfOutput(hidden_size)
    def forward(self, hidden_states, attention_mask=None):
        # Pass through the self-attention with the attention mask
        attention_output = self.self_attention(hidden_states, attention_mask)
        output = self.output(attention_output)
        return output

#----------------------------------------------------------------------------------------------------------------
# Scaled dot-product attention:
class ModelSdpaSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=16):
        super(ModelSdpaSelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads  # Dividing the hidden size into multiple heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
    #
    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_size)
    #
    def forward(self, hidden_states, attention_mask=None):
        # Compute Query, Key, and Value
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        # Transpose to shape (batch_size, num_heads, seq_len, head_size)
        query = self.transpose_for_scores(query) #query.shape: [128, 16, 3, 48]
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value) #value.shape: [128, 16, 3, 48]
        # Scaled Dot-Product Attention
        #attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (hidden_states.size(-1) ** 0.5)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.attention_head_size ** 0.5) 
        #attention_scores.shape: [128, 16, 3, 3]. key.transpose(-1, -2).shape: [128, 16, 48, 3]
        # Apply attention mask (if provided)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Shape: [128, 1, 1, 3]
            attention_mask = attention_mask.expand(-1, self.num_attention_heads, query.shape[2], query.shape[2])  # [128, 16, 3, 3]
            # Add a very large negative value to the attention scores where the mask is 1 (padding tokens)
            attention_scores = attention_scores + (attention_mask * -1e9)
        #
        attention_probs = F.softmax(attention_scores, dim=-1)  # Normalizing the scores
        attention_probs = self.dropout(attention_probs)
        # Apply attention weights to the values
        attention_output = torch.matmul(attention_probs, value) #attention_output.shape: [128, 16, 3, 48]
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous() #attention_output.shape: [128, 3, 16, 48]
        new_shape = attention_output.size()[:-2] + (self.all_head_size,) #new_shape: torch.Size([128, 3, 768])
        attention_output = attention_output.view(*new_shape)
        return attention_output
    
#----------------------------------------------------------------------------------------------------------------
# Output sub-layer of attention:
class ModelSelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super(ModelSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
    #
    def forward(self, attention_output):
        output = self.dense(attention_output)
        output = self.LayerNorm(output + attention_output)  # Residual connection
        return self.dropout(output)

#----------------------------------------------------------------------------------------------------------------
# Feed-forward sub-layer (intermediate layer)
class ModelIntermediate(nn.Module):
    def __init__(self, hidden_size):
        super(ModelIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size*4)  #intermediate layer size
        self.activation = nn.GELU()
    #
    def forward(self, attention_output):
        return self.activation(self.dense(attention_output))

#----------------------------------------------------------------------------------------------------------------
# Output layer of the feed-forward sub-layer
class ModelOutput(nn.Module):
    def __init__(self, hidden_size):
        super(ModelOutput, self).__init__()
        self.dense = nn.Linear(hidden_size*4, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
    #
    def forward(self, intermediate_output, attention_output):
        output = self.dense(intermediate_output)
        output = self.dropout(output)
        return self.LayerNorm(output + attention_output)  # Residual connection

#----------------------------------------------------------------------------------------------------------------
# Pooler for the CLS token
class ModelPooler(nn.Module):
    def __init__(self, hidden_size):
        super(ModelPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
    #
    def forward(self, hidden_states):
        # Pool the output of the first token ([CLS])
        cls_output = hidden_states[:, 0]
        pooled_output = self.dense(cls_output)
        return self.activation(pooled_output)
    

#----------------------------------------------------------------------------------------------------------------
# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, input_X, input_Y, task_id):
        self.input_X = input_X 
        self.input_Y = input_Y
        self.task_id = task_id
    def __len__(self):
        return len(self.input_Y)
    def __getitem__(self, idx):
        input_data = self.input_X[idx]
        label = self.input_Y[idx]  
        task_id = self.task_id
        return {
                'input_data': torch.tensor(input_data, dtype=torch.float32).to(device),  
                'labels': torch.tensor(label, dtype=torch.float32).to(device),
                'task_id': task_id
        }

#---------------------------------------------------------------------------------------
#import the whole training data for all 7 tasks:
#########################
#task 1: 'drug-induced_gene_exp'
#order of modality: drug, cellline, gene, time, dose
X_training_drug_induced_gene_exp = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_drug-induced_gene_exp_use_GNN_subset_0.1_portion.npy', mmap_mode='r')
#torch.Size([4491309, 858])
Y_training_drug_induced_gene_exp = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_drug-induced_gene_exp_use_GNN_subset_0.1_portion.npy', mmap_mode='r')

X_val_drug_induced_gene_exp = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_drug-induced_gene_exp_use_GNN.pt')
Y_val_drug_induced_gene_exp = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_drug-induced_gene_exp_use_GNN.pt')

#########################
#task 2: 'drug-protein_binding'
#order of modality: drug, protein
X_training_drug_protein_binding = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_drug-protein_binding_use_GNN_subset_0.1_portion.npy', mmap_mode='r')
#torch.Size([253119, 1624])
Y_training_drug_protein_binding = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_drug-protein_binding_use_GNN_subset_0.1_portion.npy', mmap_mode='r')

X_val_drug_protein_binding = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_drug-protein_binding_use_GNN.pt')
Y_val_drug_protein_binding = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_drug-protein_binding_use_GNN.pt')

#########################
#task 3: 'TF-gene_regulation' 
#order of modality: gene, protein
#X_training_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_TF-gene_regulation.pt')
X_training_TF_gene_regulation = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_TF-gene_regulation_balanced_data_subset_0.1_portion.npy', mmap_mode='r')
#torch.Size([115315, 1152])
#Y_training_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_TF-gene_regulation.pt')
Y_training_TF_gene_regulation = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_TF-gene_regulation_balanced_data_subset_0.1_portion.npy', mmap_mode='r')

X_val_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_TF-gene_regulation.pt')
Y_val_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_TF-gene_regulation.pt')


#########################
#task 4: 'drug_sensitivity'
#order of modality: drug, cellline
X_training_drug_sensitivity = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_drug_sensitivity_use_GNN_subset_0.1_portion.npy', mmap_mode='r')
#torch.Size([55342, 728])
Y_training_drug_sensitivity = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_drug_sensitivity_use_GNN_subset_0.1_portion.npy', mmap_mode='r')

X_val_drug_sensitivity = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_drug_sensitivity_use_GNN.pt')
Y_val_drug_sensitivity = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_drug_sensitivity_use_GNN.pt')

#########################
#task 5: 'gene_effect_score'
#order of modality: cellline, gene
X_training_gene_effect_score = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_gene_effect_score_subset_0.1_portion.npy', mmap_mode='r')
#torch.Size([1761902, 256])
Y_training_gene_effect_score = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_gene_effect_score_subset_0.1_portion.npy', mmap_mode='r')

X_val_gene_effect_score = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_gene_effect_score.pt')
Y_val_gene_effect_score = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_gene_effect_score.pt')

#########################
#task 6: 'gene_mutation'
#order of modality: cellline, gene
#X_training_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_gene_mutation.pt')
X_training_gene_mutation = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_gene_mutation_balanced_data_subset_0.1_portion.npy', mmap_mode='r')
#torch.Size([173359, 256])
#Y_training_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_gene_mutation.pt')
Y_training_gene_mutation = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_gene_mutation_balanced_data_subset_0.1_portion.npy', mmap_mode='r')

X_val_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_gene_mutation.pt')
Y_val_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_gene_mutation.pt')

#########################
#task 7: 'gene_CNV'
#order of modality: cellline, gene
X_training_gene_CNV = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_gene_CNV_subset_0.1_portion.npy', mmap_mode='r')
#torch.Size([3561922, 256])
Y_training_gene_CNV = np.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_gene_CNV_subset_0.1_portion.npy', mmap_mode='r')

X_val_gene_CNV = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_gene_CNV.pt')
Y_val_gene_CNV = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_gene_CNV.pt')

#----------------------------------------------------------------------------------------------------------------
# Create DataLoader
train_dataset_drug_induced_gene_exp = CustomDataset(X_training_drug_induced_gene_exp, Y_training_drug_induced_gene_exp, 
                                                   task_id = 'drug-induced_gene_exp'
                                                   )

train_dataset_drug_protein_binding = CustomDataset(X_training_drug_protein_binding, Y_training_drug_protein_binding, 
                                                   task_id = 'drug-protein_binding'
                                                   )

train_dataset_TF_gene_regulation = CustomDataset(X_training_TF_gene_regulation, Y_training_TF_gene_regulation, 
                                                   task_id = 'TF-gene_regulation'
                                                   )

train_dataset_drug_sensitivity = CustomDataset(X_training_drug_sensitivity, Y_training_drug_sensitivity, 
                                               task_id = 'drug_sensitivity'
                                               )

train_dataset_gene_effect_score = CustomDataset(X_training_gene_effect_score, Y_training_gene_effect_score, 
                                               task_id = 'gene_effect_score'
                                               )

train_dataset_gene_mutation = CustomDataset(X_training_gene_mutation, Y_training_gene_mutation, 
                                               task_id = 'gene_mutation'
                                               )

train_dataset_gene_CNV = CustomDataset(X_training_gene_CNV, Y_training_gene_CNV, 
                                               task_id = 'gene_CNV'
                                               )

val_dataset_drug_induced_gene_exp = CustomDataset(X_val_drug_induced_gene_exp, Y_val_drug_induced_gene_exp, 
                                                   task_id = 'drug-induced_gene_exp'
                                                   )

val_dataset_drug_protein_binding = CustomDataset(X_val_drug_protein_binding, Y_val_drug_protein_binding, 
                                                   task_id = 'drug-protein_binding'
                                                   )

val_dataset_TF_gene_regulation = CustomDataset(X_val_TF_gene_regulation, Y_val_TF_gene_regulation, 
                                                   task_id = 'TF-gene_regulation'
                                                   )

val_dataset_drug_sensitivity = CustomDataset(X_val_drug_sensitivity, Y_val_drug_sensitivity, 
                                               task_id = 'drug_sensitivity'
                                               )

val_dataset_gene_effect_score = CustomDataset(X_val_gene_effect_score, Y_val_gene_effect_score, 
                                               task_id = 'gene_effect_score'
                                               )

val_dataset_gene_mutation = CustomDataset(X_val_gene_mutation, Y_val_gene_mutation, 
                                               task_id = 'gene_mutation'
                                               )

val_dataset_gene_CNV = CustomDataset(X_val_gene_CNV, Y_val_gene_CNV, 
                                               task_id = 'gene_CNV'
                                               )

#---------------------------------------------------------------------------------------
#changeable parameters:
fraction = 1/5000 #the fraction of training samples to be taken from each task to form a batch
#batch_size = 128
epoch = 100 #maximum epoch
rdseed = 10 
out_dir = '/egr/research-aidd/chenruo4/self-built_transformer/model_trial_0.1_portion_of_whole_training_data_20241204/demo/'

#---------------------------------------------------------------------------------------
#training set:
num_sample_per_batch_1 = int(len(train_dataset_drug_induced_gene_exp)*fraction)
num_sample_per_batch_2 = int(len(train_dataset_drug_protein_binding)*fraction)
num_sample_per_batch_3 = int(len(train_dataset_TF_gene_regulation)*fraction)
num_sample_per_batch_4 = int(len(train_dataset_drug_sensitivity)*fraction)
num_sample_per_batch_5 = int(len(train_dataset_gene_effect_score)*fraction)
num_sample_per_batch_6 = int(len(train_dataset_gene_mutation)*fraction)
num_sample_per_batch_7 = int(len(train_dataset_gene_CNV)*fraction)

train_loader_1 = DataLoader(train_dataset_drug_induced_gene_exp, batch_size = num_sample_per_batch_1, shuffle=True)
train_loader_2 = DataLoader(train_dataset_drug_protein_binding, batch_size = num_sample_per_batch_2, shuffle=True)
train_loader_3 = DataLoader(train_dataset_TF_gene_regulation, batch_size = num_sample_per_batch_3, shuffle=True)
train_loader_4 = DataLoader(train_dataset_drug_sensitivity, batch_size = num_sample_per_batch_4, shuffle=True)
train_loader_5 = DataLoader(train_dataset_gene_effect_score, batch_size = num_sample_per_batch_5, shuffle=True)
train_loader_6 = DataLoader(train_dataset_gene_mutation, batch_size = num_sample_per_batch_6, shuffle=True)
train_loader_7 = DataLoader(train_dataset_gene_CNV, batch_size = num_sample_per_batch_7, shuffle=True)

#validation set:
fraction_val = 1/5000
num_sample_per_batch_val_1 = int(len(val_dataset_drug_induced_gene_exp)*fraction_val)
num_sample_per_batch_val_2 = int(len(val_dataset_drug_protein_binding)*fraction_val)
num_sample_per_batch_val_3 = int(len(val_dataset_TF_gene_regulation)*fraction_val)
num_sample_per_batch_val_4 = int(len(val_dataset_drug_sensitivity)*fraction_val)
num_sample_per_batch_val_5 = int(len(val_dataset_gene_effect_score)*fraction_val)
num_sample_per_batch_val_6 = int(len(val_dataset_gene_mutation)*fraction_val)
num_sample_per_batch_val_7 = int(len(val_dataset_gene_CNV)*fraction_val)

val_loader_1 = DataLoader(val_dataset_drug_induced_gene_exp, batch_size = num_sample_per_batch_val_1, shuffle = False)
val_loader_2 = DataLoader(val_dataset_drug_protein_binding, batch_size = num_sample_per_batch_val_2, shuffle = False)
val_loader_3 = DataLoader(val_dataset_TF_gene_regulation, batch_size = num_sample_per_batch_val_3, shuffle = False)
val_loader_4 = DataLoader(val_dataset_drug_sensitivity, batch_size = num_sample_per_batch_val_4, shuffle = False)
val_loader_5 = DataLoader(val_dataset_gene_effect_score, batch_size = num_sample_per_batch_val_5, shuffle = False)
val_loader_6 = DataLoader(val_dataset_gene_mutation, batch_size = num_sample_per_batch_val_6, shuffle = False)
val_loader_7 = DataLoader(val_dataset_gene_CNV, batch_size = num_sample_per_batch_val_7, shuffle = False)

#---------------------------------------------------------------------------------------
#CLS token:
seed_everything(2000)
CLS = nn.Embedding(1, 768) 
CLS = CLS(torch.LongTensor([0]))

#---------------------------------------------------------------------------------------
# Initialize model, optimizer, and loss function
seed_everything(rdseed)
model = Model().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
loss_regression = nn.MSELoss()
loss_classification = nn.BCELoss() 

#Record training time:
torch.cuda.synchronize() # wait for warm-up to finish
times = []

# Init early stop
patience = 5
best_score=None
early_stop=pd.Series([False]*7, index = ['drug-induced_gene_exp', 'drug-protein_binding', 'TF-gene_regulation', 'drug_sensitivity', 'gene_effect_score', 'gene_mutation', 'gene_CNV'])
estop_counter=pd.Series(np.zeros(7), index = ['drug-induced_gene_exp', 'drug-protein_binding', 'TF-gene_regulation', 'drug_sensitivity', 'gene_effect_score', 'gene_mutation', 'gene_CNV'])

val_loss=pd.DataFrame(np.zeros(shape = (7, epoch)),index=['drug-induced_gene_exp', 'drug-protein_binding', 'TF-gene_regulation', 'drug_sensitivity', 'gene_effect_score', 'gene_mutation', 'gene_CNV'])

#---------------------------------------------------------------------------------------
#Run model:
for e in range(epoch):
    # Internal validation:      
    model.eval()
    val_outputs_1 = []
    val_outputs_2 = []
    val_outputs_3 = []
    val_outputs_4 = []
    val_outputs_5 = []
    val_outputs_6 = []
    val_outputs_7 = []
    count = 0
    for batch_1, batch_2, batch_3, batch_4, batch_5, batch_6, batch_7 in zip(val_loader_1, val_loader_2, val_loader_3, val_loader_4, val_loader_5, val_loader_6, val_loader_7):
        #
        #task 1: 'drug-induced_gene_exp':
        input_data_1 = batch_1['input_data']
        labels_1 = batch_1['labels']
        task_id_1 = batch_1['task_id']
        #
        #task 2: 'drug-protein_binding':
        input_data_2 = batch_2['input_data']
        labels_2 = batch_2['labels']
        task_id_2 = batch_2['task_id']
        #
        #task 3: 'TF-gene_regulation':
        input_data_3 = batch_3['input_data']
        labels_3 = batch_3['labels']
        task_id_3 = batch_3['task_id']
        #
        #task 4: 'drug_sensitivity':
        input_data_4 = batch_4['input_data']
        labels_4 = batch_4['labels']
        task_id_4 = batch_4['task_id']
        #
        #task 5: 'gene_effect_score':
        input_data_5 = batch_5['input_data']
        labels_5 = batch_5['labels']
        task_id_5 = batch_5['task_id']
        #
        #task 6: 'gene_mutation':
        input_data_6 = batch_6['input_data']
        labels_6 = batch_6['labels']
        task_id_6 = batch_6['task_id']
        #
        #task 7: 'gene_CNV':
        input_data_7 = batch_7['input_data']
        labels_7 = batch_7['labels']
        task_id_7 = batch_7['task_id']
        #
        outputs_1, outputs_2, outputs_3, outputs_4, outputs_5, outputs_6, outputs_7 = model(inputs_embeds = [input_data_1.to(device), input_data_2.to(device), input_data_3.to(device), input_data_4.to(device), input_data_5.to(device), input_data_6.to(device), input_data_7.to(device)], 
                          input_task_id = [task_id_1, task_id_2, task_id_3, task_id_4, task_id_5, task_id_6, task_id_7], 
                          CLS_token = CLS.to(device))
        #
        outputs_1 = outputs_1.to('cpu').view(-1).detach().numpy().flatten().tolist()
        outputs_2 = outputs_2.to('cpu').view(-1).detach().numpy().flatten().tolist()
        outputs_3 = outputs_3.to('cpu').view(-1).detach().numpy().flatten().tolist()
        outputs_4 = outputs_4.to('cpu').view(-1).detach().numpy().flatten().tolist()
        outputs_5 = outputs_5.to('cpu').view(-1).detach().numpy().flatten().tolist()
        outputs_6 = outputs_6.to('cpu').view(-1).detach().numpy().flatten().tolist()
        outputs_7 = outputs_7.to('cpu').view(-1).detach().numpy().flatten().tolist()
        val_outputs_1.extend(outputs_1)
        val_outputs_2.extend(outputs_2)
        val_outputs_3.extend(outputs_3)
        val_outputs_4.extend(outputs_4)
        val_outputs_5.extend(outputs_5)
        val_outputs_6.extend(outputs_6)
        val_outputs_7.extend(outputs_7)
        count = count + 1
        print(count)
    #
    #task 1: 'drug-induced_gene_exp':
    val_loss_e_1 = loss_regression(torch.tensor(val_outputs_1).view(-1), Y_val_drug_induced_gene_exp[:len(val_outputs_1)].view(-1))
    val_loss_e_1 = val_loss_e_1.to('cpu').detach().flatten().numpy()[0]
    df = pd.DataFrame({"y_pred":val_outputs_1,'y_truth':Y_val_drug_induced_gene_exp[:len(val_outputs_1)].detach().numpy().flatten()})
    df.to_csv(out_dir + 'self-built_transformer_multi-task_drug_induced_gene_exp_use_GNN_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250210.csv')
    rmse = np.sqrt(mean_squared_error(df['y_truth'], df['y_pred']))
    cor = df.corr().iloc[0,1]
    all_val_set_performance = pd.DataFrame({'pearson_cor': cor, 'RMSE': rmse}, index = range(0, 1))
    all_val_set_performance.to_csv(out_dir + 'self-built_transformer_multi-task_drug_induced_gene_exp_use_GNN_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_performance_20250210.csv')
    all_val_set_loss = pd.DataFrame({'val_loss': val_loss_e_1}, index = range(0, 1))
    all_val_set_loss.to_csv(out_dir + 'self-built_transformer_multi-task_drug_induced_gene_exp_use_GNN_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_loss_20250210.csv')
    #
    #task 2: 'drug-protein_binding':
    val_loss_e_2 = loss_regression(torch.tensor(val_outputs_2).view(-1), Y_val_drug_protein_binding[:len(val_outputs_2)].view(-1))
    val_loss_e_2 = val_loss_e_2.to('cpu').detach().flatten().numpy()[0]
    df = pd.DataFrame({"y_pred":val_outputs_2,'y_truth':Y_val_drug_protein_binding[:len(val_outputs_2)].detach().numpy().flatten()})
    df.to_csv(out_dir + 'self-built_transformer_multi-task_drug_protein_binding_use_GNN_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250210.csv')
    rmse = np.sqrt(mean_squared_error(df['y_truth'], df['y_pred']))
    cor = df.corr().iloc[0,1]
    all_val_set_performance = pd.DataFrame({'pearson_cor': cor, 'RMSE': rmse}, index = range(0, 1))
    all_val_set_performance.to_csv(out_dir + 'self-built_transformer_multi-task_drug_protein_binding_use_GNN_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_performance_20250210.csv')
    all_val_set_loss = pd.DataFrame({'val_loss': val_loss_e_2}, index = range(0, 1))
    all_val_set_loss.to_csv(out_dir + 'self-built_transformer_multi-task_drug_protein_binding_use_GNN_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_loss_20250210.csv')
    #
    #task 3: 'TF-gene_regulation':
    val_loss_e_3 = loss_classification(torch.tensor(val_outputs_3).view(-1), torch.tensor(Y_val_TF_gene_regulation, dtype=torch.float32)[:len(val_outputs_3)].view(-1))
    val_loss_e_3 = val_loss_e_3.to('cpu').detach().flatten().numpy()[0]
    df = pd.DataFrame({"y_pred":val_outputs_3,'y_truth':torch.tensor(Y_val_TF_gene_regulation, dtype=torch.float32)[:len(val_outputs_3)].detach().numpy().flatten()})
    df.to_csv(out_dir + 'self-built_transformer_multi-task_TF_gene_regulation_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250210.csv')
    AUROC = roc_auc_score(df['y_truth'], df['y_pred']) #AUROC
    AUPRC = average_precision_score(df['y_truth'], df['y_pred']) #AUPRC
    df['y_pred_class'] = 0
    df.loc[df['y_pred'] > 0.5, 'y_pred_class'] = 1
    f1 = f1_score(df['y_truth'], df['y_pred_class'])
    acc = accuracy_score(df['y_truth'], df['y_pred_class'])
    all_val_set_performance = pd.DataFrame({'AUROC': AUROC, 'AUPRC': AUPRC, 'f1': f1, 'accuracy': acc}, index = range(0, 1))
    all_val_set_performance.to_csv(out_dir + 'self-built_transformer_multi-task_TF_gene_regulation_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_performance_20250210.csv')
    all_val_set_loss = pd.DataFrame({'val_loss': val_loss_e_3}, index = range(0, 1))
    all_val_set_loss.to_csv(out_dir + 'self-built_transformer_multi-task_TF_gene_regulation_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_loss_20250210.csv')
    #
    #task 4: 'drug_sensitivity':
    val_loss_e_4 = loss_regression(torch.tensor(val_outputs_4).view(-1), Y_val_drug_sensitivity[:len(val_outputs_4)].view(-1))
    val_loss_e_4 = val_loss_e_4.to('cpu').detach().flatten().numpy()[0]
    df = pd.DataFrame({"y_pred":val_outputs_4,'y_truth':Y_val_drug_sensitivity[:len(val_outputs_4)].detach().numpy().flatten()})
    df.to_csv(out_dir + 'self-built_transformer_multi-task_drug_sensitivity_use_GNN_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250210.csv')
    rmse = np.sqrt(mean_squared_error(df['y_truth'], df['y_pred']))
    cor = df.corr().iloc[0,1]
    all_val_set_performance = pd.DataFrame({'pearson_cor': cor, 'RMSE': rmse}, index = range(0, 1))
    all_val_set_performance.to_csv(out_dir + 'self-built_transformer_multi-task_drug_sensitivity_use_GNN_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_performance_20250210.csv')
    all_val_set_loss = pd.DataFrame({'val_loss': val_loss_e_4}, index = range(0, 1))
    all_val_set_loss.to_csv(out_dir + 'self-built_transformer_multi-task_drug_sensitivity_use_GNN_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_loss_20250210.csv')
    #
    #task 5: 'gene_effect_score':
    val_loss_e_5 = loss_regression(torch.tensor(val_outputs_5).view(-1), Y_val_gene_effect_score[:len(val_outputs_5)].view(-1))
    val_loss_e_5 = val_loss_e_5.to('cpu').detach().flatten().numpy()[0]
    df = pd.DataFrame({"y_pred":val_outputs_5,'y_truth':Y_val_gene_effect_score[:len(val_outputs_5)].detach().numpy().flatten()})
    df.to_csv(out_dir + 'self-built_transformer_multi-task_gene_effect_score_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250210.csv')
    rmse = np.sqrt(mean_squared_error(df['y_truth'], df['y_pred']))
    cor = df.corr().iloc[0,1]
    all_val_set_performance = pd.DataFrame({'pearson_cor': cor, 'RMSE': rmse}, index = range(0, 1))
    all_val_set_performance.to_csv(out_dir + 'self-built_transformer_multi-task_gene_effect_score_use_GNN_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_performance_20250210.csv')
    all_val_set_loss = pd.DataFrame({'val_loss': val_loss_e_5}, index = range(0, 1))
    all_val_set_loss.to_csv(out_dir + 'self-built_transformer_multi-task_gene_effect_score_use_GNN_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_loss_20250210.csv')
    #
    #task 6: 'gene_mutation':
    val_loss_e_6 = loss_classification(torch.tensor(val_outputs_6).view(-1), torch.tensor(Y_val_gene_mutation, dtype=torch.float32)[:len(val_outputs_6)].view(-1))
    val_loss_e_6 = val_loss_e_6.to('cpu').detach().flatten().numpy()[0]
    df = pd.DataFrame({"y_pred":val_outputs_6,'y_truth':torch.tensor(Y_val_gene_mutation, dtype=torch.float32)[:len(val_outputs_6)].detach().numpy().flatten()})
    df.to_csv(out_dir + 'self-built_transformer_multi-task_gene_mutation_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250210.csv')
    AUROC = roc_auc_score(df['y_truth'], df['y_pred']) #AUROC
    AUPRC = average_precision_score(df['y_truth'], df['y_pred']) #AUPRC
    df['y_pred_class'] = 0
    df.loc[df['y_pred'] > 0.5, 'y_pred_class'] = 1
    f1 = f1_score(df['y_truth'], df['y_pred_class'])
    acc = accuracy_score(df['y_truth'], df['y_pred_class'])
    all_val_set_performance = pd.DataFrame({'AUROC': AUROC, 'AUPRC': AUPRC, 'f1': f1, 'accuracy': acc}, index = range(0, 1))
    all_val_set_performance.to_csv(out_dir + 'self-built_transformer_multi-task_gene_mutation_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_performance_20250210.csv')
    all_val_set_loss = pd.DataFrame({'val_loss': val_loss_e_6}, index = range(0, 1))
    all_val_set_loss.to_csv(out_dir + 'self-built_transformer_multi-task_TF_gene_mutation_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_loss_20250210.csv')
    #
    #task 7: 'gene_CNV':
    val_loss_e_7 = loss_regression(torch.tensor(val_outputs_7).view(-1), Y_val_gene_CNV[:len(val_outputs_7)].view(-1))
    val_loss_e_7 = val_loss_e_7.to('cpu').detach().flatten().numpy()[0]
    df = pd.DataFrame({"y_pred":val_outputs_7,'y_truth':Y_val_gene_CNV[:len(val_outputs_7)].detach().numpy().flatten()})
    df.to_csv(out_dir + 'self-built_transformer_multi-task_gene_CNV_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250210.csv')
    rmse = np.sqrt(mean_squared_error(df['y_truth'], df['y_pred']))
    cor = df.corr().iloc[0,1]
    all_val_set_performance = pd.DataFrame({'pearson_cor': cor, 'RMSE': rmse}, index = range(0, 1))
    all_val_set_performance.to_csv(out_dir + 'self-built_transformer_multi-task_gene_CNV_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_performance_20250210.csv')
    all_val_set_loss = pd.DataFrame({'val_loss': val_loss_e_7}, index = range(0, 1))
    all_val_set_loss.to_csv(out_dir + 'self-built_transformer_multi-task_gene_CNV_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_internal_seen_unseen_mixture_val_set_loss_20250210.csv')
    #
    val_loss.iloc[0, e]=val_loss_e_1 
    val_loss.iloc[1, e]=val_loss_e_2
    val_loss.iloc[2, e]=val_loss_e_3
    val_loss.iloc[3, e]=val_loss_e_4
    val_loss.iloc[4, e]=val_loss_e_5
    val_loss.iloc[5, e]=val_loss_e_6
    val_loss.iloc[6, e]=val_loss_e_7
    #
    if best_score is None:
        best_score = val_loss.iloc[:, e].tolist()
    else:
        for i in range(0, 7):
            if val_loss.iloc[i, e] > (best_score[i]-0.001) and (not early_stop[i]):
                estop_counter[i] += 1
                if estop_counter[i] >= patience:
                    early_stop[i] = True
            else:
                best_score[i] = val_loss.iloc[i, e]
                estop_counter[i] = 0
    #
    if all(early_stop):
        break
    #
    # Training loop:
    epoch_loss_1 = []
    epoch_loss_2 = []
    epoch_loss_3 = []
    epoch_loss_4 = []
    epoch_loss_5 = []
    epoch_loss_6 = []
    epoch_loss_7 = []
    epoch_loss_all = []
    #
    model.train()
    count = 0
    #
    start_epoch = time.time()
    #
    for batch_1, batch_2, batch_3, batch_4, batch_5, batch_6, batch_7 in zip(train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5, train_loader_6, train_loader_7):
        #
        #task 1: 'drug-induced_gene_exp':
        input_data_1 = batch_1['input_data']
        labels_1 = batch_1['labels']
        task_id_1 = batch_1['task_id']
        #
        #task 2: 'drug-protein_binding':
        input_data_2 = batch_2['input_data']
        labels_2 = batch_2['labels']
        task_id_2 = batch_2['task_id']
        #
        #task 3: 'TF-gene_regulation':
        input_data_3 = batch_3['input_data']
        labels_3 = batch_3['labels']
        task_id_3 = batch_3['task_id']
        #
        #task 4: 'drug_sensitivity':
        input_data_4 = batch_4['input_data']
        labels_4 = batch_4['labels']
        task_id_4 = batch_4['task_id']
        #
        #task 5: 'gene_effect_score':
        input_data_5 = batch_5['input_data']
        labels_5 = batch_5['labels']
        task_id_5 = batch_5['task_id']
        #
        #task 6: 'gene_mutation':
        input_data_6 = batch_6['input_data']
        labels_6 = batch_6['labels']
        task_id_6 = batch_6['task_id']
        #
        #task 7: 'gene_CNV':
        input_data_7 = batch_7['input_data']
        labels_7 = batch_7['labels']
        task_id_7 = batch_7['task_id']
        #
        optimizer.zero_grad()
        #
        outputs_1, outputs_2, outputs_3, outputs_4, outputs_5, outputs_6, outputs_7 = model(inputs_embeds = [input_data_1.to(device), input_data_2.to(device), input_data_3.to(device), input_data_4.to(device), input_data_5.to(device), input_data_6.to(device), input_data_7.to(device)], 
                          input_task_id = [task_id_1, task_id_2, task_id_3, task_id_4, task_id_5, task_id_6, task_id_7], 
                          CLS_token = CLS.to(device))
        #
        loss_1 = loss_regression(outputs_1.view(-1), labels_1.to(device).view(-1))
        loss_2 = loss_regression(outputs_2.view(-1), labels_2.to(device).view(-1))
        loss_3 = loss_classification(outputs_3.view(-1), labels_3.to(device).view(-1))
        loss_4 = loss_regression(outputs_4.view(-1), labels_4.to(device).view(-1))
        loss_5 = loss_regression(outputs_5.view(-1), labels_5.to(device).view(-1))
        loss_6 = loss_classification(outputs_6.view(-1), labels_6.to(device).view(-1))
        loss_7 = loss_regression(outputs_7.view(-1), labels_7.to(device).view(-1))
        #weight the loss to around 1:
        loss = (loss_1) + (loss_2) + (loss_3) + (loss_4) + (loss_5) + (loss_6) + (loss_7)
        #print(f"loss_1: {loss_1}, loss_2: {loss_2}, loss_3: {loss_3}, loss_4: {loss_4}, loss_5: {loss_5}, loss_6: {loss_6}, loss_7: {loss_7}")
        #print(f"total_loss: {loss}")
        loss.backward(retain_graph=True)
        optimizer.step()
        count = count + 1
        if count % 100 == 99:
            print(f"loss_1: {loss_1}, loss_2: {loss_2}, loss_3: {loss_3}, loss_4: {loss_4}, loss_5: {loss_5}, loss_6: {loss_6}, loss_7: {loss_7}")
            print(f'Epoch {e + 1}/{epoch}, total_loss: {loss.item()}')
        epoch_loss_1.append(loss_1.to('cpu').detach().flatten().numpy()[0])
        epoch_loss_2.append(loss_2.to('cpu').detach().flatten().numpy()[0])
        epoch_loss_3.append(loss_3.to('cpu').detach().flatten().numpy()[0])
        epoch_loss_4.append(loss_4.to('cpu').detach().flatten().numpy()[0])
        epoch_loss_5.append(loss_5.to('cpu').detach().flatten().numpy()[0])
        epoch_loss_6.append(loss_6.to('cpu').detach().flatten().numpy()[0])
        epoch_loss_7.append(loss_7.to('cpu').detach().flatten().numpy()[0])
        epoch_loss_all.append(loss.to('cpu').detach().flatten().numpy()[0])
    #
    torch.cuda.synchronize()
    end_epoch = time.time()
    elapsed = end_epoch - start_epoch
    times.append(elapsed)
    #
    # Save the model:
    torch.save(model.state_dict(), out_dir + 'self-built_transformer_multi-task_all_7_tasks_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_20250210')
    # Save training loss:
    df = pd.DataFrame({'epoch_loss_1': epoch_loss_1, 'epoch_loss_2': epoch_loss_2, 'epoch_loss_3': epoch_loss_3, 'epoch_loss_4': epoch_loss_4, 'epoch_loss_5': epoch_loss_5, 'epoch_loss_6': epoch_loss_6, 'epoch_loss_7': epoch_loss_7, 'epoch_total_loss': epoch_loss_all}, index = range(0, len(epoch_loss_all)))
    df.to_csv(out_dir + 'training_loss_self-built_transformer_multi-task_all_7_tasks_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_epoch_' + str(e) + '_20250210.csv')

#Save the final model:
torch.save(model.state_dict(), out_dir + 'self-built_transformer_multi-task_all_7_tasks_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_final_20250210')

#Save training time:
pd.DataFrame(times).to_csv(out_dir + 'self-built_transformer_multi-task_all_7_tasks_lr0.00001_GELU_subset_0.1_portion_BatchNum_5000_equal_weighted_loss_training_time_per_epoch_20250210.csv')
    
#------------------------------------------------------------
#-----------------------------------------------------------
#draft:
