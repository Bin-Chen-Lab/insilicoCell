#Author: Ruoqiao Chen

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd
import random
import os


device = torch.device( f'cuda:{3}' if torch.cuda.is_available() else 'cpu')

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
        self.embedding_time1 = nn.Linear(128, 384)
        self.embedding_time2 = nn.Linear(384, hidden_size)
        self.embedding_dose1 = nn.Linear(128, 384)
        self.embedding_dose2 = nn.Linear(384, hidden_size)
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
    def forward(self, inputs_embeds, input_task_id = None, modality_tokens = None, CLS_token = None):
        # Gather each modality's data from all tasks together:
        #task_id == 'drug-induced_gene_exp', #order of modality: drug (600), cellline (128), gene (128), time (128), dose (128)
        embeddings = inputs_embeds[0]
        task_id = input_task_id[0]
        sample_size = embeddings.shape[0]
        drug, cellline, gene, time, dose = embeddings[:, :600], embeddings[:, 600:728], embeddings[:, 728:856], embeddings[:, 856:984], embeddings[:, 984:]
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
        all_time = self.embedding_time2(F.gelu(self.embedding_time1(all_time)))
        all_dose = self.embedding_dose2(F.gelu(self.embedding_dose1(all_dose)))
        all_protein = self.embedding_protein2(F.gelu(self.embedding_protein1(all_protein)))
        #
        # Add modality tokens (order: drug/gene/cellline/protein/time/dose):
        all_drug = all_drug + modality_tokens[0]
        all_gene = all_gene + modality_tokens[1]
        all_cellline = all_cellline + modality_tokens[2]
        all_protein = all_protein + modality_tokens[3]
        all_time = all_time + modality_tokens[4]
        all_dose = all_dose + modality_tokens[5]
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
        #task_id == 'drug-induced_gene_exp', #order of modality: drug (600), cellline (128), gene (128), time (128), dose (128)
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
        self.input_X = input_X.float()  
        self.input_Y = input_Y.float()  
        self.task_id = task_id
    def __len__(self):
        return len(self.input_Y)
    def __getitem__(self, idx):
        input_data = self.input_X[idx]
        label = self.input_Y[idx]  
        task_id = self.task_id
        return {
            'input_data': input_data,  
            'labels': label,
            'task_id': task_id 
        }

#---------------------------------------------------------------------------------------
# Shuffle the dataset:
def shuffle_dataset_with_torch(dataset):
    # Generate a random permutation of indices
    indices = torch.randperm(len(dataset)).tolist()
    return torch.utils.data.Subset(dataset, indices)


#---------------------------------------------------------------------------------------
# Repeat the dataset to match the maximum sample size:
def repeat_samples(dataset, target_size):
    num_samples = len(dataset)
    if num_samples >= target_size:
        return dataset  
    # Repeat the dataset until it reaches the target size
    repeat_count = target_size // num_samples  
    remainder = target_size % num_samples     
    repeated_indices = list(range(num_samples)) * repeat_count + list(range(remainder))
    return Subset(dataset, repeated_indices)

#---------------------------------------------------------------------------------------
#import the whole training data for all 7 tasks:

#########################
#task 1: 'drug-induced_gene_exp'
#order of modality: drug, cellline, gene, time, dose
X_training_drug_induced_gene_exp = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_drug-induced_gene_exp.pt')
#torch.Size([22456548, 1112])
Y_training_drug_induced_gene_exp = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_drug-induced_gene_exp.pt')


#########################
#task 2: 'drug-protein_binding'
#order of modality: drug, protein
X_training_drug_protein_binding = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_drug-protein_binding.pt')
#torch.Size([1265596, 1624])
Y_training_drug_protein_binding = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_drug-protein_binding.pt')


#########################
#task 3: 'TF-gene_regulation'
#order of modality: gene, protein
X_training_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_TF-gene_regulation.pt')
#torch.Size([2774198, 1152])
Y_training_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_TF-gene_regulation.pt')


#########################
#task 4: 'drug_sensitivity'
#order of modality: drug, cellline
X_training_drug_sensitivity = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_drug_sensitivity.pt')
#torch.Size([276712, 728])
Y_training_drug_sensitivity = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_drug_sensitivity.pt')


#########################
#task 5: 'gene_effect_score'
#order of modality: cellline, gene
X_training_gene_effect_score = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_gene_effect_score.pt')
#torch.Size([8809512, 256])
Y_training_gene_effect_score = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_gene_effect_score.pt')


#########################
#task 6: 'gene_mutation'
#order of modality: cellline, gene
X_training_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_gene_mutation.pt')
#torch.Size([16743322, 256])
Y_training_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_gene_mutation.pt')


#########################
#task 7: 'gene_CNV'
#order of modality: cellline, gene
X_training_gene_CNV = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_gene_CNV.pt')
#torch.Size([17809611, 256])
Y_training_gene_CNV = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_gene_CNV.pt')

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

#----------------------------------------------------------------------------------------------------------------
#Repeat the samples for tasks with fewer sample sizes. Make the sample size the same for all 7 tasks:
target_size = max(len(train_dataset_drug_induced_gene_exp), len(train_dataset_drug_protein_binding),
                  len(train_dataset_TF_gene_regulation), len(train_dataset_drug_sensitivity),
                  len(train_dataset_gene_effect_score), len(train_dataset_gene_mutation), len(train_dataset_gene_CNV))

# Create the oversampled dataset
train_dataset_drug_induced_gene_exp = repeat_samples(train_dataset_drug_induced_gene_exp, target_size)
train_dataset_drug_protein_binding = repeat_samples(train_dataset_drug_protein_binding, target_size)
train_dataset_TF_gene_regulation = repeat_samples(train_dataset_TF_gene_regulation, target_size)
train_dataset_drug_sensitivity = repeat_samples(train_dataset_drug_sensitivity, target_size)
train_dataset_gene_effect_score = repeat_samples(train_dataset_gene_effect_score, target_size)
train_dataset_gene_mutation = repeat_samples(train_dataset_gene_mutation, target_size)
train_dataset_gene_CNV = repeat_samples(train_dataset_gene_CNV, target_size)

#---------------------------------------------------------------------------------------
#changeable parameters:
batch_size = 128
epoch = 200
rdseed = 10 
out_dir = '/egr/research-aidd/chenruo4/self-built_transformer/model_trial_1.0_portion_of_whole_training_data_20241204/'

#---------------------------------------------------------------------------------------
#Modality tokens for drug/gene/cellline/protein/time/dose:
seed_everything(1000)
modality_tokens = nn.Embedding(6, 768) # 6 maximum modalities 
modality_tokens = modality_tokens(torch.LongTensor([0, 1, 2, 3, 4, 5]))

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

# Training loop
for e in range(epoch):
    #shuffle the data:
    train_dataset_drug_induced_gene_exp = shuffle_dataset_with_torch(train_dataset_drug_induced_gene_exp)
    train_dataset_drug_protein_binding = shuffle_dataset_with_torch(train_dataset_drug_protein_binding)
    train_dataset_TF_gene_regulation = shuffle_dataset_with_torch(train_dataset_TF_gene_regulation)
    train_dataset_drug_sensitivity = shuffle_dataset_with_torch(train_dataset_drug_sensitivity)
    train_dataset_gene_effect_score = shuffle_dataset_with_torch(train_dataset_gene_effect_score)
    train_dataset_gene_mutation = shuffle_dataset_with_torch(train_dataset_gene_mutation)
    train_dataset_gene_CNV = shuffle_dataset_with_torch(train_dataset_gene_CNV)
    indices = list(range(len(train_dataset_drug_induced_gene_exp)))
    #
    model.train()
    count = 0
    #
    for i in range(0, len(train_dataset_drug_induced_gene_exp), batch_size):
        #
        #task 1: 'drug-induced_gene_exp':
        batch_indices = indices[i:i+batch_size]
        input_data_1 = torch.stack([train_dataset_drug_induced_gene_exp[idx]['input_data'] for idx in batch_indices])  #torch.Size([128, 1112])
        labels_1 = torch.tensor([train_dataset_drug_induced_gene_exp[idx]['labels'] for idx in batch_indices])  #torch.Size([128])
        task_id_1 = train_dataset_drug_induced_gene_exp[batch_indices[0]]['task_id']
        #
        #task 2: 'drug-protein_binding':
        input_data_2 = torch.stack([train_dataset_drug_protein_binding[idx]['input_data'] for idx in batch_indices])  #torch.Size([128, 1624])
        labels_2 = torch.tensor([train_dataset_drug_protein_binding[idx]['labels'] for idx in batch_indices])  #torch.Size([128])
        task_id_2 = train_dataset_drug_protein_binding[batch_indices[0]]['task_id'] #'drug-protein_binding'
        #
        #task 3: 'TF-gene_regulation':
        input_data_3 = torch.stack([train_dataset_TF_gene_regulation[idx]['input_data'] for idx in batch_indices])  
        labels_3 = torch.tensor([train_dataset_TF_gene_regulation[idx]['labels'] for idx in batch_indices])  
        task_id_3 = train_dataset_TF_gene_regulation[batch_indices[0]]['task_id']
        #
        #task 4: 'drug_sensitivity':
        input_data_4 = torch.stack([train_dataset_drug_sensitivity[idx]['input_data'] for idx in batch_indices])  
        labels_4 = torch.tensor([train_dataset_drug_sensitivity[idx]['labels'] for idx in batch_indices])  
        task_id_4 = train_dataset_drug_sensitivity[batch_indices[0]]['task_id']
        #
        #task 5: 'gene_effect_score':
        input_data_5 = torch.stack([train_dataset_gene_effect_score[idx]['input_data'] for idx in batch_indices])  
        labels_5 = torch.tensor([train_dataset_gene_effect_score[idx]['labels'] for idx in batch_indices])  
        task_id_5 = train_dataset_gene_effect_score[batch_indices[0]]['task_id']
        #
        #task 6: 'gene_mutation':
        input_data_6 = torch.stack([train_dataset_gene_mutation[idx]['input_data'] for idx in batch_indices])  
        labels_6 = torch.tensor([train_dataset_gene_mutation[idx]['labels'] for idx in batch_indices])  
        task_id_6 = train_dataset_gene_mutation[batch_indices[0]]['task_id']
        #
        #task 7: 'gene_CNV':
        input_data_7 = torch.stack([train_dataset_gene_CNV[idx]['input_data'] for idx in batch_indices])  
        labels_7 = torch.tensor([train_dataset_gene_CNV[idx]['labels'] for idx in batch_indices])  
        task_id_7 = train_dataset_gene_CNV[batch_indices[0]]['task_id']
        #
        optimizer.zero_grad()
        #
        outputs_1, outputs_2, outputs_3, outputs_4, outputs_5, outputs_6, outputs_7 = model(inputs_embeds = [input_data_1.to(device), input_data_2.to(device), input_data_3.to(device), input_data_4.to(device), input_data_5.to(device), input_data_6.to(device), input_data_7.to(device)], 
                          input_task_id = [task_id_1, task_id_2, task_id_3, task_id_4, task_id_5, task_id_6, task_id_7], 
                          modality_tokens = modality_tokens.to(device), 
                          CLS_token = CLS.to(device))
        #
        loss_1 = loss_regression(outputs_1.view(-1), labels_1.to(device).view(-1))
        loss_2 = loss_regression(outputs_2.view(-1), labels_2.to(device).view(-1))
        loss_3 = loss_classification(outputs_3.view(-1), labels_3.to(device).view(-1))
        loss_4 = loss_regression(outputs_4.view(-1), labels_4.to(device).view(-1))
        loss_5 = loss_regression(outputs_5.view(-1), labels_5.to(device).view(-1))
        loss_6 = loss_classification(outputs_6.view(-1), labels_6.to(device).view(-1))
        loss_7 = loss_regression(outputs_7.view(-1), labels_7.to(device).view(-1))
        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
        #print(f"loss_1: {loss_1}, loss_2: {loss_2}, loss_3: {loss_3}, loss_4: {loss_4}, loss_5: {loss_5}, loss_6: {loss_6}, loss_7: {loss_7}")
        #print(f"total_loss: {loss}")
        loss.backward(retain_graph=True)
        optimizer.step()
        count = count + 1
        if count % 100 == 99:
            print(f"loss_1: {loss_1}, loss_2: {loss_2}, loss_3: {loss_3}, loss_4: {loss_4}, loss_5: {loss_5}, loss_6: {loss_6}, loss_7: {loss_7}")
            print(f'Epoch {e + 1}/{epoch}, total_loss: {loss.item()}')
    # Save the model:
    torch.save(model.state_dict(), out_dir + 'self-built_transformer_multitask_lr0.00001_all_7_tasks_' + str(epoch) + '_20241220')
    
        

            




    
    




#------------------------------------------------------------
#-----------------------------------------------------------
#draft:
