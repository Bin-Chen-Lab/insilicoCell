import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import numpy as np
import pandas as pd
import random
import os
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, f1_score, accuracy_score



device = torch.device( f'cuda:{4}' if torch.cuda.is_available() else 'cpu')

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
        embeddings = inputs_embeds[0]
        task_id = input_task_id[0]
        sample_size = embeddings.shape[0]
        #
        #task_id == 'drug-induced_gene_exp', #order of modality: drug (600), cellline (128), gene (128), time (1), dose (1)
        if task_id[0] == 'drug-induced_gene_exp':
            drug, cellline, gene, time, dose = embeddings[:, :600], embeddings[:, 600:728], embeddings[:, 728:856], embeddings[:, 856], embeddings[:, 857]
            all_drug = drug
            all_cellline = cellline
            all_gene = gene
            all_time = time
            all_dose = dose
            #
            all_drug = self.embedding_drug2(F.gelu(self.embedding_drug1(all_drug)))
            all_cellline = self.embedding_cellline2(F.gelu(self.embedding_cellline1(all_cellline)))
            all_gene = self.embedding_gene2(F.gelu(self.embedding_gene1(all_gene)))
            all_time = self.embedding_time1((all_time/48).unsqueeze(1))
            all_dose = self.embedding_dose1((all_dose/20).unsqueeze(1))
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
            # Add CLS token & padding tokens, concatenate all tokens together for each task:
            embeddings = torch.cat((CLS_token.repeat(sample_size, 1), all_drug[:sample_size, :], all_gene[:sample_size, :], all_cellline[:sample_size, :], all_time[:sample_size, :], all_dose[:sample_size, :]), dim=1)
            embeddings = embeddings.reshape(sample_size, 6, self.hidden_size) #torch.Size([128, 6, 768])
            # Add attention mask:
            attention_mask = (torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)).repeat(sample_size, 1).to(device)
        #
        #task_id == 'drug-protein_binding': #order of modality: drug (600), protein (1024)
        if task_id[0] == 'drug-protein_binding':
            drug, protein = embeddings[:, :600], embeddings[:, 600:]
            all_drug = drug
            all_protein = protein
            #
            all_drug = self.embedding_drug2(F.gelu(self.embedding_drug1(all_drug)))
            all_protein = self.embedding_protein2(F.gelu(self.embedding_protein1(all_protein)))
            #
            # Normalize + dropout:
            all_drug = self.LayerNorm_drug(all_drug)
            all_drug = self.dropout_drug(all_drug)
            all_protein = self.LayerNorm_protein(all_protein)
            all_protein = self.dropout_protein(all_protein)
            # Add CLS token & padding tokens, concatenate all tokens together for each task:
            embeddings = torch.cat((CLS_token.repeat(sample_size, 1), all_drug[:sample_size, :], all_protein[:sample_size, :]), dim=1)
            embeddings = embeddings.reshape(sample_size, 3, self.hidden_size) #torch.Size([128, 3, 768])
            # Add attention mask:
            attention_mask = (torch.tensor([0, 0, 0], dtype=torch.float32)).repeat(sample_size, 1).to(device)
        #
        #task_id == 'TF-gene_regulation': #order of modality: gene (128), protein (1024)
        if task_id[0] == 'TF-gene_regulation':
            gene, protein = embeddings[:, :128], embeddings[:, 128:]
            all_gene = gene
            all_protein = protein
            #
            all_gene = self.embedding_gene2(F.gelu(self.embedding_gene1(all_gene)))
            all_protein = self.embedding_protein2(F.gelu(self.embedding_protein1(all_protein)))
            #
            # Normalize + dropout:
            all_gene = self.LayerNorm_gene(all_gene)
            all_gene = self.dropout_gene(all_gene)
            all_protein = self.LayerNorm_protein(all_protein)
            all_protein = self.dropout_protein(all_protein)
            # Add CLS token & padding tokens, concatenate all tokens together for each task:
            embeddings = torch.cat((CLS_token.repeat(sample_size, 1), all_gene[:sample_size, :], all_protein[:sample_size, :]), dim=1)
            embeddings = embeddings.reshape(sample_size, 3, self.hidden_size) #torch.Size([128, 3, 768])
            # Add attention mask:
            attention_mask = (torch.tensor([0, 0, 0], dtype=torch.float32)).repeat(sample_size, 1).to(device)
        #
        #task_id == 'drug_sensitivity': #order of modality: drug (600), cellline (128)
        if task_id[0] == 'drug_sensitivity':
            drug, cellline = embeddings[:, :600], embeddings[:, 600:]
            all_drug = drug
            all_cellline = cellline
            #
            all_drug = self.embedding_drug2(F.gelu(self.embedding_drug1(all_drug)))
            all_cellline = self.embedding_cellline2(F.gelu(self.embedding_cellline1(all_cellline)))
            #
            # Normalize + dropout:
            all_drug = self.LayerNorm_drug(all_drug)
            all_drug = self.dropout_drug(all_drug)
            all_cellline = self.LayerNorm_cellline(all_cellline)
            all_cellline = self.dropout_cellline(all_cellline)
            # Add CLS token & padding tokens, concatenate all tokens together for each task:
            embeddings = torch.cat((CLS_token.repeat(sample_size, 1), all_drug[:sample_size, :], all_cellline[:sample_size, :]), dim=1)
            embeddings = embeddings.reshape(sample_size, 3, self.hidden_size) #torch.Size([128, 3, 768])
            # Add attention mask:
            attention_mask = (torch.tensor([0, 0, 0], dtype=torch.float32)).repeat(sample_size, 1).to(device)
        #
        #task_id == 'gene_effect_score': #order of modality: cellline (128), gene (128)
        if task_id[0] == 'gene_effect_score':
            cellline, gene = embeddings[:, :128], embeddings[:, 128:]
            all_cellline = cellline
            all_gene = gene
            #
            all_cellline = self.embedding_cellline2(F.gelu(self.embedding_cellline1(all_cellline)))
            all_gene = self.embedding_gene2(F.gelu(self.embedding_gene1(all_gene)))
            #
            # Normalize + dropout:
            all_gene = self.LayerNorm_gene(all_gene)
            all_gene = self.dropout_gene(all_gene)
            all_cellline = self.LayerNorm_cellline(all_cellline)
            all_cellline = self.dropout_cellline(all_cellline)
            # Add CLS token & padding tokens, concatenate all tokens together for each task:
            embeddings = torch.cat((CLS_token.repeat(sample_size, 1), all_cellline[:sample_size, :], all_gene[:sample_size, :]), dim=1)
            embeddings = embeddings.reshape(sample_size, 3, self.hidden_size) #torch.Size([128, 3, 768])
            # Add attention mask:
            attention_mask = (torch.tensor([0, 0, 0], dtype=torch.float32)).repeat(sample_size, 1).to(device)
        #
        #task_id == 'gene_mutation': #order of modality: cellline (128), gene (128)
        if task_id[0] == 'gene_mutation':
            cellline, gene = embeddings[:, :128], embeddings[:, 128:]
            all_cellline = cellline
            all_gene = gene
            #
            all_cellline = self.embedding_cellline2(F.gelu(self.embedding_cellline1(all_cellline)))
            all_gene = self.embedding_gene2(F.gelu(self.embedding_gene1(all_gene)))
            #
            # Normalize + dropout:
            all_gene = self.LayerNorm_gene(all_gene)
            all_gene = self.dropout_gene(all_gene)
            all_cellline = self.LayerNorm_cellline(all_cellline)
            all_cellline = self.dropout_cellline(all_cellline)
            # Add CLS token & padding tokens, concatenate all tokens together for each task:
            embeddings = torch.cat((CLS_token.repeat(sample_size, 1), all_cellline[:sample_size, :], all_gene[:sample_size, :]), dim=1)
            embeddings = embeddings.reshape(sample_size, 3, self.hidden_size) #torch.Size([128, 3, 768])
            # Add attention mask:
            attention_mask = (torch.tensor([0, 0, 0], dtype=torch.float32)).repeat(sample_size, 1).to(device)
        #
        #task_id == 'gene_CNV': #order of modality: cellline (128), gene (128)
        if task_id[0] == 'gene_CNV':
            cellline, gene = embeddings[:, :128], embeddings[:, 128:]
            all_cellline = cellline
            all_gene = gene
            #
            all_cellline = self.embedding_cellline2(F.gelu(self.embedding_cellline1(all_cellline)))
            all_gene = self.embedding_gene2(F.gelu(self.embedding_gene1(all_gene)))
            #
            # Normalize + dropout:
            all_gene = self.LayerNorm_gene(all_gene)
            all_gene = self.dropout_gene(all_gene)
            all_cellline = self.LayerNorm_cellline(all_cellline)
            all_cellline = self.dropout_cellline(all_cellline)
            # Add CLS token & padding tokens, concatenate all tokens together for each task:
            embeddings = torch.cat((CLS_token.repeat(sample_size, 1), all_cellline[:sample_size, :], all_gene[:sample_size, :]), dim=1)
            embeddings = embeddings.reshape(sample_size, 3, self.hidden_size) #torch.Size([128, 3, 768])
            # Add attention mask:
            attention_mask = (torch.tensor([0, 0, 0], dtype=torch.float32)).repeat(sample_size, 1).to(device)
        #
        # Pass through the encoder layers: 
        for layer in self.encoder_layers:
            embeddings = layer(embeddings, attention_mask)
        # Pool the output of the [CLS] token
        pooled_output = self.pooler(embeddings) #for classification tasks, consider using Sigmoid activation
        #
        #Split the data by task again:
        pooled_output = pooled_output[:sample_size, :]
        # Output layer for regression and classification tasks:
        if task_id[0] == 'drug-induced_gene_exp':
            out = self.regressor_1(pooled_output)
        #
        if task_id[0] == 'drug-protein_binding':
            out = self.regressor_2(pooled_output)
        #
        if task_id[0] == 'TF-gene_regulation':
            out = F.sigmoid(self.regressor_3(pooled_output))
        #
        if task_id[0] == 'drug_sensitivity':
            out = self.regressor_4(pooled_output)
        #
        if task_id[0] == 'gene_effect_score':
            out = self.regressor_5(pooled_output)
        #
        if task_id[0] == 'gene_mutation':
            out = F.sigmoid(self.regressor_6(pooled_output))
        #
        if task_id[0] == 'gene_CNV':
            out = self.regressor_7(pooled_output)
        return out

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
#import the 10% randomly holdout validation data for all 7 tasks:
#import the unseen test data for all 7 tasks:

#########################
#task 1: 'drug-induced_gene_exp'
#order of modality: drug, cellline, gene, time, dose
X_val_drug_induced_gene_exp = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_drug-induced_gene_exp_use_GNN.pt')
Y_val_drug_induced_gene_exp = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_drug-induced_gene_exp_use_GNN.pt')

X_test_drug_induced_gene_exp = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_unseen_task_drug-induced_gene_exp_use_GNN.pt')
Y_test_drug_induced_gene_exp = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_unseen_task_drug-induced_gene_exp_use_GNN.pt')

#########################
#task 2: 'drug-protein_binding'
#order of modality: drug, protein
X_val_drug_protein_binding = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_drug-protein_binding_use_GNN.pt')
Y_val_drug_protein_binding = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_drug-protein_binding_use_GNN.pt')

X_test_drug_protein_binding = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_unseen_task_drug-protein_binding_use_GNN.pt')
Y_test_drug_protein_binding = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_unseen_task_drug-protein_binding_use_GNN.pt')

#########################
#task 3: 'TF-gene_regulation'
#order of modality: gene, protein
X_val_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_TF-gene_regulation.pt')
Y_val_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_TF-gene_regulation.pt')

X_test_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_unseen_task_TF-gene_regulation.pt')
Y_test_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_unseen_task_TF-gene_regulation.pt')

#########################
#task 4: 'drug_sensitivity'
#order of modality: drug, cellline
X_val_drug_sensitivity = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_drug_sensitivity_use_GNN.pt')
Y_val_drug_sensitivity = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_drug_sensitivity_use_GNN.pt')

X_test_drug_sensitivity = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_unseen_task_drug_sensitivity_use_GNN.pt')
Y_test_drug_sensitivity = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_unseen_task_drug_sensitivity_use_GNN.pt')

#########################
#task 5: 'gene_effect_score'
#order of modality: cellline, gene
X_val_gene_effect_score = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_gene_effect_score.pt')
Y_val_gene_effect_score = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_gene_effect_score.pt')

X_test_gene_effect_score = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_unseen_task_gene_effect_score.pt')
Y_test_gene_effect_score = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_unseen_task_gene_effect_score.pt')

#########################
#task 6: 'gene_mutation'
#order of modality: cellline, gene
X_val_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_gene_mutation.pt')
Y_val_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_gene_mutation.pt')

X_test_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_unseen_task_gene_mutation.pt')
Y_test_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_unseen_task_gene_mutation.pt')

#########################
#task 7: 'gene_CNV'
#order of modality: cellline, gene
X_val_gene_CNV = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_seen_unseen_mixture_task_gene_CNV.pt')
Y_val_gene_CNV = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_seen_unseen_mixture_task_gene_CNV.pt')

X_test_gene_CNV = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_test_set_unseen_task_gene_CNV.pt')
Y_test_gene_CNV = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_test_set_unseen_task_gene_CNV.pt')

#----------------------------------------------------------------------------------------------------------------
# Create DataLoader
val_dataset_drug_induced_gene_exp = CustomDataset(X_val_drug_induced_gene_exp, Y_val_drug_induced_gene_exp, 
                                                   task_id = 'drug-induced_gene_exp'
                                                   )

test_dataset_drug_induced_gene_exp = CustomDataset(X_test_drug_induced_gene_exp, Y_test_drug_induced_gene_exp, 
                                                   task_id = 'drug-induced_gene_exp'
                                                   )

val_dataset_drug_protein_binding = CustomDataset(X_val_drug_protein_binding, Y_val_drug_protein_binding, 
                                                   task_id = 'drug-protein_binding'
                                                   )

test_dataset_drug_protein_binding = CustomDataset(X_test_drug_protein_binding, Y_test_drug_protein_binding, 
                                                   task_id = 'drug-protein_binding'
                                                   )

val_dataset_TF_gene_regulation = CustomDataset(X_val_TF_gene_regulation, Y_val_TF_gene_regulation, 
                                                   task_id = 'TF-gene_regulation'
                                                   )

test_dataset_TF_gene_regulation = CustomDataset(X_test_TF_gene_regulation, Y_test_TF_gene_regulation, 
                                                   task_id = 'TF-gene_regulation'
                                                   )

val_dataset_drug_sensitivity = CustomDataset(X_val_drug_sensitivity, Y_val_drug_sensitivity, 
                                               task_id = 'drug_sensitivity'
                                               )

test_dataset_drug_sensitivity = CustomDataset(X_test_drug_sensitivity, Y_test_drug_sensitivity, 
                                               task_id = 'drug_sensitivity'
                                               )

val_dataset_gene_effect_score = CustomDataset(X_val_gene_effect_score, Y_val_gene_effect_score, 
                                               task_id = 'gene_effect_score'
                                               )

test_dataset_gene_effect_score = CustomDataset(X_test_gene_effect_score, Y_test_gene_effect_score, 
                                               task_id = 'gene_effect_score'
                                               )

val_dataset_gene_mutation = CustomDataset(X_val_gene_mutation, Y_val_gene_mutation, 
                                               task_id = 'gene_mutation'
                                               )

test_dataset_gene_mutation = CustomDataset(X_test_gene_mutation, Y_test_gene_mutation, 
                                               task_id = 'gene_mutation'
                                               )

val_dataset_gene_CNV = CustomDataset(X_val_gene_CNV, Y_val_gene_CNV, 
                                               task_id = 'gene_CNV'
                                               )

test_dataset_gene_CNV = CustomDataset(X_test_gene_CNV, Y_test_gene_CNV, 
                                               task_id = 'gene_CNV'
                                               )
#---------------------------------------------------------------------------------------
#changeable parameters:
final_epoch = 9
rdseed = 10 
out_dir = '/egr/research-aidd/chenruo4/self-built_transformer/model_trial_1.0_portion_of_whole_training_data_20241204/'

#---------------------------------------------------------------------------------------
num_sample_per_batch = 2000

val_loader_1 = DataLoader(val_dataset_drug_induced_gene_exp, batch_size = num_sample_per_batch, shuffle = False)
test_loader_1 = DataLoader(test_dataset_drug_induced_gene_exp, batch_size = num_sample_per_batch, shuffle = False)

val_loader_2 = DataLoader(val_dataset_drug_protein_binding, batch_size = num_sample_per_batch, shuffle = False)
test_loader_2 = DataLoader(test_dataset_drug_protein_binding, batch_size = num_sample_per_batch, shuffle = False)

val_loader_3 = DataLoader(val_dataset_TF_gene_regulation, batch_size = num_sample_per_batch, shuffle = False)
test_loader_3 = DataLoader(test_dataset_TF_gene_regulation, batch_size = num_sample_per_batch, shuffle = False)

val_loader_4 = DataLoader(val_dataset_drug_sensitivity, batch_size = num_sample_per_batch, shuffle = False)
test_loader_4 = DataLoader(test_dataset_drug_sensitivity, batch_size = num_sample_per_batch, shuffle = False)

val_loader_5 = DataLoader(val_dataset_gene_effect_score, batch_size = num_sample_per_batch, shuffle = False)
test_loader_5 = DataLoader(test_dataset_gene_effect_score, batch_size = num_sample_per_batch, shuffle = False)

val_loader_6 = DataLoader(val_dataset_gene_mutation, batch_size = num_sample_per_batch, shuffle = False)
test_loader_6 = DataLoader(test_dataset_gene_mutation, batch_size = num_sample_per_batch, shuffle = False)

val_loader_7 = DataLoader(val_dataset_gene_CNV, batch_size = num_sample_per_batch, shuffle = False)
test_loader_7 = DataLoader(test_dataset_gene_CNV, batch_size = num_sample_per_batch, shuffle = False)

#---------------------------------------------------------------------------------------
#CLS token:
seed_everything(2000)
CLS = nn.Embedding(1, 768) 
CLS = CLS(torch.LongTensor([0]))

#---------------------------------------------------------------------------------------
# Load model:
seed_everything(rdseed)
model = Model().to(device)
model.load_state_dict(torch.load(out_dir + 'self-built_transformer_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_20241230'))

model.eval()

#---------------------------------------------------------------------------------------
# predict:
#task 1: 'drug-induced_gene_exp':
val_outputs = []
count = 0
#for batch in val_loader_1:
for batch in test_loader_1:
    input_data = batch['input_data']
    labels = batch['labels']
    task_id = batch['task_id']
    #
    outputs = model(inputs_embeds = [input_data.to(device)], 
                        input_task_id = [task_id], 
                        CLS_token = CLS.to(device))
    #
    outputs = outputs.to('cpu').view(-1).detach().numpy().flatten().tolist()
    val_outputs.extend(outputs)
    count = count + 1
    print(count)
#
#df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_val_drug_induced_gene_exp.detach().numpy().flatten()})
df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_test_drug_induced_gene_exp.detach().numpy().flatten()})
#df.to_csv(out_dir + 'self-built_transformer_drug-induced_gene_exp_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250102.csv')
df.to_csv(out_dir + 'self-built_transformer_drug-induced_gene_exp_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_pred_vs_truth_20250102.csv')
rmse = np.sqrt(mean_squared_error(df['y_truth'], df['y_pred']))
cor = df.corr().iloc[0,1]
all_val_set_performance = pd.DataFrame({'pearson_cor': cor, 'RMSE': rmse}, index = range(0, 1))
#all_val_set_performance.to_csv(out_dir + 'self-built_transformer_drug-induced_gene_exp_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_performance_20250102.csv')
all_val_set_performance.to_csv(out_dir + 'self-built_transformer_drug-induced_gene_exp_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_performance_20250102.csv')

#---------------------------------------------------------------------------------------
# predict:
#task 2: 'drug-protein_binding':
val_outputs = []
count = 0
#for batch in val_loader_2:
for batch in test_loader_2:
    input_data = batch['input_data']
    labels = batch['labels']
    task_id = batch['task_id']
    #
    outputs = model(inputs_embeds = [input_data.to(device)], 
                        input_task_id = [task_id], 
                        CLS_token = CLS.to(device))
    #
    outputs = outputs.to('cpu').view(-1).detach().numpy().flatten().tolist()
    val_outputs.extend(outputs)
    count = count + 1
    print(count)
#
#df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_val_drug_protein_binding.detach().numpy().flatten()})
df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_test_drug_protein_binding.detach().numpy().flatten()})
#df.to_csv(out_dir + 'self-built_transformer_drug-protein_binding_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250102.csv')
df.to_csv(out_dir + 'self-built_transformer_drug-protein_binding_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_pred_vs_truth_20250102.csv')
rmse = np.sqrt(mean_squared_error(df['y_truth'], df['y_pred']))
cor = df.corr().iloc[0,1]
all_val_set_performance = pd.DataFrame({'pearson_cor': cor, 'RMSE': rmse}, index = range(0, 1))
#all_val_set_performance.to_csv(out_dir + 'self-built_transformer_drug-protein_binding_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_performance_20250102.csv')
all_val_set_performance.to_csv(out_dir + 'self-built_transformer_drug-protein_binding_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_performance_20250102.csv')

#---------------------------------------------------------------------------------------
# predict:
#task 3: 'TF-gene_regulation'
val_outputs = []
count = 0
#for batch in val_loader_3:
for batch in test_loader_3:
    input_data = batch['input_data']
    labels = batch['labels']
    task_id = batch['task_id']
    #
    outputs = model(inputs_embeds = [input_data.to(device)], 
                        input_task_id = [task_id], 
                        CLS_token = CLS.to(device))
    #
    outputs = outputs.to('cpu').view(-1).detach().numpy().flatten().tolist()
    val_outputs.extend(outputs)
    count = count + 1
    print(count)
#
#df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_val_TF_gene_regulation.detach().numpy().flatten()})
df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_test_TF_gene_regulation.detach().numpy().flatten()})
#df.to_csv(out_dir + 'self-built_transformer_TF-gene_regulation_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250102.csv')
df.to_csv(out_dir + 'self-built_transformer_TF-gene_regulation_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_pred_vs_truth_20250102.csv')
AUROC = roc_auc_score(df['y_truth'], df['y_pred']) #AUROC
AUPRC = average_precision_score(df['y_truth'], df['y_pred']) #AUPRC
df['y_pred_class'] = 0
df.loc[df['y_pred'] > 0.5, 'y_pred_class'] = 1
f1 = f1_score(df['y_truth'], df['y_pred_class'])
acc = accuracy_score(df['y_truth'], df['y_pred_class'])
all_val_set_performance = pd.DataFrame({'AUROC': AUROC, 'AUPRC': AUPRC, 'f1': f1, 'accuracy': acc}, index = range(0, 1))
#all_val_set_performance.to_csv(out_dir + 'self-built_transformer_TF-gene_regulation_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_performance_20250102.csv')
all_val_set_performance.to_csv(out_dir + 'self-built_transformer_TF-gene_regulation_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_performance_20250102.csv')


#---------------------------------------------------------------------------------------
# predict:
#task 4: 'drug_sensitivity'
val_outputs = []
count = 0
#for batch in val_loader_4:
for batch in test_loader_4:
    input_data = batch['input_data']
    labels = batch['labels']
    task_id = batch['task_id']
    #
    outputs = model(inputs_embeds = [input_data.to(device)], 
                        input_task_id = [task_id], 
                        CLS_token = CLS.to(device))
    #
    outputs = outputs.to('cpu').view(-1).detach().numpy().flatten().tolist()
    val_outputs.extend(outputs)
    count = count + 1
    print(count)
#
#df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_val_drug_sensitivity.detach().numpy().flatten()})
df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_test_drug_sensitivity.detach().numpy().flatten()})
#df.to_csv(out_dir + 'self-built_transformer_drug_sensitivity_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250102.csv')
df.to_csv(out_dir + 'self-built_transformer_drug_sensitivity_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_pred_vs_truth_20250102.csv')
rmse = np.sqrt(mean_squared_error(df['y_truth'], df['y_pred']))
cor = df.corr().iloc[0,1]
all_val_set_performance = pd.DataFrame({'pearson_cor': cor, 'RMSE': rmse}, index = range(0, 1))
#all_val_set_performance.to_csv(out_dir + 'self-built_transformer_drug_sensitivity_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_performance_20250102.csv')
all_val_set_performance.to_csv(out_dir + 'self-built_transformer_drug_sensitivity_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_performance_20250102.csv')

#---------------------------------------------------------------------------------------
# predict:
#task 5: 'gene_effect_score'
val_outputs = []
count = 0
#for batch in val_loader_5:
for batch in test_loader_5:
    input_data = batch['input_data']
    labels = batch['labels']
    task_id = batch['task_id']
    #
    outputs = model(inputs_embeds = [input_data.to(device)], 
                        input_task_id = [task_id], 
                        CLS_token = CLS.to(device))
    #
    outputs = outputs.to('cpu').view(-1).detach().numpy().flatten().tolist()
    val_outputs.extend(outputs)
    count = count + 1
    print(count)
#
#df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_val_gene_effect_score.detach().numpy().flatten()})
df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_test_gene_effect_score.detach().numpy().flatten()})
#df.to_csv(out_dir + 'self-built_transformer_gene_effect_score_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250102.csv')
df.to_csv(out_dir + 'self-built_transformer_gene_effect_score_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_pred_vs_truth_20250102.csv')
rmse = np.sqrt(mean_squared_error(df['y_truth'], df['y_pred']))
cor = df.corr().iloc[0,1]
all_val_set_performance = pd.DataFrame({'pearson_cor': cor, 'RMSE': rmse}, index = range(0, 1))
#all_val_set_performance.to_csv(out_dir + 'self-built_transformer_gene_effect_score_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_performance_20250102.csv')
all_val_set_performance.to_csv(out_dir + 'self-built_transformer_gene_effect_score_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_performance_20250102.csv')

#---------------------------------------------------------------------------------------
# predict:
#task 6: 'gene_mutation'
val_outputs = []
count = 0
#for batch in val_loader_6:
for batch in test_loader_6:
    input_data = batch['input_data']
    labels = batch['labels']
    task_id = batch['task_id']
    #
    outputs = model(inputs_embeds = [input_data.to(device)], 
                        input_task_id = [task_id], 
                        CLS_token = CLS.to(device))
    #
    outputs = outputs.to('cpu').view(-1).detach().numpy().flatten().tolist()
    val_outputs.extend(outputs)
    count = count + 1
    print(count)
#
#df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_val_gene_mutation.detach().numpy().flatten()})
df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_test_gene_mutation.detach().numpy().flatten()})
#df.to_csv(out_dir + 'self-built_transformer_gene_mutation_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250102.csv')
df.to_csv(out_dir + 'self-built_transformer_gene_mutation_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_pred_vs_truth_20250102.csv')
AUROC = roc_auc_score(df['y_truth'], df['y_pred']) #AUROC
AUPRC = average_precision_score(df['y_truth'], df['y_pred']) #AUPRC
df['y_pred_class'] = 0
df.loc[df['y_pred'] > 0.5, 'y_pred_class'] = 1
f1 = f1_score(df['y_truth'], df['y_pred_class'])
acc = accuracy_score(df['y_truth'], df['y_pred_class'])
all_val_set_performance = pd.DataFrame({'AUROC': AUROC, 'AUPRC': AUPRC, 'f1': f1, 'accuracy': acc}, index = range(0, 1))
#all_val_set_performance.to_csv(out_dir + 'self-built_transformer_gene_mutation_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_performance_20250102.csv')
all_val_set_performance.to_csv(out_dir + 'self-built_transformer_gene_mutation_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_performance_20250102.csv')

#---------------------------------------------------------------------------------------
# predict:
#task 7: 'gene_CNV'
val_outputs = []
count = 0
#for batch in val_loader_7:
for batch in test_loader_7:
    input_data = batch['input_data']
    labels = batch['labels']
    task_id = batch['task_id']
    #
    outputs = model(inputs_embeds = [input_data.to(device)], 
                        input_task_id = [task_id], 
                        CLS_token = CLS.to(device))
    #
    outputs = outputs.to('cpu').view(-1).detach().numpy().flatten().tolist()
    val_outputs.extend(outputs)
    count = count + 1
    print(count)
#
#df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_val_gene_CNV.detach().numpy().flatten()})
df = pd.DataFrame({"y_pred":val_outputs,'y_truth':Y_test_gene_CNV.detach().numpy().flatten()})
#df.to_csv(out_dir + 'self-built_transformer_gene_CNV_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_pred_vs_truth_20250102.csv')
df.to_csv(out_dir + 'self-built_transformer_gene_CNV_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_pred_vs_truth_20250102.csv')
rmse = np.sqrt(mean_squared_error(df['y_truth'], df['y_pred']))
cor = df.corr().iloc[0,1]
all_val_set_performance = pd.DataFrame({'pearson_cor': cor, 'RMSE': rmse}, index = range(0, 1))
#all_val_set_performance.to_csv(out_dir + 'self-built_transformer_gene_CNV_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_internal_seen_unseen_mixture_val_set_performance_20250102.csv')
all_val_set_performance.to_csv(out_dir + 'self-built_transformer_gene_CNV_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(final_epoch) + '_epoch_unseen_test_set_performance_20250102.csv')



    
    




#------------------------------------------------------------
#-----------------------------------------------------------
#draft:
