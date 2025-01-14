import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import numpy as np
import pandas as pd
import random
import os


local_rank = int(os.getenv('LOCAL_RANK', 0)) #0,1,2,3,4
rank = int(os.getenv('RANK', 0)) #index of node
world_size = int(os.getenv('WORLD_SIZE', 1)) #5

if torch.cuda.is_available():
    print(f"CUDA is available. Running on local_rank {local_rank}")
    if torch.cuda.device_count() > 1:
        print(f"More than one GPU is available. Device count: {torch.cuda.device_count()}")
else:
    print("No GPU available")

torch.distributed.init_process_group('nccl', rank = rank, world_size = world_size)
torch.cuda.set_device(local_rank)

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 nohup torchrun --nproc_per_node=7 'train_multi-task_self-built_transformer_subset_1.0_portion_of_training_samples_all_7_tasks_multi-GPU_20250113.py' > 'train_multi-task_self-built_transformer_subset_1.0_portion_of_training_samples_all_7_tasks_multi-GPU_20250113.log' 2>&1 &

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
        attention_mask_1 = (torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)).repeat(sample_size_1, 1).cuda(local_rank)
        #
        #task_id == 'drug-protein_binding': #order of modality: drug (600), protein (1024)
        embeddings = inputs_embeds[1]
        task_id = input_task_id[1]
        sample_size_2 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_2, self.hidden_size)).repeat(1, 3).cuda(local_rank)
        embeddings_2 = torch.cat((CLS_token.repeat(sample_size_2, 1), all_drug[(sample_size_1):(sample_size_1+sample_size_2), :], all_protein[:sample_size_2, :], padding), dim=1)
        embeddings_2 = embeddings_2.reshape(sample_size_2, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_2 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_2, 1).cuda(local_rank)
        #
        #task_id == 'TF-gene_regulation': #order of modality: gene (128), protein (1024)
        embeddings = inputs_embeds[2]
        task_id = input_task_id[2]
        sample_size_3 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_3, self.hidden_size)).repeat(1, 3).cuda(local_rank)
        embeddings_3 = torch.cat((CLS_token.repeat(sample_size_3, 1), all_gene[(sample_size_1):(sample_size_1+sample_size_3), :], all_protein[(sample_size_2):(sample_size_2+sample_size_3), :], padding), dim=1)
        embeddings_3 = embeddings_3.reshape(sample_size_3, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_3 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_3, 1).cuda(local_rank)
        #
        #task_id == 'drug_sensitivity': #order of modality: drug (600), cellline (128)
        embeddings = inputs_embeds[3]
        task_id = input_task_id[3]
        sample_size_4 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_4, self.hidden_size)).repeat(1, 3).cuda(local_rank)
        embeddings_4 = torch.cat((CLS_token.repeat(sample_size_4, 1), all_drug[(sample_size_1 + sample_size_2):(sample_size_1 + sample_size_2 + sample_size_4), :], all_cellline[(sample_size_1):(sample_size_1+sample_size_4), :], padding), dim=1)
        embeddings_4 = embeddings_4.reshape(sample_size_4, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_4 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_4, 1).cuda(local_rank)
        #
        #task_id == 'gene_effect_score': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[4]
        task_id = input_task_id[4]
        sample_size_5 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_5, self.hidden_size)).repeat(1, 3).cuda(local_rank)
        embeddings_5 = torch.cat((CLS_token.repeat(sample_size_5, 1), all_cellline[(sample_size_1 + sample_size_4):(sample_size_1 + sample_size_4 + sample_size_5), :], all_gene[(sample_size_1 + sample_size_3):(sample_size_1 + sample_size_3 + sample_size_5), :], padding), dim=1)
        embeddings_5 = embeddings_5.reshape(sample_size_5, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_5 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_5, 1).cuda(local_rank)
        #
        #task_id == 'gene_mutation': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[5]
        task_id = input_task_id[5]
        sample_size_6 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_6, self.hidden_size)).repeat(1, 3).cuda(local_rank)
        embeddings_6 = torch.cat((CLS_token.repeat(sample_size_6, 1), all_cellline[(sample_size_1 + sample_size_4 + sample_size_5):(sample_size_1 + sample_size_4 + sample_size_5 + sample_size_6), :], all_gene[(sample_size_1 + sample_size_3 + sample_size_5):(sample_size_1 + sample_size_3 + sample_size_5 + sample_size_6), :], padding), dim=1)
        embeddings_6 = embeddings_6.reshape(sample_size_6, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_6 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_6, 1).cuda(local_rank)
        #
        #task_id == 'gene_CNV': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[6]
        task_id = input_task_id[6]
        sample_size_7 = embeddings.shape[0]
        padding = (torch.zeros(sample_size_7, self.hidden_size)).repeat(1, 3).cuda(local_rank)
        embeddings_7 = torch.cat((CLS_token.repeat(sample_size_7, 1), all_cellline[(sample_size_1 + sample_size_4 + sample_size_5 + sample_size_6):(sample_size_1 + sample_size_4 + sample_size_5 + sample_size_6 + sample_size_7), :], all_gene[(sample_size_1 + sample_size_3 + sample_size_5 + sample_size_6):(sample_size_1 + sample_size_3 + sample_size_5 + sample_size_6 + sample_size_7), :], padding), dim=1)
        embeddings_7 = embeddings_7.reshape(sample_size_7, 6, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_7 = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)).repeat(sample_size_7, 1).cuda(local_rank)
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
#import the whole training data for all 7 tasks:

#########################
#task 1: 'drug-induced_gene_exp'
#order of modality: drug, cellline, gene, time, dose
X_training_drug_induced_gene_exp = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_drug-induced_gene_exp_use_GNN.pt')
#torch.Size([22456548, 858])
Y_training_drug_induced_gene_exp = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_drug-induced_gene_exp_use_GNN.pt')


#########################
#task 2: 'drug-protein_binding'
#order of modality: drug, protein
X_training_drug_protein_binding = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_drug-protein_binding_use_GNN.pt')
#torch.Size([1265596, 1624])
Y_training_drug_protein_binding = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_drug-protein_binding_use_GNN.pt')


#########################
#task 3: 'TF-gene_regulation' 
#order of modality: gene, protein
#X_training_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_TF-gene_regulation.pt')
X_training_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_TF-gene_regulation_balanced_data.pt')
#torch.Size([576576, 1152])
#Y_training_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_TF-gene_regulation.pt')
Y_training_TF_gene_regulation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_TF-gene_regulation_balanced_data.pt')


#########################
#task 4: 'drug_sensitivity'
#order of modality: drug, cellline
X_training_drug_sensitivity = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_drug_sensitivity_use_GNN.pt')
#torch.Size([276712, 728])
Y_training_drug_sensitivity = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_drug_sensitivity_use_GNN.pt')


#########################
#task 5: 'gene_effect_score'
#order of modality: cellline, gene
X_training_gene_effect_score = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_gene_effect_score.pt')
#torch.Size([8809512, 256])
Y_training_gene_effect_score = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_gene_effect_score.pt')


#########################
#task 6: 'gene_mutation'
#order of modality: cellline, gene
#X_training_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_gene_mutation.pt')
X_training_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_gene_mutation_balanced_data.pt')
#torch.Size([866798, 256])
#Y_training_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_gene_mutation.pt')
Y_training_gene_mutation = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_gene_mutation_balanced_data.pt')


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

#---------------------------------------------------------------------------------------
#changeable parameters:
fraction = 1/50000 #the fraction of training samples to be taken from each task to form a batch
#batch_size = 128
epoch = 50
rdseed = 10 
out_dir = '/egr/research-aidd/chenruo4/self-built_transformer/model_trial_1.0_portion_of_whole_training_data_20241204/'

#---------------------------------------------------------------------------------------
#Use DataLoader
sampler_1 = torch.utils.data.distributed.DistributedSampler(train_dataset_drug_induced_gene_exp)
sampler_2 = torch.utils.data.distributed.DistributedSampler(train_dataset_drug_protein_binding)
sampler_3 = torch.utils.data.distributed.DistributedSampler(train_dataset_TF_gene_regulation)
sampler_4 = torch.utils.data.distributed.DistributedSampler(train_dataset_drug_sensitivity)
sampler_5 = torch.utils.data.distributed.DistributedSampler(train_dataset_gene_effect_score)
sampler_6 = torch.utils.data.distributed.DistributedSampler(train_dataset_gene_mutation)
sampler_7 = torch.utils.data.distributed.DistributedSampler(train_dataset_gene_CNV)

num_sample_per_batch_1 = int(len(train_dataset_drug_induced_gene_exp)*fraction)
num_sample_per_batch_2 = int(len(train_dataset_drug_protein_binding)*fraction)
num_sample_per_batch_3 = int(len(train_dataset_TF_gene_regulation)*fraction)
num_sample_per_batch_4 = int(len(train_dataset_drug_sensitivity)*fraction)
num_sample_per_batch_5 = int(len(train_dataset_gene_effect_score)*fraction)
num_sample_per_batch_6 = int(len(train_dataset_gene_mutation)*fraction)
num_sample_per_batch_7 = int(len(train_dataset_gene_CNV)*fraction)

train_loader_1 = DataLoader(train_dataset_drug_induced_gene_exp, batch_size = num_sample_per_batch_1, sampler = sampler_1, shuffle=(sampler_1 is None))
train_loader_2 = DataLoader(train_dataset_drug_protein_binding, batch_size = num_sample_per_batch_2, sampler = sampler_2, shuffle=(sampler_2 is None))
train_loader_3 = DataLoader(train_dataset_TF_gene_regulation, batch_size = num_sample_per_batch_3, sampler = sampler_3, shuffle=(sampler_3 is None))
train_loader_4 = DataLoader(train_dataset_drug_sensitivity, batch_size = num_sample_per_batch_4, sampler = sampler_4, shuffle=(sampler_4 is None))
train_loader_5 = DataLoader(train_dataset_gene_effect_score, batch_size = num_sample_per_batch_5, sampler = sampler_5, shuffle=(sampler_5 is None))
train_loader_6 = DataLoader(train_dataset_gene_mutation, batch_size = num_sample_per_batch_6, sampler = sampler_6, shuffle=(sampler_6 is None))
train_loader_7 = DataLoader(train_dataset_gene_CNV, batch_size = num_sample_per_batch_7, sampler = sampler_7, shuffle=(sampler_7 is None))

#---------------------------------------------------------------------------------------
#CLS token:
seed_everything(2000)
CLS = nn.Embedding(1, 768) 
CLS = CLS(torch.LongTensor([0]))

#---------------------------------------------------------------------------------------
# Initialize model, optimizer, and loss function
seed_everything(rdseed)
model = Model().cuda(local_rank)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
loss_regression = nn.MSELoss()
loss_classification = nn.BCELoss() 

# Training loop
for e in range(epoch):
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
    sampler_1.set_epoch(e)
    sampler_2.set_epoch(e)
    sampler_3.set_epoch(e)
    sampler_4.set_epoch(e)
    sampler_5.set_epoch(e)
    sampler_6.set_epoch(e)
    sampler_7.set_epoch(e)
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
        outputs_1, outputs_2, outputs_3, outputs_4, outputs_5, outputs_6, outputs_7 = model(inputs_embeds = [input_data_1.cuda(local_rank), input_data_2.cuda(local_rank), input_data_3.cuda(local_rank), input_data_4.cuda(local_rank), input_data_5.cuda(local_rank), input_data_6.cuda(local_rank), input_data_7.cuda(local_rank)], 
                          input_task_id = [task_id_1, task_id_2, task_id_3, task_id_4, task_id_5, task_id_6, task_id_7], 
                          CLS_token = CLS.cuda(local_rank))
        #
        loss_1 = loss_regression(outputs_1.view(-1), labels_1.cuda(local_rank).view(-1))
        loss_2 = loss_regression(outputs_2.view(-1), labels_2.cuda(local_rank).view(-1))
        loss_3 = loss_classification(outputs_3.view(-1), labels_3.cuda(local_rank).view(-1))
        loss_4 = loss_regression(outputs_4.view(-1), labels_4.cuda(local_rank).view(-1))
        loss_5 = loss_regression(outputs_5.view(-1), labels_5.cuda(local_rank).view(-1))
        loss_6 = loss_classification(outputs_6.view(-1), labels_6.cuda(local_rank).view(-1))
        loss_7 = loss_regression(outputs_7.view(-1), labels_7.cuda(local_rank).view(-1))
        #weight the loss to around 1:
        loss = (loss_1 * (1/3)) + (loss_2 * (1/7)) + (loss_3 * 2) + (loss_4 * (1/7)) + (loss_5 * 5) + (loss_6 * 2) + (loss_7)
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
    # Save the model:
    torch.save(model.state_dict(), out_dir + 'self-built_transformer_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(e) + '_20241230')
    # Save training loss:
    df = pd.DataFrame({'epoch_loss_1': epoch_loss_1, 'epoch_loss_2': epoch_loss_2, 'epoch_loss_3': epoch_loss_3, 'epoch_loss_4': epoch_loss_4, 'epoch_loss_5': epoch_loss_5, 'epoch_loss_6': epoch_loss_6, 'epoch_loss_7': epoch_loss_7, 'epoch_total_loss': epoch_loss_all}, index = range(0, len(epoch_loss_all)))
    df.to_csv(out_dir + 'training_loss_self-built_transformer_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(e) + '_20241230.csv')

    
        