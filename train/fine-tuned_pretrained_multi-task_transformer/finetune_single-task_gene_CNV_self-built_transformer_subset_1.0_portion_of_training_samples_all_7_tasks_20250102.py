import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
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
        #task_id == 'gene_CNV': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[0]
        task_id = input_task_id[0]
        sample_size = embeddings.shape[0]
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
        #
        # Split the data back into each task:
        # Add CLS token & padding tokens, concatenate all tokens together for each task:
        #
        #task_id == 'gene_CNV': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[0]
        task_id = input_task_id[0]
        sample_size_7 = embeddings.shape[0]
        embeddings_7 = torch.cat((CLS_token.repeat(sample_size_7, 1), all_cellline[:sample_size_7, :], all_gene[:sample_size_7, :]), dim=1)
        embeddings_7 = embeddings_7.reshape(sample_size_7, 3, self.hidden_size) #torch.Size([128, 6, 768])
        # Add attention mask:
        attention_mask_7 = (torch.tensor([0, 0, 0], dtype=torch.float32)).repeat(sample_size_7, 1).to(device)
        #
        #Gather all tasks' data again for transfomer encoder:
        embeddings = embeddings_7
        attention_mask = attention_mask_7
        #
        # Pass through the encoder layers: 
        for layer in self.encoder_layers:
            embeddings = layer(embeddings, attention_mask)
        # Pool the output of the [CLS] token
        pooled_output = self.pooler(embeddings) #for classification tasks, consider using Sigmoid activation
        #
        #Split the data by task again:
        pooled_output_7 = pooled_output[:sample_size_7, :]
        # Output layer for regression and classification tasks:
        out_7 = self.regressor_7(pooled_output_7)
        return out_7

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
#task 7: 'gene_CNV'
#order of modality: cellline, gene
X_training_gene_CNV = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/X_training_set_task_gene_CNV.pt')
#torch.Size([17809611, 256])
Y_training_gene_CNV = torch.load('/egr/research-aidd/chenruo4/self-built_transformer/input_representations/concat_input_per_task/Y_training_set_task_gene_CNV.pt')

#----------------------------------------------------------------------------------------------------------------
# Create DataLoader
train_dataset_gene_CNV = CustomDataset(X_training_gene_CNV, Y_training_gene_CNV, 
                                               task_id = 'gene_CNV'
                                               )

#---------------------------------------------------------------------------------------
#changeable parameters:
pretrained_final_epoch = 9
finetune_epoch = 10
rdseed = 10 
num_sample_per_batch = 128
out_dir = '/egr/research-aidd/chenruo4/self-built_transformer/model_trial_1.0_portion_of_whole_training_data_20241204/'

#---------------------------------------------------------------------------------------

train_loader_7 = DataLoader(train_dataset_gene_CNV, batch_size = num_sample_per_batch, shuffle = True)

#---------------------------------------------------------------------------------------
#CLS token:
seed_everything(2000)
CLS = nn.Embedding(1, 768) 
CLS = CLS(torch.LongTensor([0]))

#---------------------------------------------------------------------------------------
# Initialize model, optimizer, and loss function
seed_everything(rdseed)
model = Model().to(device)
model.load_state_dict(torch.load(out_dir + 'self-built_transformer_multitask_no_modality_tokens_use_GNN_drugs_lr0.00001_all_7_tasks_1.0_portion_' + str(pretrained_final_epoch) + '_20241230'))

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
loss_regression = nn.MSELoss()
loss_classification = nn.BCELoss() 

# Training loop
for e in range(finetune_epoch):
    epoch_loss_all = []
    #
    model.train()
    count = 0
    #
    for batch_7 in train_loader_7:
        #
        #task 7: 'gene_CNV':
        input_data_7 = batch_7['input_data']
        labels_7 = batch_7['labels']
        task_id_7 = batch_7['task_id']
        #
        optimizer.zero_grad()
        #
        outputs_7 = model(inputs_embeds = [input_data_7.to(device)], 
                          input_task_id = [task_id_7], 
                          CLS_token = CLS.to(device))
        #
        loss_7 = loss_regression(outputs_7.view(-1), labels_7.to(device).view(-1))
        #weight the loss to around 1.5:
        loss = loss_7
        #print(f"loss_1: {loss_1}, loss_2: {loss_2}, loss_3: {loss_3}, loss_4: {loss_4}, loss_5: {loss_5}, loss_6: {loss_6}, loss_7: {loss_7}")
        #print(f"total_loss: {loss}")
        loss.backward(retain_graph=True)
        optimizer.step()
        count = count + 1
        if count % 100 == 99:
            #print(f"loss_1: {loss_1}, loss_2: {loss_2}, loss_3: {loss_3}, loss_4: {loss_4}, loss_5: {loss_5}, loss_6: {loss_6}, loss_7: {loss_7}")
            print(f'Epoch {e + 1}/{finetune_epoch}, total_loss: {loss.item()}')
        epoch_loss_all.append(loss.to('cpu').detach().flatten().numpy()[0])
    # Save the model:
    torch.save(model.state_dict(), out_dir + 'finetune_self-built_transformer_no_modality_tokens_single-task_gene_CNV_lr0.00001_all_7_tasks_subset_1.0_portion_' + str(e) + '_20241229')
    # Save training loss:
    df = pd.DataFrame({'epoch_total_loss': epoch_loss_all}, index = range(0, len(epoch_loss_all)))
    df.to_csv(out_dir + 'finetune_loss_self-built_transformer_no_modality_tokens_single-task_gene_CNV_lr0.00001_all_7_tasks_subset_1.0_portion_' + str(e) + '_20241229.csv')

    
        

            




    
    




#------------------------------------------------------------
#-----------------------------------------------------------
#draft:
