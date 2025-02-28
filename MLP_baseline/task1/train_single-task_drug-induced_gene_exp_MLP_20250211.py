import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import numpy as np
import pandas as pd
import random
import os
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy


device = torch.device( f'cuda:{7}' if torch.cuda.is_available() else 'cpu')

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

seed_everything(43)

#----------------------------------------------------------------------------------------------------------------
# Model:
class MLPModel(nn.Module):
    def __init__(self, hidden_size = 768):
        super(MLPModel, self).__init__()
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
        self.projection = nn.Linear(hidden_size * 5, hidden_size)
        # Encoder layers (12 layers)
        self.encoder_layers = nn.ModuleList([ModelLayer(hidden_size) for _ in range(12)])
        # Regression head for each task:
        self.regressor_1 = nn.Linear(hidden_size, 1)
        self.regressor_2 = nn.Linear(hidden_size, 1)
        self.regressor_3 = nn.Linear(hidden_size, 1)
        self.regressor_4 = nn.Linear(hidden_size, 1)
        self.regressor_5 = nn.Linear(hidden_size, 1)
        self.regressor_6 = nn.Linear(hidden_size, 1)
        self.regressor_7 = nn.Linear(hidden_size, 1)
    #
    def forward(self, inputs_embeds, input_task_id = None):
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
        #
        # Split the data back into each task:
        #
        #task_id == 'drug-induced_gene_exp', #order of modality: drug (600), cellline (128), gene (128), time (1), dose (1)
        embeddings = inputs_embeds[0]
        task_id = input_task_id[0]
        sample_size_1 = embeddings.shape[0]
        embeddings_1 = torch.cat((all_drug[:sample_size_1, :], all_gene[:sample_size_1, :], all_cellline[:sample_size_1, :], all_time[:sample_size_1, :], all_dose[:sample_size_1, :]), dim=1)
        #
        #Gather all tasks' data again for MLP encoder:
        embeddings = embeddings_1
        embeddings = self.projection(embeddings)
        #
        # Pass through the encoder layers: 
        for layer in self.encoder_layers:
            embeddings = layer(embeddings)
        # Output layer for regression and classification tasks:
        out_1 = self.regressor_1(embeddings)
        return out_1

#----------------------------------------------------------------------------------------------------------------
# Model layer (length of hidden layer set to hidden_size * 6 to match the paraters of transformer):
class ModelLayer(nn.Module):
    def __init__(self, hidden_size):
        super(ModelLayer, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 6)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_size * 6, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states_residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        # Res connect
        return self.LayerNorm(hidden_states + hidden_states_residual)

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
X_training_drug_induced_gene_exp = torch.load('../../../X_training_set_task_drug-induced_gene_exp_use_GNN.pt')
#torch.Size([22456548, 858])
Y_training_drug_induced_gene_exp = torch.load('../../../Y_training_set_task_drug-induced_gene_exp_use_GNN.pt')

X_val_drug_induced_gene_exp = torch.load('../../../X_test_set_seen_unseen_mixture_task_drug-induced_gene_exp_use_GNN.pt')
Y_val_drug_induced_gene_exp = torch.load('../../../Y_test_set_seen_unseen_mixture_task_drug-induced_gene_exp_use_GNN.pt')

X_test_drug_induced_gene_exp = torch.load('../../../X_test_set_unseen_task_drug-induced_gene_exp_use_GNN.pt')
Y_test_drug_induced_gene_exp = torch.load('../../../Y_test_set_unseen_task_drug-induced_gene_exp_use_GNN.pt')

#----------------------------------------------------------------------------------------------------------------
# Create DataLoader
train_dataset_drug_induced_gene_exp = CustomDataset(X_training_drug_induced_gene_exp, Y_training_drug_induced_gene_exp, 
                                                   task_id = 'drug-induced_gene_exp'
                                                   )

val_dataset_drug_induced_gene_exp = CustomDataset(X_val_drug_induced_gene_exp, Y_val_drug_induced_gene_exp, 
                                                   task_id = 'drug-induced_gene_exp'
                                                   )

test_dataset_drug_induced_gene_exp = CustomDataset(X_test_drug_induced_gene_exp, Y_test_drug_induced_gene_exp, 
                                                   task_id = 'drug-induced_gene_exp'
                                                   )

#---------------------------------------------------------------------------------------
#changeable parameters:
#batch_size = 2048
epoch = 100
rdseed = 10 
out_dir = '/egr/research-aidd/molinqin/insilicoCell_Ruoqiao/concat_input_per_task/mlp/optimized_singletask_MLP/task1/'

#---------------------------------------------------------------------------------------
num_sample_per_batch_1 = 2048

train_loader_1 = DataLoader(train_dataset_drug_induced_gene_exp, batch_size = num_sample_per_batch_1, shuffle = True)
val_loader_1 = DataLoader(val_dataset_drug_induced_gene_exp, batch_size = num_sample_per_batch_1, shuffle = False)
test_loader_1 = DataLoader(test_dataset_drug_induced_gene_exp, batch_size = num_sample_per_batch_1, shuffle = False)

#---------------------------------------------------------------------------------------
# Initialize model, optimizer, and loss function

seed_everything(rdseed)
model = MLPModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002)
loss_regression = nn.MSELoss()
loss_classification = nn.BCELoss() 

all_performance = []  
train_loss_list = []
val_loss_list = []
test_loss_list = []

# Init early stop
patience = 5
best_score = None
best_model_wts = None
best_epoch = None  
early_stop = False
estop_counter = 0

# Training loop
for e in range(epoch):
    epoch_loss_all = []
    #
    model.train()
    running_train_loss = 0.0
    total_batches = len(train_loader_1)
    #
    for batch_1 in tqdm(train_loader_1, desc=f"Epoch {e+1}/{epoch} - Training", total=total_batches):
        #
        #task 1: 'drug-induced_gene_exp':
        input_data_1 = batch_1['input_data']
        labels_1 = batch_1['labels']
        task_id_1 = batch_1['task_id']
        #
        optimizer.zero_grad()
        #
        outputs_1 = model(inputs_embeds = [input_data_1.to(device)], 
                          input_task_id = [task_id_1])
        #
        loss_1 = loss_regression(outputs_1.view(-1), labels_1.to(device).view(-1))
        loss = loss_1
        loss.backward(retain_graph=True)
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / total_batches
    train_loss_list.append(avg_train_loss)

    model.eval()

    running_val_loss = 0.0
    val_outputs = []
    val_labels_list = []
    with torch.no_grad():
        for batch in val_loader_1:
            input_data = batch['input_data']
            labels = batch['labels']
            task_id = batch['task_id']
            #
            outputs = model(inputs_embeds = [input_data.to(device)], 
                                input_task_id = [task_id])
            loss_val = loss_regression(outputs.view(-1), labels.to(device).view(-1))
            running_val_loss += loss_val.item()
            #
            outputs_np = outputs.to('cpu').view(-1).detach().numpy().tolist()
            val_outputs.extend(outputs_np)
    avg_val_loss = running_val_loss / len(val_loader_1)
    val_loss_list.append(avg_val_loss)
    #
    df_val = pd.DataFrame({"y_pred": val_outputs, "y_truth": Y_val_drug_induced_gene_exp.detach().numpy().flatten()})
    rmse_val = np.sqrt(mean_squared_error(df_val['y_truth'], df_val['y_pred']))
    cor_val = df_val.corr().iloc[0,1]

    running_test_loss = 0.0
    test_outputs = []
    test_labels_list = []
    with torch.no_grad():
        for batch in test_loader_1:
            input_data = batch['input_data']
            labels = batch['labels']
            task_id = batch['task_id']
            #
            outputs = model(inputs_embeds = [input_data.to(device)], 
                                input_task_id = [task_id])
            loss_test = loss_regression(outputs.view(-1), labels.to(device).view(-1))
            running_test_loss += loss_test.item()
            #
            outputs_np = outputs.to('cpu').view(-1).detach().numpy().tolist()
            test_outputs.extend(outputs_np)
    avg_test_loss = running_test_loss / len(test_loader_1)
    test_loss_list.append(avg_test_loss)
    #
    df_test = pd.DataFrame({"y_pred": test_outputs, "y_truth": Y_test_drug_induced_gene_exp.detach().numpy().flatten()})
    rmse_test = np.sqrt(mean_squared_error(df_test['y_truth'], df_test['y_pred']))
    cor_test = df_test.corr().iloc[0,1]

    performance_dict = {
        'epoch': e + 1,
        'train_loss': round(avg_train_loss, 4),
        'val_loss': round(avg_val_loss, 4),
        'test_loss': round(avg_test_loss, 4),
        'val_rmse': round(rmse_val, 4),
        'val_cor': round(cor_val, 4),
        'test_rmse': round(rmse_test, 4),
        'test_cor': round(cor_test, 4)
    }
    all_performance.append(performance_dict)

    print(f"Epoch {e+1}: Train Loss = {avg_train_loss:.3f}, Val Loss = {avg_val_loss:.3f}, Test Loss = {avg_test_loss:.3f}")

    if best_score is None or avg_val_loss < best_score:
        best_score = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = e + 1   
        estop_counter = 0  
    else:
        estop_counter += 1
        if estop_counter >= patience:
            early_stop = True
    if early_stop:
        print("Early stopping triggered.")
        break

df_all_performance = pd.DataFrame(all_performance)
df_all_performance.to_csv(out_dir + 'MLP_drug-induced_gene_exp_performance_all_epochs_20250211.csv', index=False)

# Plot loss curve
plt.figure(figsize=(10, 6))
epochs_range = range(1, len(train_loss_list) + 1)
plt.plot(epochs_range, train_loss_list, label='Train Loss', marker='o')
plt.plot(epochs_range, val_loss_list, label='Validation Loss', marker='o')
plt.plot(epochs_range, test_loss_list, label='Test Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation and Test Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(out_dir + 'loss_curves_20250211.png')
plt.show()

# Save the best model with the best epoch in the filename:
torch.save(best_model_wts, out_dir + f'MLP_drug-induced_gene_exp_best_model_epoch_{best_epoch}_20250211.pth')


    
        

            




    
    




#------------------------------------------------------------
#-----------------------------------------------------------
#draft:
