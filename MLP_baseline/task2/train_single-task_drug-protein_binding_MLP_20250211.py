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
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
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
        #task_id == 'drug-protein_binding': #order of modality: drug (600), protein (1024)
        embeddings = inputs_embeds[0]
        task_id = input_task_id[0]
        sample_size = embeddings.shape[0]
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
        #
        # Split the data back into each task:
        #
        #task_id == 'drug-protein_binding': #order of modality: drug (600), protein (1024)
        embeddings = inputs_embeds[0]
        task_id = input_task_id[0]
        sample_size_2 = embeddings.shape[0]
        embeddings_2 = torch.cat((all_drug[:sample_size_2, :], all_protein[:sample_size_2, :]), dim=1)
        #
        #Gather all tasks' data again for MLP encoder:
        embeddings = embeddings_2
        embeddings = self.projection(embeddings)
        #
        # Pass through the encoder layers: 
        for layer in self.encoder_layers:
            embeddings = layer(embeddings)
        # Output layer for regression and classification tasks:
        out_2 = self.regressor_2(embeddings)
        return out_2

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
X_training_drug_protein_binding = torch.load('../../../X_training_set_task_drug-protein_binding_use_GNN.pt')
#torch.Size([1265596, 1624])
Y_training_drug_protein_binding = torch.load('../../../Y_training_set_task_drug-protein_binding_use_GNN.pt')

X_val_drug_protein_binding = torch.load('../../../X_test_set_seen_unseen_mixture_task_drug-protein_binding_use_GNN.pt')
Y_val_drug_protein_binding = torch.load('../../../Y_test_set_seen_unseen_mixture_task_drug-protein_binding_use_GNN.pt')

X_test_drug_protein_binding = torch.load('../../../X_test_set_unseen_task_drug-protein_binding_use_GNN.pt')
Y_test_drug_protein_binding = torch.load('../../../Y_test_set_unseen_task_drug-protein_binding_use_GNN.pt')

#----------------------------------------------------------------------------------------------------------------
# Create DataLoader
train_dataset_drug_protein_binding = CustomDataset(X_training_drug_protein_binding, Y_training_drug_protein_binding, 
                                                   task_id = 'drug-protein_binding'
                                                   )

val_dataset_drug_protein_binding = CustomDataset(X_val_drug_protein_binding, Y_val_drug_protein_binding, 
                                                   task_id = 'drug-protein_binding'
                                                   )

test_dataset_drug_protein_binding = CustomDataset(X_test_drug_protein_binding, Y_test_drug_protein_binding, 
                                                   task_id = 'drug-protein_binding'
                                                   )

#---------------------------------------------------------------------------------------
#changeable parameters:
#batch_size = 2048
epoch = 100
rdseed = 10 
out_dir = '/egr/research-aidd/molinqin/insilicoCell_Ruoqiao/concat_input_per_task/mlp/optimized_singletask_MLP/task2/'

#---------------------------------------------------------------------------------------
num_sample_per_batch_2 = 2048

train_loader_2 = DataLoader(train_dataset_drug_protein_binding, batch_size = num_sample_per_batch_2, shuffle = True)
val_loader_2 = DataLoader(val_dataset_drug_protein_binding, batch_size = num_sample_per_batch_2, shuffle = False)
test_loader_2 = DataLoader(test_dataset_drug_protein_binding, batch_size = num_sample_per_batch_2, shuffle = False)

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
best_score=None
best_model_wts = None
early_stop=False
estop_counter=0

# Training loop
for e in range(epoch):
    epoch_loss_all = []
    #
    model.train()
    running_train_loss = 0.0
    total_batches = len(train_loader_2)
    #
    for batch_2 in tqdm(train_loader_2, desc=f"Epoch {e+1}/{epoch} - Training", total=total_batches):
        #
        #task 2: 'drug-protein_binding':
        input_data_2 = batch_2['input_data']
        labels_2 = batch_2['labels']
        task_id_2 = batch_2['task_id']
        #
        optimizer.zero_grad()
        #
        outputs_2 = model(inputs_embeds = [input_data_2.to(device)], 
                          input_task_id = [task_id_2])
        #
        loss_2 = loss_regression(outputs_2.view(-1), labels_2.to(device).view(-1))
        loss = loss_2
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
        for batch in val_loader_2:
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
    avg_val_loss = running_val_loss / len(val_loader_2)
    val_loss_list.append(avg_val_loss)
    #
    df_val = pd.DataFrame({"y_pred": val_outputs, "y_truth": Y_val_drug_protein_binding.detach().numpy().flatten()})
    rmse_val = np.sqrt(mean_squared_error(df_val['y_truth'], df_val['y_pred']))
    cor_val = df_val.corr().iloc[0,1]

    running_test_loss = 0.0
    test_outputs = []
    test_labels_list = []
    with torch.no_grad():
        for batch in test_loader_2:
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
    avg_test_loss = running_test_loss / len(test_loader_2)
    test_loss_list.append(avg_test_loss)
    #
    df_test = pd.DataFrame({"y_pred": test_outputs, "y_truth": Y_test_drug_protein_binding.detach().numpy().flatten()})
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
        estop_counter = 0  
    else:
        estop_counter += 1
        if estop_counter >= patience:
            early_stop = True
    if early_stop:
        print("Early stopping triggered.")
        break

df_all_performance = pd.DataFrame(all_performance)
df_all_performance.to_csv(out_dir + 'MLP_drug-protein_binding_performance_all_epochs_20250211.csv', index=False)

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

# Save the best model:
torch.save(best_model_wts, out_dir + 'MLP_drug-protein_binding_best_model_20250211.pth')

    
        

            




    
    




#------------------------------------------------------------
#-----------------------------------------------------------
#draft:
