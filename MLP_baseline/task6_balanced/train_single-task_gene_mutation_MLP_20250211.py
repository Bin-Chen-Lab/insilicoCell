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


device = torch.device( f'cuda:{5}' if torch.cuda.is_available() else 'cpu')

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
        #task_id == 'gene_mutation': #order of modality: cellline (128), gene (128)
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
        #
        #task_id == 'gene_mutation': #order of modality: cellline (128), gene (128)
        embeddings = inputs_embeds[0]
        task_id = input_task_id[0]
        sample_size_6 = embeddings.shape[0]
        embeddings_6 = torch.cat((all_cellline[:sample_size_6, :], all_gene[:sample_size_6, :]), dim=1)
        #
        #Gather all tasks' data again for MLP encoder:
        embeddings = embeddings_6
        embeddings = self.projection(embeddings)
        #
        # Pass through the encoder layers: 
        for layer in self.encoder_layers:
            embeddings = layer(embeddings)
        # Output layer for regression and classification tasks:
        out_6 = F.sigmoid(self.regressor_6(embeddings))
        return out_6

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
#task 6: 'gene_mutation'
#order of modality: cellline, gene
X_training_gene_mutation = torch.load('../../../X_training_set_task_gene_mutation_balanced_data.pt')
#torch.Size([16743322, 256])
Y_training_gene_mutation = torch.load('../../../Y_training_set_task_gene_mutation_balanced_data.pt')

X_val_gene_mutation = torch.load('../../../X_test_set_seen_unseen_mixture_task_gene_mutation.pt')
Y_val_gene_mutation = torch.load('../../../Y_test_set_seen_unseen_mixture_task_gene_mutation.pt')

X_test_gene_mutation = torch.load('../../../X_test_set_unseen_task_gene_mutation.pt')
Y_test_gene_mutation = torch.load('../../../Y_test_set_unseen_task_gene_mutation.pt')

#----------------------------------------------------------------------------------------------------------------
# Create DataLoader
train_dataset_gene_mutation = CustomDataset(X_training_gene_mutation, Y_training_gene_mutation, 
                                               task_id = 'gene_mutation'
                                               )

val_dataset_gene_mutation = CustomDataset(X_val_gene_mutation, Y_val_gene_mutation, 
                                               task_id = 'gene_mutation'
                                               )

test_dataset_gene_mutation = CustomDataset(X_test_gene_mutation, Y_test_gene_mutation, 
                                               task_id = 'gene_mutation'
                                               )

#---------------------------------------------------------------------------------------
#changeable parameters:
#batch_size = 2048
epoch = 100
rdseed = 10 
out_dir = '/egr/research-aidd/molinqin/insilicoCell_Ruoqiao/concat_input_per_task/mlp/optimized_singletask_MLP/task6_balanced/'

#---------------------------------------------------------------------------------------
num_sample_per_batch_6 = 1024

train_loader_6 = DataLoader(train_dataset_gene_mutation, batch_size = num_sample_per_batch_6, shuffle = True)
val_loader_6 = DataLoader(val_dataset_gene_mutation, batch_size = num_sample_per_batch_6, shuffle = False)
test_loader_6 = DataLoader(test_dataset_gene_mutation, batch_size = num_sample_per_batch_6, shuffle = False)

#---------------------------------------------------------------------------------------
# Initialize model, optimizer, and loss function

seed_everything(rdseed)
model = MLPModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.000001)
loss_regression = nn.MSELoss()
loss_classification = nn.BCELoss() 

all_performance = []  
train_loss_list = []
val_loss_list = []
test_loss_list = []

# Init early stop
patience = 10
best_score = None
best_model_wts = None
best_epoch = None  
early_stop = False
estop_counter = 0

# Training loop
for e in range(epoch):
    #
    model.train()
    running_train_loss = 0.0
    total_batches = len(train_loader_6)
    #
    for batch_6 in tqdm(train_loader_6, desc=f"Epoch {e+1}/{epoch} - Training", total=total_batches):
        #
        #task 6: 'gene_mutation':
        input_data_6 = batch_6['input_data']
        labels_6 = batch_6['labels']
        task_id_6 = batch_6['task_id']
        #
        optimizer.zero_grad()
        #
        outputs_6 = model(inputs_embeds = [input_data_6.to(device)], 
                          input_task_id = [task_id_6])
        #
        loss_6 = loss_classification(outputs_6.view(-1), labels_6.to(device).view(-1))
        loss = loss_6
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
        for batch in val_loader_6:
            input_data = batch['input_data']
            labels = batch['labels']
            task_id = batch['task_id']
            #
            outputs = model(inputs_embeds = [input_data.to(device)], 
                                input_task_id = [task_id])
            loss_val = loss_classification(outputs.view(-1), labels.to(device).view(-1))
            running_val_loss += loss_val.item()
            #
            outputs_np = outputs.to('cpu').view(-1).detach().numpy().tolist()
            val_outputs.extend(outputs_np)
            labels_np = labels.to('cpu').view(-1).detach().numpy().tolist()
            val_labels_list.extend(labels_np)
    avg_val_loss = running_val_loss / len(val_loader_6)
    val_loss_list.append(avg_val_loss)
    #
    df_val = pd.DataFrame({"y_pred": val_outputs, "y_truth": val_labels_list})
    AUROC_val = roc_auc_score(df_val['y_truth'], df_val['y_pred'])
    AUPRC_val = average_precision_score(df_val['y_truth'], df_val['y_pred'])
    df_val['y_pred_class'] = (df_val['y_pred'] > 0.5).astype(int)
    f1_val = f1_score(df_val['y_truth'], df_val['y_pred_class'])
    acc_val = accuracy_score(df_val['y_truth'], df_val['y_pred_class'])


    running_test_loss = 0.0
    test_outputs = []
    test_labels_list = []
    with torch.no_grad():
        for batch in test_loader_6:
            input_data = batch['input_data']
            labels = batch['labels']
            task_id = batch['task_id']
            #
            outputs = model(inputs_embeds = [input_data.to(device)], 
                                input_task_id = [task_id])
            loss_test = loss_classification(outputs.view(-1), labels.to(device).view(-1))
            running_test_loss += loss_test.item()
            #
            outputs_np = outputs.to('cpu').view(-1).detach().numpy().tolist()
            test_outputs.extend(outputs_np)
            labels_np = labels.to('cpu').view(-1).detach().numpy().tolist()
            test_labels_list.extend(labels_np)
    avg_test_loss = running_test_loss / len(test_loader_6)
    test_loss_list.append(avg_test_loss)
    #
    df_test = pd.DataFrame({"y_pred": test_outputs, "y_truth": test_labels_list})
    AUROC_test = roc_auc_score(df_test['y_truth'], df_test['y_pred'])
    AUPRC_test = average_precision_score(df_test['y_truth'], df_test['y_pred'])
    df_test['y_pred_class'] = (df_test['y_pred'] > 0.5).astype(int)
    f1_test = f1_score(df_test['y_truth'], df_test['y_pred_class'])
    acc_test = accuracy_score(df_test['y_truth'], df_test['y_pred_class'])

    performance_dict = {
        'epoch': e + 1,
        'train_loss': round(avg_train_loss, 4),
        'val_loss': round(avg_val_loss, 4),
        'test_loss': round(avg_test_loss, 4),
        'val_AUROC': round(AUROC_val, 4),
        'val_AUPRC': round(AUPRC_val, 4),
        'val_f1': round(f1_val, 4),
        'val_accuracy': round(acc_val, 4),
        'test_AUROC': round(AUROC_test, 4),
        'test_AUPRC': round(AUPRC_test, 4),
        'test_f1': round(f1_test, 4),
        'test_accuracy': round(acc_test, 4)
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
df_all_performance.to_csv(out_dir + 'MLP_gene_mutation_performance_all_epochs_20250211.csv', index=False)

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
torch.save(best_model_wts, out_dir + f'MLP_gene_mutation_best_model_epoch_{best_epoch}_20250211.pth')



    
        

            




    
    




#------------------------------------------------------------
#-----------------------------------------------------------
#draft:
