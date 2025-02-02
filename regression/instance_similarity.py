import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import wandb
from matplotlib import pyplot as plt
import os

from system_configuration import SystemConfiguration

parameter_metadata = ["innodb_buffer_pool_size", "innodb_read_io_threads", "innodb_write_io_threads", 
                   "innodb_flush_log_at_trx_commit", "innodb_adaptive_hash_index", "sync_binlog",
                   "innodb_lru_scan_depth", "innodb_buffer_pool_instances", "innodb_change_buffer_max_size",
                   "innodb_io_capacity", "innodb_log_file_size", "table_open_cache"]

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, inputs, targets, device='cuda'):
        self.inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    

class VectorPredictor(nn.Module):
    """
    input: database metrics
    output: important parameter vector, normalized
    """
    def __init__(self, input_size, num_hidden_layer, hidden_size, output_size):
        super(VectorPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_hidden_layer)])
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output(x)
        x = F.softmax(x, dim=1)
        return x
    
    
class Agent:
    def __init__(self, system_configuration):
        self.feature_columns = None
        self.system = system_configuration
    
    def train_model(self, workload_label, hardware_label):
        
        data = pd.read_csv("dataset/metric_learning/train.csv")
        data = data.drop(data.columns[data.isnull().any()], axis=1)
        
        # Data Preprocessing
        # Split the input features and the target label
        features = data.drop(columns=['label', 'workload_label'])
        self.feature_columns = features.columns
        labels = data['label'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

        scaler = MinMaxScaler()

        normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=self.feature_columns)
        # Recombine the normalized features with the label column
        data = pd.concat([normalized_features, labels, data['workload_label']], axis=1)
        train_data = data[data['workload_label'] != workload_label]
        train_data = train_data[train_data["hardware_label"] != hardware_label]
        
        wandb.init(
            project="AutoDB",
            group="CV",
            config={
                "input_size": features.shape[1],
                "output_size": np.vstack(labels.values).shape[1],
                "num_hidden_layer": 5,
                "hidden_size": 128,
                "batch_size": 32,
                "learning_rate": 0.001,
                "num_epochs": 10
            }
        )
        
        input_size = wandb.config.input_size    # Input size
        output_size = wandb.config.output_size  # Output size


        # Hyperparameters
        num_hidden_layer = wandb.config.num_hidden_layer
        hidden_size = wandb.config.hidden_size
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        num_epochs = wandb.config.num_epochs

        
        train_dataset = CustomDataset(train_data.drop(columns=['workload_label', 'label', 'hardware_label']).values, np.vstack(train_data['label'].values))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # Instantiate the model
        model = VectorPredictor(input_size, num_hidden_layer, hidden_size, output_size).to(device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
        
        return model


    def get_similar_datasets(self, target_input, workload_label, hardware_label, n=2):
        """
        Get the n most similar datasets to the given dataset
        """
        
        model = self.train_model(workload_label, hardware_label)
        
        # get all the csv files in dataset/metric_learning
        files = os.listdir("dataset/metric_learning")
        files = [f for f in files if f.endswith(".csv")]
        dfs = [pd.read_csv(df) for df in files if df != "train.csv"]
        
            
        model.eval()
        with torch.no_grad():
            target_input = target_input[self.feature_columns]
            
            scaler = MinMaxScaler()
            target_input = pd.DataFrame(scaler.fit_transform(target_input), columns=self.feature_columns)
            target_output = model(target_input)
            target_output_np = target_output.cpu().numpy()
            
            similarities = []
            for df in dfs:
                hardware = f"{df["num_cores"][0]}-{df["memory_size"][0]}"
                if hardware == hardware_label:
                    continue
                df = df.drop(df.columns[df.isnull().any()], axis=1)
                # get unique workload labels
                workloads = df['workload_label'].unique()
                for workload in workloads:
                    if workload == workload_label:
                        continue
                    data = df[df['workload_label'] == workload]
                    data = data.drop(data.columns[data.isnull().any()], axis=1)
                    for param in self.system.get_param_names():
                        data = data[data[param] == self.system.get_default_param_values()[param]]
                    features = data[self.feature_columns]
                    input_data = pd.DataFrame(scaler.fit_transform(features), columns=self.feature_columns)
                    output = model(input_data)
                    output_np = output.cpu().numpy()
                    similarity = np.dot(target_output_np, output_np) / (np.linalg.norm(target_output_np) * np.linalg.norm(output_np))
                    similarities.append((data, workload, hardware, similarity))
        
        similarities = sorted(similarities, key=lambda x: x[3], reverse=True)
        return similarities[:n]
        
        
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")