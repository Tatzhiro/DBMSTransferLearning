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


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the data
data = pd.read_csv("/home/dataset/train.csv")
# remove columns with missing values
data = data.drop(data.columns[data.isnull().any()], axis=1)
workload_labels = data['workload_label'].unique()

# Data Preprocessing
# Split the input features and the target label
features = data.drop(columns=['label', 'workload_label'])
labels = data['label'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

scaler = MinMaxScaler()

normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# Recombine the normalized features with the label column
data = pd.concat([normalized_features, labels, data['workload_label']], axis=1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


losses = {w: 0 for w in workload_labels}
errors = {w: 0 for w in workload_labels}
# Iterate over each unique workload_label (Leave-One-Workload-Out CV)
for workload_label in workload_labels:
    print(f"Training with workload_label '{workload_label}' as test set")

    # Separate the test set (data with the current workload label)
    test_data = data[data['workload_label'] == workload_label]
    train_val_data = data[data['workload_label'] != workload_label]

    # Split the remaining data into train and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

    # Convert data to PyTorch Datasets
    train_dataset = CustomDataset(train_data.drop(columns=['workload_label', 'label']).values, np.vstack(train_data['label'].values))
    val_dataset = CustomDataset(val_data.drop(columns=['workload_label', 'label']).values, np.vstack(val_data['label'].values))
    test_dataset = CustomDataset(test_data.drop(columns=['workload_label', 'label']).values, np.vstack(test_data['label'].values))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    model = VectorPredictor(input_size, num_hidden_layer, hidden_size, output_size).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
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
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Testing on the left-out workload_label (test set)
    model.eval()
    test_loss = 0.0
    error = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += loss.item() * inputs.size(0)
            
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            dot_product = np.sum(outputs_np * targets_np, axis=1)  # Dot product for each pair
            norm_outputs = np.linalg.norm(outputs_np, axis=1)  # Norm for each output
            norm_targets = np.linalg.norm(targets_np, axis=1)  # Norm for each target
            # Avoid division by zero
            cosine_similarity = dot_product / (norm_outputs * norm_targets + 1e-8)
            error += np.sum(1 - cosine_similarity)

    test_loss /= len(test_loader.dataset)
    error /= len(test_loader.dataset)
    print(f"Test Loss for workload_label '{workload_label}': {test_loss:.4f}\n")
    print(f"Sample output: {outputs[0]}")
    print(f"Target: {targets[0]}")
    output = [outputs[0].cpu().numpy()]
    target = [targets[0].cpu().numpy()]
    # make a bar plot with parameter_metadata as x-axis and output and target as y-axis
    plt.figure(figsize=(10, 5))
    plt.bar(parameter_metadata, output[0], label='Output')
    plt.bar(parameter_metadata, target[0], alpha=0.5, label='Target')
    plt.xticks(ticks=range(len(parameter_metadata)), labels=parameter_metadata, rotation=90)
    plt.legend()
    plt.title(f"Output vs Target for workload_label '{workload_label}'")
    plt.tight_layout()
    wandb.log({f"output vs target for {workload_label}": plt})
    plt.close()
    
    
    losses[workload_label] = test_loss
    errors[workload_label] = error

print(f"Average Test Loss: {np.mean(list(losses.values())):.4f}")
# data = [[k, v] for k, v in losses.items()]
data = [[k, v] for k, v in errors.items()]
table = wandb.Table(data=data, columns=["workload_label", "test_loss"])
wandb.log({"test_loss": wandb.plot.bar(table, "workload_label", "test_loss", title="Test Loss per Workload Label")})

print(f"Average error: {np.mean(list(errors.values())):.4f}")

print(f"Test Losses for each workload_label:")
for workload_label, loss in losses.items():
    print(f"{workload_label}: {loss:.4f}")
print("Cross-validation complete.")
