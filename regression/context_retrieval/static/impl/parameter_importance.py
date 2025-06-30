from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import glob
import json
import random
from scipy.stats import entropy

from regression.utils import read_data_csv
from regression.context_retrieval import StaticContextRetrieval, Context, ContextSimilarity


class ParameterImportanceRetrieval(StaticContextRetrieval):
    """
    Finds similar datasets by learning a neural net that maps database metrics 
    to important parameter vectors, then comparing these vectors (e.g., via Mahalanobis distance).
    """
    _model = None
    _feature_columns = None
    _scaler = MinMaxScaler()
    
    # NEW: We'll store inverse covariance of labels here
    _label_cov_inv = None

    class CustomDataset(Dataset):
        def __init__(self, inputs, targets, device='cpu'):
            self.inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            self.targets = torch.tensor(targets, dtype=torch.float32).to(device)
        def __len__(self):
            return len(self.inputs)
        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]

    class VectorPredictor(nn.Module):
        """
        A simple feedforward network that outputs a softmax-normalized vector.
        """
        def __init__(self, input_size, num_hidden_layer, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layer)]
            )
            self.output = nn.Linear(hidden_size, output_size)

            self._initialize_weights()

        def _initialize_weights(self):
            for layer in [self.fc1, *self.hidden_layers, self.output]:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        def forward(self, x, give_logit=False):
            x = F.leaky_relu(self.fc1(x))
            for layer in self.hidden_layers:
                x = F.leaky_relu(layer(x))
            logit = self.output(x)
            if give_logit:
                return logit
            # Softmax across dim=1
            prob = F.softmax(logit, dim=1)
            return prob

    def __init__(self, system, data_dir: str, train_dir: str, seed: int = 42, excluding_factors=["hardware", "workload"]):
        self.system = system
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.config = None
        self.excluding_factors = excluding_factors
        self.set_seed(seed)

    @property
    def model(self):
        return ParameterImportanceRetrieval._model

    @model.setter
    def model(self, val):
        ParameterImportanceRetrieval._model = val

    @property
    def feature_columns(self):
        return ParameterImportanceRetrieval._feature_columns

    @feature_columns.setter
    def feature_columns(self, val):
        ParameterImportanceRetrieval._feature_columns = val

    @property
    def scaler(self):
        return ParameterImportanceRetrieval._scaler

    # NEW: Covariance inverse getter/setter
    @property
    def label_cov_inv(self):
        return ParameterImportanceRetrieval._label_cov_inv

    @label_cov_inv.setter
    def label_cov_inv(self, val):
        ParameterImportanceRetrieval._label_cov_inv = val

    def set_hyper_parameters(self, config):
        self.config = config
    
    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train_model(self, workload_label, hardware_label):
        """
        Trains a neural network on 'train.csv', excluding data that 
        matches the provided workload_label or hardware_label.
        Also computes and stores the label covariance inverse for Mahalanobis.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        EPS = 1e-7
        
        train_path = os.path.join(self.train_dir, "train.csv")
        data = pd.read_csv(train_path)
        data = data.drop(columns=data.columns[data.isnull().any()], errors="ignore")

        data['hardware_label'] = data.apply(
            lambda row: f"{row['num_cpu']}c{row['mem_size']}g", axis=1
        )
        # Filter out data matching hardware_label or workload_label
        if "hardware" in self.excluding_factors:
            data = data[data["hardware_label"] != hardware_label]
        if "workload" in self.excluding_factors:
            data = data[data['workload_label'] != workload_label]

        # Choose columns as you wish
        columns = [
            "tps",
            "Average Memory Usage Percentage",
            "InnoDB Buffer Pool Cache Hit Rate",
            "InnoDB Dirty Buffer Pages", 
            "Current QPS (Queries Per Second)",
            "Max CPU Usage (100 - Idle)",
            "InnoDB Rows Deleted (60s Rate)", "InnoDB Rows Inserted (60s Rate)",
            "InnoDB Rows Read (60s Rate)", "InnoDB Rows Updated (60s Rate)",
            "Average Disk IOPS (Read)", "Average Disk IOPS (Write)", 
        ]
        features = data[columns]
        self.feature_columns = features.columns

        # Convert label string -> np.array
        labels = data['label'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        # Scale features
        normalized_features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=self.feature_columns,
            index=features.index
        )
        train_data = pd.concat([normalized_features, labels.rename("label")], axis=1)

        config = self.config
        if config is None:
            config = {
                "batch_size": 256,
                "hidden_size": 64,
                "learning_rate": 0.001,
                "num_epochs": 100,
                "num_hidden_layer": 50,
                "weight_decay": 0.001
            }
            
            
        os.makedirs(f"neural_network/weights/{json.dumps(config)}/exclude-{'-'.join(self.excluding_factors)}", exist_ok=True)
        model_path = f"neural_network/weights/{json.dumps(config)}/exclude-{'-'.join(self.excluding_factors)}/{workload_label}_{hardware_label}.pt"
        if os.path.exists(model_path):
            print(f"Model already exists at {model_path}. Loading...")
            checkpoint = torch.load(model_path)
            config = checkpoint['config']
            input_size = checkpoint['input_size']
            output_size = checkpoint['output_size']

            model = self.VectorPredictor(
                input_size,
                config["num_hidden_layer"],
                config["hidden_size"],
                output_size
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            self.model = model
            return self.model
        
            
        input_size = features.shape[1]
        # Build a 2D array of labels
        label_mat = np.vstack(train_data['label'].values)
        output_size = label_mat.shape[1] if len(label_mat) > 0 else 1

        # Create PyTorch Dataset
        input_data = train_data.drop(columns=['label']).values
        train_dataset = self.CustomDataset(input_data, label_mat, device=device)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

        # Instantiate the model
        model = self.VectorPredictor(
            input_size,
            config["num_hidden_layer"],
            config["hidden_size"],
            output_size
        ).to(device)

        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

        for epoch in range(config["num_epochs"]):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                logits = model(inputs, give_logit=True)
                log_probs = F.log_softmax(logits, dim=1)
                loss = criterion(log_probs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {epoch_loss:.4f}")
            
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'output_size': output_size,
            'config': config
        }, model_path)


        return model

    def augment_data(self, data, target_workloads, num_copies=2, noise_scale=0.01):
        """
        Augments the input data by adding Gaussian noise to numeric columns.
        """
        mask = data['workload_label'].isin(target_workloads)
        subset = data[mask].copy()
        numeric_cols = subset.select_dtypes(include=[np.number]).columns
        df_list = [data]

        for _ in range(num_copies):
            new_rows = subset.copy()
            new_rows[numeric_cols] += np.random.normal(
                loc=0.0, 
                scale=noise_scale,
                size=new_rows[numeric_cols].shape
            )
            df_list.append(new_rows)
        data = pd.concat(df_list, ignore_index=True)

    def get_sample(self, target_data_path, workload_label):
        target_df = pd.read_csv(target_data_path)
        target_df = target_df[target_df['workload_label'] == workload_label]

        if len(target_df) < 2:
            raise ValueError("Not enough rows after filtering on default param values.")
        return target_df.iloc[[1]]


    def _prepare_target_data(self, target_data_path, workload_label):
        """
        Retrieves the target input, hardware label, and computes the target output vector
        if not already provided.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        file = os.path.basename(target_data_path)
        metric_path = os.path.join(self.data_dir, file)
        target_input = self.get_sample(metric_path, workload_label)
        hardware_label = f"{target_input['num_cpu'].iloc[0]}c{target_input['mem_size'].iloc[0]}g"

        # Ensure the model is trained
        if self.model is None:
            self.model = self.train_model(workload_label, hardware_label)

        # Compute target output vector if not provided
        self.model.eval()
        with torch.no_grad():
            input_features = [col for col in self.feature_columns if col in target_input.columns]
            target_input_sub = target_input[input_features]
            scaled_input = pd.DataFrame(
                self.scaler.transform(target_input_sub),
                columns=input_features,
                index=target_input_sub.index
            )
            target_tensor = torch.tensor(scaled_input.values, dtype=torch.float32).to(device)
            target_output = self.model(target_tensor)
            target_output_np = target_output.cpu().numpy()
            
        return target_input, hardware_label, target_output_np


    def retrieve_contexts(
        self, 
        target_context: Context,
        target_output_np=None
    ) -> list[ContextSimilarity]:
        """
        Retrieves the n "most similar" datasets by comparing the model outputs 
        (parameter-importance vectors) via *Mahalanobis distance*.
        Similarity is computed as: 1 / (1 + distance).
        """
        target_data_path = target_context.hardware
        workload_label = target_context.workload
        
        if target_output_np is None:
            _, hardware_label, target_output_np = self._prepare_target_data(
                target_data_path, workload_label
            )

        # Gather CSV files from data directory
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        dfs = []
        for file_path in csv_files:
            if "train.csv" in file_path or "88c190g" in file_path:
                continue
            dfs.append(pd.read_csv(file_path))

        divergences = []
        for df in dfs:
            file_hardware = f"{df['num_cpu'].iloc[0]}c{df['mem_size'].iloc[0]}g"
            if file_hardware == hardware_label and "hardware" in self.excluding_factors:
                continue

            df = df.drop(columns=df.columns[df.isnull().any()], errors="ignore")
            workloads = df['workload_label'].unique()

            for wl in workloads:
                if wl == workload_label and "workload" in self.excluding_factors:
                    continue
                subset = df[df['workload_label'] == wl]
                if 'label' not in subset.columns or len(subset) == 0:
                    continue

                try:
                    output_np = np.fromstring(subset.iloc[0]['label'].strip('[]'), sep=' ')
                except Exception:
                    continue

                # Compute divergence using KL divergence between vectors
                kl_div = entropy(target_output_np.flatten(), output_np)
                
                param_df_path = f"dataset/transfer_learning/mysql/chimera_tech/{file_hardware}-result.csv"
                param_df = read_data_csv(param_df_path, self.system, workload_label)
                
                divergences.append(ContextSimilarity(param_df, wl, file_hardware, kl_div, -kl_div))
                

        # Sort datasets by increasing divergence
        divergences.sort(key=lambda x: x.distance)
        return [divergence for divergence in divergences]

    def get_parameter_importance(self, target_data_path, workload_label):
        """
        Returns the parameter importance vector (model output) for the target dataset.
        Output elements are in the order defined by self.system.get_param_names().
        """
        _, _, target_output_np = self._prepare_target_data(target_data_path, workload_label)
        return target_output_np.flatten()

        
    def loocv(self, target_data_path, workloads):
        """
        Leave-one-out cross-validation for the target dataset.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load the sample row

        total_errors = 0
        results = {}
        for workload_label in workloads:
            target_input = self.get_sample(target_data_path, workload_label)
            true_output = np.fromstring(target_input.iloc[0]['label'].strip('[]'), sep=' ')
            hardware_label = f"{target_input['num_cpu'].iloc[0]}c{target_input['mem_size'].iloc[0]}g"
            # Train the model if not already loaded
            self.model = self.train_model(workload_label, hardware_label)

            # Generate or reuse the target output vector
            self.model.eval()
            with torch.no_grad():
                # Keep only known feature columns
                input_features = [
                    col for col in self.feature_columns if col in target_input.columns
                ]
                target_input_sub = target_input[input_features]

                # Scale
                scaled_input = pd.DataFrame(
                    self.scaler.transform(target_input_sub),
                    columns=input_features,
                    index=target_input_sub.index
                )

                # Convert to tensor and run the model
                target_tensor = torch.tensor(scaled_input.values, dtype=torch.float32).to(device)
                predicted_output = self.model(target_tensor)
                predicted_output_np = predicted_output.cpu().numpy()
            
            kl_div = entropy(true_output, predicted_output_np.flatten())
            # Compute KL divergence against uniform distribution
            print(f"Workload: {workload_label}, kl_div: {kl_div}")
            results[workload_label] = kl_div
            total_errors += kl_div

        average_div = total_errors / len(workloads)
        print(f"Average kl_div: {average_div}")
        # dict of workload_label -> kl_div
        df = pd.DataFrame(results, index=[0]).T
        df.columns = ['kl_divergence']
        df.index.name = 'workload_label'
        print(df)
        df.to_csv("loocv.csv", index=True)
        return average_div, results
