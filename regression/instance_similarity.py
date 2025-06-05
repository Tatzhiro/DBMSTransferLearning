import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import json

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from scipy.stats import entropy
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from regression.system_configuration import MySQLConfiguration
from regression.utils import read_data_csv


class InstanceSimilarity(ABC):
    """
    Abstract base class that defines the interface for dataset similarity.
    """
    
    class DatasetMetadata:
        def __init__(self, df, workload_label, hardware_label, distance):
            self.df = df
            self.workload_label = workload_label
            self.hardware_label = hardware_label
            self.distance = distance

    @abstractmethod
    def get_similar_datasets(self, target_data_path, workload_label=None, n=2, metadata=False) -> list:
        """
        Returns n most similar datasets to the target.
        """
        pass


import glob
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from typing import List, Dict

class OtterTuneSimilarity(InstanceSimilarity):
    """
    Finds similar datasets by pruning metrics (via FactorAnalysis) and comparing 
    Euclidean distances in the pruned metric space, with OtterTune-style decile binning.
    """
    _distinct_metrics = None

    def __init__(self, system: MySQLConfiguration, data_dir: str, excluding_factors=["hardware", "workload"]):
        """
        Parameters
        ----------
        system : MySQLConfiguration
            A system configuration object that handles parameter preprocessing.
        data_dir : str
            Path to the directory containing CSV files.
        """
        self.system = system
        self.data_dir = data_dir
        self.excluding_factors = excluding_factors

    @property
    def distinct_metrics(self):
        """Getter for the class-level distinct_metrics cache."""
        return OtterTuneSimilarity._distinct_metrics

    @distinct_metrics.setter
    def distinct_metrics(self, metrics):
        """Setter for the class-level distinct_metrics cache."""
        OtterTuneSimilarity._distinct_metrics = metrics

    def prune_metrics(self, workload_label, hardware_label):
        """
        Prunes metrics using FactorAnalysis, then KMeans clustering, and returns 
        a subset of metric names that are representative of each cluster centroid.
        """
        # Gather all CSV files in data_dir
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))

        dfs = []
        for file_path in csv_files:
            if "train.csv" in file_path or "88c190g" in file_path:
                # Skip specific files if needed
                continue
            df = pd.read_csv(file_path)
            # Generate the hardware label from the first row
            file_hardware = f"{df['num_cpu'].iloc[0]}c{df['mem_size'].iloc[0]}g"

            # Exclude matching hardware
            if file_hardware == hardware_label and "hardware" in self.excluding_factors:
                continue

            # Exclude matching workload
            if "workload" in self.excluding_factors:
                df = df[df["workload_label"] != workload_label]
                
            dfs.append(df)

        if not dfs:
            # Fallback if no data is found
            return []

        concat_df = pd.concat(dfs, ignore_index=True)
        
        # concat_df = pd.read_csv("dataset/metric_learning/train.csv")
        if "workload" in self.excluding_factors:
            concat_df = concat_df[concat_df['workload_label'] != workload_label]
        if "hardware" in self.excluding_factors:
            # Generate hardware label for filtering
            concat_df["hardware_label"] = concat_df.apply(
                lambda row: f"{row['num_cpu']}c{row['mem_size']}g", axis=1
            )
            concat_df = concat_df[concat_df['hardware_label'] != hardware_label]

        # Drop columns with NaNs
        X = concat_df.drop(columns=concat_df.columns[concat_df.isnull().any()])
        # Drop non-metric columns
        drop_cols = ["workload_label", "hardware_label", "id", "label"] + self.system.get_param_names()
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

        # Factor Analysis
        fa = FactorAnalysis()
        X_t = X.T  # shape => (n_features, n_samples)
        U = fa.fit_transform(X_t)  # shape => (n_features, n_components)

        # Choose a suitable k using the method below
        k = self.select_k_pham_dimov_nguyen(U)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(U)

        # Get indices of metrics closest to each centroid
        metrics_indices = self.get_closest_samples_to_centroids(U, kmeans)
        metrics = X_t.iloc[metrics_indices].index.values
        return metrics

    def get_similar_datasets(self, target_data_path, workload_label=None, n=2, metadata=False) -> list:
        """
        Given a target CSV, finds n most similar datasets based on pruned metrics 
        (decile-binned) and Euclidean distance in that space.
        """
        # Read target data
        file = os.path.basename(target_data_path)
        metric_path = os.path.join(self.data_dir, file)
        target_df = pd.read_csv(metric_path)
        hardware_label = f"{target_df['num_cpu'].iloc[0]}c{target_df['mem_size'].iloc[0]}g"


        metrics_path = f"neural_network/metrics/exclude-{'-'.join(self.excluding_factors)}/{workload_label}_{hardware_label}.txt"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = f.read().splitlines()
            self.distinct_metrics = metrics
        else:
            self.distinct_metrics = self.prune_metrics(workload_label, hardware_label)
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                for metric in self.distinct_metrics:
                    f.write(f"{metric}\n")

        target_df = target_df[target_df['workload_label'] == workload_label]

        # Focus only on the pruned metrics
        valid_metrics = [m for m in self.distinct_metrics if m in target_df.columns]
        # If no valid target rows exist, raise an error
        if len(target_df) == 0:
            raise ValueError("No valid target rows found after preprocessing and param filtering.")

        # We'll compare using the first row as the target
        target_metrics = target_df.iloc[[0]][valid_metrics].copy()

        # -----------------------------------------
        # 1) LOAD CANDIDATE DATA
        # -----------------------------------------
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        dfs = []
        for file_path in csv_files:
            if "train.csv" in file_path or "88c190g" in file_path:
                continue
            df = pd.read_csv(file_path)
            dfs.append(df)
            
        candidate_dfs = []
        for df in dfs:
            # Skip same hardware
            file_hardware = f"{df['num_cpu'].iloc[0]}c{df['mem_size'].iloc[0]}g"
            if file_hardware == hardware_label and "hardware" in self.excluding_factors:
                continue

            # Drop NaN columns
            df = df.drop(columns=df.columns[df.isnull().any()], errors="ignore")
           
            workloads = df['workload_label'].unique()
            for wl in workloads:
                if wl == workload_label and "workload" in self.excluding_factors:
                    continue

                data = df[df['workload_label'] == wl]
                if data.empty:
                    continue

                candidate_dfs.append((data, file_hardware, wl))
                
        # -----------------------------------------
        # 2) COMBINE TARGET + CANDIDATES TO COMPUTE DECILES
        # -----------------------------------------
        # Build one combined DataFrame (only the columns we need for binning).
        # We'll then apply those binning edges to each dataset individually.
        combined_df = pd.concat([target_metrics] + [d[0][valid_metrics] for d in candidate_dfs], ignore_index=True)

        # Compute decile bin edges for each metric
        bin_edges_dict = self._compute_decile_edges(combined_df, valid_metrics)

        # -----------------------------------------
        # 4) BIN THE TARGET ROW
        # -----------------------------------------
        binned_target = target_metrics.copy()
        for metric in valid_metrics:
            edges = bin_edges_dict[metric]
            # Use np.digitize to map the raw values to bin indices [1..10]
            binned_target[metric] = np.digitize(binned_target[metric], edges, right=True)

        # -----------------------------------------
        # 5) FOR EACH CANDIDATE, BIN + COMPUTE DISTANCE
        # -----------------------------------------
        distances = []
        for (data, file_hw, wl) in candidate_dfs:
            # Binning each row in this candidate segment
            data_for_dist = data[valid_metrics].copy()
            for metric in valid_metrics:
                edges = bin_edges_dict[metric]
                data_for_dist[metric] = np.digitize(data_for_dist[metric], edges, right=True)

            # Now compute distance (Euclidean) between binned_target and the entire candidate set.
            # You might decide to average across rows, or just pick the first row, etc.
            # Below, we flatten them so we can do a direct norm. Another approach is to pick
            # a single representative row from `data_for_dist`.
            
            # get the single representative row
            data_for_dist = data_for_dist.iloc[[0]]
            dist = np.linalg.norm(binned_target.values - data_for_dist.values)
            
            param_df_path = f"dataset/transfer_learning/mysql/chimera_tech/{file_hw}-result.csv"
            param_df = read_data_csv(param_df_path, self.system, wl)
            # We store (data, wl, hardware_label, distance)
            distances.append(self.DatasetMetadata(param_df, wl, file_hw, dist))

        distances.sort(key=lambda x: x.distance)

        if metadata:
            return distances[:n]
        else:
            return [s[0] for s in distances[:n]]

    # -------------------------------------------------------------------------
    # HELPER FOR COMPUTING DECILE EDGES
    # -------------------------------------------------------------------------
    def _compute_decile_edges(self, df: pd.DataFrame, metrics: List[str]) -> Dict[str, np.ndarray]:
        """
        For each metric in `metrics`, compute the bin edges for deciles using the
        data in `df`. Returns a dict of {metric_name: bin_edges}.
        """
        bin_edges_dict = {}
        for metric in metrics:
            # Drop NaNs just in case
            vals = df[metric].dropna().values
            # 10 deciles => 0%, 10%, 20%, ..., 100%
            # That gives 11 edges total
            edges = np.percentile(vals, np.linspace(0, 100, 11))
            bin_edges_dict[metric] = edges
        return bin_edges_dict

    @staticmethod
    def get_closest_samples_to_centroids(U, kmeans):
        """
        Given data U (shape: n_samples x n_features) and a fitted KMeans model, 
        returns a list of sample indices closest to each cluster centroid.
        """
        if not isinstance(U, np.ndarray):
            U = U.values

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        k = centroids.shape[0]

        closest_dist = [np.inf] * k
        closest_index = [-1] * k

        # Iterate over each sample
        for i in range(U.shape[0]):
            cluster_id = labels[i]
            dist = np.linalg.norm(U[i] - centroids[cluster_id])
            if dist < closest_dist[cluster_id]:
                closest_dist[cluster_id] = dist
                closest_index[cluster_id] = i
        return closest_index

    @staticmethod
    def select_k_pham_dimov_nguyen(X, k_max=10, alpha=1.0, threshold=1.0, random_state=42):
        """
        Selects the number of clusters K in K-means according to
        the method of Pham, Dimov, and Nguyen (2005).

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        k_max : int, default=10
        alpha : float, default=1.0
        threshold : float, default=1.0
        random_state : int, default=42

        Returns
        -------
        best_k : int
            The selected number of clusters.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        f_values = []
        for k in range(1, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
            kmeans.fit(X)
            f_values.append(kmeans.inertia_)

        # Delta values
        delta_values = [0]  # delta(1) = 0 placeholder
        for k in range(2, k_max + 1):
            numerator = f_values[k - 2] - f_values[k - 1]
            denominator = f_values[k - 2]
            delta_k = numerator / denominator if denominator != 0 else 0
            delta_values.append(delta_k)

        # Ratio values
        ratio_values = [0, 0]  # ratio(1) and ratio(2) placeholders
        for k in range(3, k_max + 1):
            if delta_values[k - 2] > 0:
                ratio_k = delta_values[k - 1] / (delta_values[k - 2] ** alpha)
            else:
                ratio_k = 0
            ratio_values.append(ratio_k)

        best_k = 1
        for k in range(3, k_max + 1):
            if ratio_values[k - 1] >= threshold:
                best_k = k

        # Edge case if best_k remains 1
        if best_k == 1 and k_max >= 2:
            # Optionally check delta_values[1] (k=2)
            if delta_values[1] > 0.0:
                best_k = 2

        return best_k


class ParameterImportanceSimilarity(InstanceSimilarity):
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
        return ParameterImportanceSimilarity._model

    @model.setter
    def model(self, val):
        ParameterImportanceSimilarity._model = val

    @property
    def feature_columns(self):
        return ParameterImportanceSimilarity._feature_columns

    @feature_columns.setter
    def feature_columns(self, val):
        ParameterImportanceSimilarity._feature_columns = val

    @property
    def scaler(self):
        return ParameterImportanceSimilarity._scaler

    # NEW: Covariance inverse getter/setter
    @property
    def label_cov_inv(self):
        return ParameterImportanceSimilarity._label_cov_inv

    @label_cov_inv.setter
    def label_cov_inv(self, val):
        ParameterImportanceSimilarity._label_cov_inv = val

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


    def get_similar_datasets(
        self, 
        target_data_path, 
        workload_label, 
        n=2, 
        metadata=False, 
        target_output_np=None
    ) -> list:
        """
        Retrieves the n "most similar" datasets by comparing the model outputs 
        (parameter-importance vectors) via *Mahalanobis distance*.
        Similarity is computed as: 1 / (1 + distance).
        """
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
                
                divergences.append(self.DatasetMetadata(param_df, wl, file_hardware, kl_div))
                

        # Sort datasets by increasing divergence
        divergences.sort(key=lambda x: x.distance)

        if metadata:
            return divergences[:n]
        else:
            return [s.df for s in divergences[:n]]
    

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
        results = {"kl_div": [], "random_kl_div": []}
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
            random_kl_div = entropy(true_output, np.ones_like(true_output) / len(true_output))
            print(f"Workload: {workload_label}, kl_div: {kl_div}, random_kl_div: {random_kl_div}")
            results["kl_div"].append(kl_div)
            results["random_kl_div"].append(random_kl_div)
            total_errors += kl_div
            plot_pi(
                true_output, 
                [
                    (workload_label, hardware_label, kl_div, predicted_output_np.flatten()),
                    (workload_label, hardware_label, random_kl_div, np.ones_like(true_output) / len(true_output))
                ], 
                self.system, workload_label
            )
        average_div = total_errors / len(workloads)
        print(f"Average kl_div: {average_div}")
        # dict of workload_label -> kl_div, random_kl_div
        df = pd.DataFrame(results, index=workloads)
        print(df)
        df.to_csv("loocv.csv", index=True)
        return average_div, results
        

def evaluate_similarity(target_csv, workload_label, similar_datasets, system, data_dir, plot=False, excluding_factors=["hardware", "workload"]):
    """
    Outputs the actual similarity scores and the most similar datasets.
    """
    target_df = pd.read_csv(target_csv)
    target_df = target_df[target_df['workload_label'] == workload_label]
    hardware_label = f"{target_df['num_cpu'].iloc[0]}c{target_df['mem_size'].iloc[0]}g"

    # Use the second row if the first row might be invalid, etc.
    if len(target_df) < 2:
        raise ValueError("Not enough rows after filtering on default param values.")
    
    target_output = np.fromstring(target_df.iloc[1]['label'].strip('[]'), sep=' ')
    
    # Compute similarities
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    dfs = []
    for file_path in csv_files:
        # Exclude special files if desired
        if "train.csv" in file_path or "88c190g" in file_path:
            continue
        dfs.append(pd.read_csv(file_path))
        
        
    divergences = []
    actual_divergences = []
    similar_context = [(entry.workload_label, entry.hardware_label) for entry in similar_datasets]
    for df in dfs:
        file_hardware = f"{df['num_cpu'].iloc[0]}c{df['mem_size'].iloc[0]}g"
        if file_hardware == hardware_label and "hardware" in excluding_factors:
            # Skip same hardware
            continue

        df = df.drop(columns=df.columns[df.isnull().any()], errors="ignore")

        workloads = df['workload_label'].unique()
        for wl in workloads:
            if wl == workload_label and "workload" in excluding_factors:
                continue
            subset = df[df['workload_label'] == wl]
            if 'label' not in subset.columns or len(subset) == 0:
                continue

            # Convert the first row's label to a vector
            try:
                past_output = np.fromstring(subset.iloc[0]['label'].strip('[]'), sep=' ')
            except:
                continue

            kl_div = entropy(target_output, past_output)
            divergences.append((wl, file_hardware, kl_div, past_output))
            
            if (wl, file_hardware) in similar_context:
                actual_divergences.append((wl, file_hardware, kl_div, past_output))

    print("\n[Actual Similarities]")
    for entry in actual_divergences:
        wl, hw, kl_div, vec = entry
        print(f"Workload: {wl}, Hardware: {hw}, KL Divergence: {kl_div}")
    
    # Sort ascending by divergence
    divergences.sort(key=lambda x: x[2], reverse=False)
    print("\n[Actual Similar Datasets]")
    for entry in divergences[:2]:
        wl, hw, kl_div, vec = entry
        print(f"Workload: {wl}, Hardware: {hw}, KL Divergence: {kl_div}")
        
    if plot:
        plot_pi(target_output, divergences[:2] + actual_divergences[:2], system, workload_label)
        
    return actual_divergences
    
    
def plot_pi(target_output, similar_datasets, system, workload_label):
    """
    Plot grouped bar chart of target_output and similar datasets.
    """
    labels = system.get_param_names()
    target_output = target_output
    similar_outputs = [entry[3] for entry in similar_datasets]
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.bar(x - width, target_output, width, label='Target')
    for i, output in enumerate(similar_outputs):
        ax.bar(x + width * i, output, width, label=f'Similar {i+1}')
        
    x_labels = [param.replace("innodb_", "") for param in system.get_param_names()]
    ax.set_xticks(x, x_labels, rotation=45)
        
    ax.set_ylabel('Parameter Importance')
    plt.legend()
    plt.savefig(workload_label + ".pdf")
    plt.close()


def main():
    """
    Example usage of the refactored similarity classes.
    """
    data_dir = "dataset/metric_learning/chimera_tech"
    hardware_label = "88c190g"
    target_csv = os.path.join(data_dir, f"{hardware_label}-result.csv")
    workloads = [
        "64-1000000-4-oltp_read_only-0.2",
        "64-1000000-4-oltp_read_only-0.6",
        "64-1000000-4-oltp_read_only-1.0",
        "64-1000000-4-oltp_write_only-0.2",
        "64-1000000-4-oltp_write_only-0.6",
        "64-1000000-4-oltp_write_only-1.0",
        "64-1000000-4-oltp_read_write_95-0.2",
        "64-1000000-4-oltp_read_write_95-0.6",
        "64-1000000-4-oltp_read_write_95-1.0",
        "64-1000000-4-oltp_read_write_80-0.2",
        "64-1000000-4-oltp_read_write_80-0.6",
        "64-1000000-4-oltp_read_write_80-1.0",
        "64-1000000-4-oltp_read_write_50-0.2",
        "64-1000000-4-oltp_read_write_50-0.6",
        "64-1000000-4-oltp_read_write_50-1.0",
        "64-1000000-4-oltp_read_write_20-0.2",
        "64-1000000-4-oltp_read_write_20-0.6",
        "64-1000000-4-oltp_read_write_20-1.0",
        "64-1000000-4-oltp_read_write_5-0.2",
        "64-1000000-4-oltp_read_write_5-0.6",
        "64-1000000-4-oltp_read_write_5-1.0",
        "10-4-4-tpcc-nan",
        "100-4-4-tpcc-nan",
    ]
    excluding_factors = ["hardware", "workload"]
    output_name = f"similarity_scores_exclude-{'-'.join(excluding_factors)}_target-{hardware_label}"
    
    param_similarity = ParameterImportanceSimilarity(
        system=MySQLConfiguration(),
        data_dir=data_dir,
        train_dir="dataset/metric_learning",
        excluding_factors=excluding_factors
    )
    param_similarity.loocv(target_csv, workloads)
    ottertune_similarity = OtterTuneSimilarity(
        system=MySQLConfiguration(), 
        data_dir=data_dir, 
        excluding_factors=excluding_factors
    )
    
    
    data = {"ChimeraTech (Proposed)": [], "OtterTune": []}
    for workload_label in workloads:
        print(f"\nTarget CSV: {target_csv}, Workload: {workload_label}")

        ottertune_similarity.distinct_metrics = None
        similar_datasets = ottertune_similarity.get_similar_datasets(
            target_data_path=target_csv,
            workload_label=workload_label,
            n=1,
            metadata=True
        )
        print("\n[OtterTuneSimilarity] Found similar datasets:")
        for entry in similar_datasets:
            print(f"Workload: {entry.workload_label}, Hardware: {entry.hardware_label}, Euclidean Distance: {entry.distance}")
            
        # Evaluate similarity scores
        ottertune_result = evaluate_similarity(target_csv, workload_label, similar_datasets, MySQLConfiguration(), data_dir, excluding_factors=excluding_factors)
        data["OtterTune"].append(ottertune_result[0][2])

        # Example usage of ParameterImportanceSimilarity
        param_similarity.model = None
        similar_datasets_param = param_similarity.get_similar_datasets(
            target_data_path=target_csv,
            workload_label=workload_label,
            n=1,
            metadata=True
        )
        print("\n[ParameterImportanceSimilarity] Found similar datasets (metadata):")
        for entry in similar_datasets_param:
            print(f"Workload: {entry.workload_label}, Hardware: {entry.hardware_label}, Similarity: {entry.distance}")
            
        # Evaluate similarity scores
        param_sim_result = evaluate_similarity(target_csv, workload_label, similar_datasets_param, MySQLConfiguration(), data_dir, excluding_factors=excluding_factors)
        data["ChimeraTech (Proposed)"].append(param_sim_result[0][2])
        
    df = pd.DataFrame(data, index=workloads)
    df.to_csv(f"{output_name}.csv")
    
    data = pd.read_csv(f"{output_name}.csv", index_col=0)
        # Categorize into workload groups
    def categorize(workload):
        if "read_only" in workload:
            return "RO"
        elif "write_only" in workload:
            return "WO"
        elif "read_write_95" in workload:
            return "RW_95"
        elif "read_write_80" in workload:
            return "RW_80"
        elif "read_write_50" in workload:
            return "RW_50"
        elif "read_write_20" in workload:
            return "RW_20"
        elif "read_write_5" in workload:
            return "RW_5"
        elif "tpcc" in workload:
            return "TPC-C"
        else:
            return "Other"

    data["Group"] = data.index.map(categorize)

    # Group by category and take mean
    grouped_df = data.groupby("Group")[["ChimeraTech (Proposed)", "OtterTune"]].mean().reset_index()

    # Custom order
    custom_order = ["TPC-C", "RO", "WO", "RW_95", "RW_80", "RW_50", "RW_20", "RW_5"]
    grouped_df = grouped_df.set_index("Group").loc[custom_order].reset_index()
    plot_kl_divergence(grouped_df, custom_order, save_path=f"{output_name}.pdf")
    
    
def wl():
    data_dir = "dataset/metric_learning/chimera_tech"
    hardware_label = "32c64g"
    target_csv = os.path.join(data_dir, f"{hardware_label}-result.csv")
    workloads = [
        "64-1000000-4-oltp_read_only-0.2",
        "64-1000000-4-oltp_read_only-0.6",
        "64-1000000-4-oltp_read_only-1.0",
        "64-1000000-4-oltp_write_only-0.2",
        "64-1000000-4-oltp_write_only-0.6",
        "64-1000000-4-oltp_write_only-1.0",
        "64-1000000-4-oltp_read_write_95-0.2",
        "64-1000000-4-oltp_read_write_95-0.6",
        "64-1000000-4-oltp_read_write_95-1.0",
        "64-1000000-4-oltp_read_write_80-0.2",
        "64-1000000-4-oltp_read_write_80-0.6",
        "64-1000000-4-oltp_read_write_80-1.0",
        "64-1000000-4-oltp_read_write_50-0.2",
        "64-1000000-4-oltp_read_write_50-0.6",
        "64-1000000-4-oltp_read_write_50-1.0",
        "64-1000000-4-oltp_read_write_20-0.2",
        "64-1000000-4-oltp_read_write_20-0.6",
        "64-1000000-4-oltp_read_write_20-1.0",
        "64-1000000-4-oltp_read_write_5-0.2",
        "64-1000000-4-oltp_read_write_5-0.6",
        "64-1000000-4-oltp_read_write_5-1.0",
        "10-4-4-tpcc-nan",
        "100-4-4-tpcc-nan",
    ]
    excluding_factors = ["workload"]
    
    param_similarity = ParameterImportanceSimilarity(
        system=MySQLConfiguration(),
        data_dir=data_dir,
        train_dir="dataset/metric_learning",
        excluding_factors=excluding_factors
    )
    param_similarity.loocv(target_csv, workloads)
    ottertune_similarity = OtterTuneSimilarity(
        system=MySQLConfiguration(), 
        data_dir=data_dir,
        excluding_factors=excluding_factors
    )
    
    
    data = {"ChimeraTech (Proposed)": [], "OtterTune": []}
    for workload_label in workloads:
        print(f"\nTarget CSV: {target_csv}, Workload: {workload_label}")

        ottertune_similarity.distinct_metrics = None
        similar_datasets = ottertune_similarity.get_similar_datasets(
            target_data_path=target_csv,
            workload_label=workload_label,
            n=1,
            metadata=True
        )
        print("\n[OtterTuneSimilarity] Found similar datasets:")
        for entry in similar_datasets:
            print(f"Workload: {entry.workload_label}, Hardware: {entry.hardware_label}, Euclidean Distance: {entry.distance}")
            
        # Evaluate similarity scores
        ottertune_result = evaluate_similarity(target_csv, workload_label, similar_datasets, MySQLConfiguration(), data_dir, excluding_factors=excluding_factors)
        data["OtterTune"].append(ottertune_result[0][2])

        # Example usage of ParameterImportanceSimilarity
        param_similarity.model = None
        similar_datasets_param = param_similarity.get_similar_datasets(
            target_data_path=target_csv,
            workload_label=workload_label,
            n=1,
            metadata=True
        )
        print("\n[ParameterImportanceSimilarity] Found similar datasets (metadata):")
        for entry in similar_datasets_param:
            print(f"Workload: {entry.workload_label}, Hardware: {entry.hardware_label}, Similarity: {entry.distance}")
            
        # Evaluate similarity scores
        param_sim_result = evaluate_similarity(target_csv, workload_label, similar_datasets_param, MySQLConfiguration(), data_dir, excluding_factors=excluding_factors)
        data["ChimeraTech (Proposed)"].append(param_sim_result[0][2])
        
    df = pd.DataFrame(data, index=workloads)
    output_name = f"similarity_scores_exclude-{'-'.join(excluding_factors)}_target-{hardware_label}"
    df.to_csv(f"{output_name}.csv")
    
    data = pd.read_csv(f"{output_name}.csv", index_col=0)
        # Categorize into workload groups
    def categorize(workload):
        if "read_only" in workload:
            return "RO"
        elif "write_only" in workload:
            return "WO"
        elif "read_write" in workload:
            return "RW"
        elif "tpcc" in workload:
            return "TPC-C"
        else:
            return "Other"

    data["Group"] = data.index.map(categorize)

    # Group by category and take mean
    grouped_df = data.groupby("Group")[["ChimeraTech (Proposed)", "OtterTune"]].mean().reset_index()

    # Custom order
    custom_order = ["TPC-C", "RO", "WO", "RW"]
    grouped_df = grouped_df.set_index("Group").loc[custom_order].reset_index()
    plot_kl_divergence(grouped_df, custom_order, save_path=f"{output_name}.pdf")
    
    
def hw():
    """
    Example usage of the refactored similarity classes.
    """
    data_dir = "dataset/metric_learning/chimera_tech"
    hardwares = [
        "4c6g",
        "8c12g",
        "12c16g",
        "16c24g",
        "24c32g",
        "32c64g",
        "88c190g",
    ]
    workload_label = "100-4-4-tpcc-nan"
    excluding_factors = ["hardware"]
    
    param_similarity = ParameterImportanceSimilarity(
        system=MySQLConfiguration(),
        data_dir=data_dir,
        train_dir="dataset/metric_learning",
        excluding_factors=excluding_factors
    )
    # param_similarity.loocv(target_csv, workloads)
    ottertune_similarity = OtterTuneSimilarity(
        system=MySQLConfiguration(), 
        data_dir=data_dir,
        excluding_factors=excluding_factors
    )
    
    
    data = {"ChimeraTech (Proposed)": [], "OtterTune": []}
    for hardware in hardwares:
        target_csv = os.path.join(data_dir, f"{hardware}-result.csv")
        print(f"\nTarget CSV: {target_csv}, Workload: {workload_label}, Hardware: {hardware}")

        ottertune_similarity.distinct_metrics = None
        similar_datasets = ottertune_similarity.get_similar_datasets(
            target_data_path=target_csv,
            workload_label=workload_label,
            n=1,
            metadata=True
        )
        print("\n[OtterTuneSimilarity] Found similar datasets:")
        for entry in similar_datasets:
            print(f"Workload: {entry.workload_label}, Hardware: {entry.hardware_label}, Euclidean Distance: {entry.distance}")
            
        # Evaluate similarity scores
        ottertune_result = evaluate_similarity(target_csv, workload_label, similar_datasets, MySQLConfiguration(), data_dir, excluding_factors=excluding_factors)
        data["OtterTune"].append(ottertune_result[0][2])

        # Example usage of ParameterImportanceSimilarity
        param_similarity.model = None
        similar_datasets_param = param_similarity.get_similar_datasets(
            target_data_path=target_csv,
            workload_label=workload_label,
            n=1,
            metadata=True
        )
        print("\n[ParameterImportanceSimilarity] Found similar datasets (metadata):")
        for entry in similar_datasets_param:
            print(f"Workload: {entry.workload_label}, Hardware: {entry.hardware_label}, Similarity: {entry.distance}")
            
        # Evaluate similarity scores
        param_sim_result = evaluate_similarity(target_csv, workload_label, similar_datasets_param, MySQLConfiguration(), data_dir, excluding_factors=excluding_factors)
        data["ChimeraTech (Proposed)"].append(param_sim_result[0][2])
    
    hw_labels = [
        "small",
        "medium",
        "large",
        "xlarge",
        "xxlarge",
        "xxxlarge",
        "prod"
    ]
    
    df = pd.DataFrame(data, index=hw_labels)
    output_name = f"similarity_scores_exclude-{'-'.join(excluding_factors)}_target-{workload_label}"
    df.to_csv(f"{output_name}.csv")
    
    data = pd.read_csv(f"{output_name}.csv", index_col=0)
    plot_kl_divergence(data, hw_labels, save_path=f"{output_name}.pdf")
    

def plot_kl_divergence(
    data,
    indices, 
    save_path="kl_divergence_plot.pdf"
):
    """
    Plots the KL divergence between the target output and similar datasets.
    """
    # Plot the results
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    x = np.arange(len(indices))
    width = 0.35
    ax.bar(x - width/2, data["ChimeraTech (Proposed)"], width, label='ChimeraTech (Proposed)', color='red', zorder=3)
    ax.bar(x + width/2, data["OtterTune"], width, label='OtterTune', color='blue', zorder=3)
    ax.set_ylabel('KL Divergence', fontsize=24)
    ax.set_xticks(x)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_xticklabels(indices, fontsize=20)
    ax.grid(True, axis='y', zorder=0)
    # ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()
    # hw()
    # wl()
