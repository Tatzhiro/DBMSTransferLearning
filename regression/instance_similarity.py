import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from regression.system_configuration import MySQLConfiguration


class InstanceSimilarity(ABC):
    """
    Abstract base class that defines the interface for dataset similarity.
    """
    @abstractmethod
    def get_similar_datasets(self, target_data_path, workload_label=None, n=2) -> list:
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

    def __init__(self, system: MySQLConfiguration, data_dir: str):
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
            file_hardware = f"{df['num_cpu'].iloc[0]}-{df['mem_size'].iloc[0]}"

            # Exclude matching hardware
            if file_hardware == hardware_label:
                continue

            # Preprocess param values
            df = self.system.preprocess_param_values(df)
            # Exclude matching workload
            df = df[df["workload_label"] != workload_label]
            dfs.append(df)

        if not dfs:
            # Fallback if no data is found
            return []

        concat_df = pd.concat(dfs, ignore_index=True)

        # Drop columns with NaNs
        X = concat_df.drop(columns=concat_df.columns[concat_df.isnull().any()])
        # Drop non-metric columns
        drop_cols = ["workload_label", "id", "label"] + self.system.get_param_names()
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

        # Factor Analysis
        fa = FactorAnalysis(random_state=42)
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
        target_df = pd.read_csv(target_data_path)
        hardware_label = f"{target_df['num_cpu'].iloc[0]}-{target_df['mem_size'].iloc[0]}"

        # Compute distinct_metrics if not already cached
        if self.distinct_metrics is None:
            self.distinct_metrics = self.prune_metrics(workload_label, hardware_label)

        # Preprocess target
        target_df = self.system.preprocess_param_values(target_df)
        target_df = target_df[target_df['workload_label'] == workload_label]

        # Keep only rows that match default param values
        for param in self.system.get_param_names():
            default_val = self.system.get_default_param_values().get(param)
            if default_val is not None:
                target_df = target_df[target_df[param] == default_val]

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
            file_hardware = f"{df['num_cpu'].iloc[0]}-{df['mem_size'].iloc[0]}"
            if file_hardware == hardware_label:
                continue

            df = self.system.preprocess_param_values(df)
            # Drop NaN columns
            df = df.drop(columns=df.columns[df.isnull().any()], errors="ignore")
           
            workloads = df['workload_label'].unique()
            for wl in workloads:
                if wl == workload_label:
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
        similarities = []
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

            # We store (data, wl, hardware_label, distance)
            similarities.append((data, wl, file_hw, dist))

        # -----------------------------------------
        # 6) CONVERT DISTANCE TO SIMILARITY + SORT
        # -----------------------------------------
        if not similarities:
            return [] if not metadata else []

        max_dist = max(s[3] for s in similarities)
        # similarity = 1 - dist/max_dist
        # If max_dist == 0, it means identical data somewhere. Handle that safely:
        if max_dist == 0:
            similarities = [(s[0], s[1], s[2], 1.0) for s in similarities]
        else:
            similarities = [(s[0], s[1], s[2], 1 - s[3]/max_dist) for s in similarities]

        similarities.sort(key=lambda x: x[3], reverse=True)

        if metadata:
            return similarities[:n]
        else:
            return [s[0] for s in similarities[:n]]

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


class ParameterImportanceSimilarity:
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

        def forward(self, x):
            x = F.leaky_relu(self.fc1(x))
            for layer in self.hidden_layers:
                x = F.leaky_relu(layer(x))
            x = self.output(x)
            # Softmax across dim=1
            x = F.softmax(x, dim=1)
            return x

    def __init__(self, system, data_dir: str, train_dir: str, seed: int = 42):
        self.system = system
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.config = None
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
            lambda row: f"{row['num_cpu']}-{row['mem_size']}", axis=1
        )
        # Filter out data matching hardware_label or workload_label
        data = data[data["hardware_label"] != hardware_label]
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
                "learning_rate": 0.0005,
                "num_epochs": 30,
                "num_hidden_layer": 20,
                "weight_decay": 1e-3
            }
            
        input_size = features.shape[1]
        # Build a 2D array of labels
        label_mat = np.vstack(train_data['label'].values)
        output_size = label_mat.shape[1] if len(label_mat) > 0 else 1

        # NEW: Compute covariance matrix of label vectors and invert it
        # We add a small regularization on the diagonal to avoid singular inversions.
        if len(label_mat) > 1:
            cov = np.cov(label_mat, rowvar=False)  # shape (d, d)
            # Add EPS to the diagonal for stability
            cov += EPS * np.eye(cov.shape[0], dtype=cov.dtype)

            cov_inv = np.linalg.inv(cov)
            self.label_cov_inv = cov_inv
        else:
            self.label_cov_inv = None  # not enough data to form a covariance matrix

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

        criterion = nn.MSELoss()
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
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {epoch_loss:.4f}")

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
        target_df = self.system.preprocess_param_values(target_df)
        target_df = target_df[target_df['workload_label'] == workload_label]

        for param in self.system.get_param_names():
            default_val = self.system.get_default_param_values().get(param)
            if default_val is not None:
                target_df = target_df[target_df[param] == default_val]

        if len(target_df) < 2:
            raise ValueError("Not enough rows after filtering on default param values.")
        return target_df.iloc[[1]]

    # NEW: Utility for Mahalanobis distance
    def mahalanobis_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        x, y: 1D vectors of the same dimensionality.
        returns: scalar distance
        """
        if self.label_cov_inv is None:
            # If there's no covariance info (e.g., not enough data), fallback
            # to e.g. Euclidean or just return 0. Or raise an error.
            diff = x - y
            return float(np.sqrt(np.sum(diff**2)))  # fallback: Euclidean

        diff = x - y
        # (1 x d) @ (d x d) @ (d x 1) => scalar
        dist_sq = diff @ self.label_cov_inv @ diff.T
        return float(np.sqrt(dist_sq))

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
        We'll convert that distance to similarity = 1 / (1 + distance).
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        target_input = self.get_sample(target_data_path, workload_label)
        hardware_label = f"{target_input['num_cpu'].iloc[0]}-{target_input['mem_size'].iloc[0]}"

        # Train the model if not already loaded
        if self.model is None:
            self.model = self.train_model(workload_label, hardware_label)

        # Generate or reuse the target output vector
        if target_output_np is None:
            self.model.eval()
            with torch.no_grad():
                input_features = [
                    col for col in self.feature_columns if col in target_input.columns
                ]
                target_input_sub = target_input[input_features]

                scaled_input = pd.DataFrame(
                    self.scaler.transform(target_input_sub),
                    columns=input_features,
                    index=target_input_sub.index
                )
                target_tensor = torch.tensor(scaled_input.values, dtype=torch.float32).to(device)
                target_output = self.model(target_tensor)
                target_output_np = target_output.cpu().numpy()  # shape: (1, output_size)

        # Gather CSV files
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        dfs = []
        for file_path in csv_files:
            if "train.csv" in file_path or "88c190g" in file_path:
                continue
            dfs.append(pd.read_csv(file_path))

        similarities = []
        for df in dfs:
            file_hardware = f"{df['num_cpu'].iloc[0]}-{df['mem_size'].iloc[0]}"
            if file_hardware == hardware_label:
                continue

            df = self.system.preprocess_param_values(df)
            df = df.drop(columns=df.columns[df.isnull().any()], errors="ignore")
            workloads = df['workload_label'].unique()

            for wl in workloads:
                if wl == workload_label:
                    continue
                subset = df[df['workload_label'] == wl]
                if 'label' not in subset.columns or len(subset) == 0:
                    continue

                try:
                    output_np = np.fromstring(subset.iloc[0]['label'].strip('[]'), sep=' ')
                except:
                    continue

                # NEW: Mahalanobis distance
                d_mah = self.mahalanobis_distance(target_output_np.flatten(), output_np)
                # Convert distance to similarity measure
                sim = 1.0 / (1.0 + d_mah)

                similarities.append((subset, wl, file_hardware, sim, output_np))

        # Sort descending by sim => "most similar" = highest similarity
        similarities.sort(key=lambda x: x[3], reverse=True)

        if metadata:
            return similarities[:n]
        else:
            return [s[0] for s in similarities[:n]]

        
    def loocv(self, target_data_path, workloads):
        """
        Leave-one-out cross-validation for the target dataset.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load the sample row

        total_errors = 0
        result = {}
        for workload_label in workloads:
            target_input = self.get_sample(target_data_path, workload_label)
            true_output = np.fromstring(target_input.iloc[0]['label'].strip('[]'), sep=' ')
            hardware_label = f"{target_input['num_cpu'].iloc[0]}-{target_input['mem_size'].iloc[0]}"
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
                target_output = self.model(target_tensor)
                target_output_np = target_output.cpu().numpy()
                
            d_mah = self.mahalanobis_distance(true_output, target_output_np)
                # Convert distance to similarity measure
            sim = 1.0 / (1.0 + d_mah)
            
            error = 1 - sim
            print(f"Workload: {workload_label}, error: {error}")
            result[workload_label] = error
            total_errors += error
        average_error = total_errors / len(workloads)
        print(f"Average Error: {average_error}")
        return average_error, result
        

def evaluate_similarity(target_csv, workload_label, similar_datasets, system, data_dir, plot=False):
    """
    Outputs the actual similarity scores and the most similar datasets.
    """
    target_df = pd.read_csv(target_csv)
    target_df = system.preprocess_param_values(target_df)
    target_df = target_df[target_df['workload_label'] == workload_label]
    hardware_label = f"{target_df['num_cpu'].iloc[0]}-{target_df['mem_size'].iloc[0]}"

    # Keep only rows that match the default param values
    for param in system.get_param_names():
        default_val = system.get_default_param_values().get(param)
        if default_val is not None:
            target_df = target_df[target_df[param] == default_val]

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
        
        
    similarities = []
    actual_similarities = []
    similar_context = [(entry[1], entry[2]) for entry in similar_datasets]
    for df in dfs:
        file_hardware = f"{df['num_cpu'].iloc[0]}-{df['mem_size'].iloc[0]}"
        if file_hardware == hardware_label:
            # Skip same hardware
            continue

        df = system.preprocess_param_values(df)
        df = df.drop(columns=df.columns[df.isnull().any()], errors="ignore")

        workloads = df['workload_label'].unique()
        for wl in workloads:
            if wl == workload_label:
                continue
            subset = df[df['workload_label'] == wl]
            if 'label' not in subset.columns or len(subset) == 0:
                continue

            # Convert the first row's label to a vector
            try:
                output_np = np.fromstring(subset.iloc[0]['label'].strip('[]'), sep=' ')
            except:
                continue

            # Cosine similarity
            sim = np.dot(target_output, output_np.T) / (
                np.linalg.norm(target_output) * np.linalg.norm(output_np)
            )
            similarities.append((wl, file_hardware, sim, output_np))
            
            if (wl, file_hardware) in similar_context:
                actual_similarities.append((wl, file_hardware, sim, output_np))

    print("\n[Actual Similarities]")
    for entry in actual_similarities:
        wl, hw, sim, vec = entry
        print(f"Workload: {wl}, Hardware: {hw}, Similarity: {sim}")
    
    # Sort descending by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    print("\n[Actual Similar Datasets]")
    for entry in similarities[:2]:
        wl, hw, sim, vec = entry
        print(f"Workload: {wl}, Hardware: {hw}, Similarity: {sim}")
        
    if plot:
        plot_pi(target_output, similarities[:2] + actual_similarities[:2], system, workload_label)
    
    
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
    data_dir = "dataset/transfer_learning/mysql/chimera_tech"
    target_csv = os.path.join(data_dir, "88c190g-result.csv")
    workloads = [
        "64-1000000-4-oltp_read_only-0.2",
        "64-1000000-4-oltp_read_only-0.6",
        "64-1000000-4-oltp_write_only-0.2",
        "64-1000000-4-oltp_write_only-0.6",
        "64-1000000-4-oltp_read_write_50-0.2",
        "64-1000000-4-oltp_read_write_50-0.6",
        "10-4-4-tpcc-nan",
        "100-4-4-tpcc-nan",
        
    ]
    
    param_similarity = ParameterImportanceSimilarity(
        system=MySQLConfiguration(),
        data_dir=data_dir,
        train_dir="dataset/metric_learning"
    )
    param_similarity.loocv(target_csv, workloads)
    ottertune_similarity = OtterTuneSimilarity(system=MySQLConfiguration(), data_dir=data_dir)
    
    
    for workload_label in workloads:
        print(f"Target CSV: {target_csv}, Workload: {workload_label}")

        ottertune_similarity.distinct_metrics = None
        similar_datasets = ottertune_similarity.get_similar_datasets(
            target_data_path=target_csv,
            workload_label=workload_label,
            n=2,
            metadata=True
        )
        print("\n[OtterTuneSimilarity] Found similar datasets:")
        for entry in similar_datasets:
            df, wl, hw, sim = entry
            print(f"Workload: {wl}, Hardware: {hw}, Similarity: {sim}")
            
        # Evaluate similarity scores
        evaluate_similarity(target_csv, workload_label, similar_datasets, MySQLConfiguration(), data_dir)

        # Example usage of ParameterImportanceSimilarity
        param_similarity.model = None
        similar_datasets_param = param_similarity.get_similar_datasets(
            target_data_path=target_csv,
            workload_label=workload_label,
            n=2,
            metadata=True
        )
        print("\n[ParameterImportanceSimilarity] Found similar datasets (metadata):")
        for entry in similar_datasets_param:
            df, wl, hw, sim, vec = entry
            print(f"Workload: {wl}, Hardware: {hw}, Similarity: {sim}")
            
        # Evaluate similarity scores
        evaluate_similarity(target_csv, workload_label, similar_datasets_param, MySQLConfiguration(), data_dir)
    


if __name__ == "__main__":
    main()
