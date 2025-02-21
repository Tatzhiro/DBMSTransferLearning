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


class OtterTuneSimilarity(InstanceSimilarity):
    """
    Finds similar datasets by pruning metrics (via FactorAnalysis) and comparing 
    Euclidean distances in the pruned metric space.
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
        and Euclidean distance in that metric space.
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
        target_df = target_df[valid_metrics]

        # Use a single row as the reference
        if len(target_df) == 0:
            raise ValueError("No valid target rows found after preprocessing and param filtering.")
        target_metrics = target_df.iloc[[0]]  # pick the first row or whichever row you prefer

        # Load candidate files
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
                # Skip same hardware
                continue
            # Preprocess
            df = self.system.preprocess_param_values(df)
            df = df.drop(columns=df.columns[df.isnull().any()], errors="ignore")

            workloads = df['workload_label'].unique()
            for wl in workloads:
                if wl == workload_label:
                    continue
                data = df[df['workload_label'] == wl]
                # Keep only relevant metrics
                data_metrics = data[valid_metrics].values
                # Euclidean distance between target and the candidate data
                dist = np.linalg.norm(target_metrics.values - data_metrics)
                similarities.append((data, wl, file_hardware, dist))

        # normalize distances and convert to similarity scores
        max_dist = max([s[3] for s in similarities])
        similarities = [(s[0], s[1], s[2], 1 - s[3] / max_dist) for s in similarities]
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[3], reverse=True)

        
        if metadata:
            return similarities[:n]
        else:
            return [s[0] for s in similarities[:n]]

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
    to important parameter vectors, then comparing these vectors (e.g., via cosine similarity).
    """
    _model = None
    _feature_columns = None
    _scaler = MinMaxScaler()

    class CustomDataset(Dataset):
        """
        Custom PyTorch Dataset for input -> (label) vector regression.
        """
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

        def forward(self, x):
            x = F.relu(self.fc1(x))
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
            x = self.output(x)
            # Softmax across dim=1
            x = F.softmax(x, dim=1)
            return x

    def __init__(self, system: MySQLConfiguration, data_dir: str, train_dir: str, seed: int = 42):
        """
        Parameters
        ----------
        system : MySQLConfiguration
        data_dir : str
            Directory containing CSV files for metric learning (e.g. train.csv).
        """
        self.system = system
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.set_seed(seed)

    @property
    def model(self):
        """Model getter."""
        return ParameterImportanceSimilarity._model

    @model.setter
    def model(self, val):
        """Model setter."""
        ParameterImportanceSimilarity._model = val

    @property
    def feature_columns(self):
        """Feature columns getter."""
        return ParameterImportanceSimilarity._feature_columns

    @feature_columns.setter
    def feature_columns(self, val):
        """Feature columns setter."""
        ParameterImportanceSimilarity._feature_columns = val

    @property
    def scaler(self):
        """Scaler getter."""
        return ParameterImportanceSimilarity._scaler
    
    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train_model(self, workload_label, hardware_label):
        """
        Trains a neural network on 'train.csv' in self.train_dir, excluding data that 
        matches the provided workload_label or hardware_label.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_path = os.path.join(self.train_dir, "train.csv")

        data = pd.read_csv(train_path)
        # Drop columns with NaNs
        data = data.drop(columns=data.columns[data.isnull().any()], errors="ignore")

        data['hardware_label'] = data.apply(
            lambda row: f"{row['num_cpu']}-{row['mem_size']}", axis=1
        )
        # Filter out data matching hardware_label or workload_label
        data = data[data["hardware_label"] != hardware_label]
        data = data[data['workload_label'] != workload_label]

        # Extract features and labels
        # 'label' is expected to hold a string like "[x y z ...]"
        features = data.drop(columns=['label', 'workload_label', 'hardware_label'], errors="ignore")
        self.feature_columns = features.columns

        # Convert label string -> np.array
        labels = data['label'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

        # Scale features
        normalized_features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=self.feature_columns,
            index=features.index
        )

        # Re-combine into a single dataframe if needed
        train_data = pd.concat([normalized_features, labels.rename("label")], axis=1)

        config = {
            "input_size": features.shape[1],
            "output_size": np.vstack(labels.values).shape[1] if len(labels) > 0 else 1,
            "num_hidden_layer": 10,
            "hidden_size": 128,
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 20
        }

        # Create PyTorch Dataset
        input_data = train_data.drop(columns=['label']).values
        target_data = np.vstack(train_data['label'].values)
        train_dataset = self.CustomDataset(input_data, target_data, device=device)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

        # Instantiate the model
        model = self.VectorPredictor(
            config["input_size"],
            config["num_hidden_layer"],
            config["hidden_size"],
            config["output_size"]
        ).to(device)

        # Define loss & optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)

        # Training loop
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
            # Optional: print or log epoch_loss if desired
            # print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {epoch_loss:.4f}")

        return model

    def get_sample(self, target_data_path, workload_label):
        """
        Loads and preprocesses a single row from the target CSV, matching the 
        default parameter values for the given workload_label.
        """
        target_df = pd.read_csv(target_data_path)
        target_df = self.system.preprocess_param_values(target_df)
        target_df = target_df[target_df['workload_label'] == workload_label]

        # Keep only rows that match the default param values
        for param in self.system.get_param_names():
            default_val = self.system.get_default_param_values().get(param)
            if default_val is not None:
                target_df = target_df[target_df[param] == default_val]

        # Use the second row if the first row might be invalid, etc.
        if len(target_df) < 2:
            raise ValueError("Not enough rows after filtering on default param values.")
        return target_df.iloc[[1]]

    def get_similar_datasets(
        self, 
        target_data_path, 
        workload_label, 
        n=2, 
        metadata=False, 
        target_output_np=None
    ) -> list:
        """
        Retrieves the n most similar datasets by comparing the model outputs 
        (parameter-importance vectors) via cosine similarity.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the sample row
        target_input = self.get_sample(target_data_path, workload_label)
        hardware_label = f"{target_input['num_cpu'].iloc[0]}-{target_input['mem_size'].iloc[0]}"

        # Train the model if not already loaded
        if self.model is None:
            self.model = self.train_model(workload_label, hardware_label)

        # Generate or reuse the target output vector
        if target_output_np is None:
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
                target_output_np = target_output.cpu().numpy()  # shape: (1, output_size)

        # Gather all candidate CSV files
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        dfs = []
        for file_path in csv_files:
            # Exclude special files if desired
            if "train.csv" in file_path or "88c190g" in file_path:
                continue
            dfs.append(pd.read_csv(file_path))

        # Compute similarities
        similarities = []
        for df in dfs:
            file_hardware = f"{df['num_cpu'].iloc[0]}-{df['mem_size'].iloc[0]}"
            if file_hardware == hardware_label:
                # Skip same hardware
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

                # Convert the first row's label to a vector
                try:
                    output_np = np.fromstring(subset.iloc[0]['label'].strip('[]'), sep=' ')
                except:
                    continue

                # Cosine similarity
                sim = np.dot(target_output_np, output_np.T) / (
                    np.linalg.norm(target_output_np) * np.linalg.norm(output_np)
                )
                similarities.append((subset, wl, file_hardware, sim, output_np))

        # Sort descending by similarity
        similarities.sort(key=lambda x: x[3], reverse=True)


        # Return either raw metadata or just the DataFrames
        if metadata:
            return similarities[:n]
        else:
            return [s[0] for s in similarities[:n]]
        

def evaluate_similarity(target_csv, workload_label, similar_datasets, system, data_dir):
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
    
    plot_pi(target_output, actual_similarities[:2], system)
        
    # Sort descending by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    print("\n[Actual Similar Datasets]")
    for entry in similarities[:2]:
        wl, hw, sim, vec = entry
        print(f"Workload: {wl}, Hardware: {hw}, Similarity: {sim}")
    
    
def plot_pi(target_output, similar_datasets, system):
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
    plt.show()
    plt.close()


def main():
    """
    Example usage of the refactored similarity classes.
    """
    data_dir = "dataset/transfer_learning/mysql/chimera_tech"
    target_csv = os.path.join(data_dir, "88c190g-result.csv")
    workload_label = "100-4-4-tpcc-nan"

    similarity = OtterTuneSimilarity(system=MySQLConfiguration(), data_dir=data_dir)
    similar_datasets = similarity.get_similar_datasets(
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
    param_similarity = ParameterImportanceSimilarity(
        system=MySQLConfiguration(),
        data_dir=data_dir,
        train_dir="dataset/metric_learning"
    )
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
