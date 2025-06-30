import glob
import os
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis

from regression.utils import read_data_csv
from regression.system_configuration import SystemConfiguration
from regression.context_retrieval import StaticContextRetrieval, Context, ContextSimilarity


class MetricSimRetrieval(StaticContextRetrieval):
    """
    Finds similar datasets by pruning metrics (via FactorAnalysis) and comparing 
    Euclidean distances in the pruned metric space, with OtterTune-style decile binning.
    """
    _distinct_metrics = None

    def __init__(self, system: SystemConfiguration, data_dir: str, excluding_factors=["hardware", "workload"]):
        """
        Parameters
        ----------
        system : SystemConfiguration
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
        return MetricSimRetrieval._distinct_metrics

    @distinct_metrics.setter
    def distinct_metrics(self, metrics):
        """Setter for the class-level distinct_metrics cache."""
        MetricSimRetrieval._distinct_metrics = metrics

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

    def retrieve_contexts(self, target_context: Context) -> list[ContextSimilarity]:
        """
        Given a target CSV, finds n most similar datasets based on pruned metrics 
        (decile-binned) and Euclidean distance in that space.
        """
        target_data_path = target_context.hardware
        workload_label = target_context.workload
        
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
            distances.append(ContextSimilarity(param_df, wl, file_hw, dist, -dist))

        distances.sort(key=lambda x: x.distance)

        return [distance for distance in distances]

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
