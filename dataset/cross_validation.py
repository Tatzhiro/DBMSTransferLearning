import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

from train import EmbeddingNet, train_model_function 

# Define the parameter columns used for configuration matching
CONFIG_PARAMS = [
    'innodb_buffer_pool_size', 'innodb_read_io_threads', 'innodb_write_io_threads',
    'innodb_flush_log_at_trx_commit', 'innodb_adaptive_hash_index', 'sync_binlog',
    'innodb_lru_scan_depth', 'innodb_buffer_pool_instances', 'innodb_change_buffer_max_size',
    'innodb_io_capacity', 'innodb_log_file_size', 'table_open_cache'
]

# --- Helper Functions ---

def parse_context_id(ctx_id):
    """Splits context_id into (hardware, workload)."""
    parts = ctx_id.split('_', 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (None, None)

def get_context_df(ctx_id):
    """Loads the result CSV for a specific context and filters by workload."""
    hw, wl = parse_context_id(ctx_id)
    file_path = f"{hw}-result.csv"
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    # Ensure consistent column naming for workload
    return df[df['workload_label'] == wl].copy()

def is_excluded(candidate_id, target_id, target_hw, target_wl, exclusion_level):
    """Centralized exclusion logic."""
    if candidate_id == target_id:
        return True
    if exclusion_level == 0:
        return False
    
    curr_hw, curr_wl = parse_context_id(candidate_id)
    if exclusion_level == 1: return curr_hw == target_hw
    if exclusion_level == 2: return curr_wl == target_wl
    if exclusion_level == 3: return curr_hw == target_hw or curr_wl == target_wl
    return False

# --- New Downstream Task Evaluation ---

def evaluate_tuning_transfer(target_id, predicted_id, k_configs=5):
    """
    Evaluates how well the top-k configurations from the predicted context 
    perform on the target context.
    """
    target_df = get_context_df(target_id)
    pred_df = get_context_df(predicted_id)

    if target_df is None or pred_df is None:
        print(f"      [Error] Could not load data for {target_id} or {predicted_id}")
        return 0.0

    # 1. Find Top-K configurations in the Predicted Context based on TPS
    top_pred_configs = pred_df.sort_values('tps', ascending=False).head(k_configs)
    
    # 2. Find the absolute best TPS possible in the Target Context (Ground Truth)
    target_max_tps = target_df['tps'].max()
    
    # 3. Test those Top-K configurations on the Target Context
    # We merge on the parameter values to find the target performance for those specific configs
    transfer_results = pd.merge(top_pred_configs[CONFIG_PARAMS], target_df[CONFIG_PARAMS + ['tps']], 
                                on=CONFIG_PARAMS, how='inner')
    
    if transfer_results.empty:
        print(f"      [Warning] No matching configurations found between contexts.")
        return 0.0

    # We take the best TPS achieved among the transferred configurations
    best_transferred_tps = transfer_results['tps'].max()
    tps_ratio = best_transferred_tps / target_max_tps
    
    return tps_ratio

# --- Existing Evaluation Logic (Modified for Top-1 predicted_id return) ---

def prepare_triplet_data(target_id, full_triplet_df, output_path):
    t_hw, t_wl = parse_context_id(target_id)
    def keep_id(cid):
        return not is_excluded(cid, target_id, t_hw, t_wl, exclusion_level=3)

    filtered_df = full_triplet_df[
        full_triplet_df['anchor_id'].apply(keep_id) &
        full_triplet_df['pos_id'].apply(keep_id) &
        full_triplet_df['neg_id'].apply(keep_id)
    ]
    return filtered_df


def find_nearest_context(target_id, metrics_df, model_path, feature_cols):
    """
    Loads the model, generates embeddings, and finds the nearest neighbor context_id
    that satisfies the exclusion criteria.
    """
    all_context_ids = metrics_df['context_id'].tolist()
    
    # Data Preparation
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(metrics_df[feature_cols])
    
    # Model Inference
    model = EmbeddingNet(input_dim=len(feature_cols), embedding_dim=16)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        embeddings = model(torch.tensor(features_scaled, dtype=torch.float32)).numpy()
    
    # Locate Target Embedding
    if target_id not in all_context_ids:
        print(f"      [Error] Target {target_id} not found in metrics.")
        return None

    idx = all_context_ids.index(target_id)
    target_emb = embeddings[idx].reshape(1, -1)
    
    # Calculate Distances
    distances = cdist(target_emb, embeddings, metric='euclidean').flatten()
    sorted_indices = np.argsort(distances)
    
    # Identify Prediction (Excluding strict matches based on Level 3)
    target_hw, target_wl = parse_context_id(target_id)
    predicted_id = None
    
    for i in sorted_indices:
        cid = all_context_ids[i]
        # We use exclusion_level=3 here to ensure we don't pick the same HW or Workload
        if not is_excluded(cid, target_id, target_hw, target_wl, exclusion_level=3):
            predicted_id = cid
            break
            
    return predicted_id

def evaluate_target_concordance(target_id, predicted_id):
    """
    Calculates the ratio of the score of the predicted_id vs the best possible score 
    in the answer_df (Ground Truth).
    Does NOT run inference.
    """
    if predicted_id is None:
        return 0.0
    
    answer_df = pd.read_csv('concordant_pair_ranking.csv')

    target_hw, target_wl = parse_context_id(target_id)

    # 1. Filter the Ground Truth (answer_df) to find the valid pool for this target
    #    We must apply the same exclusion logic to the ground truth to find the "Ideal" candidate.
    truth_pool = answer_df[answer_df['context_id'] == target_id].copy()
    
    # Remove invalid candidates from the ground truth pool
    truth_pool = truth_pool[~truth_pool['similar_context_id'].apply(
        lambda x: is_excluded(x, target_id, target_hw, target_wl, exclusion_level=3)
    )]
    
    if truth_pool.empty:
        return 0.0

    # 2. Get Score of the Predicted ID
    #    We check if our predicted_id exists in the ground truth pool and get its score
    score_map = dict(zip(truth_pool['similar_context_id'], truth_pool['similarity_score']))
    pred_score = score_map.get(predicted_id, 0)
    
    # 3. Get Score of the Ideal ID (Best possible in the pool)
    ideal_score = sorted(score_map.values(), reverse=True)[0]
    
    # 4. Calculate Ratio
    concordance = pred_score / ideal_score if ideal_score > 0 else 0
    return concordance

# --- Main Cross-Validation Loop ---

def run_cross_validation(target_ids, metrics_path, train_type):
    datasets = {
        "transfer": 'full_triplet_data_transfer.csv',
        "concordance": 'full_triplet_data_concordance.csv',
    }
    
    if train_type not in datasets:
        print(f"[Error] Unknown training type: {train_type}")
        return
    
    triplet_path = datasets[train_type]
    
    metrics_df = pd.read_csv(metrics_path)
    full_triplet_df = pd.read_csv(triplet_path)
    
    feature_cols = ['tps', 'Average Memory Usage Percentage', 'InnoDB Buffer Pool Cache Hit Rate',
                    'InnoDB Dirty Buffer Pages', 'Current QPS (Queries Per Second)',
                    'Max CPU Usage (100 - Idle)', 'InnoDB Rows Deleted (60s Rate)',
                    'InnoDB Rows Inserted (60s Rate)', 'InnoDB Rows Read (60s Rate)',
                    'InnoDB Rows Updated (60s Rate)', 'Average Disk IOPS (Read)',
                    'Average Disk IOPS (Write)']

    concordance_scores = []
    tuning_ratios = []

    for target_id in target_ids:
        print(f"\n>>> Starting CV for Target: {target_id}")
        
        # 1. Training (Skipped if model exists)
        tmp_triplet_csv = f"triplet_data_exclude_{target_id}.csv"
        triplet_df = prepare_triplet_data(target_id, full_triplet_df, tmp_triplet_csv)
        directory = f'models_{train_type}'
        os.makedirs(directory, exist_ok=True)
        model_name = f"{directory}/context_model_exclude_{target_id}.pth"
        
        if not os.path.exists(model_name):
            print(f"    Training model: {model_name}...")
            model = train_model_function(triplet_df.drop(columns=['anchor_id', 'pos_id', 'neg_id']))
            torch.save(model.state_dict(), model_name)
            
        
        # 2. Inference: Get the Predicted Context ID
        predicted_id = find_nearest_context(target_id, metrics_df, model_name, feature_cols)
        
        # 3. Evaluation Step A: Concordance (Ranking Quality)
        concordance = evaluate_target_concordance(target_id, predicted_id)
        concordance_scores.append(concordance)
        
        # 4. Evaluation Step B: Tuning Transfer (Downstream Task Performance)
        tps_ratio = evaluate_tuning_transfer(target_id, predicted_id, k_configs=5)
        tuning_ratios.append(tps_ratio)
        
        print(f"    - Predicted ID: {predicted_id}")
        print(f"    - Concordance Ratio: {concordance:.2%}")
        print(f"    - Tuning TPS Ratio:  {tps_ratio:.2%}")

    print("\n" + "="*40)
    print(f"CROSS VALIDATION COMPLETE")
    # print mean and variance of results
    mean_concordance = np.mean(concordance_scores)
    var_concordance = np.var(concordance_scores)
    mean_tuning = np.mean(tuning_ratios)
    var_tuning = np.var(tuning_ratios)
    print(f"Concordance - Mean: {mean_concordance:.2%}, Variance: {var_concordance:.4f}")
    print(f"Tuning TPS  - Mean: {mean_tuning:.2%}, Variance: {var_tuning:.4f}")
    print("="*40)
    
    
# --- Execution ---
targets = [
    "88c190g_10-4-4-tpcc-nan", "88c190g_100-4-4-tpcc-nan",
    "88c190g_64-1000000-4-oltp_write_only-0.2", "88c190g_64-1000000-4-oltp_write_only-0.6", "88c190g_64-1000000-4-oltp_write_only-1.0",
    "88c190g_64-1000000-4-oltp_read_only-0.2", "88c190g_64-1000000-4-oltp_read_only-0.6", "88c190g_64-1000000-4-oltp_read_only-1.0",
    "88c190g_64-1000000-4-oltp_read_write_5-0.2", "88c190g_64-1000000-4-oltp_read_write_5-0.6", "88c190g_64-1000000-4-oltp_read_write_5-1.0",
    "88c190g_64-1000000-4-oltp_read_write_20-0.2", "88c190g_64-1000000-4-oltp_read_write_20-0.6", "88c190g_64-1000000-4-oltp_read_write_20-1.0",
    "88c190g_64-1000000-4-oltp_read_write_50-0.2", "88c190g_64-1000000-4-oltp_read_write_50-0.6", "88c190g_64-1000000-4-oltp_read_write_50-1.0",
    "88c190g_64-1000000-4-oltp_read_write_80-0.2", "88c190g_64-1000000-4-oltp_read_write_80-0.6", "88c190g_64-1000000-4-oltp_read_write_80-1.0",
    "88c190g_64-1000000-4-oltp_read_write_95-0.2", "88c190g_64-1000000-4-oltp_read_write_95-0.6", "88c190g_64-1000000-4-oltp_read_write_95-1.0",
] # Add your CV targets here


run_cross_validation(
    targets, 
    'context_default_metrics_all.csv', 
    train_type='concordance'  # Choose 'transfer' or 'concordance'
)
