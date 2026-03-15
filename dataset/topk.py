import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

from train import EmbeddingNet


def parse_context_id(ctx_id):
    """
    Parses context_id (e.g., '88c190g_10-4-4-tpcc-nan') 
    into (hardware, workload).
    """
    parts = ctx_id.split('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


def parse_context_id(ctx_id):
    """Splits context_id into (hardware, workload)."""
    parts = ctx_id.split('_', 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (None, None)


def is_excluded(candidate_id, target_hw, target_wl, exclusion_level):
    """
    Centralized logic to determine if a candidate context should be filtered out.
    """
    if exclusion_level == 0:
        return False
    
    curr_hw, curr_wl = parse_context_id(candidate_id)
    
    if exclusion_level == 1:
        return curr_hw == target_hw
    if exclusion_level == 2:
        return curr_wl == target_wl
    if exclusion_level == 3:
        return curr_hw == target_hw or curr_wl == target_wl
    return False


def get_top_k_for_context(target_id, context_ids, embeddings, k=5, exclusion_level=0):
    if target_id not in context_ids:
        return []
    
    target_hw, target_wl = parse_context_id(target_id)
    idx = context_ids.index(target_id)
    target_emb = embeddings[idx].reshape(1, -1)
    
    distances = cdist(target_emb, embeddings, metric='euclidean').flatten()
    sorted_indices = np.argsort(distances)
    
    filtered_top_k = []
    for i in sorted_indices:
        candidate_id = context_ids[i]
        
        # Always exclude the target itself
        if candidate_id == target_id:
            continue
            
        if is_excluded(candidate_id, target_hw, target_wl, exclusion_level):
            continue
            
        filtered_top_k.append(candidate_id)
        if len(filtered_top_k) == k:
            break
            
    return filtered_top_k


def calculate_overlap_for_context(target_id, context_ids, embeddings, answer_df, k=5, exclusion_level=0):
    target_hw, target_wl = parse_context_id(target_id)

    # 1. Get Predicted Top-K
    predicted = set(get_top_k_for_context(target_id, context_ids, embeddings, k=k, exclusion_level=exclusion_level))
    
    # 2. Get Ground Truth Top-K (Filtered by the same rules)
    target_truth_df = answer_df[answer_df['context_id'] == target_id].copy()
    
    # Apply exclusion to truth
    target_truth_df = target_truth_df[
        target_truth_df['similar_context_id'].apply(
            lambda cid: not is_excluded(cid, target_hw, target_wl, exclusion_level)
        )
    ]
    
    truth = set(target_truth_df.sort_values('rank').head(k)['similar_context_id'])
    
    # 3. Output and Intersection
    print(f"Target: {target_id} | Predicted: {predicted} | Truth: {truth}")
    return len(predicted.intersection(truth))


def run_global_evaluation(metrics_path, answer_path, model_path, target_ids=None, k=5, exclusion_level=0):
    # Load Data
    metrics_df = pd.read_csv(metrics_path)
    answer_df = pd.read_csv(answer_path)
    
    # Define feature columns (ensure 'hardware' and 'workload' exist in CSV but aren't in features)
    feature_cols = ['tps', 'Average Memory Usage Percentage', 'InnoDB Buffer Pool Cache Hit Rate',
                    'InnoDB Dirty Buffer Pages', 'Current QPS (Queries Per Second)',
                    'Max CPU Usage (100 - Idle)', 'InnoDB Rows Deleted (60s Rate)',
                    'InnoDB Rows Inserted (60s Rate)', 'InnoDB Rows Read (60s Rate)',
                    'InnoDB Rows Updated (60s Rate)', 'Average Disk IOPS (Read)',
                    'Average Disk IOPS (Write)']
    
    all_context_ids = metrics_df['context_id'].tolist()
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(metrics_df[feature_cols])
    
    model = EmbeddingNet(input_dim=len(feature_cols), embedding_dim=16)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        embeddings = model(torch.tensor(features_scaled, dtype=torch.float32)).numpy()
    
    if target_ids is None:
        target_ids = all_context_ids
    else:
        target_ids = [tid for tid in target_ids if tid in all_context_ids]

    total_hits = 0
    for ctx_id in target_ids:
        total_hits += calculate_overlap_for_context(ctx_id, all_context_ids, embeddings, answer_df, k=k, exclusion_level=exclusion_level)
    
    avg_hit_rate = total_hits / (len(target_ids) * k)
    print(f"\n--- Evaluation Results (Exclusion Level {exclusion_level}) ---")
    print(f"Hit Rate: {avg_hit_rate:.2%}")
    
    return avg_hit_rate

# --- Execution ---
my_targets = {'88c190g_10-4-4-tpcc-nan'}
hit_rate = run_global_evaluation(
    'context_default_metrics_all.csv', 
    'concordance_ranking.csv', 
    'context_model_exclude_88c190g_10-4-4-tpcc-nan.pth', 
    target_ids=my_targets, 
    k=3,
    exclusion_level=3 # <--- Set your exclusion level here
)