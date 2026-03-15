import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from train import EmbeddingNet

def parse_context_id(ctx_id):
    """Splits context_id into (hardware, workload)."""
    parts = ctx_id.split('_', 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (None, None)

def is_excluded(candidate_id, target_id, target_hw, target_wl, exclusion_level):
    """Centralized exclusion logic."""
    if candidate_id == target_id:
        return True
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

def get_concordance_metrics(target_id, context_ids, embeddings, answer_df, k=5, exclusion_level=0):
    """
    Calculates the average ground truth score (concordance) for the predicted top-k.
    """
    target_hw, target_wl = parse_context_id(target_id)
    idx = context_ids.index(target_id)
    target_emb = embeddings[idx].reshape(1, -1)
    
    # 1. Filter Ground Truth for this target based on exclusion level
    # We assume 'answer_df' has columns: [context_id, similar_context_id, score]
    truth_pool = answer_df[answer_df['context_id'] == target_id].copy()
    truth_pool = truth_pool[~truth_pool['similar_context_id'].apply(
        lambda x: is_excluded(x, target_id, target_hw, target_wl, exclusion_level)
    )]
    
    # Create a lookup map for ground truth scores
    score_map = dict(zip(truth_pool['similar_context_id'], truth_pool['similarity_score']))
    
    # 2. Get Predicted Top-K (Distance-based)
    distances = cdist(target_emb, embeddings, metric='euclidean').flatten()
    sorted_indices = np.argsort(distances)
    
    predicted_ids = []
    for i in sorted_indices:
        candidate_id = context_ids[i]
        if is_excluded(candidate_id, target_id, target_hw, target_wl, exclusion_level):
            continue
        predicted_ids.append(candidate_id)
        if len(predicted_ids) == k:
            break
            
    # 3. Calculate Scores
    # We find the ground truth scores of our predictions
    # If a predicted ID isn't in the truth pool, score is 0
    predicted_scores = [score_map.get(pid, 0) for pid in predicted_ids]
    avg_predicted_score = np.mean(predicted_scores) if predicted_scores else 0
    
    # 4. Ideal Score (For normalization/reference)
    ideal_scores = sorted(score_map.values(), reverse=True)[:k]
    avg_ideal_score = np.mean(ideal_scores) if ideal_scores else 1e-9 # avoid div by zero
    
    # Normalized Concordance Score (0.0 to 1.0)
    concordance_ratio = avg_predicted_score / avg_ideal_score
    
    return {
        "avg_score": avg_predicted_score,
        "ideal_score": avg_ideal_score,
        "ratio": concordance_ratio
    }

def run_concordance_evaluation(metrics_path, answer_path, model_path, target_ids=None, k=5, exclusion_level=0):
    metrics_df = pd.read_csv(metrics_path)
    answer_df = pd.read_csv(answer_path) # Assumes this has the 'score' column
    
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
    
    targets = target_ids if target_ids else all_context_ids
    results = []
    
    for tid in targets:
        if tid in all_context_ids:
            m = get_concordance_metrics(tid, all_context_ids, embeddings, answer_df, k=k, exclusion_level=exclusion_level)
            results.append(m)
            print(f"ID: {tid} | Score: {m['avg_score']:.4f} | Ideal: {m['ideal_score']:.4f} | Ratio: {m['ratio']:.2%}")

    avg_ratio = np.mean([r['ratio'] for r in results])
    print(f"\n--- Global Concordance Report (Level {exclusion_level}) ---")
    print(f"Mean Normalized Concordance: {avg_ratio:.2%}")
    
    return avg_ratio

# --- Execution ---
my_targets = {'88c190g_10-4-4-tpcc-nan'}
run_concordance_evaluation(
    'context_default_metrics_all.csv', 
    'concordance_ranking.csv', 
    'context_model_exclude_88c190g_10-4-4-tpcc-nan.pth', 
    target_ids=my_targets, 
    k=3, 
    exclusion_level=3
)
