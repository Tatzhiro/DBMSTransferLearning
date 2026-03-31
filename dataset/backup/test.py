import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

from train import EmbeddingNet


def get_top_k_similar_contexts(csv_path, model_path, k=3):
    # Load the metrics data
    df = pd.read_csv(csv_path)
    context_ids = df['context_id'].tolist()
    
    feature_cols = [
        'tps',
        'Average Memory Usage Percentage', 'InnoDB Buffer Pool Cache Hit Rate',
        'InnoDB Dirty Buffer Pages', 'Current QPS (Queries Per Second)',
        'Max CPU Usage (100 - Idle)', 'InnoDB Rows Deleted (60s Rate)',
        'InnoDB Rows Inserted (60s Rate)', 'InnoDB Rows Read (60s Rate)',
        'InnoDB Rows Updated (60s Rate)', 'Average Disk IOPS (Read)',
        'Average Disk IOPS (Write)'
    ]
    features = df[feature_cols].values
    
    # 2. Preprocessing: Scale features to [0, 1] 
    # NOTE: Ideally, use the scaler object saved during training
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 3. Load the Model
    model = EmbeddingNet(input_dim=len(feature_cols), embedding_dim=16)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 4. Generate Embeddings for all contexts
    with torch.no_grad():
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        embeddings = model(input_tensor).numpy()
    
    # 5. Calculate Pairwise Euclidean Distances
    # Result is an [N x N] matrix
    dist_matrix = cdist(embeddings, embeddings, metric='euclidean')
    
    # 6. Extract Top-K closest neighbors
    results = []
    for i in range(len(context_ids)):
        # Get distances from the current context to all others
        distances = dist_matrix[i]
        
        # Sort indices by distance. 
        # index 0 will be the context itself (dist=0), so we take 1 to k+1
        closest_indices = np.argsort(distances)[1:k+1]
        
        neighbors = []
        for idx in closest_indices:
            neighbors.append({
                'context_id': context_ids[idx],
                'distance': round(float(distances[idx]), 4)
            })
            
        results.append({
            'target_context': context_ids[i],
            'neighbors': neighbors
        })
        
    return results

def evaluate_top_k_overlap(metrics_path, answer_path, model_path, k=5):
    # --- Load Data ---
    metrics_df = pd.read_csv(metrics_path)
    answer_df = pd.read_csv(answer_path)
    
    # Feature columns used in training (including tps if it was in your triplet code)
    feature_cols = [
        'tps', 'Average Memory Usage Percentage', 'InnoDB Buffer Pool Cache Hit Rate',
        'InnoDB Dirty Buffer Pages', 'Current QPS (Queries Per Second)',
        'Max CPU Usage (100 - Idle)', 'InnoDB Rows Deleted (60s Rate)',
        'InnoDB Rows Inserted (60s Rate)', 'InnoDB Rows Read (60s Rate)',
        'InnoDB Rows Updated (60s Rate)', 'Average Disk IOPS (Read)',
        'Average Disk IOPS (Write)'
    ]
    
    context_ids = metrics_df['context_id'].tolist()
    features = metrics_df[feature_cols].values
    
    # Scale features (MinMax)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # --- Load Model and Generate Embeddings ---
    model = EmbeddingNet(input_dim=len(feature_cols), embedding_dim=16)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        embeddings = model(input_tensor).numpy()
    
    # --- Calculate Predicted Distances ---
    # Pairwise distances [N x N]
    dist_matrix = cdist(embeddings, embeddings, metric='euclidean')
    
    # --- Calculate Overlap ---
    total_overlap = 0
    context_count = len(context_ids)
    
    for i, target_id in enumerate(context_ids):
        # 1. Get Ground Truth Top-K
        # Filter answer dataset for current context and take top k rows
        true_top_k = set(answer_df[answer_df['context_id'] == target_id]
                         .sort_values('rank')
                         .head(k)['similar_context_id'])
        
        # 2. Get Predicted Top-K
        distances = dist_matrix[i]
        # Sort indices, exclude self (index i)
        pred_indices = np.argsort(distances)
        pred_indices = [idx for idx in pred_indices if context_ids[idx] != target_id][:k]
        pred_top_k = set([context_ids[idx] for idx in pred_indices])
        
        # 3. Intersection
        overlap = len(true_top_k.intersection(pred_top_k))
        total_overlap += overlap
        
    # Final Metric: Average Precision @ K
    avg_hit_rate = total_overlap / (context_count * k)
    
    print(f"--- Evaluation for Top-K Overlap (K={k}) ---")
    print(f"Average Hit Rate: {avg_hit_rate:.2%}")
    print(f"Total Matches: {total_overlap} out of {context_count * k} possible")
    
    return avg_hit_rate

# --- Execution ---
k_neighbors = get_top_k_similar_contexts('context_default_metrics.csv', 'context_model_exclude_88c190g_10-4-4-tpcc-nan.pth', k=3)

for neighbor in k_neighbors:
    if neighbor['target_context'] == '88c190g_10-4-4-tpcc-nan':
        print(f"Top matches for {neighbor['target_context']}:")
        for n in neighbor['neighbors']:
            print(f" -> {n['context_id']} (Dist: {n['distance']})")

# # Print example for the first context
# print(f"Top matches for {k_neighbors[0]['target_context']}:")
# for n in k_neighbors[0]['neighbors']:
#     print(f" -> {n['context_id']} (Dist: {n['distance']})")
    
# print(f"Top matches for {k_neighbors[1]['target_context']}:")
# for n in k_neighbors[1]['neighbors']:
#     print(f" -> {n['context_id']} (Dist: {n['distance']})")