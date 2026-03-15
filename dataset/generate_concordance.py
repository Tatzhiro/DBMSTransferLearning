import pandas as pd
import numpy as np
from tqdm import tqdm

files = [
    "88c190g-result.csv", "32c64g-result.csv", "24c32g-result.csv",
    "16c24g-result.csv", "4c6g-result.csv", "12c16g-result.csv", "8c12g-result.csv"
]

params = [
    'innodb_buffer_pool_size', 'innodb_read_io_threads', 'innodb_write_io_threads',
    'innodb_flush_log_at_trx_commit', 'innodb_adaptive_hash_index', 'sync_binlog',
    'innodb_lru_scan_depth', 'innodb_buffer_pool_instances', 'innodb_change_buffer_max_size',
    'innodb_io_capacity', 'innodb_log_file_size', 'table_open_cache'
]

dfs = []
for f in files:
    df = pd.read_csv(f)
    df['hardware'] = f.split('-')[0]
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data['context_id'] = all_data['hardware'] + "_" + all_data['workload_label']
all_data['config_key'] = all_data[params].apply(tuple, axis=1)

tps_pivot = all_data.pivot_table(index='config_key', columns='context_id', values='tps')
context_ids = tps_pivot.columns.tolist()
n_contexts = len(context_ids)

# Precompute sign matrices for each context
# This avoids O(N^2) loops inside the context pair loop
sign_matrices = {}
valid_configs_masks = {}

for ctx in context_ids:
    series = tps_pivot[ctx]
    valid_mask = series.notna().values
    valid_indices = np.where(valid_mask)[0]
    values = series.values[valid_indices]
    
    # Sign matrix for existing values
    # S[i, j] = sign(values[i] - values[j])
    diffs = values[:, None] - values[None, :]
    sign_matrix = np.sign(diffs)
    
    sign_matrices[ctx] = sign_matrix
    valid_configs_masks[ctx] = valid_indices

ranking_results = []

print(f"Calculating Concordant Pairs Ratio for {n_contexts} contexts...")

for i in tqdm(range(n_contexts)):
    target_ctx = context_ids[i]
    target_signs = sign_matrices[target_ctx]
    target_indices = valid_configs_masks[target_ctx]
    target_idx_map = {idx: pos for pos, idx in enumerate(target_indices)}
    
    similarities = []
    
    for j in range(n_contexts):
        if i == j:
            continue
            
        compare_ctx = context_ids[j]
        compare_signs = sign_matrices[compare_ctx]
        compare_indices = valid_configs_masks[compare_ctx]
        
        # Find intersection of configs
        common_indices = np.intersect1d(target_indices, compare_indices, assume_unique=True)
        k = len(common_indices)
        
        if k < 2:
            similarities.append((compare_ctx, 0, 0.0))
            continue
            
        # Map common indices to positions in the sign matrices
        t_pos = [target_idx_map[idx] for idx in common_indices]
        # Map for compare_ctx
        comp_idx_map = {idx: pos for pos, idx in enumerate(compare_indices)}
        c_pos = [comp_idx_map[idx] for idx in common_indices]
        
        # Extract sub-matrices
        s1 = target_signs[np.ix_(t_pos, t_pos)]
        s2 = compare_signs[np.ix_(c_pos, c_pos)]
        
        # Calculate concordant pairs
        # logic: (t1 > t2 and c1 > c2) or (t1 < t2 and c1 < c2) or (t1 == t2 and c1 == c2)
        # which is exactly s1 == s2
        # only count upper triangle
        concordant = (np.triu(s1 == s2, 1)).sum()
        total_valid_pairs = k * (k - 1) // 2
        
        ratio = concordant / total_valid_pairs if total_valid_pairs > 0 else 0
        similarities.append((compare_ctx, int(concordant), float(ratio)))

    # Rank by Ratio Descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (sim_ctx, concordant, score) in enumerate(similarities, start=1):
        ranking_results.append({
            'context_id': target_ctx,
            'similar_context_id': sim_ctx,
            'concordant_pairs': concordant,
            'similarity_score': score,
            'rank': rank
        })

output_df = pd.DataFrame(ranking_results)
output_df.to_csv('concordant_pair_ranking.csv', index=False)
print("Finished. Saved to concordant_pair_ranking.csv")
print(output_df.head())