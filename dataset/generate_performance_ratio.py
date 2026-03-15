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

# Align TPS data
tps_pivot = all_data.pivot_table(index='config_key', columns='context_id', values='tps')
context_ids = tps_pivot.columns.tolist()
n_contexts = len(context_ids)

k_configs = 1
transfer_ranking_results = []

print(f"Calculating Tuning Transfer Ratio for {n_contexts} contexts...")

for i in tqdm(range(n_contexts)):
    target_id = context_ids[i]
    target_series = tps_pivot[target_id].dropna()
    
    if target_series.empty:
        continue
        
    # Absolute best TPS in Target (for shared configs context)
    # Note: We calculate this relative to the configurations common with the candidate
    # but the user said "Find the absolute best TPS possible in the Target Context (Ground Truth)"
    # Usually, this means the global maximum in the target data.
    target_global_max = target_series.max()
    
    similarities = []
    
    for j in range(n_contexts):
        if i == j:
            continue
            
        candidate_id = context_ids[j]
        candidate_series = tps_pivot[candidate_id].dropna()
        
        # Find common configurations
        common_configs = target_series.index.intersection(candidate_series.index)
        
        if len(common_configs) == 0:
            similarities.append((candidate_id, 0.0))
            continue
            
        # 1. Get candidate TPS for common configs and find Top-K
        candidate_common_tps = candidate_series.loc[common_configs]
        top_k_configs = candidate_common_tps.sort_values(ascending=False).head(k_configs).index
        
        # 2. Find the best performance of these Top-K configs in the Target Context
        target_transferred_tps = target_series.loc[top_k_configs].max()
        
        # 3. Ratio relative to Target's absolute best
        # Note: If target_global_max is 0 (shouldn't happen), ratio is 0
        ratio = target_transferred_tps / target_global_max if target_global_max > 0 else 0.0
        
        similarities.append((candidate_id, float(ratio)))
        
    # 4. Rank by Ratio Descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (sim_ctx, score) in enumerate(similarities, start=1):
        transfer_ranking_results.append({
            'context_id': target_id,
            'similar_context_id': sim_ctx,
            'similarity_score': score,
            'rank': rank
        })

output_df = pd.DataFrame(transfer_ranking_results)
output_df.to_csv('tuning_transfer_ranking.csv', index=False)
print("Finished. Saved to tuning_transfer_ranking.csv")
print(output_df.head(10))