import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from sklearn.preprocessing import MinMaxScaler

# --- Configuration & Data Loading ---
FILES = [
    "88c190g-result.csv", "32c64g-result.csv", "24c32g-result.csv",
    "16c24g-result.csv", "4c6g-result.csv", "12c16g-result.csv", "8c12g-result.csv"
]

METRICS = [
    'tps', 'Average Memory Usage Percentage', 'InnoDB Buffer Pool Cache Hit Rate',
    'InnoDB Dirty Buffer Pages', 'Current QPS (Queries Per Second)',
    'Max CPU Usage (100 - Idle)', 'InnoDB Rows Deleted (60s Rate)',
    'InnoDB Rows Inserted (60s Rate)', 'InnoDB Rows Read (60s Rate)',
    'InnoDB Rows Updated (60s Rate)', 'Average Disk IOPS (Read)',
    'Average Disk IOPS (Write)',
]

PARAMS = [
    'innodb_buffer_pool_size', 'innodb_read_io_threads', 'innodb_write_io_threads',
    'innodb_flush_log_at_trx_commit', 'innodb_adaptive_hash_index', 'sync_binlog',
    'innodb_lru_scan_depth', 'innodb_buffer_pool_instances', 'innodb_change_buffer_max_size',
    'innodb_io_capacity', 'innodb_log_file_size', 'table_open_cache'
]

# --- Core Logic Functions ---

def generate_triplets_kendall(context_ids, context_defaults_scaled, tps_pivot, k=5):
    """Generates triplets using Kendall Tau calculated on the fly."""
    triplets = []
    n = len(context_ids)
    
    # Pre-calculate similarity matrix
    sim_matrix = np.full((n, n), -2.0)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = context_ids[i], context_ids[j]
            s1, s2 = tps_pivot[c1], tps_pivot[c2]
            mask = s1.notna() & s2.notna()
            if mask.sum() > 10:
                tau, _ = kendalltau(s1[mask], s2[mask])
                sim_matrix[i, j] = sim_matrix[j, i] = tau if not np.isnan(tau) else -2.0

    for i in range(n):
        anchor_id = context_ids[i]
        scores = sim_matrix[i]
        sorted_idx = np.argsort(scores)
        
        pos_indices = sorted_idx[-k:]
        neg_indices = sorted_idx[:k]
        
        anchor_vec = context_defaults_scaled.loc[anchor_id].tolist()
        for p_idx in pos_indices:
            if scores[p_idx] > -1.5:
                pos_vec = context_defaults_scaled.iloc[p_idx].tolist()
                for n_idx in neg_indices:
                    neg_vec = context_defaults_scaled.iloc[n_idx].tolist()
                    triplets.append([anchor_id, context_ids[p_idx], context_ids[n_idx]] + anchor_vec + pos_vec + neg_vec)
    return triplets

def generate_triplets_concordance_csv(context_ids, context_defaults_scaled, ranking_path='concordance_ranking.csv', k=5):
    """Generates triplets using precalculated Concordant Ranking Pairs from CSV."""
    ranking_df = pd.read_csv(ranking_path)
    triplets = []
    
    for anchor_id in context_ids:
        # Get pre-ranked similarities for this anchor
        anchor_ranks = ranking_df[ranking_df['context_id'] == anchor_id].sort_values('similarity_score', ascending=False)
        
        if anchor_ranks.empty: continue
            
        pos_ids = anchor_ranks.head(k)['similar_context_id'].tolist()
        neg_ids = anchor_ranks.tail(k)['similar_context_id'].tolist()
        
        anchor_vec = context_defaults_scaled.loc[anchor_id].tolist()
        for p_id in pos_ids:
            pos_vec = context_defaults_scaled.loc[p_id].tolist()
            for n_id in neg_ids:
                neg_vec = context_defaults_scaled.loc[n_id].tolist()
                triplets.append([anchor_id, p_id, n_id] + anchor_vec + pos_vec + neg_vec)
    return triplets

def generate_triplets_transfer_csv(context_ids, context_defaults_scaled, ranking_path='tuning_transfer_ranking.csv', k=5):
    """Choice C: Generates triplets using precalculated Tuning Transfer Ratios from CSV."""
    ranking_df = pd.read_csv(ranking_path)
    triplets = []
    for anchor_id in context_ids:
        # Sort by Transfer Ratio (highest is most similar/useful)
        anchor_ranks = ranking_df[ranking_df['context_id'] == anchor_id].sort_values('similarity_score', ascending=False)
        if anchor_ranks.empty: continue
        
        pos_ids = anchor_ranks.head(k)['similar_context_id'].tolist()
        neg_ids = anchor_ranks.tail(k)['similar_context_id'].tolist()
        
        anchor_vec = context_defaults_scaled.loc[anchor_id].tolist()
        for p_id in pos_ids:
            pos_vec = context_defaults_scaled.loc[p_id].tolist()
            for n_id in neg_ids:
                neg_vec = context_defaults_scaled.loc[n_id].tolist()
                triplets.append([anchor_id, p_id, n_id] + anchor_vec + pos_vec + neg_vec)
    return triplets

# --- Main Execution Block ---

def main():
    # 1. Load and Clean Data
    dfs = []
    for f in FILES:
        df = pd.read_csv(f)
        df['hardware'] = f.split('-')[0]
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)
    all_data['context_id'] = all_data['hardware'] + "_" + all_data['workload_label']
    all_data['config_key'] = all_data[PARAMS].apply(tuple, axis=1)

    # 2. Extract Features and Normalize
    default_config_vals = all_data[all_data['id'] == 0][PARAMS].iloc[0].values
    context_defaults = all_data[all_data[PARAMS].apply(lambda x: np.array_equal(x.values, default_config_vals), axis=1)]
    context_defaults = context_defaults.groupby('context_id')[METRICS].mean().dropna()
    
    scaler = MinMaxScaler()
    context_defaults_scaled = pd.DataFrame(scaler.fit_transform(context_defaults), index=context_defaults.index, columns=METRICS)
    
    context_ids = context_defaults.index.tolist()
    tps_pivot = all_data.pivot_table(index='config_key', columns='context_id', values='tps')

    # --- SWITCH HERE ---
    # Choice A: Kendall Tau
    # triplets_list = generate_triplets_kendall(context_ids, context_defaults_scaled, tps_pivot, k=5)
    # output_name = 'full_triplet_data_kendall.csv'
    
    # Choice B: Precalculated Concordance CSV
    # triplets_list = generate_triplets_concordance_csv(context_ids, context_defaults_scaled, k=5)
    # output_name = 'full_triplet_data_concordance.csv'

    # Choice C: Precalculated Tuning Transfer CSV
    triplets_list = generate_triplets_transfer_csv(context_ids, context_defaults_scaled, k=5)
    output_name = 'full_triplet_data_transfer.csv'
    # ------------------

    # 3. Save Output
    id_cols = ['anchor_id', 'pos_id', 'neg_id']
    vec_cols = [f'anchor_{m}' for m in METRICS] + [f'pos_{m}' for m in METRICS] + [f'neg_{m}' for m in METRICS]
    pd.DataFrame(triplets_list, columns=id_cols + vec_cols).to_csv(output_name, index=False)
    print(f"Generated {len(triplets_list)} triplets.")

if __name__ == "__main__":
    main()