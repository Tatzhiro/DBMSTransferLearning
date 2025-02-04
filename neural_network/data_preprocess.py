import pandas as pd
import glob
import re

def rename_columns(df):
    metric_mapping = {
        "Top Command Counters {'metric': 'topk(5 'command': 'set_option'}": "Top 5 Command Usage (Set Option)",
        "MySQL Table Locks {'metric': 'sum(rate(mysql_global_status_table_locks_waited[60s]))'}": "MySQL Table Locks Waited (60s Rate)",
        "Top Command Counters {'metric': 'topk(5 'command': 'commit'}": "Top 5 Command Usage (Commit)",
        "Top Command Counters {'metric': 'topk(5 'command': 'show_slave_status'}": "Top 5 Command Usage (Show Slave Status)",
        "InnoDB Buffer Pool {'metric': 'mysql_global_status_buffer_pool_pages' 'state': 'free'}": "InnoDB Buffer Pool (Free Pages)",
        "Avg CPU {'metric': 'avg without(cpu) (rate(node_cpu_seconds_total[60s]) * 100)' 'mode': 'softirq'}": "Average CPU Usage (SoftIRQ Mode)",
        "Top Command Counters {'metric': 'topk(5 'command': 'begin'}": "Top 5 Command Usage (Begin Transaction)",
        "MySQL Connections {'metric': 'mysql_global_variables_max_connections'}": "MySQL Max Connections",
        "Avg Memory usage {'metric': 'node_vmstat_pswpout'}": "Average Memory Swap Out",
        "MySQL Client Thread Activity {'metric': 'sum(max_over_time(mysql_global_status_threads_connected[60s]))'}": "MySQL Client Threads Connected (Max Over Time)",
        "Avg CPU {'metric': 'avg without(cpu) (rate(node_cpu_seconds_total[60s]) * 100)' 'mode': 'system'}": "Average CPU Usage (System Mode)",
        "MySQL Select Types {'metric': 'sum(rate(mysql_global_status_select_full_range_join[60s]))'}": "MySQL Full Range Join Selects (60s Rate)",
        "MySQL Transaction Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'savepoint_rollback'}": "MySQL Transaction Handlers (Savepoint Rollback)",
        "MySQL Sorts {'metric': 'sum(rate(mysql_global_status_sort_merge_passes[60s]))'}": "MySQL Sort Merge Passes (60s Rate)",
        "Avg CPU {'metric': 'avg without(cpu) (rate(node_cpu_seconds_total[60s]) * 100)' 'mode': 'steal'}": "Average CPU Usage (Steal Mode)",
        "MySQL Select Types {'metric': 'sum(rate(mysql_global_status_select_scan[60s]))'}": "MySQL Full Table Scans (60s Rate)",
        "Top Command Counters {'metric': 'topk(5 'command': 'show_replica_status'}": "Top 5 Command Usage (Show Replica Status)",
        "MySQL Connections {'metric': 'mysql_global_status_max_used_connections'}": "MySQL Max Used Connections",
        "disk usage (root volume) {'metric': 'node_filesystem_size_bytes - node_filesystem_avail_bytes' 'mountpoint': '/'}": "Root Volume Disk Usage",
        "disk usage (percent) root volume {'metric': '(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100' 'mountpoint': '/'}": "Root Volume Disk Usage Percentage",
        "MySQL Thread Cache {'metric': 'sum(rate(mysql_global_status_threads_created[60s]))'}": "MySQL Threads Created (60s Rate)",
        "Top Command Counters {'metric': 'topk(5 'command': 'insert'}": "Top 5 Command Usage (Insert)",
        "Avg CPU {'metric': 'avg without(cpu) (rate(node_cpu_seconds_total[60s]) * 100)' 'mode': 'idle'}": "Average CPU Usage (Idle Mode)",
        "MySQL Table Open Cache Status {'metric': 'rate(mysql_global_status_table_open_cache_hits[60s])/(rate(mysql_global_status_table_open_cache_hits[60s])+rate(mysql_global_status_table_open_cache_misses[60s]))'}": "MySQL Table Open Cache Efficiency",
        "MySQL Select Types {'metric': 'sum(rate(mysql_global_status_select_full_join[60s]))'}": "MySQL Full Join Selects (60s Rate)",
        "Avg Network sent {'metric': 'rate(node_network_transmit_bytes_total[60s])' 'device': 'eth0'}": "Average Network Traffic Sent (Eth0)",
        "MySQL Client Thread Activity {'metric': 'sum(avg_over_time(mysql_global_status_threads_running[60s]))'}": "MySQL Threads Running (Max Over Time)",
        "Avg disk IOPS {'metric': 'rate(node_disk_reads_completed_total[60s])' 'device': 'sr0'}": "Average Disk IOPS (Read)",
        "MySQL Connections {'metric': 'max_over_time(mysql_global_status_threads_connected[60s])'}": "MySQL Threads Connected (Max Over Time)",
        "Avg Memory usage {'metric': 'node_vmstat_pswpin'}": "Average Memory Swap In",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'write'}": "MySQL Handlers (Write)",
        "Avg pages swap in/out {'metric': 'rate(node_vmstat_pswpin[60s])'}": "Average Pages Swap In/Out (60s Rate)",
        "MySQL Sorts {'metric': 'sum(rate(mysql_global_status_sort_rows[60s]))'}": "MySQL Rows Sorted (60s Rate)",
        "Avg Network recived {'metric': 'rate(node_network_receive_bytes_total[60s])' 'device': 'lo'}": "Average Network Traffic Received (LO Interface)",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'delete'}": "MySQL Handlers (Delete)",
        "MySQL Network Traffic {'metric': 'sum(rate(mysql_global_status_bytes_sent[60s]))'}": "MySQL Network Traffic (Bytes Sent)",
        "Avg pages swap in/out {'metric': 'rate(node_vmstat_pswpout[60s])'}": "Average Pages Swap Out (60s Rate)",
        "MySQL Table Definition Cache {'metric': 'rate(mysql_global_status_opened_table_definitions[60s])'}": "MySQL Opened Table Definitions (60s Rate)",
        "Top Command Counters {'metric': 'topk(5 'command': 'stmt_close'}": "Top 5 Command Usage (Statement Close)",
        "MySQL Transaction Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'rollback'}": "MySQL Transaction Handlers (Rollback)",
        "Avg Network recived {'metric': 'rate(node_network_receive_bytes_total[60s])' 'device': 'eth0'}": "Average Network Traffic Received (Eth0)",
        "Avg Memory usage {'metric': 'node_memory_MemTotal_bytes'}": "Total Memory",
        "Cache hit rate {'metric': '(1- (mysql_global_status_innodb_buffer_pool_reads / mysql_global_status_innodb_buffer_pool_read_requests)) * 100'}": "InnoDB Buffer Pool Cache Hit Rate",
        "MySQL Transaction Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'savepoint'}": "MySQL Transaction Handlers (Savepoint)",
        "Top Command Counters {'metric': 'topk(5 'command': 'admin_commands'}": "Top 5 Command Usage (Admin Commands)",
        "MySQL Temporary Objects {'metric': 'sum(rate(mysql_global_status_created_tmp_files[60s]))'}": "MySQL Temporary Files Created (60s Rate)",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'read_rnd_next'}": "MySQL Handlers (Read Next Record)",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'read_key'}": "MySQL Handlers (Key Read)",
        "MySQL Table Definition Cache {'metric': 'mysql_global_status_open_table_definitions'}": "MySQL Table Definition Cache",
        "MySQL Select Types {'metric': 'sum(rate(mysql_global_status_select_range_check[60s]))'}": "MySQL Range Selects (60s Rate)",
        "MySQL Table Open Cache Status {'metric': 'rate(mysql_global_status_table_open_cache_misses[60s])'}": "MySQL Open Cache Misses (60s Rate)",
        "MySQL Transaction Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'prepare'}": "MySQL Transaction Handlers (Prepare)",
        "MySQL Thread Cache {'metric': 'sum(mysql_global_status_threads_cached)'}": "MySQL Threads Cached",
        "MySQL Open Files {'metric': 'mysql_global_variables_open_files_limit'}": "MySQL Open Files Limit",
        "MySQL Temporary Objects {'metric': 'sum(rate(mysql_global_status_created_tmp_tables[60s]))'}": "MySQL Temporary Tables Created (60s Rate)",
        "MySQL Table Open Cache Status {'metric': 'rate(mysql_global_status_table_open_cache_overflows[60s])'}": "MySQL Open Cache Overflows (60s Rate)",
        "MySQL Thread Cache {'metric': 'sum(mysql_global_variables_thread_cache_size)'}": "MySQL Thread Cache Size",
        "Avg Memory usage {'metric': '100 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)*100'}": "Average Memory Usage Percentage",
        "MySQL Table Locks {'metric': 'sum(rate(mysql_global_status_table_locks_immediate[60s]))'}": "MySQL Table Locks Immediate (60s Rate)",
        "Top Command Counters {'metric': 'topk(5 'command': 'select'}": "Top 5 Command Usage (Select)",
        "MySQL Temporary Objects {'metric': 'sum(rate(mysql_global_status_created_tmp_disk_tables[60s]))'}": "MySQL Temporary Disk Tables Created (60s Rate)",
        "MySQL Aborted Connections {'metric': 'sum(rate(mysql_global_status_aborted_connects[60s]))'}": "MySQL Aborted Connections (60s Rate)",
        "Avg disk busy {'metric': 'rate(node_disk_io_time_seconds_total[60s]) * 100' 'device': 'sr0'}": "Average Disk Busy (Sr0)",
        "MySQL InnoDB rows updated {'metric': 'rate(mysql_global_status_innodb_row_ops_total[60s])' 'operation': 'inserted'}": "InnoDB Rows Inserted (60s Rate)",
        "Avg CPU {'metric': 'avg without(cpu) (rate(node_cpu_seconds_total[60s]) * 100)' 'mode': 'irq'}": "Average CPU Usage (IRQ Mode)",
        "MySQL Internal Memory Overview {'metric': 'sum(mysql_global_variables_key_buffer_size)'}": "MySQL Internal Memory (Key Buffer Size)",
        "MySQL Internal Memory Overview {'metric': 'sum(mysql_global_status_innodb_page_size * on (instance) mysql_global_status_buffer_pool_pages)'}": "InnoDB Buffer Pool Total Pages",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'external_lock'}": "MySQL Handlers (External Lock)",
        "MySQL Table Open Cache Status {'metric': 'rate(mysql_global_status_table_open_cache_hits[60s])'}": "MySQL Table Open Cache Hits (60s Rate)",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'discover'}": "MySQL Handlers (Discover)",
        "MySQL Network Traffic {'metric': 'sum(rate(mysql_global_status_bytes_received[60s]))'}": "MySQL Network Traffic (Bytes Received)",
        "MySQL Table Definition Cache {'metric': 'mysql_global_variables_table_definition_cache'}": "MySQL Table Definition Cache Size",
        "Avg disk IOPS {'metric': 'rate(node_disk_writes_completed_total[60s])' 'device': 'vda'}": "Average Disk IOPS (Write)",
        "Avg Network retrans rate {'metric': '(node_netstat_Tcp_RetransSegs / node_netstat_Tcp_OutSegs)*100'}": "Average Network Retransmit Rate",
        "MySQL Table Open Cache Status {'metric': 'rate(mysql_global_status_opened_tables[60s])'}": "MySQL Opened Tables (60s Rate)",
        "MySQL Sorts {'metric': 'sum(rate(mysql_global_status_sort_scan[60s]))'}": "MySQL Sort Scans (60s Rate)",
        "MySQL Client Thread Activity {'metric': 'sum(max_over_time(mysql_global_status_threads_running[60s]))'}": "MySQL Threads Running (Max Over Time)",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'mrr_init'}": "MySQL Handlers (MRR Init)",
        "InnoDB Buffer Pool {'metric': 'mysql_global_status_buffer_pool_pages' 'state': 'data'}": "InnoDB Buffer Pool (Data Pages)",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'read_rnd'}": "MySQL Handlers (Read Random)",
        "MySQL Internal Memory Overview {'metric': 'sum(mysql_global_variables_innodb_log_buffer_size)'}": "InnoDB Log Buffer Size",
        "MySQL Select Types {'metric': 'sum(rate(mysql_global_status_select_range[60s]))'}": "MySQL Range Selects (60s Rate)",
        "MySQL Sorts {'metric': 'sum(rate(mysql_global_status_sort_range[60s]))'}": "MySQL Sort Range (60s Rate)",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'read_prev'}": "MySQL Handlers (Read Previous Record)",
        "MySQL InnoDB rows updated {'metric': 'rate(mysql_global_status_innodb_row_ops_total[60s])' 'operation': 'read'}": "InnoDB Rows Read (60s Rate)",
        "Avg Memory usage {'metric': 'node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes'}": "Total Memory Usage",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'read_last'}": "MySQL Handlers (Read Last Record)",
        "MySQL Open Tables {'metric': 'mysql_global_variables_table_open_cache'}": "MySQL Table Open Cache Size",
        "Top Command Counters {'metric': 'topk(5 'command': 'delete'}": "Top 5 Command Usage (Delete)",
        "MySQL Open Tables {'metric': 'mysql_global_status_open_tables'}": "MySQL Open Tables",
        "MySQL Transaction Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'commit'}": "MySQL Transaction Handlers (Commit)",
        "MySQL Aborted Connections {'metric': 'sum(rate(mysql_global_status_aborted_clients[60s]))'}": "MySQL Aborted Clients (60s Rate)",
        "MySQL Open Files {'metric': 'mysql_global_status_open_files'}": "MySQL Open Files",
        "Uptime {'metric': 'mysql_global_status_uptime'}": "MySQL Uptime",
        "MySQL File Openings {'metric': 'rate(mysql_global_status_opened_files[60s])'}": "MySQL File Openings (60s Rate)",
        "Avg disk IOPS {'metric': 'rate(node_disk_reads_completed_total[60s])' 'device': 'vda'}": "Average Disk IOPS (Read, VDA)",
        "Top Command Counters {'metric': 'topk(5 'command': 'stmt_prepare'}": "Top 5 Command Usage (Statement Prepare)",
        "Top Command Counters {'metric': 'topk(5 'command': 'update'}": "Top 5 Command Usage (Update)",
        "MySQL InnoDB rows updated {'metric': 'rate(mysql_global_status_innodb_row_ops_total[60s])' 'operation': 'deleted'}": "InnoDB Rows Deleted (60s Rate)",
        "Avg Network sent {'metric': 'rate(node_network_transmit_bytes_total[60s])' 'device': 'lo'}": "Average Network Traffic Sent (LO Interface)",
        "InnoDB Buffer Pool {'metric': 'mysql_global_status_buffer_pool_pages' 'state': 'misc'}": "InnoDB Buffer Pool (Misc Pages)",
        "Avg disk busy {'metric': 'rate(node_disk_io_time_seconds_total[60s]) * 100' 'device': 'vda'}": "Average Disk Busy (VDA)",
        "Top Command Counters {'metric': 'topk(5 'command': 'show_status'}": "Top 5 Command Usage (Show Status)",
        "InnoDB Buffer Pool(dirty) {'metric': 'mysql_global_status_buffer_pool_dirty_pages'}": "InnoDB Dirty Buffer Pages",
        "Avg CPU {'metric': 'avg without(cpu) (rate(node_cpu_seconds_total[60s]) * 100)' 'mode': 'nice'}": "Average CPU Usage (Nice Mode)",
        "Max CPU(100 - idle) {'metric': '(1 - (min without(cpu) (rate(node_cpu_seconds_total[60s]))))*100' 'mode': 'idle'}": "Max CPU Usage (100 - Idle)",
        "MySQL Slow Queries {'metric': 'sum(rate(mysql_global_status_slow_queries[60s]))'}": "MySQL Slow Queries (60s Rate)",
        "Current QPS {'metric': 'rate(mysql_global_status_queries[60s])'}": "Current QPS (Queries Per Second)",
        "Avg CPU {'metric': 'avg without(cpu) (rate(node_cpu_seconds_total[60s]) * 100)' 'mode': 'user'}": "Average CPU Usage (User Mode)",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'read_first'}": "MySQL Handlers (Read First Record)",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'read_next'}": "MySQL Handlers (Read Next Record)",
        "Top Command Counters {'metric': 'topk(5 'command': 'stmt_execute'}": "Top 5 Command Usage (Statement Execute)",
        "Avg CPU {'metric': 'avg without(cpu) (rate(node_cpu_seconds_total[60s]) * 100)' 'mode': 'iowait'}": "Average CPU Usage (IO Wait Mode)",
        "Avg Swap used percent {'metric': '(1 - (node_memory_SwapFree_bytes / node_memory_SwapTotal_bytes)) * 100'}": "Average Swap Usage Percentage",
        "Avg disk IOPS {'metric': 'rate(node_disk_writes_completed_total[60s])' 'device': 'sr0'}": "Average Disk IOPS (Write, Sr0)",
        "MySQL Handlers {'metric': 'rate(mysql_global_status_handlers_total[60s])' 'handler': 'update'}": "MySQL Handlers (Update)",
        "Top Command Counters {'metric': 'topk(5 'command': 'show_variables'}": "Top 5 Command Usage (Show Variables)",
        "MySQL Open Files {'metric': 'mysql_global_status_innodb_num_open_files'}": "MySQL Open Files (InnoDB)",
        "MySQL InnoDB rows updated {'metric': 'rate(mysql_global_status_innodb_row_ops_total[60s])' 'operation': 'updated'}": "InnoDB Rows Updated (60s Rate)",
        
        "Avg Network recived {'metric': 'rate(node_network_receive_bytes_total[60s])' 'device': 'eno1'}": "Average Network Traffic Received (Eth0)",
        "Avg Network sent {'metric': 'rate(node_network_transmit_bytes_total[60s])' 'device': 'eno1'}": "Average Network Traffic Sent (Eth0)",
        "Avg disk IOPS {'metric': 'rate(node_disk_reads_completed_total[60s])' 'device': 'sda'}": "Average Disk IOPS",
        
    }
    
    df.columns = [metric_mapping.get(col, col) for col in df.columns]
    return df


def unify_metrics(df):
    pattern_mapping = {
        # Example: unify "Avg Network Traffic Sent" for any 'device' 
        "Average Network Traffic Sent": r"^Avg Network sent \{.*'device': '.*'.*\}$",
        
        # Example: unify "Avg Network Traffic Received" for any 'device'
        "Average Network Traffic Received": r"^Avg Network recived \{.*'device': '.*'.*\}$",
        
        # Example: unify "Average Disk IOPS (Read)" for any 'device'
        "Average Disk IOPS (Read)": r"^Avg disk IOPS \{.*rate\(node_disk_reads_completed_total\[60s\]\).*\}$",
        
        # Example: unify "Average Disk IOPS (Write)" for any 'device'
        "Average Disk IOPS (Write)": r"^Avg disk IOPS \{.*rate\(node_disk_writes_completed_total\[60s\]\).*\}$",
        
        # Add more patterns as needed...
        "Average Disk Busy": r"^Avg disk busy \{.*\}$",
    }
    grouped_cols = {new_name: [] for new_name in pattern_mapping}
    
    for col in df.columns:
        # Check each pattern; if matched, group it under that new_name
        for new_name, regex_pat in pattern_mapping.items():
            if re.search(regex_pat, col):
                grouped_cols[new_name].append(col)
                break
        # If a column doesn't match any pattern, we simply leave it alone
        # or you can handle it however you like (e.g. keep, drop, etc.)
    
    # ----------------------------------------------------------------------
    # 3) Remove columns that have all 0 or contain any NaNs
    #    Then 4) average the remaining columns into a single column
    # ----------------------------------------------------------------------
    for new_name, old_cols in grouped_cols.items():
        # Filter out columns with any NaNs or all zero
        valid_cols = []
        for c in old_cols:
            series = df[c]
            if not series.isnull().any() and not (series == 0).all():
                valid_cols.append(c)
                
        if len(valid_cols) == 0:
            # No valid columns remain, skip creating a new column
            continue
        elif len(valid_cols) == 1:
            # Only one valid column -> just rename (if you want to unify anyway)
            df[new_name] = df[valid_cols[0]]
        else:
            # Multiple valid columns -> average them
            df[new_name] = df[valid_cols].mean(axis=1)
        
        # (Optional) Drop the original columns
        df.drop(columns=old_cols, inplace=True)
    
    return df


metric_file_list = sorted(glob.glob('dataset/metric_learning/*.csv'))
metric_file_list = [file for file in metric_file_list if "train" not in file]

for file in metric_file_list:
    metric_df = pd.read_csv(file)
    metric_df = unify_metrics(metric_df)
    metric_df = rename_columns(metric_df)
    
    param_df = pd.read_csv(f"dataset/transfer_learning/mysql/chimera_tech/{file.split('/')[-1]}")
    df = pd.merge(metric_df, param_df, on='id', how='inner')
    if "workload_label" not in df.columns:
        df["workload_label"] = df["num_table"].astype(str) + "-" + df["table_size"].astype(str) + "-" + df["num_client"].astype(str) + "-" + df["workload"].astype(str) + "-" + df["skew"].astype(str)
        df = df.drop(["num_table", "table_size", "num_client", "workload", "skew"], axis=1)
    if "num_cpu" not in df.columns:
        num_cpu = file.split('/')[-1].split("c")[0]
        mem_size = file.split('/')[-1].split("c")[1].split("g")[0]
        df["mem_size"] = mem_size
        df["num_cpu"] = num_cpu
    df.to_csv(f"dataset/transfer_learning/mysql/merge/{file.split('/')[-1]}", index=False)
    
    
    # import pandas as pd
# import numpy as np
# from lineairdb.regression.system_configuration import MySQLConfiguration
# from lineairdb.regression.jamshidi import L2SFeatureSelector, LassoFeatureSelector
# from lineairdb.regression.utils import drop_unimportant_parameters
# import glob
# import os
# from IPython import embed

# class FeatureImportance:
#     def __init__(self, df: pd.DataFrame, system: MySQLConfiguration):
#         self.df = df
#         self.system = system
        
        
#     def get_range(self, df: pd.DataFrame):
#         min = df[self.system.get_perf_metric()].min()
#         max = df[self.system.get_perf_metric()].max()
#         return max - min
        
    
#     def get_parameter_vector(self):
#         vector = np.zeros(len(self.system.get_param_names()))
        
#         for i, p in enumerate(self.system.get_param_names()):
#             df = drop_unimportant_parameters(self.df, [p], self.system)
#             range = self.get_range(df)
#             vector[i] = range
        
#         vector = vector / np.sum(vector)
#         return vector
            

# selector = LassoFeatureSelector()

# path_to_this_file = os.path.dirname(os.path.realpath(__file__))

# files = glob.glob(path_to_this_file + "/dataset/raw/concat/*result.csv")
# os.makedirs(path_to_this_file + "/dataset/raw/train", exist_ok=True)
# dfs = []
# for file in files:
#     group = file.split("/")[-1].split("-")[0]
#     param_df = pd.read_csv(file)
#     param_df = MySQLConfiguration().preprocess_param_values(param_df)
#     train_df = pd.read_csv(path_to_this_file + "/dataset/raw/concat/" + group + "-result_metric.csv")
#     df = pd.merge(param_df, train_df, on="id")
    
#     # get set of workload parameters
#     workloads = df[["num_table", "table_size", "num_client", "workload", "skew"]]
#     workloads = workloads.drop_duplicates()
#     # remove first row because it might be a warm-up
#     df = df.iloc[1:]
#     for workload in workloads.iterrows():
#         mask = (
#             (df["num_table"] == workload[1]["num_table"]) &
#             (df["table_size"] == workload[1]["table_size"]) &
#             (df["num_client"] == workload[1]["num_client"]) &
#             (df["workload"] == workload[1]["workload"]) &
#             (
#                 (pd.isna(df["skew"]) & pd.isna(workload[1]["skew"])) | 
#                 (df["skew"] == workload[1]["skew"])
#             )
#         )
#         df_workload = df[mask]
#         # selector.select_important_features(df_workload, MySQLConfiguration())
#         # vector = selector.get_parameter_vector()
#         fi = FeatureImportance(df_workload, MySQLConfiguration())
#         vector = fi.get_parameter_vector()
#         df.loc[mask, "label"] = df.loc[mask].apply(lambda x: vector, axis=1)
#     # for param in MySQLConfiguration().get_param_names():
#     #     df = df[df[param] == MySQLConfiguration().get_default_param_values()[param]]
#     # remove unnecessary columns
#     # df = df.drop(MySQLConfiguration().get_param_names(), axis=1)
    
#     # make a column that concatenates "num_table", "table_size", "num_client", "workload", "skew"
#     df["workload_label"] = df["num_table"].astype(str) + "-" + df["table_size"].astype(str) + "-" + df["num_client"].astype(str) + "-" + df["workload"].astype(str) + "-" + df["skew"].astype(str)
#     df = df.drop(["num_table", "table_size", "num_client", "workload", "skew"], axis=1)
    
#     # group is {num_cpu}c{mem_size}g
#     num_cpu = group.split("c")[0]
#     mem_size = group.split("c")[1].split("g")[0]
#     df["mem_size"] = mem_size
#     df["num_cpu"] = num_cpu
#     dfs.append(df)
#     df.to_csv(path_to_this_file + "/dataset/raw/train/" + file.split("/")[-1], index=False)
# # df = pd.concat(dfs)
# # df.to_csv(path_to_this_file + "/dataset/raw/train/" + "train.csv", index=False)    

        
    