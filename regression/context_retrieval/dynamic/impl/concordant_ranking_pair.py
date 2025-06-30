import glob
import os
import pandas as pd
import numpy as np

from regression.system_configuration import SystemConfiguration
from regression.context_retrieval import DynamicContextRetrieval, Context, ContextSimilarity
    

class ConcordantRankingPairRetrieval(DynamicContextRetrieval):
    """
    Class for ResTune dynamic similarity measure.
    """
    def __init__(self, system, data_dir):
        """
        Initialize the dynamic similarity measure with the directory containing dataset files.
        
        :param data_dir: Directory containing (parameter -> performance) CSV files.
        """
        self.system: SystemConfiguration = system
        self.data_dir = data_dir
        self.all_contexts = self._load_all_contexts()
        

    def retrieve_contexts(self, target_context: Context) -> list[ContextSimilarity]:
        """
        Get similar datasets based on the ResTune method.
        
        :param target_df: DataFrame containing the target dataset.
        :param target_wl: Workload label for the target dataset.
        :param target_hw: Hardware label for the target dataset.
        :return: List of similar datasets ranked by concordant pairs.
        """
        target_df = target_context.df
        target_wl = target_context.workload
        target_hw = target_context.hardware

        df_ranking = []
        for (hw, wl), candidate_df in self.all_contexts.items():
            if hw == target_hw or wl == target_wl:
                continue  # Skip same hardware or workload

            # Calculate concordant pairs
            concordant_pairs = self.calculate_concordant_pairs(candidate_df, target_df)
            
            df_ranking.append(ContextSimilarity(
                df=candidate_df,
                workload=wl,
                hardware=hw,
                distance=-concordant_pairs,
                similarity=concordant_pairs
            ))
                
        # Sort the DataFrame ranking by distance (negative of concordant pairs)
        np.random.shuffle(df_ranking)
        df_ranking.sort(key=lambda x: x.distance)
        return [df for df in df_ranking]
    
            
    def calculate_concordant_pairs(self, df, target_df):
        """
        Calculate the number of concordant pairs in the DataFrame.
        
        :param df: DataFrame containing the dataset.
        :return: Number of concordant pairs.
        """
        if len(df) < 2 or len(target_df) < 2:
            # If there are less than 2 rows, no pairs can be formed
            return 0

        perf_metric = self.system.get_perf_metric()
        
        merge_df = pd.merge(df, target_df, on=self.system.get_param_names(), suffixes=('_source', '_target'))
        p1 = merge_df[f'{perf_metric}_source'].values
        p2 = merge_df[f'{perf_metric}_target'].values
        diff1 = p1[:, None] - p1
        diff2 = p2[:, None] - p2

        # concordant ⇔ product of diffs  > 0   (ignores ties → 0)
        concordant = (diff1 * diff2) > 0

        # count *upper-triangular* part to avoid double-count
        return int(np.triu(concordant, k=1).sum())


    def _load_all_contexts(self) -> dict:
        """
        Load all candidate datasets, preprocess them, and group by (hardware, workload).
        Returns:
            dict[(hardware, workload)] = preprocessed context DataFrame
        """
        all_contexts = {}
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))

        for file_path in csv_files:
            if "train.csv" in file_path or "88c190g" in file_path:
                continue

            hardware = os.path.basename(file_path)
            df = pd.read_csv(file_path)

            for workload in df['workload_label'].unique():
                key = (hardware, workload)
                sub_df = df[df['workload_label'] == workload]
                sub_df = sub_df.groupby(self.system.get_param_names())[self.system.get_perf_metric()].mean().reset_index()
                sub_df = self.system.preprocess_param_values(sub_df)
                all_contexts[key] = sub_df

        return all_contexts
