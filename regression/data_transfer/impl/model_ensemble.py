

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.base import clone

from regression.data_transfer import DataTransfer
from regression.context_retrieval import ContextSimilarity


class ModelEnsemble(DataTransfer):
    """
    Class for model ensemble data transfer technique.
    It is only compatible with ConcordantRankingPairRetrieval.
    """
    def __init__(self, system, model):
        self.system = system
        self.model = model
        self.model_cache = {}
        super().__init__()
        
        
    class CacheEntry:
        def __init__(self, model, concordance):
            self.model = model
            self.concordance = concordance
            self.weights = 0
        

    def initialize(self, target_df):
        np.random.seed(self.seed)
        self.target_df = target_df
        self.target_data_population = target_df.groupby(self.system.get_param_names())[self.system.get_perf_metric()].mean().reset_index()
        self.sampled_data = pd.DataFrame()
        self.model_cache = {}
        self.iter = 0
        self.terminated = False
        
        
    def build_base_models(self, source_contexts: list[ContextSimilarity]) -> None:
        """
        Build base models from the source contexts.
        
        :param source_contexts: List of source contexts containing data for model training.
        """
        for context in source_contexts:
            src_df = context.df
            src_wl = context.workload
            src_hw = context.hardware
            src_sim = context.similarity
            key = (src_wl, src_hw)
            if key in self.model_cache:
                self.model_cache[key].concordance = src_sim
                continue
            
            src_df = src_df.groupby(self.system.get_param_names())[self.system.get_perf_metric()].mean().reset_index()
            parameters = self.system.get_param_names()
            perf_metric = self.system.get_perf_metric()
            self.model.fit(src_df[parameters], src_df[perf_metric])
            self.model_cache[key] = self.CacheEntry(self.model, src_sim)
    
    
    def receive_contexts(self, source_contexts: list[ContextSimilarity]) -> None:
        self.build_base_models(source_contexts)


    def run_next_iteration(self) -> None:
        while True:
            if len(self.target_data_population) == 0:
                print("No more data to sample")
                self.terminated = True
                return
            
            sampled_row = self.target_data_population.sample(n=1, random_state=self.seed)
                
            self.target_data_population = self.target_data_population.drop(sampled_row.index)
            duplicate = self.sampled_data[self.sampled_data.eq(sampled_row.iloc[0]).all(axis=1)]
            if len(duplicate) == 0:
                break
        self.sampled_data = pd.concat([self.sampled_data, sampled_row])


    def fit(self) -> None:
        # self.sampled_data: # DataFrame containing the collected target data for training
        # self.src_model_cache: # base learners for each source context
        parameters = self.system.get_param_names()
        perf_metric = self.system.get_perf_metric()
        
        X = self.sampled_data[parameters]
        y = self.sampled_data[perf_metric].to_numpy()
        
        loo = LeaveOneOut()
        
        y_pred = cross_val_predict(
            clone(self.model), X, y, cv=loo, method="predict"
        ).ravel()
        
        diff_pred  = y_pred[:, None] - y_pred[None, :]
        diff_true  = y[:, None] - y[None, :]

        # A pair is concordant if differences have the same sign
        concordant = (diff_pred * diff_true) > 0
        # Ignore diagonal & lower triangle
        target_concordance = int(np.triu(concordant, k=1).sum())
        
        self.model.fit(self.sampled_data[parameters], self.sampled_data[perf_metric])
        self.model_cache["target"] = self.CacheEntry(self.model, target_concordance)
        sum_concordance = sum(entry.concordance for entry in self.model_cache.values())
        for _, entry in self.model_cache.items():
            entry.weights = entry.concordance / sum_concordance
        

    def predict(self, df: pd.DataFrame) -> float:
        parameters = self.system.get_param_names()
        df = self.system.preprocess_param_values(df)
        pred = 0.0
        for _, entry in self.model_cache.items():
            model = entry.model
            pred += entry.weights * model.predict(df[parameters])
        return pred
